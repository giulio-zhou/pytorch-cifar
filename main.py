'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import cPickle as pickle
import json
import pprint as pp
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import random

from models import *
from utils import progress_bar

import lib.backproppers
import lib.datasets
import lib.loggers
import lib.selectors

def set_random_seeds(seed):
    if seed:
        print("Setting static random seeds to {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    return

def get_stat(data):
    stat = {}
    stat["average"] = np.average(data)
    stat["p50"] = np.percentile(data, 50)
    stat["p75"] = np.percentile(data, 75)
    stat["p90"] = np.percentile(data, 90)
    stat["max"] = max(data)
    stat["min"] = min(data)
    return stat

class State:

    def __init__(self, num_images, pickle_dir, pickle_prefix):
        self.num_images_backpropped = 0
        self.num_images_skipped = 0
        self.num_images_seen = 0
        self.sum_sp = 0
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix

        self.init_target_confidences()

    def init_target_confidences(self):
        self.target_confidences = {}

        target_confidences_pickle_dir = os.path.join(self.pickle_dir, "target_confidences")
        self.target_confidences_pickle_file = os.path.join(target_confidences_pickle_dir,
                                                           "{}_target_confidences.pickle".format(self.pickle_prefix))

        # Make images hist pickle path
        if not os.path.exists(target_confidences_pickle_dir):
            os.mkdir(target_confidences_pickle_dir)

    def update_target_confidences(self, epoch, confidences, num_images_backpropped):
        if epoch not in self.target_confidences.keys():
            self.target_confidences[epoch] = {"confidences": []}
        self.target_confidences[epoch]["confidences"] += confidences
        self.target_confidences[epoch]["num_backpropped"] = num_images_backpropped

    def write_summaries(self):
        with open(self.target_confidences_pickle_file, "wb") as handle:
            print(self.target_confidences_pickle_file)
            pickle.dump(self.target_confidences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_sum_sp(self, sp):
        self.num_images_seen += 1
        self.sum_sp += sp

    @property
    def average_sp(self):
        if self.num_images_seen == 0:
            return 1
        return self.sum_sp / float(self.num_images_seen)


class Example(object):
    # TODO: Add ExampleCollection class
    def __init__(self,
                 loss=None,
                 softmax_output=None,
                 target=None,
                 datum=None,
                 image_id=None,
                 select_probability=None):
        self.loss = loss.detach()
        self.softmax_output = softmax_output.detach()
        self.target = target.detach()
        self.datum = datum.detach()
        self.image_id = image_id.detach()
        self.select_probability = select_probability
        self.backpropped_loss = None   # Populated after backprop

    @property
    def predicted(self):
        _, predicted = self.softmax_output.max(0)
        return predicted

    @property
    def is_correct(self):
        return self.predicted.eq(self.target)


class Trainer(object):
    def __init__(self,
                 device,
                 net,
                 selector,
                 backpropper,
                 batch_size,
                 max_num_backprops=float('inf'),
                 lr_schedule=None):
        self.device = device
        self.net = net
        self.selector = selector
        self.backpropper = backpropper
        self.batch_size = batch_size
        self.backprop_queue = []
        self.forward_pass_handlers = []
        self.backward_pass_handlers = []
        self.global_num_backpropped = 0
        self.max_num_backprops = max_num_backprops
        self.on_backward_pass(self.update_num_backpropped)
        if lr_schedule:
            self.load_lr_schedule(lr_schedule)
            self.on_backward_pass(self.update_learning_rate)

    def update_num_backpropped(self, batch):
        self.global_num_backpropped += sum([1 for e in batch if e.select])

    def on_forward_pass(self, handler):
        self.forward_pass_handlers.append(handler)

    def on_backward_pass(self, handler):
        self.backward_pass_handlers.append(handler)

    def emit_forward_pass(self, batch):
        for handler in self.forward_pass_handlers:
            handler(batch)

    def emit_backward_pass(self, batch):
        for handler in self.backward_pass_handlers:
            handler(batch)

    # TODO move to a LRScheduler object or to backpropper
    def load_lr_schedule(self, schedule_path):
        with open(schedule_path, "r") as f:
            data = json.load(f)
        self.lr_schedule = {}
        for k in data:
            self.lr_schedule[int(k)] = data[k]

    def set_learning_rate(self, lr):
        print("Setting learning rate to {} at {} backprops".format(lr,
                                                                   self.global_num_backpropped))
        for param_group in self.backpropper.optimizer.param_groups:
            param_group['lr'] = lr

    def update_learning_rate(self, batch):
        for start_num_backprop in reversed(sorted(self.lr_schedule)):
            lr = self.lr_schedule[start_num_backprop]
            if self.global_num_backpropped >= start_num_backprop:
                if self.backpropper.optimizer.param_groups[0]['lr'] is not lr:
                    self.set_learning_rate(lr)
                break

    @property
    def stopped(self):
        return self.global_num_backpropped >= self.max_num_backprops

    def train(self, trainloader):
        for i, batch in enumerate(trainloader):
            if self.stopped: break
            if i == len(trainloader) - 1:
                self.train_batch(batch, final=True)
            else:
                self.train_batch(batch, final=False)

    def train_batch(self, batch, final):
        forward_pass_batch = self.forward_pass(*batch)
        annotated_forward_batch = self.selector.mark(forward_pass_batch)
        self.emit_forward_pass(annotated_forward_batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)

    def forward_pass(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)

        self.net.eval()
        with torch.no_grad():
            outputs = self.net(data)

        losses = nn.CrossEntropyLoss(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)

        examples = zip(losses, softmax_outputs, targets, data, image_ids)
        return [Example(*example) for example in examples]

    def get_batch(self, final):
        num_images_to_backprop = 0
        for index, example in enumerate(self.backprop_queue):
            num_images_to_backprop += int(example.select)
            if num_images_to_backprop == self.batch_size:
                # Note: includes item that should and shouldn't be backpropped
                backprop_batch = self.backprop_queue[:index+1]
                self.backprop_queue = self.backprop_queue[index+1:]
                return backprop_batch
        if final:
            backprop_batch = self.backprop_queue
            self.backprop_queue = []
            return backprop_batch
        return None


def test(args,
         net,
         testloader,
         device,
         epoch,
         state,
         logger):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, image_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            softmax_outputs = nn.Softmax()(outputs)
            targets_array = targets.cpu().numpy()
            outputs_array = softmax_outputs.cpu().numpy()
            confidences = [o[t] for t, o in zip(targets_array, outputs_array)]
            state.update_target_confidences(epoch,
                                            confidences[:10],
                                            logger.global_num_backpropped)

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                logger.global_num_backpropped,
                logger.global_num_skipped,
                test_loss,
                100.*correct/total,
                time.time()))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    net_state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'num_backpropped': logger.global_num_backpropped,
    }
    checkpoint_dir = os.path.join(args.pickle_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, args.pickle_prefix + "_ckpt.t7")
    print("Saving checkpoint at {}".format(checkpoint_file))
    torch.save(net_state, checkpoint_file)


def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-sched', default=None, help='Path to learning rate schedule')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--augment', '-a', dest='augment', action='store_true',
                        help='turn on data augmentation for CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--dataset', default="cifar10", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--write-images', default=False, type=bool,
                        help='whether or not write png images by id')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for randomization; None to not set seed')

    parser.add_argument('--sb-strategy', default="deterministic", metavar='N',
                        help='Selective backprop strategy among {baseline, deterministic, sampling}')
    parser.add_argument('--sb-start-epoch', type=int, default=0,
                        help='epoch to start selective backprop')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--max-num-backprops', type=int, default=float('inf'), metavar='N',
                        help='how many images to backprop total')

    parser.add_argument('--sampling-strategy', default="square", metavar='N',
                        help='Selective backprop sampling strategy among {recenter, translate, nosquare, square}')
    parser.add_argument('--sampling-min', type=float, default=1,
                        help='Minimum sampling rate for sampling strategy')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    set_random_seeds(args.seed)

    # Model
    print('==> Building model..')
    if args.net == "resnet":
        net = ResNet18()
    elif args.net == "vgg":
        net = VGG('VGG19')
    elif args.net == "preact_resnet":
        net = PreActResNet18()
    elif args.net == "googlenet":
        net = GoogLeNet()
    elif args.net == "densenet":
        net = DenseNet121()
    elif args.net == "resnext":
        net = ResNeXt29_2x64d()
    elif args.net == "mobilenet":
        net = MobileNet()
    elif args.net == "mobilenet_v2":
        net = MobileNetV2()
    elif args.net == "dpn":
        net = DPN92()
    elif args.net == "shufflenet":
        net = ShuffleNetG2()
    elif args.net == "senet":
        net = SENet18()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint_dir = os.path.join(args.pickle_dir, "checkpoint")
        checkpoint_file = os.path.join(checkpoint_dir, args.pickle_prefix + "_ckpt.t7")
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        print("Loading checkpoint at {}".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    if args.dataset == "cifar10":
        dataset = lib.datasets.CIFAR10(net, args.test_batch_size, args.batch_size * 100, args.augment)
    elif args.dataset == "mnist":
        dataset = lib.datasets.MNIST(device, args.test_batch_size, args.batch_size * 100)
    else:
        print("Only cifar10 is implemented")
        exit()

    optimizer = optim.SGD(dataset.model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.decay)

    state = State(dataset.num_training_images, args.pickle_dir, args.pickle_prefix)
    if args.write_images:
        image_writer = lib.loggers.ImageWriter('./data', args.dataset, dataset.unnormalizer)
        for partition in dataset.partitions:
            image_writer.write_partition(partition)

    ## Setup Trainer ##
    square = args.sampling_strategy in ["square", "translate", "recenter"]
    translate = args.sampling_strategy in ["translate", "recenter"]
    recenter = args.sampling_strategy == "recenter"

    probability_calculator = lib.selectors.SelectProbabiltyCalculator(args.sampling_min,
                                                                      len(dataset.classes),
                                                                      device,
                                                                      square=square,
                                                                      translate=translate)
    if args.sb_strategy == "sampling":
        final_selector = lib.selectors.SamplingSelector(probability_calculator)
        final_backpropper = lib.backproppers.SamplingBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 recenter=recenter)
    elif args.sb_strategy == "deterministic":
        final_selector = lib.selectors.DeterministicSamplingSelector(probability_calculator,
                                                                     initial_sum=1)
        final_backpropper = lib.backproppers.SamplingBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 recenter=recenter)
    elif args.sb_strategy == "baseline":
        final_selector = lib.selectors.BaselineSelector()
        final_backpropper = lib.backproppers.BaselineBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer)
    else:
        print("Use sb-strategy in {sampling, deterministic, baseline}")
        exit()

    selector = lib.selectors.PrimedSelector(lib.selectors.BaselineSelector(),
                                            final_selector,
                                            args.sb_start_epoch)

    backpropper = lib.backproppers.PrimedBackpropper(lib.backproppers.BaselineBackpropper(device,
                                                                                          dataset.model,
                                                                                          optimizer),
                                                     final_backpropper,
                                                     args.sb_start_epoch)

    trainer = Trainer(device,
                      dataset.model,
                      selector,
                      backpropper,
                      args.batch_size,
                      max_num_backprops=args.max_num_backprops,
                      lr_schedule=args.lr_sched)
    logger = lib.loggers.Logger(log_interval = args.log_interval)
    image_id_hist_logger = lib.loggers.ImageIdHistLogger(args.pickle_dir,
                                                         args.pickle_prefix,
                                                         dataset.num_training_images)
    probability_by_image_logger = lib.loggers.ProbabilityByImageLogger(args.pickle_dir,
                                                                       args.pickle_prefix)
    trainer.on_forward_pass(logger.handle_forward_batch)
    trainer.on_backward_pass(logger.handle_backward_batch)
    trainer.on_backward_pass(image_id_hist_logger.handle_backward_batch)
    trainer.on_backward_pass(probability_by_image_logger.handle_backward_batch)
    stopped = False


    for epoch in range(start_epoch, start_epoch+5000):

        if stopped: break

        trainloader = torch.utils.data.DataLoader(dataset.trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        test(args, dataset.model, dataset.testloader, device, epoch, state, logger)

        trainer.train(trainloader)
        logger.next_partition()
        if trainer.stopped:
            stopped = True
            break

        logger.next_epoch()
        image_id_hist_logger.next_epoch()
        probability_by_image_logger.next_epoch()
        selector.next_epoch()
        backpropper.next_epoch()

if __name__ == '__main__':
    main()
