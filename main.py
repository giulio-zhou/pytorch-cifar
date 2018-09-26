'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import pickle
import pprint as pp
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


def get_stat(data):
    # TODO: Add num backpropped
    stat = {}
    stat["average"] = np.average(data)
    stat["p50"] = np.percentile(data, 50)
    stat["p75"] = np.percentile(data, 75)
    stat["p90"] = np.percentile(data, 90)
    stat["max"] = max(data)
    stat["min"] = min(data)
    return stat


def update_batch_stats(batch_stats, num_backpropped, num_skipped,
                       pool_losses=None, chosen_losses=None, pool_sps=None, chosen_sps=None):
    '''
    batch_stats = [{'chosen_losses': {stat},
                   'pool_losses': {stat}}]
    '''
    snapshot = {}
    snapshot["num_backpropped"] = num_backpropped
    snapshot["num_skipped"] = num_skipped
    if chosen_losses:
        snapshot["chosen_losses"] = get_stat(chosen_losses)
    if pool_losses:
        snapshot["pool_losses"] = get_stat(pool_losses)
    if chosen_sps:
        snapshot["chosen_sps"] = get_stat(chosen_sps)
    if pool_sps:
        snapshot["pool_sps"] = get_stat(pool_sps)
    batch_stats.append(snapshot)

# Training
def train_topk(args,
               net,
               trainloader,
               device,
               optimizer,
               epoch,
               total_num_images_backpropped,
               total_num_images_skipped,
               images_hist,
               batch_stats = None):

    print('\nEpoch: %d in train_topk' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    num_backprop = 0
    num_skipped = 0
    loss_reduction = None

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        output = net(data)
        loss = nn.CrossEntropyLoss(reduce=True)(output, targets)

        losses_pool.append(loss.item())
        data_pool.append(data)
        targets_pool.append(targets)
        ids_pool.append(image_id.item())

        if len(losses_pool) == args.pool_size:

            # Choose frames from pool to backprop
            indices = np.array(losses_pool).argsort()[-args.top_k:]
            chosen_data = [data_pool[i] for i in indices]
            chosen_targets = [targets_pool[i] for i in indices]
            chosen_ids = [ids_pool[i] for i in indices]
            chosen_losses = [losses_pool[i] for i in indices]
            num_skipped += len(data_pool) - len(chosen_data)

            data_batch = torch.stack(chosen_data, dim=1)[0]
            targets_batch = torch.cat(chosen_targets)
            output_batch = net(data_batch) # redundant

            for chosen_id in chosen_ids:
                images_hist[chosen_id] += 1

            # Get stats for batches
            if batch_stats is not None:
                update_batch_stats(batch_stats,
                                   total_num_images_backpropped,
                                   total_num_images_skipped,
                                   pool_losses = losses_pool, 
                                   chosen_losses = chosen_losses,
                                   pool_sps = [],
                                   chosen_sps = [])

            # Note: This will only work for batch size of 1
            loss_reduction = nn.CrossEntropyLoss(reduce=True)(output_batch, targets_batch)

            optimizer.zero_grad()
            loss_reduction.backward()
            optimizer.step()
            train_loss += loss_reduction.item()
            num_backprop += len(chosen_data)

            losses_pool = []
            data_pool = []
            targets_pool = []
            ids_pool = []

            output = output_batch
            targets = targets_batch

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and loss_reduction is not None:
            print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        total_num_images_backpropped + num_backprop,
                        total_num_images_skipped + num_skipped,
                        loss_reduction.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

        # Stop epoch rightaway if we've exceeded maximum number of epochs
        if args.max_num_backprops:
            if args.max_num_backprops <= total_num_images_backpropped + num_backprop:
                return num_backprop

    return num_backprop, num_skipped


# Training
def train_sampling(args,
                   net,
                   trainloader,
                   device,
                   optimizer,
                   epoch,
                   total_num_images_backpropped,
                   total_num_images_skipped,
                   images_hist,
                   batch_stats = None):

    print('\nEpoch: %d in train_sampling' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    sps_pool = []
    chosen_data = []
    chosen_losses = []
    chosen_targets = []
    chosen_ids = []
    chosen_sps = []
    num_backprop = 0
    num_skipped = 0
    select_probs = -1
    loss_reduction = None

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        output = net(data)
        loss = nn.CrossEntropyLoss(reduce=True)(output, targets)

        # Prepare output for L2 distance
        softmax_output = nn.Softmax()(output)
        #print("Softmax: ", softmax_output)

        # Prepare target for L2 distance
        target_vector = np.zeros(len(output.data[0]))
        target_vector[targets.item()] = 1
        target_tensor = torch.Tensor(target_vector)
        #print("Target: ", target_tensor)

        l2_dist = torch.dist(target_tensor.to(device), softmax_output)
        #print("L2 Dist: ", l2_dist.item())

        l2_dist *= l2_dist
        #print("L2 Dist Squared: ", l2_dist.item())

        select_probs = torch.clamp(l2_dist, min=args.sampling_min, max=1)
        #print("Chosen Probs: ", select_probs.item())

        draw = np.random.uniform(0, 1)
        if draw < select_probs.item() or epoch == 0:
            # Do the backprop
            if epoch == 0:
                # Don't do importance sampling on first epoch
                loss_normalized = loss
            else:
                loss_normalized = loss / select_probs.item()

            optimizer.zero_grad()
            loss_normalized.backward()
            optimizer.step()
            train_loss += loss_normalized.item()
            num_backprop += 1
            images_hist[image_id.item()] += 1

            # Add to batch for logging purposes
            chosen_losses.append(loss.item())
            chosen_data.append(data)
            chosen_targets.append(targets)
            chosen_ids.append(image_id.item())
            chosen_sps.append(select_probs.item())

        else:
            num_skipped += 1
            print("Skipping image with sp {}".format(select_probs))

        # Add to batch for logging purposes
        losses_pool.append(loss.item())
        data_pool.append(data.data[0])
        targets_pool.append(targets)
        ids_pool.append(image_id.item())
        sps_pool.append(select_probs.item())

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and num_backprop > 0:
            print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        total_num_images_backpropped + num_backprop,
                        total_num_images_skipped + num_skipped,
                        np.average(losses_pool),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

            # Record stats for batch
            if batch_stats is not None:
                update_batch_stats(batch_stats,
                                   total_num_images_backpropped,
                                   total_num_images_skipped,
                                   pool_losses = losses_pool, 
                                   chosen_losses = chosen_losses,
                                   pool_sps = sps_pool,
                                   chosen_sps = chosen_sps)

            losses_pool = []
            data_pool = []
            targets_pool = []
            ids_pool = []
            sps_pool = []

            chosen_data = []
            chosen_losses = []
            chosen_targets = []
            chosen_ids = []
            chosen_sps = []

        # Stop epoch rightaway if we've exceeded maximum number of epochs
        if args.max_num_backprops:
            if args.max_num_backprops <= total_num_images_backpropped + num_backprop:
                return num_backprop

    return num_backprop, num_skipped

def test(args,
         net,
         testloader,
         device,
         epoch,
         total_num_images_backpropped,
         total_num_images_skipped):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.nll_loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                total_num_images_backpropped,
                total_num_images_skipped,
                test_loss,
                100.*correct/total,
                time.time()))

    # Save checkpoint.
    '''
    global best_acc
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    '''


def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')

    parser.add_argument('--sb-strategy', default="topk", metavar='N',
                        help='Selective backprop strategy among {topk, sampling}')
    parser.add_argument('--sampling-min', type=float, default=0.05,
                        help='Minimum sampling rate for sampling strategy')
    parser.add_argument('--top-k', type=int, default=8, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pool-size', type=int, default=16, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--max-num-backprops', type=int, default=None, metavar='N',
                        help='how many images to backprop total')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    ## Selective backprop setup ##

    assert(args.pool_size >= args.top_k)

    # Partition training set to get more test datapoints
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainset = [t + (i,) for i, t in enumerate(trainset)]
    chunk_size = args.pool_size * 10
    partitions = [trainset[i:i + chunk_size] for i in xrange(0, len(trainset), chunk_size)]

    # Store frequency of each image getting backpropped
    keys = range(len(trainset))
    images_hist = dict(zip(keys, [0] * len(keys)))
    batch_stats = []

    # Make images hist pickle path
    image_id_pickle_dir = os.path.join(args.pickle_dir, "image_id_hist")
    if not os.path.exists(image_id_pickle_dir):
        os.mkdir(image_id_pickle_dir)
    image_id_pickle_file = os.path.join(image_id_pickle_dir,
                                        "{}_images_hist.pickle".format(args.pickle_prefix))

    # Make batch stats pickle path
    batch_stats_pickle_dir = os.path.join(args.pickle_dir, "batch_stats")
    if not os.path.exists(batch_stats_pickle_dir):
        os.mkdir(batch_stats_pickle_dir)
    batch_stats_pickle_file = os.path.join(batch_stats_pickle_dir,
                                           "{}_batch_stats.pickle".format(args.pickle_prefix))

    total_num_images_backpropped = 0
    total_num_images_skipped = 0

    for epoch in range(start_epoch, start_epoch+500):
        for partition in partitions:
            trainloader = torch.utils.data.DataLoader(partition, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test(args, net, testloader, device, epoch, total_num_images_backpropped, total_num_images_skipped)

            # Stop training rightaway if we've exceeded maximum number of epochs
            if args.max_num_backprops:
                if args.max_num_backprops <= total_num_images_backpropped:
                    return

            if args.sb_strategy == "topk":
                trainer = train_topk
            elif args.sb_strategy == "sampling":
                trainer = train_sampling
            else:
                print("Unknown selective backprop strategy {}".format(args.sb_strategy))
                exit()

            num_images_backpropped, num_images_skipped = trainer(args,
                                                               net,
                                                               trainloader,
                                                               device,
                                                               optimizer,
                                                               epoch,
                                                               total_num_images_backpropped,
                                                               total_num_images_skipped,
                                                               images_hist,
                                                               batch_stats=batch_stats)

            total_num_images_backpropped += num_images_backpropped
            total_num_images_skipped += num_images_skipped

            # Write out summary statistics

            with open(image_id_pickle_file, "wb") as handle:
                print(image_id_pickle_file)
                pickle.dump(images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(batch_stats_pickle_file, "wb") as handle:
                print(batch_stats_pickle_file)
                pickle.dump(batch_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
