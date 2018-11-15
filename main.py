'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import random
import argparse

from models import *
from utils import progress_bar

DEBUG = False

def set_random_seeds(seed):
    if seed:
        print("Setting static random seeds to {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    return

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--augment', '-a', action='store_true', help='Turn on data augmentation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', '-e', default=150, type=int, help='number of epochs')
parser.add_argument('--checkpoint', '-c', default="ckpt.t7", type=str, help='checkpoint filename')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for randomization; None to not set seed')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

global_num_backpropped = 0 # start from epoch 0 or last checkpoint num_backpropped

# Data
print('==> Preparing data..')
if args.augment:
    print("Performing data augmentation on CIFAR10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    print("Not performing data augmentation on CIFAR10")
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.{}.t7'.format(args.checkpoint))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    global_num_backpropped = checkpoint['num_backpropped']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    global global_num_backpropped
    global DEBUG
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if DEBUG:
            print("[DEBUG train] output:", outputs.data.cpu().numpy()[0])
            print("[DEBUG train] targets:", targets.data.cpu().numpy()[0])
            print("[DEBUG train] loss:", loss.item())

        loss.backward()
        optimizer.step()
        global_num_backpropped += len(inputs)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # DEBUG
        # if batch_idx % 10 == 0:
        print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
            epoch,
            global_num_backpropped,
            0,
            train_loss/(batch_idx+1),
            train_loss/(batch_idx+1),
            0,
            100.*correct/total))
    if DEBUG:
        s = torch.sum(net.module.conv1.weight.data)
        print("[DEBUG train] Weight sum:", s.item())
        s = torch.sum(net.module.bn1.weight.data)
        print("[DEBUG train] Weight sum:", s.item())
        s = torch.sum(net.module.linear.weight.data)
        print("[DEBUG train] Weight sum:", s.item())

def test(epoch):
    global best_acc
    global global_num_backpropped
    global DEBUG
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if DEBUG:
        s = torch.sum(net.module.conv1.weight.data)
        print("[DEBUG test] Weight sum:", s.item())
        s = torch.sum(net.module.bn1.weight.data)
        print("[DEBUG test] Weight sum:", s.item())
        s = torch.sum(net.module.linear.weight.data)
        print("[DEBUG test] Weight sum:", s.item())

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            if DEBUG:
                print("[DEBUG test] output:", outputs.data.cpu().numpy()[-1])
                print("[DEBUG test] BS:", len(inputs.data.cpu().numpy()))
                print("[DEBUG test] target:", targets.data.cpu().numpy()[-1])
                print("[DEBUG test] loss:", loss.item())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
        epoch,
        global_num_backpropped,
        0,
        test_loss / len(testloader.dataset),
        100.*correct/total,
        0))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch + 1,
        'num_backpropped': global_num_backpropped,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.{}.t7'.format(args.checkpoint))
    best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
