import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import lib.cifar
import lib.mnist

class CIFAR10:
    def __init__(self, model, test_batch_size, augment, shuffle_labels):

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.model = model
        self.augment = augment

        # Testing set
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = lib.cifar.CIFAR10(root='./data',
                                    train=False,
                                    download=False,
                                    transform=transform_test,
                                    shuffle_labels=shuffle_labels)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        # Training set
        if self.augment:
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
        self.trainset = lib.cifar.CIFAR10(root='./data',
                                          train=True,
                                          download=False,
                                          transform=transform_train,
                                          shuffle_labels=shuffle_labels)

        self.num_training_images = len(self.trainset)

        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                       transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                            std = [ 1., 1., 1. ])
                                      ])

class MNIST:
    def __init__(self, device, test_batch_size):

        self.model = MNISTNet().to(device)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # Testing set
        self.testloader = torch.utils.data.DataLoader(
            lib.mnist.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=test_batch_size, shuffle=False, num_workers=0)

        # Training set
        self.trainset = lib.mnist.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.unnormalizer = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.3081]),
                                       transforms.Normalize(mean = [ -0.1307],
                                                            std = [ 1., 1., 1. ])
                                      ])

        self.num_training_images = len(self.trainset)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import torchvision.datasets as datasets
class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        retval = super(datasets.ImageFolder, self).__getitem__(index)
        return retval + (index,)

class TinyImageNet:
    def __init__(self, model, test_batch_size, partition_size, augment):
        self.model = model
        self.augment = augment

        # Data loading code
        import glob
        import numpy as np
        import os
        root_dir = 'tiny_imagenet_data'
        traindir = os.path.join(root_dir, 'train')
        testdir = os.path.join(root_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.classes = sorted(glob.glob(traindir + '/*'))

        # Training set
        if self.augment:
            print("Performing data augmentation on Tiny ImageNet")
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0., 0., 0.), (1.0, 1.0, 1.0)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not performing data augmentation on Tiny ImageNet")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0., 0., 0.), (1.0, 1.0, 1.0)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset = IndexedImageFolder(traindir,
                                                    transform_train)
        self.num_training_images = len(self.trainset)

        # Test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1.0, 1.0, 1.0)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = IndexedImageFolder(testdir,
                                              transform_test)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0)

        self.unnormalizer = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                 std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
            transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                 std = [ 1., 1., 1. ])
        ])

