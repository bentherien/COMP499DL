#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from sys import argv
from torchvision import datasets, transforms


def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if display:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

if __name__=="__main__":
    if len(argv)==1:
        input_dir = '.'
        output_dir = '.'
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)


    ##################### YOUR CODE GOES HERE
    os.system("pip install scipy")
    os.system("pip install --upgrade pillow")


    from torchvision import datasets, transforms
    from wide_resnet import WideResNet
    from auto_augment import AutoAugment, Cutout
    import os
    import sys

    ### Preparation
    print("[Preparation] Start...")
    # select device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset: normalize and convert to tensor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    dataAugmentation = [ 
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    transform_train = transforms.Compose([transforms.ToTensor(), normalize] + dataAugmentation) 

    # dataset: load cifar10 data
    print(os.listdir(os.path.join(input_dir)))
    cifar_data = torchvision.datasets.ImageFolder(root=os.path.join(input_dir, 'train'), transform=transform_train)

    # dataset: initialize dataloaders for train and validation set
    train_loader = torch.utils.data.DataLoader(cifar_data, batch_size=128, shuffle=True)

    # model: initialize model
    model = WideResNet(50, 20, num_classes=10)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.1, momentum=0.9,
                                weight_decay=0.0005)
    print("[Preparation] Done")




    ### Training
    # model: training loop
    print(type(train_loader), file=sys.stderr)

    print("[Training] Start...\n")
    for epoch in range(140):
        train(model, device, train_loader, optimizer, epoch, display=True)
    print("\n[Training] Done")









    ##################### END OF YOUR CODE


    ### Saving Outputs
    print("[Saving Outputs] Start...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # test evaluation: make predictions
    print("[Saving Outputs] Test set...")
    test_data = torchvision.datasets.ImageFolder(root=os.path.join(input_dir, 'test'), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_predictions.extend(pred.squeeze().cpu().tolist())

    # test evaluation: save predictions
    test_str = '\n'.join(list(map(str, test_predictions)))
    with open(os.path.join(output_dir, 'answer_test.txt'), 'w') as result_file:
        result_file.write(test_str)
    print("[Saving Outputs] Done")

    print("All done!")

