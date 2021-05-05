import PIL
import gc
import torch
import torchvision
import os
import sys

import numpy as np
import matplotlib.pyplot as plt  
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import Subset
from IPython.core.display import display, HTML
from numpy.random import RandomState
from wide_resnet import WideResNet
from auto_augment import AutoAugment, Cutout
from efficientnet_pytorch import EfficientNet
from cifar_loader import SmallSampleController

sys.path.insert(0,'glico-learning-small-sample/glico_model')

from tester import runGlico



torch.cuda.empty_cache()
gc.collect()

OPTIM = "Adam"
MODEL = "WideResNet28"
EPOCH_NUM = 1000
TRAIN_SAMPLE_NUM = 100
VAL_SAMPLE_NUM = 2000
BATCH_SIZE = 32
VALIDATION_SET_NUM = 5
AUGMENT = True
VAL_DISPLAY_DIVISOR = 4
CIFAR_TRAIN = False

#cifar-10:
#mean = (0.4914, 0.4822, 0.4465)
#std = (0.247, 0.243, 0.261)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
if AUGMENT:
    dataAugmentation = [ 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(),
        Cutout()
    ]
    augment = "Crop,Flip,AutoAugment,Cutout"
else: 
    dataAugmentation = []
    augment = "Nothing"



transform_train = transforms.Compose(dataAugmentation + [transforms.ToTensor(), normalize]) 
transform_val = transforms.Compose([transforms.ToTensor(), normalize]) #careful to keep this one same

cifar_train = datasets.CIFAR10(root='.',train=CIFAR_TRAIN, transform=transform_train, download=True)
cifar_val = datasets.CIFAR10(root='.',train=CIFAR_TRAIN, transform=transform_val, download=True)

ss = SmallSampleController(numClasses=10,trainSampleNum=TRAIN_SAMPLE_NUM, # abstract the data-loading procedure
                           valSampleNum=VAL_SAMPLE_NUM, batchSize=BATCH_SIZE, 
                           multiplier=VALIDATION_SET_NUM, trainDataset=cifar_train, 
                           valDataset=cifar_val)
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, valSets, seed = ss.generateNewSet(device,valMultiplier = VALIDATION_SET_NUM) #Sample from datasets


nagTrainer = runGlico(train_labeled_dataset=ss.trainSampler.subsets[0], classes=10)