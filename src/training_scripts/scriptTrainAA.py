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



# display(HTML("<style>.container { width:40% !important; }</style>"))



def getAcc(preds,targets):
    return np.sum([1 if preds[i] == targets[i] else 0 for i in range(len(preds))])/len(preds)

def train(model, device, train_loader, optimizer, epoch, display=True):
    """
    Summary: Implements the training procedure for a given model
    == params ==
    model: the model to test
    device: cuda or cpu 
    optimizer: the optimizer for our training
    train_loader: dataloader for our train data
    display: output flag
    == output ==
    the mean train loss, the train accuracy
    """
    
    lossTracker = []
    
    targets=[]
    preds=[]
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        lossTracker.append(loss.detach())
        with torch.no_grad():
            pred = torch.argmax(output,1).cpu().numpy()
            preds.extend(pred)
            targets.extend(target.cpu().numpy())
        
    lossTracker = [x.item() for x in lossTracker]
    meanLoss = np.mean(lossTracker)
    accuracy = getAcc(preds,targets)
    if display:
        print('Train Epoch: {} [acc: {:.0f}%]\tLoss: {:.6f}'.format(
          epoch, 100. * accuracy, meanLoss))
        
    return accuracy, meanLoss


def glicoTrain(model, device, train_loader, optimizer, epoch, glicoLoader,replaceProb=0.5,display=True):
    """
    Summary: Implements the training procedure for a given model
    == params ==
    model: the model to test
    device: cuda or cpu 
    optimizer: the optimizer for our training
    train_loader: dataloader for our train data
    display: output flag
    == output ==
    the mean train loss, the train accuracy
    """
    
    lossTracker = []
    
    targets=[]
    preds=[]
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #replace samples with samples from glico with probability replaceprob
        data = glicoLoader.replaceBatch(data,target,replaceProb) 
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        lossTracker.append(loss.detach())
        with torch.no_grad():
            pred = torch.argmax(output,1).cpu().numpy()
            preds.extend(pred)
            targets.extend(target.cpu().numpy())
        
    lossTracker = [x.item() for x in lossTracker]
    meanLoss = np.mean(lossTracker)
    accuracy = getAcc(preds,targets)
    if display:
        print('Train Epoch: {} [acc: {:.0f}%]\tLoss: {:.6f}'.format(
          epoch, 100. * accuracy, meanLoss))
        
    return accuracy, meanLoss



def test(model, device, test_loader,verbose=True):
    """
    Summary: Implements the testing procedure for a given model
    == params ==
    model: the model to test
    device: cuda or cpu 
    test_loader: dataloader for our test data
    verbose: output flag
    == output ==
    the mean test loss, the test accuracy
    """
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    meanLoss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if verbose: print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        mean_test_loss, correct, len(test_loader.dataset),
        accuracy))
        
    return accuracy, meanLoss


def checkTest(model,device,valSets,valTracker,latexTracker,epoch,
              model_name,optim_name,lr,totalTestSamples,seed,augment,verbose=True):
    """
    Summary: checks the test accuracy, prints, and saves statistics
    """
    tempAcc = []
    tempLoss = []
    for val_loader in valSets:
        acc,loss = test(model, device, val_loader,verbose = False)
        tempAcc.append(acc)
        tempLoss.append(loss)
        
    meanAcc = np.mean(tempAcc)
    stdAcc = np.std(tempAcc)
    
    meanLoss = np.mean(tempLoss)
    if verbose:
        print('[Trained for {} epochs and tested on {} sets of 2000 images]\
        Avg Acc: {:.2f} +- {:.2f} , Avg Loss: {:.2f}'.format(
            epoch,VALIDATION_SET_NUM,meanAcc,stdAcc,meanLoss))
        
        
    tableRow = getLatexRow(architecture=model_name,epoch=epoch,accuracy=meanAcc,optim=optim_name,
                           lr=lr,totalTestSamples=totalTestSamples,dataAug=augment,
                           seed=seed,title=False)
    
    latexTracker.append(tableRow)
        
    valTracker["allLoss"].extend(tempLoss)
    valTracker["allAcc"].extend(tempAcc)
    valTracker["meanLoss"].append(meanLoss)
    valTracker["meanAcc"].append(meanAcc)
    valTracker["stdAcc"].append(stdAcc)



def getLatexRow(architecture,epoch,accuracy,optim,lr,
                totalTestSamples,dataAug,seed,title=False):
    """
    Summary: generates one row of latex for a results table
    """
    categories = ["Model","Epoch","Accuracy","Optimizer","lr","Test Sample Num",
                  "data augmentation","seed"]
    row = [str(architecture),str(epoch),str(round(accuracy,3)),str(optim),
           str(lr),str(totalTestSamples),str(dataAug),str(seed)]
    
    if title:
        c = "&".join(categories)
        r = "&".join(row)
        return "{}\\\\\n{}\\\\".format(c,r)
    else:
        r = "&".join(row)
        return "{}\\\\".format(r)
    
    
def plot(xlist,ylist,xlab,ylab,title,color,label,savedir=".",save=False):
    """
    Summary: plots the given list of numbers against its idices and 
    allows for high resolution saving
    """
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(xlist,ylist,color=color,marker=".",label=label)
    plt.legend()
    
    if save:
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        filepath = os.path.join(savedir,"{}".format(title))
        plt.savefig(filepath+".pdf")
        os.system("pdftoppm -png -r 300 {}.pdf {}.png".format(filepath,filepath))
        
    plt.show()
    

def getModel(model_name):
    if "wide" in model_name.lower():
        return WideResNet(28, 10, num_classes=10)
    elif "fix" in model_name.lower():
        return EfficientNet.from_pretrained(model_name) # change to not be pretrained
    
    
def getOptimizer128(optimizer_name,parameters):
    if "sgd" in  optimizer_name.lower():
        LR = 0.09
        optim = torch.optim.SGD(parameters, 
                                  lr=LR, momentum=0.9,
                                  weight_decay=0.0005)
        return optim, LR
    elif "adam" in optimizer_name.lower():
        LR = 0.001
        optim = torch.optim.Adam(parameters, 
                              lr=LR, weight_decay=0)
        return optim, LR
        


for seed in [301]:
    torch.cuda.empty_cache()
    gc.collect()

    OPTIM = "Adam"
    MODEL = "WideResNet28"
    EPOCH_NUM = 250
    TRAIN_SAMPLE_NUM = 100
    VAL_SAMPLE_NUM = 2000
    BATCH_SIZE = 128
    VALIDATION_SET_NUM = 5
    AUGMENT = False
    VAL_DISPLAY_DIVISOR = 25
    CIFAR_TRAIN = True
    REPLACE_PROB = 0.05
    SEED = seed

    #cifar-10:
    #mean = (0.4914, 0.4822, 0.4465)
    #std = (0.247, 0.243, 0.261)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.247, 0.243, 0.261])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])
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
    train_data, valSets, seed = ss.generateNewSet(device,valMultiplier = VALIDATION_SET_NUM,seed=SEED) #Sample from datasets










    import time
    import itertools





    model = getModel(MODEL).cuda()
    optimizer,LR = getOptimizer128(OPTIM,model.parameters())

    print(' => Total trainable parameters: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))        

    trainTracker = {"meanLoss":[],"accuracy":[]}
    valTracker = {"allLoss":[],"allAcc":[],"meanLoss":[],"meanAcc":[],"stdAcc":[]}
    latexTracker = []

    print("Begin Train for {} epochs".format(EPOCH_NUM))
    for epoch in range(EPOCH_NUM):
        acc, loss = train(model, device, train_data[0], optimizer, epoch+1, display=True)
        trainTracker["meanLoss"].append(loss)
        trainTracker["accuracy"].append(acc)
        
        if (epoch+1) % VAL_DISPLAY_DIVISOR == 0:
            checkTest(model,device,valSets,valTracker,latexTracker,epoch+1,
                model_name=MODEL,optim_name=OPTIM,lr=LR,totalTestSamples=VAL_SAMPLE_NUM*VALIDATION_SET_NUM,
                    seed=seed,augment=augment,verbose=True)
            
            
            
    dirname = latexTracker[-1][:-2] 

    def writeTex(latexTracker,dirname):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
            
        f= open(os.path.join(dirname,"latexTable.txt"),"w")
        for x in latexTracker:
            f.write(x+"\n")
        f.close()

    writeTex(latexTracker,dirname)

    for x in latexTracker:
        print(x)





    epochList = [x+1 for x in range(len(trainTracker["meanLoss"]))]

    plot(xlist=epochList,ylist=trainTracker["meanLoss"],xlab="Mean Train Loss",
        ylab="Epochs",title="Mean Train Loss over Epochs",
        color="#243A92",label="mean train loss",savedir=dirname,save=True)

    plot(xlist=epochList,ylist=trainTracker["accuracy"],xlab="Train Accuracy",
        ylab="Epochs",title="Train Accuracy Over Epochs",
        color="#34267E",label="Train Accuracy",savedir=dirname,save=True)





    epochList = [VAL_DISPLAY_DIVISOR*(x+1) for x in range(len(valTracker["meanLoss"]))]

    plot(xlist=epochList,ylist=valTracker["meanLoss"],xlab="Epochs",
        ylab="Mean Val Loss",title="Mean Val Loss over Epochs",
        color="#243A92",label="mean val loss",savedir=dirname,save=True)

    plot(xlist=epochList,ylist=valTracker["meanAcc"],xlab="Epochs",
        ylab="Val Accuracy",title="Val Accuracy Over Epochs",
        color="#34267E",label="Val Accuracy",savedir=dirname,save=True)

    plot(xlist=epochList,ylist=valTracker["stdAcc"],xlab="Epochs",
        ylab="Val Accuracy Standard Deviation",title="Val Accuracy Standard Deviation Over Epochs",
        color="#34267E",label="Val Accuracy SD",savedir=dirname,save=True)


    valSetEvalCount = VAL_DISPLAY_DIVISOR * EPOCH_NUM * VALIDATION_SET_NUM
    epochList = [VAL_DISPLAY_DIVISOR*(x+1) for x in range(len(valTracker["meanLoss"]))\
                for y in range(VALIDATION_SET_NUM)]


    plot(xlist=epochList,ylist=valTracker["allLoss"],xlab="Val Set Evaluations",
        ylab="Val Loss",title="Val loss over val set evaluations ({} \
    every {} epochs)".format(VALIDATION_SET_NUM,VAL_DISPLAY_DIVISOR),
        color="#34267E",label="Val Loss",savedir=dirname,save=True)

    plot(xlist=epochList,ylist=valTracker["allAcc"],xlab="Val Set Evaluations",
        ylab="Val Accuracy",title="Val loss over val set evaluations ({} \
    every {} epochs) ".format(VALIDATION_SET_NUM,VAL_DISPLAY_DIVISOR),
        color="#34267E",label="Val Accuracy",savedir=dirname,save=True)