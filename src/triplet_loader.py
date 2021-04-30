from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
import torch

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from numpy.random import RandomState
from torch.utils.data import Subset



import time

class TripletController:
    class Triplet:
        """Class designed to manage the batch generation of one anchor form one dataset"""
        def __init__(self,anchor,mnist,mnistDataset,svhnDataset):
            self.anchor = anchor
            self.mnist = mnist
            prng = RandomState(int(time.time()))
            
            if self.mnist:
                self.anchorIndexer = prng.permutation(np.arange(0, len(mnistDataset[self.anchor])))
                self.posIndexer = prng.permutation(np.arange(0, len(svhnDataset[self.anchor])))
                self.negIndexer = {k:prng.permutation(np.arange(0,len(v))) for k,v in svhnDataset.items()}
            else:                
                self.anchorIndexer = prng.permutation(np.arange(0, len(svhnDataset[self.anchor])))
                self.posIndexer = prng.permutation(np.arange(0, len(mnistDataset[self.anchor])))
                self.negIndexer = {k:prng.permutation(np.arange(0,len(v))) for k,v in mnistDataset.items()}
                
            self.negIndex = {k:0 for k in range(10)}
            self.posIndex = 0
            self.anchorIndex = 0
                
        def sampleAnchor(self,dataset):
            """sample from the anchors dataset, and reset the sample permutation if we wrap around"""
            if self.anchorIndex == len(self.anchorIndexer):
                prng = RandomState(int(time.time()))
                self.anchorIndexer = prng.permutation(np.arange(0, len(self.anchorIndexer)))
                self.anchorIndex = 0
                
            temp = dataset[self.anchor][self.anchorIndexer[self.anchorIndex]]
            self.anchorIndex += 1
            return temp
                
        def samplePos(self,dataset):
            """sample from the positive dataset, and reset the sample permutation if we wrap around"""
            
            if self.posIndex == len(self.posIndexer):
                prng = RandomState(int(time.time()))
                self.posIndexer = prng.permutation(np.arange(0, len(self.posIndexer)))
                self.posIndex = 0
                
            temp = dataset[self.anchor][self.posIndexer[self.posIndex]]
            self.posIndex += 1
            return temp
        
        def sampleNeg(self,sampleNum,dataset):
            """sample from the negative dataset, and reset the sample permutation if we wrap around"""
            
            if self.negIndex[sampleNum] == len(self.negIndexer):
                prng = RandomState(int(time.time()))
                self.negIndexer[sampleNum] = prng.permutation(np.arange(0, len(self.negIndexer[sampleNum])))
                self.negIndex[sampleNum] = 0
            
            temp = dataset[sampleNum][self.negIndexer[sampleNum][self.negIndex[sampleNum]]] #sample from dataset based on index
            self.negIndex[sampleNum] += 1 # increment index
            return temp
            
                
        
        def sampleBatch(self,mnistDataset,svhnDataset,sampleMul=1):
            anchors = []
            posExs = []
            negExs = []
            for _ in range(sampleMul):
                for x in [z for z in range(10) if z != self.anchor]:
                    if self.mnist:
                        anchors.append(self.sampleAnchor(mnistDataset))
                        posExs.append(self.samplePos(svhnDataset))
                        negExs.append(self.sampleNeg(x,svhnDataset))
                    else:
                        anchors.append(self.sampleAnchor(svhnDataset))
                        posExs.append(self.samplePos(mnistDataset))
                        negExs.append(self.sampleNeg(x,mnistDataset))
            return (torch.cat(anchors,0),torch.cat(posExs,0),torch.cat(negExs,0))
            
        
        
    def __init__(self,mnistDataset,svhnDataset):
        self.svhn = {}
        for x in range(10):
            indx = np.where(np.array(svhnDataset.labels) == x)[0]
            self.svhn[x] = torch.utils.data.DataLoader(Subset(svhnDataset, indx), 
                                                     batch_size=4, shuffle=False)
            self.svhn[x] = [batch for batch,_ in self.svhn[x]][:-1]

        self.mnist = {}
        for x in range(10):
            indx = np.where(np.array(mnistDataset.targets) == x)[0]
            self.mnist[x] = torch.utils.data.DataLoader(Subset(mnistDataset, indx), 
                                                     batch_size=4, shuffle=False)
            self.mnist[x] = [batch for batch,_ in self.mnist[x]][:-1]
            
        self.mnistAnchors = [TripletController.Triplet(anchor,True,self.mnist,self.svhn) for anchor in range(10)]
        self.svhnAnchors = [TripletController.Triplet(anchor,False,self.mnist,self.svhn) for anchor in range(10)]
            
            
    def sample(self,mnist,silent=True):
        anchors = []
        posExs = []
        negExs = []
        
        t1 = time.time()
        for x in range(10):
            if mnist:
                a,p,n = self.mnistAnchors[x].sampleBatch(self.mnist,self.svhn,sampleMul=1)
                anchors.append(a)
                posExs.append(p)
                negExs.append(n)
            else:
                a,p,n = self.svhnAnchors[x].sampleBatch(self.mnist,self.svhn,sampleMul=1)
                anchors.append(a)
                posExs.append(p)
                negExs.append(n)
                
        if not silent: 
            print("Sampling time elapsed: {}".format(time.time()-t1))  
            
        return (torch.cat(anchors,0),torch.cat(posExs,0),torch.cat(negExs,0))
    
    
    def toDevice(self,device):
        for k,batchList in self.svhn.items():
            for i,batch in enumerate(batchList):
                self.svhn[k][i] = self.svhn[k][i].to(device)
            
        for k,batchList in self.mnist.items():
            for i,batch in enumerate(batchList):
                self.mnist[k][i] = self.mnist[k][i].to(device)