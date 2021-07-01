import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets,transforms,utils
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
import math
import nn
import utils
from optim import Adam
from torch import tensor
from pre_processor import data_clean,load_corpus
df=data_clean()
corpus,vocab=load_corpus(df)


class CNN(nn.Model):
    def __init__(self):
        self.fc = nn.Sequential(nn.conv1(100,32,3),nn.Linear(,),nn.ReLU(),nn.Linear(98, 1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,10))
        super(CNN, self).__init__()
        self.set_name('Net')

    def construct(self):
        return [self.fc]
    
    def forward(self, obs):
        actions = self.fc(obs)
        return actions
net=CNN()
criterion=nn.MSE()
optimizer=Adam(net.parameters())
net=CNN()
criterion=nn.MSE()
optimizer=Adam(net.parameters())

train_accs=[]
train_loss=[]
test_accs=[]
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#training
for epoch in range(5):
    running_loss=0.0
    for i,data in enumerate(train_loader,0):
        print(i)
        inputs,labels=data[0],data[1]
        optimizer.zero_grad()
        inputs=inputs.reshape((inputs.shape[0],28*28))
        outputs=net(inputs)
        labels_=tensor(utils.to_one_hot(labels)).float()
        loss=criterion(outputs,labels_)
        loss.backward()
        print(loss)
        optimizer.step()
        running_loss+=loss.item()
        if i%100 == 99:
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/100))
            running_loss=0.0
        train_loss.append(loss.item())
        correct=0
        total=0
        _,predicted=torch.max(outputs.data,1)
        total=labels.size(0)
        correct=(predicted==labels).sum().item()
        train_accs.append(100*correct/total)