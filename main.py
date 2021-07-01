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

class CNN(nn.Model):
    def __init__(self):
        self.fc = nn.Sequential(nn.Linear(28*28,64*7*7),nn.ReLU(),nn.Linear(64*7*7, 1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,10))
        super(CNN, self).__init__()
        self.set_name('DQN')

    def construct(self):
        return [self.fc]
    
    def forward(self, obs):
        actions = self.fc(obs)
        return actions
    
'''class CNN(nn.Model):
    def __init__(self):
        super(CNN,self).__init__()
        
        #self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        #self.pool=nn.MaxPool2d(2,2)
        #self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        
        self.fc0=nn.Linear(1,64*7*7)
        self.fc1=nn.Linear(64*7*7, 1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,10)
    def forward(self,x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        '''
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc(\%)", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()

#read the MNIST dataset
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
train_data=datasets.MNIST(root="./data/",transform=transform,train=True,download=True)
test_data=datasets.MNIST(root="./data/",transform=transform,train=False)

train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

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
    
train_iters = range(len(train_accs))
draw_train_process('training',train_iters,train_loss,train_accs,'training loss','training acc')

PATH='./mnist_net.pth'
torch.save(net.seq_dict(),PATH)
correct=0
total=0
test_net=CNN()
test_net.load_seq_dict(torch.load(PATH))
#test
with torch.no_grad():
    for data in test_loader:
        images,labels=data
        outputs=test_net(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("Accuracy on the test images:%d%%"%(100*correct/total))

