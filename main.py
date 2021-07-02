import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import Optimizer
import math
import nn
import utils
from utils import loaders,to_one_hot,Macro_F1
from optim import Adam
from torch import Tensor
from pre_processor import data_clean,load_corpus,tokenize_star,comment_to_idx,comment
from embedding import load_GloVe,embedding
import random
SEED=5
    


    
    
#****************************************************************************************
df=data_clean()
corpus,vocab=load_corpus(df)
comments_with_stars=tokenize_star(df)
token_to_index=vocab.token_to_index

np.random.seed(SEED)
tens,labels=comment(comments_with_stars,token_to_index)
random.shuffle(tens)
random.shuffle(labels)
tens=Tensor(tens)

embedding_mat,embeddings_index=load_GloVe(vocab)
embedding_mat=Tensor(embedding_mat)
embed=embedding(embedding_mat)
data=embed(tens)
data=data.permute(0,2,1)

train_data=data[:50000]
train_labels=labels[:50000]
test_data=data[50000:]
test_labels=labels[50000:]
#****************************************************************************************
class CNN(nn.Model):
    def __init__(self):
        self.fc = nn.Sequential(nn.Conv1d(100,128,5),nn.ReLU(),nn.MaxPooling1d(2),nn.Conv1d(128,256,3),nn.ReLU(),nn.GlobalMaxPooling1d(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,6))
        super(CNN, self).__init__()
        self.set_name('Net')

    def construct(self):
        return [self.fc]
    
    def forward(self, obs):
        actions = self.fc(obs)
        return actions
#*****************************************************************************************    
net=CNN()
criterion=nn.MSE()
optimizer=Adam(net.parameters())
train_accs=[]
train_loss=[]
test_accs=[]
Macro_F1_score=[]
#training
for epoch in range(5):
    running_loss=0.0
    train_loader=loaders(train_data,train_labels,batch_size=32)
    for i,data in enumerate(train_loader,0):
        print(i)
        inputs,labels=data[0],data[1]
        inputs.cuda_required()
        optimizer.zero_grad()
        outputs=net(inputs)
        labels_=Tensor(to_one_hot(labels)).float().cuda()
        loss=criterion(outputs,labels_)
        loss.backward()
        print(loss)
        optimizer.step()
        correct=0
        total=0
        _,predicted=torch.max(outputs.data,1)
        total=labels.size(0)
        correct=(predicted==labels).sum().item()
        print("accuracy on training set")
        print(100*correct/total)
        F=Macro_F1(np.array(labels),np.array(predicted))
        print("Macro-F1 Score:")
        print(F)
        Macro_F1_score.append(F)
        train_accs.append(100*correct/total)
        running_loss+=loss.item()
        if i%50 == 49:
            print('[%d,%5d] training loss: %.3f'%(epoch+1,i+1,running_loss/50))
            running_loss=0.0
            test_outputs=net(test_data)
            test_labels_=Tensor(to_one_hot(test_labels)).float().cuda()
            test_loss=criterion(test_outputs,test_labels_)
            print('[%d,%5d] test loss: %.3f'%(epoch+1,i+1,test_loss))
            _,test_predicted=torch.max(test_outputs.data,1)
            test_total=len(test_labels)
            test_correct=(test_predicted==test_labels).sum().item()
            print("test_accuracy")
            print(100*test_correct/test_total)
            test_accs.append(100*test_correct/test_total)
        train_loss.append(loss.item())

        
