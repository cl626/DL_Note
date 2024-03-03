import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.functional as F
import matplotlib.pyplot as plt

train_data=pd.read_csv('train.csv',sep='\\s+',skiprows=None,header=None).values
test_data=np.array(pd.read_csv('test.csv',sep='\\s+').values)

class net(nn.module):
    def __init__(self,in_dim,out_dim):
        self.layer1=nn.Sequential(nn.Linear(in_dim,out_dim))

    def forward(self,input):
        x=F.relu(self.layer1(input))
        return x

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim()
