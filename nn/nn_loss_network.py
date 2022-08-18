# -*- coding: utf-8 -*-
'''
@File    :   nn_loss_network.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/16 20:23   gujiayue      1.0         None
'''
import torch
import torchvision.datasets
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../resource/datasets",download=True,train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=2)

class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()
        #通过seq的方式，等同于上面
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        # 这种利用seq的方式等同于上面一长串，更间接
        x = self.model1(x)
        return x

#交叉熵
loss = nn.CrossEntropyLoss()
my_module = My_module()
for data in dataloader:
    imgs ,targets = data
    outputs = my_module(imgs)
    result_loss = loss(outputs,targets)
    print(result_loss)