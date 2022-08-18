# -*- coding: utf-8 -*-
'''
@File    :   nn_seq.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/15 17:33   gujiayue      1.0         None
'''
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2,stride=1)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,padding=2,stride=1)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)

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
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        # 这种利用seq的方式等同于上面一长串，更间接
        # x = self.model1(x)
        return x

my_module = My_module()
print(my_module)
input = torch.ones((64,3,32,32))
output = my_module(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(my_module,input)
writer.close()