# -*- coding: utf-8 -*-
'''
@File    :   nn_linear.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/14 18:58   gujiayue      1.0         None
'''
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../resource/datasets",download=True,train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()
        self.linear = Linear(196608,10)

    def forward(self,x):
        output = self.linear(x)
        return output

my_module = My_module()

for data in dataloader:
    imgs,targets = data
    # print(imgs.shape)
    # outs = torch.reshape(imgs,(1,1,1,-1))
    # print(outs.shape)
    outs = torch.flatten(imgs)
    output = my_module(outs)
    print(output.shape)
    print(output.data)
