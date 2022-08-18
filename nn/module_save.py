# -*- coding: utf-8 -*-
'''
@File    :   module_save.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 0:24   gujiayue      1.0         None
'''
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)

#保存方式1,将网络模型和参数全部保存到当前目录
torch.save(vgg16_false,"vgg16_method1.pth")

# 保存方式2,只将模型中的参数保存为一个字典(官方推荐），不保存网络结构，节省空间
torch.save(vgg16_false.state_dict(),"vgg16_method2.pth")


# 保存方式1的陷阱
class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()
        # 通过seq的方式，等同于上面
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

    def forward(self, x):
        # 这种利用seq的方式等同于上面一长串，更间接
        x = self.model1(x)
        return x


my_model = My_module()
torch.save(my_model, "my_model.pth")