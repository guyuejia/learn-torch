# -*- coding: utf-8 -*-
'''
@File    :   model.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/21 16:43   gujiayue      1.0         None
'''
import torch

"""
单独新创建一个文件，定义网络模型
"""
from torch import nn

#搭建神经网络
class My_module(nn.Module):
    def __init__(self):
        super(My_module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            #展平
            nn.Flatten(),
            #全连接层
            #展平后的尺寸就是64*4*4，64是通道数，4*4是图像的尺寸
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    my_model = My_module()
    #创建一个张量
    input = torch.ones((64,3,32,32))
    output = my_model(input)
    print(output.shape)