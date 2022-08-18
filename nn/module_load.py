# -*- coding: utf-8 -*-
'''
@File    :   module_load.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 0:26   gujiayue      1.0         None
'''

# 加载方式1——对应保存方式1
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

model = torch.load("vgg16_method1.pth")
print(model)

# 加载方式2——对应保存方式2
model2 = torch.load("vgg16_method2.pth")
# 由于保存方式2只保存了参数，没有网络结构，因此需要先加载网络结构
vgg16 = torchvision.models.vgg16(pretrained=False)
# 加载模型状态，也就是参数
vgg16.load_state_dict(model2)
print(vgg16)


# 陷阱1,直接加载会报错，需要把网络模型结构复制过来，或者import原来的那个py文件
my_model = torch.load("my_model.pth")