# -*- coding: utf-8 -*-
'''
@File    :   model_pretrained.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/16 23:47   gujiayue      1.0         None
'''
import torchvision

# train_data = torchvision.datasets.ImageNet(root="../resource/datasets",split="train",download=True,
#                                            transform=torchvision.transforms.ToTensor())

# false的话，只使用模型的网络结果，true的话使用预训练参数，需要下载
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

dataset = torchvision.datasets.CIFAR10("../resource/datasets",download=True,train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=2)

# 由于vgg16模型，是1000分类的，而我们实际只需要10分类，
# 因此可以考虑原有模型的最后再添加一个线性层，从1000分类变为10分类
# 一般使用预训练模型，可以先print下该模型，然后大概了解其结果，然后再添加自己需要的层
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

# 也可以直接修改最后的线性层，让其输出为10个特征
# 这里用vgg16_false来演示
vgg16_false.classifier[6]  = nn.Linear(4096,10)
print(vgg16_false)