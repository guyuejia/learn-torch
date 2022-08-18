# -*- coding: utf-8 -*-
'''
@File    :   nn_loss.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/16 19:56   gujiayue      1.0         None
'''
import torch
from torch import nn
from torch.nn import L1Loss


inputs = torch.tensor([1,2,3],dtype=torch.float)
targets = torch.tensor([1,2,5],dtype=torch.float)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = L1Loss()
result = loss(inputs,targets)
print(result)

#平方差
loss_mse = nn.MSELoss()
result = loss_mse(inputs,targets)
print(result)

#交叉熵
#输入代表了，预测为第0个类别的概率，第1个类别概率和第2个类别的概率
x = torch.tensor(([0.1,0.2,0.3]))
#目标代表了，实际为第1个类别（索引从0开始）
target = torch.tensor([1])
#对于输入，必须要转换维度，第一个维度是bactch_size,第二个维度是类别数量
x = torch.reshape(x,[1,3])

loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x,target)
print(result)
