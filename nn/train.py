# -*- coding: utf-8 -*-
'''
@File    :   train.py    
@Contact :   hushishuai.fly@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/21 16:28   gujiayue      1.0         None
'''
import torch.optim
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import  nn
from model import My_module

train_data = torchvision.datasets.CIFAR10(root="../resource/datasets",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../resource/datasets",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

#利用dataloader加载数据
train_dataloader = DataLoader(dataset=train_data,batch_size=64)
test_dataloader = DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
my_model = My_module()
#创建损失函数
loss_fn = nn.CrossEntropyLoss()

#定义优化器
optim = torch.optim.SGD(my_model.parameters(),lr=0.01)

#设置训练网络的一些参数
#训练的次数
total_train_step = 0
#测试的次数
total_test_step = 0
#训练的轮数
epoch = 10
#利用tensorboard可视化
writer = SummaryWriter("logs")
for i in range(epoch):
    print("--------第{}轮训练开始------".format(i+1))
    #训练开始,
    #每一轮训练之前加上train函数，网络中有某些特定层的时候（官方文档）会起作用，
    my_model.train()
    for data in train_dataloader:
        imgs,targets = data
        #调用模型预测输出
        outputs = my_model(imgs)
        #计算损失
        loss = loss_fn(outputs,targets)
        #梯度归0
        optim.zero_grad()
        #反向传播
        loss.backward()
        #更新权重参数
        optim.step()

        #训练次数加1
        total_train_step += 1
        #每训练100次，打印一下
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss,total_train_step)

    #测试步骤开始
    #测试验证的时候统一调用eval
    my_model.eval()
    total_test_loss = 0
    #测试集上总正常预测的数量
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = my_model(imgs)
            loss = loss_fn(outputs,targets)
            #记录整个测试机的loss
            total_test_loss += loss

            #计算测试集上的正确率，一般分类问题需要计算正确率
            #output的值实际上是每个分类的概率，我们一般取最大的概率所在的分类为实际的分类
            #利用argmax可以获得最大概率所在的索引，也就是分类索引。参数1是代表从行的方向取最大值
            pre = outputs.argmax(1)
            #将预测值与实际值相比较，true的个数也就是预测正确的个数
            accuracy = (pre == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    total_test_step+=1

    #保存每一轮的训练模型
    torch.save(my_model,"mymoel_{}.pth".format(i))

writer.close()