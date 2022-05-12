import torch
import torch.nn as nn

"""
创建一个最简单的神经网络
"""
class My_model(nn.Module):
    """
    所有的神经网络都应该继承：nn.Model
    """
    def __init__(self):
        super(My_model, self).__init__()

    # 所有的神经网络都应该重写forward函数
    # 神经网络的每个输入都应用于该函数，并数据
    def forward(self,input):
        output = input+1
        return output

my_model = My_model()
x = torch.tensor(1.0)
#将x应用于自己的神经网络，本质就是调用forward函数
output = my_model(x)
print(output) # tensor(2.)

