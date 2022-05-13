import  torch
from torch import nn
import  torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../resource/datasets",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

my_model = My_model()
print(my_model)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs ,targets = data
    output = my_model(imgs)
    # print(output.shape) #torch.Size([64, 6, 30, 30])
    writer.add_images("input",imgs,step)
    # torch.Size([64, 6, 30, 30])
    # 经过卷积后的图像channel是6，不可以用tensorboard显示，会报错，需要转换,将channel转为3
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)

    step += 1

writer.close()