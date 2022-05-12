import torchvision
from torch.utils.tensorboard import SummaryWriter


#获取官方数据集CIFAR10,各个参数要看官方文档说明
#获取训练和测试数据集，不做transform

# train_set = torchvision.datasets.CIFAR10(root="../resource/datasets",train=True,download=True)
# val_set = torchvision.datasets.CIFAR10(root="../resource/datasets",train=False,download=True)
# 查看测试数据中第一个图片，是PIL图像
# print(val_set[0])
# #查看所有的类别
# print(val_set.classes)
# #数据集中的每个元素都包括2部分内容，一个是图像本身，一个是类别
# img,target = val_set[0]
# img.show()

#先定义一个transform


data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
#获取数据集的同事，做transform
train_set = torchvision.datasets.CIFAR10(root="../resource/datasets",train=True,transform=data_trans,download=True)
val_set = torchvision.datasets.CIFAR10(root="../resource/datasets",train=False,transform=data_trans,download=True)
#做过转换后，数据集的元素就不是图像了，而是tensor了
print(val_set[0])
tensor,target = val_set[0]

#利用tensorboadr显示图片,添加10张图片
writer = SummaryWriter("logs")
for i in range(10):
    img,target = val_set[i]
    writer.add_image("CIFAR10",img,i)

writer.close()