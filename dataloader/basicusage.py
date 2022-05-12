from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision
data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
#获取数据集的同时，做transform
val_set = torchvision.datasets.CIFAR10(root="../resource/datasets",train=False,transform=data_trans,download=True)

#利用Dataload处理数据，配置相关参数
#shuffle参数会打乱图片的顺序，意思是每个epoch后要不要把图片顺序洗牌，如果是False，也就是每个epoch后获取图片的顺序是一样的，如果是True，每次获取图片的顺序是不一样的。
# drop_last,意思如果是要不要丢弃最后N个图片，N= 图片数量% batch_size
val_loader = DataLoader(dataset=val_set,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
#测试数据集第一个图片信息
img,target = val_set[0]

writer = SummaryWriter("logs")
step = 0
#dataloader本身也是一个集合类的数据，集合里面的每个元素就是由batch_size个原始数据组成的。
# 对于本例来说，每个元素就是有4个图片
for data in val_loader:
    imgs,targets = data # imgs 就是4个图片的集合，target就是对应的4个图片的分类
    # print(imgs.shape) # torch.Size([4, 3, 32, 32])
    # print(targets) # tensor([7, 0, 6, 7])
    # 之前都是添加的单个图片的tensor，
    # 现在这里通过 add_images 相当于添加了批量图片的tensor
    writer.add_images("dataloader",imgs,step)
    step += 1

writer.close()

