"""
常用transform的使用
多看代码中的帮助注释
重点了解每个trans的输入和输出

"""

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os
import cv2




project_path = r"C:\Users\hushishuai\PycharmProjects\learn-torch"
img_path = r"resource/hymenoptera_data/train/ants/0013035.jpg"
img_path = os.path.join(project_path,img_path)
img = Image.open(img_path)

writer = SummaryWriter("logs")
#1. To_tensor，输入图像的像素值大小是[0,255],to_tensor后，会将像素值变为[0.0, 1.0]
#支持PIL图像，或者opencv打开的文件
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("to_tensor",img_tensor)

#2.  Normalize 标准化，指定均值和方差
# 不支持PIL文件，一般都是将to_tensor之后的数据作为输入
trans_nor = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_nor = trans_nor(img_tensor)
writer.add_image("Normalize",img_nor)


#3. Resize,如果只输入一个数字参数，那会按照最小边进行等比例缩放
# 输入支持PIL 图像,返回的结果也是一个PIL,因此转换之后还要再经过一个to_tensor才能继续应用
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
# print(img_resize)
writer.add_image("resize",img_resize)


#4. compose，将多个transfrom集成到一起，前一个trans的输出是下一个的输入，因此类型要匹配，否则会报错
trans_resize2 = transforms.Resize(256)
trans_comp = transforms.Compose([trans_resize2,trans_totensor])
img_resize2 = trans_comp(img)
writer.add_image("resize-compose",img_resize2)

#5，Random_crop 随机裁剪为给定的尺寸，注意：只是裁剪，不做缩放。如果只给一个数值，就裁剪为正方形
trans_crop = transforms.RandomCrop(256)
trans_comp = transforms.Compose([trans_crop,trans_totensor])
#裁剪10次试试
for i in range(10):
    img_crop = trans_comp(img)
    writer.add_image("Randomcrop",img_crop,i)

writer.close()
