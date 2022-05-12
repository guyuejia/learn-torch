from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os
import cv2

project_path = r"C:\Users\hushishuai\PycharmProjects\learn-torch"
img_path = r"resource/hymenoptera_data/train/ants/0013035.jpg"
img_path = os.path.join(project_path,img_path)

#将PIL打开的图片转为tensor
img = Image.open(img_path)
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
# print(img_tensor)


writer = SummaryWriter("logs")
#利用opencv打开图片，转为tensor
img_cv = cv2.imread(img_path)
img_tensor = trans_tensor(img_cv)
# print(img_tensor)

writer.add_image("tensor",img_tensor)
writer.close()

