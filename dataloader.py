import os
import re
from PIL import Image
from torch.utils.data import Dataset
#from torchvision import transforms as trans

class DataClassify(Dataset):
    def __init__(self, root, transforms=None, mode=None):
        #存放图像地址
        self.imgs = [x.path for x in os.scandir(root) if
            x.name.endswith(".jpg")]
        self.labels = [y.path for y in os.scandir(root) if
            y.name.endswith(".txt")]
        self.transforms = transforms
        
    def __getitem__(self, index):
        #读取图像数据并返回
        img_path = self.imgs[index]
        label = int(re.sub('\D', '', open(self.labels[index]).read()[-4:]))
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return len(self.imgs)
