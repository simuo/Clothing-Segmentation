from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

data_dir = r'/content/Clothing-Segmentation/photos'
label_dir = r'/content/Clothing-Segmentation/annotations/pixmask'


class MydataSet(Dataset):
    def __init__(self):
        super(MydataSet, self).__init__()
        self.dataset = os.listdir(label_dir)
        self.dataset.sort(key=lambda x: int(x.split('.')[0]))

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(data_dir, self.dataset[index]).replace('\\', '/'))
        label = Image.open(
            os.path.join(label_dir, self.dataset[index]).replace('\\', '/'))
        transformimg = transforms.Compose([transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])
        image_data = transformimg(image)
        lable_data = torch.Tensor(np.array(label)) 
        return image_data, lable_data

    def __len__(self):
        return len(self.dataset)
