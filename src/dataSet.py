from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

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
        transform = transforms.Compose([transforms.Resize((600,350),interpolation=Image.BILINEAR),
                                        transforms.ToTensor()])
        image_data = transform(image)
        lable_data = transform(label)   
        return image_data, lable_data

    def __len__(self):
        return len(self.dataset)
