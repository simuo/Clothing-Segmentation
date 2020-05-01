import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from Unet import Unet
import torch
import torchvision
from cfgnumber import colors
import cv2
import os


def plot_img_and_mask(img, mask,num):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.savefig('/content/Clothing-Segmentation/src/outImgs/{}.jpg'.format(num))
    # plt.show()

def test_singmodel(param_path,num):
  tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  net = Unet(3, 59).cuda()
  if os.path.exists(param_path):
      net.load_state_dict(torch.load(param_path))
  img = Image.open(r'/content/Clothing-Segmentation/photos/1.jpg')
  imgarray = np.array(img)
  input = torch.unsqueeze(tranform(imgarray), dim=0)
  with torch.no_grad():
    output = net(input.cuda())
    pred = output.squeeze(dim=0)
    index = torch.argmax(pred, dim=0).cpu().numpy()
    for id in range(59):
      imgarray[index==id]=colors[f'{id}']
    maskimg = Image.fromarray(imgarray, 'RGB')
    plot_img_and_mask(img, maskimg,num)


if __name__ == '__main__':
    paramPath='/content/Clothing-Segmentation/src/params'
    param_paths=os.listdir(paramPath)
    for i in range(len(param_paths)):
      # print(paramPath)
      # print(param_paths)
      print(paramPath+'/'+param_paths[i])
      if param_paths[i].endswith('pt'):
        test_singmodel(paramPath+'/'+param_paths[i],i)
    print('模型测试成功，测试结果已保存!')
    
