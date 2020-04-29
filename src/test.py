import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from Unet import Unet
import torch
import torchvision
from cfgnumber import colors
import cv2
import os


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    paramPath='../src/params/params4.pt'
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    net = Unet(3, 59).cuda()
    if os.path.exists(paramPath):
        net.load_state_dict(torch.load(paramPath))
    img = Image.open(r'../photos/2097.jpg')
    imgarray = np.array(img)
    input = torch.unsqueeze(tranform(imgarray), dim=0)
    with torch.no_grad():
        output = net(input.cuda())
        pred = output.squeeze(dim=0)
        index = torch.argmax(pred, dim=0).cpu().numpy()
        for id in range(59):
            if id==0:
                imgarray[index==id]=colors['0']
            else:
                imgarray[index==id]=colors[f'{id}']
        maskimg = Image.fromarray(imgarray, 'RGB')
        # maskimg.show()
        plot_img_and_mask(img, maskimg)
