import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from Unet import Unet
import torch
import torchvision
from cfgnumber import colors
import cv2


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
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    net = Unet(3, 59)
    # net.load_state_dict(torch.load('paramnew/net.pt-350-2240'))
    img = Image.open(r'D:\projects\clothing Segmentation\photos\1.jpg')
    imgarray = np.array(img)
    input = torch.unsqueeze(tranform(imgarray), dim=0)
    with torch.no_grad():
        output = net(input)
        pred = output.squeeze(dim=0)
        index = torch.argmax(pred, dim=0).numpy()
        for id in range(59):
            if id==0:
                imgarray[index==id]=colors['0']
            else:
                imgarray[index==id]=colors[f'{id}']
        maskimg = Image.fromarray(imgarray, 'RGB')
        # maskimg.show()
        plot_img_and_mask(img, maskimg)
                    # if index[i, j] == id:
                    #     index[i, j] = np.array(colors[f'{i}'], dtype=np.uint8)
                    #     print(index[i, j])
        # mask = pred > 0.5
        # maskimg = np.array((mask * 255)).astype(np.uint8)
        # cv2.imshow('mask',maskimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # imgcolorindex = torch.argmax(mask.squeeze(dim=0), dim=0)
        # for i in np.arange(0, 59):
        #     if i != 0:
        #         input[imgcolorindex == i] = colors[f'{i}'].values()
        #     else:
        #         imgarray[imgcolorindex == i] = imgarray[imgcolorindex == i]
        # maskimg = Image.fromarray(imgarray, 'RGB')
        # # maskimg.show()
        # plot_img_and_mask(img, maskimg)
