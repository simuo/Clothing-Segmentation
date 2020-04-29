# coding:UTF-8
import scipy.io as scio
import cv2
import os
from PIL import Image

pixDataFile = '../annotations/pixel-level'
masksavepath = '../annotations/pixmask'
masks = os.listdir(pixDataFile)
maskOri = [os.path.join(pixDataFile, path) for path in masks]
for i, item in enumerate(maskOri):
    data = scio.loadmat(item)
    # print(data)
    imgarr = data['groundtruth']
    img=Image.fromarray(imgarr).convert('L')
    cv2.imwrite(f'{masksavepath}/{i + 1}.jpg', imgarr)
    # print(imgarr.shape)
    # img = cv2.imshow('mask', imgarr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(data['tags'])

