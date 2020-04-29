import cv2
import torch

# x = torch.rand(2, 5)
# print(x)
# x=torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
# print(x)


img = cv2.imread('annotations/pixmask/56.jpg')
# print()
# img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# cv2.imshow('1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
data = torch.randint(1, 4, (3, 3))
zeros = torch.zeros(1, 59, 3, 3).permute(0, 2, 3, 1)
label = torch.scatter(dim=1, index=data[:, :], value=1)
print(data)
print(data[:])
print(zeros)
