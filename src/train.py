import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from dataSet import MydataSet
from Unet import Unet

if __name__ == '__main__':
    writer=SummaryWriter('/content/Clothing-Segmentation/src/runs')
    dataset = MydataSet()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    module = Unet(3, 59).cuda()
    if os.path.exists(r'/content/Clothing-Segmentation/src/params/params2.pt'):
      module.load_state_dict(torch.load('/content/Clothing-Segmentation/src/params/params2.pt'))
    optimizer = optim.Adam(module.parameters())
    criticizer = nn.CrossEntropyLoss()
    for epoch in range(10000):
      for i, (data, lable) in enumerate(dataloader):
        data = data.cuda()
        lable = lable.cuda()
        output = module(data)
        # print(output.shape,lable.shape)
        loss1 = criticizer(output, lable.long())
        print(loss1)
        writer.add_scalar('Loss/train', loss1.item(), i)
        # loss2 = 1 - get_DC(output.cpu(),lable.cpu())
        # print(loss1.item(), loss2.item())
        # loss = loss1 + loss2
        # print(loss1.item(),loss2)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        print(f'EPOCH {epoch}, {i} / {len(dataloader)} , loss:{loss1.item()}')
      torch.save(module.state_dict(), r'/content/Clothing-Segmentation/src/params/params{}.pt'.format(epoch))
    writer.close()
