import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataSet import MydataSet
from Unet import Unet

if __name__ == '__main__':
    writer=SummaryWriter()
    dataset = MydataSet()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    module = Unet(3, 59)  # .cuda()
    optimizer = optim.Adam(module.parameters())
    criticizer = nn.CrossEntropyLoss()
    for epoch in range(1000):
        for i, (data, lable) in enumerate(dataloader):
            data = data  # .cuda()
            lable = lable  # .cuda()
            output = module(data)
            loss1 = criticizer(output, lable.squeeze(1).type(torch.LongTensor))
            writer.add_scalar('Loss/train', loss1.item(), i)
            # loss2 = 1 - get_DC(output.cpu(),lable.cpu())
            # print(loss1.item(), loss2.item())
            # loss = loss1 + loss2
            # print(loss1.item(),loss2)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            print(f'EPOCH {epoch}, {i} / {len(dataloader)} , loss:{loss1.item()}')
        torch.save(module.state_dict(), r'params\params{}.pt'.format(epoch))
    writer.close()
