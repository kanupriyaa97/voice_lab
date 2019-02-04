

import torch.nn as nn
from BasicBlock import BasicBlock
import torch.nn.functional as F

## construct network
class ResNet(nn.Module):
    def __init__(self, channel_0, channel_1, channel_2, channel_3, channel_4, class_n):
        super(ResNet, self).__init__()
        self.featureD = 128
        self.convlayers = nn.Sequential(
            nn.Conv2d(channel_0, channel_1, 3, 2, bias=False),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(inplace=True),
            BasicBlock(channel_1),
            BasicBlock(channel_1),
            nn.Conv2d(channel_1, channel_2, 3, 2, bias=False),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(inplace=True),
            BasicBlock(channel_2),
            BasicBlock(channel_2),
            nn.Conv2d(channel_2, channel_3, 3, 2, bias=False),
            nn.BatchNorm2d(channel_3),
            nn.ReLU(inplace=True),
            BasicBlock(channel_3),
            BasicBlock(channel_3),
            nn.Conv2d(channel_3, channel_4, 3, 2, bias=False),
            nn.BatchNorm2d(channel_4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_4, self.featureD, 3, 2, bias=False),
        )
        self.bn = nn.BatchNorm1d(self.featureD, affine=False)
        #self.fc1 = AngleLinear(self.featureD, class_n)
        self.fc1 = nn.utils.weight_norm(nn.Linear(self.featureD, class_n, bias=False), name='weight')

    def forward(self, x):
        conv_o = self.convlayers(x)
        x = F.avg_pool2d(conv_o, [conv_o.size()[2], conv_o.size()[3]], stride=1)
        x = x.view(-1, self.featureD)
        return x

