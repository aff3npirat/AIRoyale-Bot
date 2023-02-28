import torch
import torch.nn as nn



class ResBlk(nn.Module):
    
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 16, (3, 3), padding=1, stride=1, bias=True)
        self.prelu = nn.PReLU(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding=1, stride=1, bias=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = x + identity
        return x


class BoardEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            *[ResBlk() for _ in range(5)]
        )
        self.downsample = nn.Conv2d(16, 16, (3, 3), padding=0, stride=2, bias=False)
        self.linear = nn.Linear(14*8*16, 512, bias=False)

    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
