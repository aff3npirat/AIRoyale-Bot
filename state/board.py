import torch
import torch.nn as nn



class IBasicBlock(nn.Module):
    pass


class BoardEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            *[IBasicBlock() for _ in range(5)]
        )

    def forward(self, x):
        return self.blocks(x)
