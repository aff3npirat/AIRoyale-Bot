import torch
import torch.nn as nn

from timing import exec_time



def name_to_activation(act_name):
    if act_name == "sigmoid":
        act_cls = nn.Sigmoid
    elif act_name == "tanh":
        act_cls = nn.Tanh
    elif act_name == "prelu":
        act_cls = nn.PReLU
    elif act_name == "relu":
        act_cls = nn.ReLU

    return act_cls


class ResBlk(nn.Module):
    
    def __init__(self):
        super().__init__()

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

    @exec_time
    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
    

class ConcatLayer(nn.Module):

    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, x):
        out = self.net(x)
        return torch.cat((x, out), dim=1)
    

class DenseNet(nn.Module):
    
    def __init__(self, net_arch, activation, feature_extractor=False, bias=True):
        super().__init__()

        self.feature_extractor = feature_extractor

        if feature_extractor:
            linear = nn.Linear(net_arch[0], net_arch[1], bias=bias)
            act = name_to_activation(activation)()
            layers = [nn.Sequential(linear, act)]
            net_arch = net_arch[1:]
        else:
            layers = []

        skip_size = net_arch[0]
        for i in range(1, len(net_arch)-1):
            linear = nn.Linear(skip_size, net_arch[i], bias=bias)
            act = name_to_activation(activation)()

            layers.append(ConcatLayer(nn.Sequential(linear, act)))

            skip_size += net_arch[i]

        # output layer
        layers.extend([
            nn.Linear(skip_size, net_arch[-1], bias=False),
            nn.ReLU(),
        ])

        self.layers = nn.Sequential(*layers)

    @exec_time
    def forward(self, x):
        out = self.layers(x)
        return out


class QNet(nn.Module):

    def __init__(self, net_arch, activation, feature_extractor=False, bias=True):
        super().__init__()

        self.board_emb = BoardEmbedding()
        self.val_net = DenseNet(net_arch=net_arch, activation=activation, feature_extractor=feature_extractor, bias=bias)

    def forward(self, x):
        board, context = x

        squeeze = (len(board.shape)==3) and (len(context.shape)==1)
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
        if len(context.shape) == 1:
            context = context.unsqueeze(0)

        board = self.board_emb(board)
        context = torch.cat((board, context), dim=-1)
        qval = self.val_net(context)

        if squeeze:
            qval = qval.squeeze(0)
        
        return qval