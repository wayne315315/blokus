import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.dense_g = nn.Linear(filters, filters, bias=False)

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        g = out.mean(dim=(2, 3))
        g = self.dense_g(g)
        g = g.view(g.size(0), g.size(1), 1, 1)
        out = out + g
        out = out + shortcut
        return F.relu(out)

class PyTorchAdvancedBlokusModel(nn.Module):
    def __init__(self, board_size=20, num_blocks=4, filters=16):
        super().__init__()
        self.board_size = board_size
        self.conv_init = nn.Conv2d(8, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(num_blocks)])
        self.conv_v = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.dense_v1 = nn.Linear(board_size * board_size, 256)
        self.dense_v2 = nn.Linear(256, 1)
        self.conv_s = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_s = nn.BatchNorm2d(1)
        self.dense_s1 = nn.Linear(board_size * board_size, 256)
        self.dense_s2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks: x = block(x)
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1) 
        v = F.relu(self.dense_v1(v))
        value_out = torch.tanh(self.dense_v2(v))
        s = F.relu(self.bn_s(self.conv_s(x)))
        s = s.view(s.size(0), -1) 
        s = F.relu(self.dense_s1(s))
        score_out = self.dense_s2(s)
        return value_out, score_out
