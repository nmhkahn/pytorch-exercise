import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()
        
        self.scale = scale

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)
        self.exit  = nn.Conv2d(64, 3, 3, 1, 1)
        
        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU()
        )
        
        # support only x2, x4, ...
        self.upsample = list()
        for _ in range(int(math.log(scale, 2))):
            self.upsample += [nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU()]
        self.upsample = nn.Sequential(*self.upsample)

    def forward(self, x):
        x = self.entry(x)
        out = self.body(x)
        out += x
        
        out = self.upsample(out)
        out = self.exit(out)

        return out

