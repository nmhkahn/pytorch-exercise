import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)
        self.exit  = nn.Conv2d(64, 3, 3, 1, 1)

        body = list()
        for i in range(5):
            body += [nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU()]
        self.body = nn.Sequential(*body) 
        
        # support only x2, x4, ...
        upsample = list()
        for _ in range(int(math.log(scale, 2))):
            upsample += [nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU()]
        self.upsample = nn.Sequential(*upsample)

    def forward(self, x):
        x = self.entry(x)
        out = self.body(x)
        out += x
        
        out = self.upsample(out)
        out = self.exit(out)

        return out

