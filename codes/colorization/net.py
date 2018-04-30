import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        layers = list()
        layers += [nn.Conv2d(1, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU()]
        layers += [nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()]
        
        layers += [nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU()]
        layers += [nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()]
        
        layers += [nn.Conv2d(256, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU()]
        layers += [nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()]
        
        layers += [nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU()]
        layers += [nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()]
        
        layers += [nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU()]
        layers += [nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()]
        
        layers += [nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()]
        layers += [nn.Conv2d(64, 2, 3, 1, 1)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x.unsqueeze(1))

        return out