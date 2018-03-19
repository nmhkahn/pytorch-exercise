import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.block1 = self._make_layers(64, 128, 2)
        self.block2 = self._make_layers(128, 128, 2)
        self.block3 = self._make_layers(128, 256, 2)

        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(256, 5)

    def _make_layers(self, 
                     in_channels, out_channels, 
                     num_layers=2):
        layers = list()
        for i in range(num_layers):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        
        out = self.block1(out)
        out = self.max_pool(out)
        
        out = self.block2(out)
        out = self.max_pool(out)
        
        out = self.block3(out)
        out = self.max_pool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
