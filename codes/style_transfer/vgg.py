import torch
import torch.nn as nn
from torchvision import models

class VGGNet(nn.Module):
    feature_table = {
        "content": ["25"],
        "style": ["0", "5", "10", "19", "28"]
    }
    
    def __init__(self):
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x, phase):
        if phase not in ["content", "style"]:
            raise ValueError("phase argument must be in [\"content\", \"style\"]")

        features = list()
        table = self.feature_table[phase]

        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in table:
                features.append(x)
                if len(features) >= len(table):
                    break
        return features
