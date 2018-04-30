import torch
import torch.nn as nn

def transpose_conv(in_channels, out_channels, 
                   kernel_size, stride=2, padding=1, 
                   act=nn.LeakyReLU(0.05, True),
                   bn=True):
    layers = list()

    layers += [nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding)
    ]
    if bn:
        layers += [nn.BatchNorm2d(out_channels)]
    if act:
        layers += [act]

    return nn.Sequential(*layers)
    

def conv(in_channels, out_channels, 
         kernel_size, stride=2, padding=1, 
         act=nn.ReLU(True),
         bn=True):
    layers = list()

    layers += [nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding)
    ]
    if bn:
        layers += [nn.BatchNorm2d(out_channels)]
    if act:
        layers += [act]

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, num_class=10, z_dim=100):
        super(Generator, self).__init__()

        self.fc = transpose_conv(num_class+z_dim, 32, 7, 1, 0, bn=False)
        self.t_conv1 = transpose_conv(32, 64, 4)
        self.t_conv2 = transpose_conv(64, 128, 4)
        self.conv3   = conv(128, 1, 3, 1, 1, act=nn.Tanh(), bn=False)

    def forward(self, y, z):
        latent = torch.cat([y, z], dim=1)
        latent = latent.view(latent.size(0), latent.size(1), 1, 1)
        
        out = self.fc(latent)
        out = self.t_conv1(out)
        out = self.t_conv2(out)
        out = self.conv3(out)

        return out
    

class Discriminator(nn.Module):
    def __init__(self, num_class=10):
        super(Discriminator, self).__init__()

        self.conv1 = conv(1, 32, 4, bn=False)
        self.conv2 = conv(32, 64, 4)
        self.fc_adv = conv(64, 2, 7, 1, 0, act=None, bn=False)
        self.fc_cls = conv(64, num_class, 7, 1, 0, act=None, bn=False)
        
    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)

        adv = self.fc_adv(out).squeeze()
        cls = self.fc_cls(out).squeeze()

        return adv, cls
