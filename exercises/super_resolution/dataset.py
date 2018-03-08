import os
import glob
import numpy as np
import PIL
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(Dataset, self).__init__()
        
        self.scale = scale
        self.paths = glob.glob(os.path.join(dirname, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = Image.open(self.paths[index])

        w, h = hr_image.size
        lr_image = hr_image.resize((int(w/self.scale), int(h/self.scale)), PIL.Image.BICUBIC)

        return self.transform(hr_image), self.transform(lr_image)

    def __len__(self):
        return len(self.paths)
