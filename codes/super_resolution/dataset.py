import os
import glob
import numpy as np
import PIL
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *

class Dataset(data.Dataset):
    def __init__(self, scale, train, **kwargs):
        super(Dataset, self).__init__()

        self.scale = scale
        self.size = kwargs.get("size", -1) # -1 stands for original resolution
        self.data_root = kwargs.get("data_root", "./data")
        
        self._prepare_dataset(self.size, self.data_root)

        if train:
            dirname = os.path.join(self.data_root, "flower/train")
        else:
            dirname = os.path.join(self.data_root, "flower/test")

        self.paths = glob.glob(os.path.join(dirname, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr_image = Image.open(self.paths[index])

        w, h = hr_image.size
        lr_image = hr_image.resize((int(w/self.scale), int(h/self.scale)), 
                                   PIL.Image.BICUBIC)

        return self.transform(hr_image), self.transform(lr_image)

    def __len__(self):
        return len(self.paths)

    def _prepare_dataset(self, size, data_root):
        check = os.path.join(data_root, "flower")
        if not os.path.isdir(check):
            download_and_convert(size, data_root)
