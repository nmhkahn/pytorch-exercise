import os
import glob
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *

class Dataset(data.Dataset):
    def __init__(self, train, **kwargs):
        super(Dataset, self).__init__()

        self.size = kwargs.get("size", None)
        self.data_root = kwargs.get("data_root", "./data")
        
        self._prepare_dataset(self.data_root)
        
        phase = "train" if train else "test"
        dirname = os.path.join(self.data_root, "flower/{}".format(phase))

        self.paths = glob.glob(os.path.join(dirname, "*.jpg"))
  
    def __getitem__(self, index):
        image_raw = Image.open(self.paths[index])
        
        # resize original images
        if self.size:
            image_raw = image_raw.resize((self.size, self.size), Image.BICUBIC)
            
        image_raw = np.array(image_raw)
        
        # convert RGB image to Lab space
        image_lab = rgb2lab(image_raw).astype(np.float32)
        image_lab = (image_lab + 128) / 255
        
        image_ab = image_lab[:, :, 1:]
        image_ab = torch.from_numpy(image_ab.transpose((2, 0, 1)))
        
        image_gray = rgb2gray(image_raw).astype(np.float32)
        image_gray = torch.from_numpy(image_gray)

        return image_gray, image_ab

    def __len__(self):
        return len(self.paths)

    def _prepare_dataset(self, data_root):
        check = os.path.join(data_root, "flower")
        if not os.path.isdir(check):
            download_and_convert(data_root)