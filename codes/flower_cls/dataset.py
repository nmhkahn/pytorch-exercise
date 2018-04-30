import os
import csv
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *

class Dataset(data.Dataset):
    # mapping table of label and index
    str2label = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}
    label2str = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}

    def __init__(self, train, **kwargs):
        super(Dataset, self).__init__()

        self.data = list()
        self.size = kwargs.get("size", None)
        self.data_root = kwargs.get("data_root", "./data")
        
        self._prepare_dataset(self.data_root)
        
        # load csv file and import file paths
        csv_name = "train.csv" if train else "test.csv"
        with open(os.path.join(self.data_root, "flower", csv_name)) as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        path, label = self.data[index]
        image = Image.open(path)
        
        # resize input images
        if self.size:
            image = image.resize((self.size, self.size), Image.BICUBIC)

        label = self.str2label[label]

        return self.transform(image), label

    def __len__(self):
        return len(self.data)

    def _prepare_dataset(self, data_root):
        check = os.path.join(data_root, "flower")
        if not os.path.isdir(check):
            download_and_convert(data_root)
