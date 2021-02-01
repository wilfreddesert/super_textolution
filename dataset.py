import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name = dirname.split("/")[-1]
        self.scale = scale
        all_files = glob.glob(os.path.join(dirname, "*.png"))
        all_files.extend(glob.glob(os.path.join(dirname, "*.jpg")))
        all_files.extend(glob.glob(os.path.join(dirname, "*.jpeg")))
        print(all_files)
        # Removed if "LR" in name part
        self.lr = [name for name in all_files]
        self.lr.sort()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        lr = Image.open(self.lr[index])
        lr = lr.convert("RGB")
        filename = self.lr[index].split("/")[-1]
        return self.transform(lr), filename

    def __len__(self):
        return len(self.lr)
