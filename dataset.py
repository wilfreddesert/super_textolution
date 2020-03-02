import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
   # h, w = lr.shape
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")
        
        self.hr = [v[:] for v in h5f["HR"].values()]
        #print(f"SELF.HR {self.hr}")
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
            #print(f"SELF.LR {self.lr}")
        
        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size
        #print(f"SIZE {size}")
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        #print(f"ITEM {item}")
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        all_files = glob.glob(os.path.join(dirname, "*.png"))
        # Removed if "LR" in name part
        self.lr = [name for name in all_files]
        self.lr.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        lr = Image.open(self.lr[index])
        lr = lr.convert("RGB")
        filename = self.lr[index].split("/")[-1]
        return self.transform(lr), filename

    def __len__(self):
        return len(self.lr)
