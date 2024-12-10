import paddle
from paddle.io import Dataset
from PIL import Image
import os
import numpy as np
import random
from copy import deepcopy

"""
If you have existing mask_data, use the former class,
else try the latter.
"""

class MyData_label(Dataset):
    def __init__(self,imgmask_path,label_path,transform=None):
        super().__init__()
        self.image_paths = os.listdir(label_path)
        self.label_paths = []
        self.imgmask_paths = []
        self.transform = transform
        for dir in imgmask_path:
            for path in self.image_paths:
                self.imgmask_paths.append(dir+path)
                self.label_paths.append(label_path+path)

    def __getitem__(self, idx):
        label = np.array(Image.open(self.label_paths[idx]).resize((512,512))).astype("int")
        imgmask = np.array(Image.open(self.imgmask_paths[idx]).resize((512,512)))
        if self.transform is not None:
            imgmask = self.transform(imgmask)
        return imgmask,label
    def __len__(self):
        return len(self.imgmask_paths)

class MyData_mask(Dataset):
    def __init__(self, image_path, label_path):
        super().__init__()
        self.image_paths = os.listdir(image_path)
        self.image_path = image_path
        self.label_path = label_path

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_path + "/" + self.image_paths[idx]).resize((512, 512)))
        label = np.array(
            Image.open(self.label_path + "/" + self.image_paths[idx][:-3] + "png").resize((512, 512))).astype(
            "int8")
        image_mask = deepcopy(image)
        locx = random.randint(0, 4)
        locy = random.randint(0, 4)
        image_mask[locx * 120:(locx + 1) * 120, locy * 120:(locy + 1) * 120, :] = 0
        return image_mask, image, label

    def __len__(self):
        return len(self.image_paths)
