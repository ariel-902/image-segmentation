import numpy as np
import pandas as pd
import torch
import os
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
from mean_iou_evaluate import *
import glob


class SegmentationDataset(Dataset):
    def __init__(self,path,transform=None,files=None):
        super(SegmentationDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.labels = read_masks(path)
        if files != None:
            self.files = files
        self.transform = transform
        # self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        label = self.labels[idx]
        im = Image.open(fname)

        if (self.transform != None):
            im = self.transform(im)
            label = self.transform(label)
        
        return im,label
            # print("im:", im.size())
            # print("label:", label.size())
        # label= transforms.ToTensor()(label)
        # if self.train:
        #     if self.rand() < 0.5:
        #         torch.flip(im, [1, 2])
        #         torch.flip(label, [0, 1])

        #     if self.rand() < 0.5:
        #         torch.flip(im, [2, 1])
        #         torch.flip(label, [1, 0])

        #     rotation = self.rand()
        #     # print("original", im.size())
        #     if rotation < 0.25:
        #         im = im
        #         label = label
        #     elif rotation < 0.5:
        #         im = torch.rot90(im, 1, [1, 2])
        #         label = torch.rot90(label, 1, [1, 2])
        #     elif rotation < 0.75:
        #         im = torch.rot90(im, 2, [1, 2])
        #         # print("rotate: ", im.size())
        #         label = torch.rot90(label, 2, [1, 2])
        #     else:
        #         im = torch.rot90(im, 3, [1, 2])
        #         # print("rotate: ", im.size())
        #         label = torch.rot90(label, 3, [1, 2])

        # return image, masks.long(), mask_path
        # print("label:", label.size())

    # def rand(self, a=0, b=1):
    #     return np.random.rand() * (b - a) + a
