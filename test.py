import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
import argparse
import glob
import imageio
import sys
from mean_iou_evaluate import *
# from p3_datasets import *

test_tfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
oc_cls = {'urban':0,
           'rangeland': 2,
           'forest':3,
           'unknown':6,
           'barreb land':5,
           'Agriculture land':1,
           'water':4}
cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

class testDataset(Dataset):
    def __init__(self, root, transform=None):
        super(testDataset).__init__()
        images = glob.glob(os.path.join(root, '*.jpg'))
        images.sort()

        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image,img_path

def viz_data(im, seg, color, inner_alpha = 0.3):
    color_mask = np.zeros((im.shape[0]*im.shape[1], 3))
    l_loc = np.where(seg.flatten() == 1)[0]
    color_mask[l_loc, : ] = color
    color_mask = np.reshape(color_mask, im.shape)
    mask = np.concatenate((seg[:,:,np.newaxis],seg[:,:,np.newaxis],seg[:,:,np.newaxis]), axis = -1)
    im_new = im*(1-mask) + im*mask + color_mask

    return im_new

def saveImage(masks, img_path, SAVE_PATH):
    np.set_printoptions(threshold=np.inf)

    #img_path = xxxx_sat.jpg
    idx = img_path.split("/")[-1].split("_")[0] # xxxx
    
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    cmap = cls_color
    cs = np.unique(masks)
    img = np.zeros((masks.shape[0],masks.shape[1], 3))

    for c in cs:
        mask = np.zeros((masks.shape[0], masks.shape[1]))
        ind = np.where(masks==c)
        mask[ind[0], ind[1]] = 1
        img = viz_data(img, mask, color=cmap[c])

    mask_path = idx + '_mask.png'
    path = os.path.join(SAVE_PATH, mask_path)
    # path = SAVE_PATH +'/' + idx + '_mask.png'
    # img.save(path)
    imageio.imsave(path, np.uint8(img)) 



if __name__ == '__main__':


    TEST_IMAGE_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    # TEST_IMAGE_PATH = "./hw1_data_4_students/hw1_data/p3_data/validation"
    # OUTPUT_PATH = "./p3_seg/"
    MODEL_LOAD_PATH = "./p3_model.ckpt" # model path
    batch_size = 1
    test_set = testDataset(root=TEST_IMAGE_PATH, transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.segmentation.deeplabv3_resnet50(weights = 'DEFAULT')
    model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(MODEL_LOAD_PATH), strict=False)
    model = model.to(device)

    # testing
    model.eval()
    for batch in tqdm(test_loader):
        with torch.no_grad():
            imgs, imgs_path = batch
            # labels = labels.squeeze(1)
            # labels = labels.to(device, dtype=torch.long)
            logits = model(imgs.to(device))['out']
            pred = logits.argmax(dim=1)
            for j in range(pred.shape[0]):
                saveImage(pred[j].cpu().numpy(), imgs_path[j], OUTPUT_PATH)
    
    # pred_masks_numpy = torch.cat(pred_masks_cpu).numpy()
    # labels_numpy = torch.cat(labels_cpu).numpy()
    # test_miou = mean_iou_score(pred_masks_numpy, labels_numpy)

    # print("test_miou: ", test_miou)