import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
from mean_iou_evaluate import *
from p3_datasets import *

trainDataPath = './hw1_data_4_students/hw1_data/p3_data/train'
testDataPath = './hw1_data_4_students/hw1_data/p3_data/validation'

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

train_tfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.ColorJitter(brightness=(0.5,2)),
            # transforms.ColorJitter(contrast=(1,5)),
            # transforms.ColorJitter(saturation=(0.1,4)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
test_tfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
num_classes = 7

# class fcn32(nn.Module):
#     def __init__(self,num_classes):
#         super(fcn32,self).__init__()
#         # self.pretrained_model=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         self.fconv=nn.Sequential(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features,
#                                 nn.Conv2d(512,4096,7),
#                                 nn.ReLU(inplace=True),
#                                 nn.Dropout2d(),
#                                 nn.Conv2d(4096,4096,1),
#                                 nn.ReLU(inplace=True),
#                                 nn.Dropout2d(),
#                                 nn.Conv2d(4096, num_classes, 1),
#                                 nn.ConvTranspose2d(num_classes, num_classes, 224, stride=32, bias=False)
#                                 )
#     def forward(self,x):
#         score = self.fconv(x)
#         return score


device = "cuda" if torch.cuda.is_available() else "cpu"
# model = fcn32(num_classes).to(device)
model = models.segmentation.deeplabv3_resnet50(weights = 'DEFAULT')
# model = models.segmentation.deeplabv3_resnet50(weights = None)
model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load('p3_B_SGD_larger_batch_best_0.729.ckpt'), strict=False)
print(model)
model = model.to(device)
# print(model)

batch_size = 5 # 調高試試看
n_epochs = 50
patience = 100
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD(model.parameters(), momentum = 0.85, lr=5e-3, weight_decay=5e-4) #表現還好

if __name__ == "__main__":
    train_set = SegmentationDataset(root=trainDataPath, transform=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    valid_set = SegmentationDataset(root=testDataPath, transform=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
    stale = 0
    best_miou = 0 #[1, 3, 512, 512]
    trainLosses = []

    for epoch in range(n_epochs):

        model.train()
        train_loss = []
        for batch in tqdm(train_loader):

            imgs, labels = batch
            labels = labels.squeeze(1).to(device, dtype=torch.long)
            logits = model(imgs.to(device))['out']
            # print(logits)
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            train_loss.append(loss.item())
            # torch.cuda.empty_cache()

        train_loss = sum(train_loss) / len(train_loss)
        trainLosses.append(train_loss)
        # scheduler.step(train_loss)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

        # # ---------- Validation ----------
        model.eval()
        valid_loss = []
        pred_mask_cpu = []
        labels_cpu = []
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                imgs, labels = batch
                labels = labels.squeeze(1).to(device, dtype=torch.long)
                logits = model(imgs.to(device))['out']
                # logits = model(imgs.to(device))
                loss = criterion(logits, labels)
                valid_loss.append(loss.item())
                pred_mask = logits.argmax(dim=1)
                pred_mask_cpu.append(pred_mask.cpu())
                labels_cpu.append(labels.cpu())


        valid_loss = sum(valid_loss) / len(valid_loss)
        pred_mask_numpy = torch.cat(pred_mask_cpu).numpy()
        labels_numpy = torch.cat(labels_cpu).numpy()
        valid_miou = mean_iou_score(pred_mask_numpy, labels_numpy)
        # torch.cuda.empty_cache()

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, valid_miou = {valid_miou:.5f}")

        # # update logs
        # if valid_acc > best_acc:
        #     with open(f"./E_log.txt","a") as F:
        #         print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        #         print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best", file=F)
        # else:
        #     # with open(f"./_log.txt","a"):
        #         print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # # save models
        # if (epoch+1 == 1):
        #     print(f"Saving model from epoch {1}")
        #     torch.save(model.state_dict(), f"p3_B_cj_1.ckpt")

        # if (epoch+1 == 25):
        #     print(f"Saving model from epoch {1}")
        #     torch.save(model.state_dict(), f"p3_B_cj_25.ckpt")

        # if (epoch+1 == 50):
        #     print(f"Saving model from epoch {50}")
        #     torch.save(model.state_dict(), f"p3_B_cj_50.ckpt")
        
        # if (epoch+1 == 100):
        #     print(f"Saving model from epoch {100}")
        #     torch.save(model.state_dict(), f"p3_B_cj_100.ckpt")

        # if valid_miou > best_miou:
        #     print(f"Best model found at epoch {epoch+1}, saving model")
        #     torch.save(model.state_dict(), f"p3_B_cj_best.ckpt") # only save best to prevent output memory exceed error
        #     best_miou = valid_miou
        #     stale = 0
        # else:
        #     stale += 1
        #     if stale > patience:
        #         print(f"No improvment {patience} consecutive epochs, early stopping")
        #         print(f"Saving model from the last epoch before early stopping")
        #         torch.save(model.state_dict(), f"p3_B_cj_last_epoch.ckpt")
        #         break






