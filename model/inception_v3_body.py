from __future__ import print_function, division
import sys
sys.path.insert(0, r"D:\BaiduNetdiskDownload\temp\multilabel")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

from data_process import dataset


root_path = r"D:\BaiduNetdiskDownload\temp\TinyMind"
train_tags_path = "tag_train.npz"
train_img_dir = "train"
train_img_order_csv = "visual_china_train1.csv"

tiny_mind_dataset = dataset.TinyMindDataset(root_path, train_tags_path, train_img_dir, train_img_order_csv)
# for img, label in tiny_mind_dataset:
#   print(type(img), type(label))
#   break

tiny_mind_dataloader = DataLoader(tiny_mind_dataset, batch_size=4,
                        shuffle=True)

# print(torch.cuda.current_device())
device = torch.device("cpu")
print(device)
# model_ft = models.inception_v3(pretrained=True)
# n = model_ft.fc.in_features
# model_ft.fc = nn.Linear(n, 6941)
model_ft = models.vgg16_bn(pretrained=True)

for parma in model_ft.parameters():
    parma.requires_grad = False

model_ft.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 6941))

# print(model_ft.state_dict())
# print(model_ft.fc.in_features)
# exit()

# print(model_ft)
model_ft = model_ft.to(device)

criterion = nn.MultiLabelSoftMarginLoss()

optimizer_ft = optim.Adam(model_ft.classifier.parameters())
# def train()

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
num_epochs = 5
for epoch in range(num_epochs):
  losses = []
  for imgs, labels in tiny_mind_dataloader:
    # print(imgs.size())
    imgs = imgs.to(device)
    # print(imgs.size())
    labels = labels.to(device)
    # print(labels.size())
    # break
    # print(type(imgs))
    # print(imgs.size())
    # exit()


    outputs = model_ft(imgs)
    loss = criterion(outputs, labels)
    optimizer_ft.zero_grad()

    loss.backward()

    optimizer_ft.step()

    losses.append(loss.data.mean())
    # print(type(outputs))
    # print(outputs.size())
    # print(outputs[0].size())
    # print(outputs[1].size())
    # torch.mean()
  print(np.mean(losses))
    # break
