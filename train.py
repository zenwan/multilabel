import sys
sys.path.insert(0, r"D:\BaiduNetdiskDownload\temp\multilabel")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_process import dataset
from torch.utils.data import DataLoader

from model import vgg16_bn_body
from data_process.train_val_spilt import train_val_split
import copy

if __name__ == "__main__":

  root_path = r"D:\BaiduNetdiskDownload\temp\TinyMind"
  train_tags_path = "tag_train.npz"
  train_img_dir = "train"
  train_img_order_csv = "visual_china_train1.csv"

  tiny_mind_dataset = dataset.TinyMindDataset(root_path, train_tags_path, train_img_dir, train_img_order_csv)

  train_loader, validation_loader = train_val_split(tiny_mind_dataset)
  # tiny_mind_dataloader = DataLoader(tiny_mind_dataset, batch_size=5, shuffle=True)

  dataloaders = {"train": train_loader,
                 "val": validation_loader,}

  dataset_sizes =  {"train": 30000,
                    "val": 5000,}
# print(torch.cuda.current_device())
#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print(device)

  model_ft = vgg16_bn_body.model_vgg(device)

# model_ft = models.inception_v3(pretrained=True)
# n = model_ft.fc.in_features
# model_ft.fc = nn.Linear(n, 6941)

  criterion = nn.MultiLabelSoftMarginLoss()

  optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=1e-5)
# def train()

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
  num_epochs = 20
  best_loss = 99999

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    for phase in ['train', 'val']:
      if phase == 'train':
        model_ft.train()  # Set model to training mode
      else:
        model_ft.eval()  # Set model to evaluate mode

      running_loss = 0.0
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model_ft(inputs)

          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer_ft.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
      epoch_loss = running_loss / dataset_sizes[phase]

      print('{} Loss: {:.4f} '.format(
        phase, epoch_loss))

      if phase == 'val' and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        print("saving ....")
        torch.save(model_ft.state_dict(), 'params.pkl')


    # print("training")
    # print('-' * 10)
    # train_running_loss = 0.0
    # model_ft.train()
    #
    # for idx, (imgs, labels) in enumerate(tiny_mind_dataloader):
    #   imgs = imgs.to(device)
    #   labels = labels.to(device)
    #
    #   # print(imgs.size())
    #   # print(labels.size())
    #
    #   optimizer_ft.zero_grad()
    #
    #   outputs = model_ft(imgs)
    #   loss = criterion(outputs, labels)
    #
    #   loss.backward()
    #   optimizer_ft.step()
    #
    #   if idx % 10 == 0:
    #     print("train_loss: {}".format(loss))
    #
    #     # val_loss = 0.0
    #     # for imgs_val, labels_val in validation_loader:
    #     #   imgs_val = imgs_val.to(device)
    #     #   labels_val = labels_val.to(device)
    #     #   outputs = model_ft(imgs_val)
    #     #   loss = criterion(outputs, labels_val)
    #     #   val_loss += loss
    #     # print("val_loss: {}".format(val_loss))
    #
    #   if idx % 5000 == 0:
    #     print("saving model")
    #     torch.save(model_ft.state_dict(), 'params.pkl')
        #


    # break

