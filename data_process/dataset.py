import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class TinyMindDataset(Dataset):
  def __init__(self, root_path, train_tags_path, train_img_dir, train_img_order_csv):
    self.transformer = transforms.ToTensor()
    self.train_tags = np.load(os.path.join(root_path, train_tags_path))["tag_train"]
    self.train_imgs = self.train_img_order(root_path, train_img_dir, train_img_order_csv)

  def img_loader(self, img_path):
    return Image.open(img_path).resize((224,224))

  def train_img_order(self, root_path, train_img_dir ,train_img_order_csv):
    train_img_paths = pd.read_csv(os.path.join(root_path, train_img_order_csv)).iloc[:,0]
    train_imgs = []
    for img_path in train_img_paths:
      train_imgs.append(os.path.join(root_path, train_img_dir, img_path))
    return train_imgs

  def __getitem__(self, index):
    img = self.img_loader(self.train_imgs[index])
    img = self.transformer(img)
    tag = torch.from_numpy(self.train_tags[index]).float()
    return img, tag

  def __len__(self):
    assert len(self.train_imgs) == self.train_tags.shape[0]
    return len(self.train_imgs)

if __name__ == '__main__':
  root_path = r"D:\BaiduNetdiskDownload\temp\TinyMind"
  train_tags_path = "tag_train.npz"
  train_img_dir = "train"
  train_img_order_csv = "visual_china_train1.csv"

  tiny_mind_dataset = TinyMindDataset(root_path, train_tags_path, train_img_dir, train_img_order_csv)
  for img, label in tiny_mind_dataset:
    print(type(img), type(label))
    break

  tiny_mind_dataloader = DataLoader(tiny_mind_dataset, batch_size=4,
                          shuffle=True)

  for imgs, labels in tiny_mind_dataloader:
    print(imgs.size())
    print(labels.size())
    break




