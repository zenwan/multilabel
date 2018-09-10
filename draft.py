# -*- coding:utf-8 -*-
from PIL import Image

img_path = r"D:\BaiduNetdiskDownload\temp\TinyMind\train_sample\00afb07c10e8f614f6e66db73a4bc09a6d630d6d.jpg"
img = Image.open(img_path).convert("RGB")
img.resize((224))


for epoch in range(num_epoch):
  for phase in ["train", "val"]: