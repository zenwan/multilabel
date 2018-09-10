from __future__ import print_function, division

import torch

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



def model_vgg(device):
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
  return model_ft

