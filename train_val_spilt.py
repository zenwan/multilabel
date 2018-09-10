import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch

def train_val_split(mydset):

  num_train = len(mydset)
  indices = list(range(num_train))
  split = 5000

  # Random, non-contiguous split
  validation_idx = np.random.choice(indices, size=split, replace=False)
  train_idx = list(set(indices) - set(validation_idx))
  print("val_len: {}".format(len(validation_idx)))
  print("train_len: {}".format(len(train_idx)))

  # Contiguous split
  # train_idx, validation_idx = indices[split:], indices[:split]

  ## define our samplers -- we use a SubsetRandomSampler because it will return
  ## a random subset of the split defined by the given indices without replaf
  train_sampler = SubsetRandomSampler(train_idx)
  validation_sampler = SubsetRandomSampler(validation_idx)

  train_loader = torch.utils.data.DataLoader(mydset,
                  batch_size=4, sampler=train_sampler)

  validation_loader = torch.utils.data.DataLoader(mydset,
                  batch_size=10, sampler=validation_sampler)

  return train_loader, validation_loader