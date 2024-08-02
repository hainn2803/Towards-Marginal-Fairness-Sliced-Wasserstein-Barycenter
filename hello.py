from dataloader.dataloader import MNISTLTDataLoader
import numpy as np
import torch

data_loader = MNISTLTDataLoader(data_dir="data/mnist", train_batch_size=1000, test_batch_size=128)
train_loader, test_loader = data_loader.create_dataloader()

for x, y in train_loader:
    for cls in range(10):
        print(cls, torch.sum(y == cls))