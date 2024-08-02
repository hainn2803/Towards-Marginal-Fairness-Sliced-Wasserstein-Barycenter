from dataloader.dataloader import STL10DataLoader
import torch
from swae.models.stl10 import STL10Autoencoder

dataloader = STL10DataLoader(data_dir="data/stl10/", train_batch_size=80, test_batch_size=80, image_size=64)

train_loader, test_loader = dataloader.create_dataloader()

for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    print("------------")
    break

model = STL10Autoencoder(embedding_dim=64*4)

x_recon, z = model(x)

print(x_recon.shape, z.shape)