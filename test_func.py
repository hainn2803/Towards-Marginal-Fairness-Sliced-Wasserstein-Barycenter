from swae.models.celeba import CelebAAutoencoder, CelebAEncoder, CelebADecoder
import torchvision.transforms as transforms
from dataloader.dataloader import CelebADataLoader
import torch
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    dataloader = CelebADataLoader(data_dir="data/celeba", train_batch_size=80, test_batch_size=80)
    train_loader, test_loader = dataloader.create_dataloader()
    for x, y in train_loader:
        print(x.shape, y.shape)

    # root_dir = "data/celeba"
    # print(os.path.exists(root_dir))
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(), 
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    # celeba_train = CelebA(root=root_dir, split='train', target_type='attr', transform=transform, download=False)
    # celeba_test = CelebA(root=root_dir, split='test', target_type='attr', transform=transform, download=False)
    
    # print(celeba_train.attr[:, 20].shape)

    # sample_index = 0
    # image, target = celeba_train[sample_index]
    # print("Sample image shape:", image.shape)
    # print("Sample attributes:", target)
    # print(target[20])
    
    # image_np = image.numpy()
    # image_np = image_np.transpose((1, 2, 0))
    # plt.imshow(image_np)
    # plt.axis('off')
    
    
    # x = torch.randn(5, 3, 64, 64)
    
    # enc = CelebAEncoder(in_channels=3, latent_dim=48, hidden_dims=None)
    # dec = CelebADecoder(in_channels=3, latent_dim=48, hidden_dims=None)
    # autoencoder = CelebAAutoencoder(in_channels=3, latent_dim=48, hidden_dims=None)
    
    # z = enc(x)
    # x_recon = dec(z)
    # x_rec = autoencoder(x)
    
    # print(z.shape, x_recon.shape, x_rec[0].shape, x_rec[1].shape)