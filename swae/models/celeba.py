import torch.nn as nn
import torch.nn.functional as F
import torch


class CelebAEncoder(nn.Module):
    
    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(CelebAEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)
    
    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z
        
class CelebADecoder(nn.Module):
    
    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(CelebADecoder, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    
    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
class CelebAAutoencoder(nn.Module):
    
    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(CelebAAutoencoder, self).__init__()

        self.encoder = CelebAEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = CelebADecoder(in_channels, latent_dim, hidden_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def generate(self, z):
        return self.decoder(z)