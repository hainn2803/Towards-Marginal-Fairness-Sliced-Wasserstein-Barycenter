import torch
import torch.nn as nn



class STL10Encoder(nn.Module):

    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=32):
        super(STL10Encoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        assert self.embedding_dim_ % 4 == 0

        self.features = nn.Sequential(
            # 64x64
            nn.Conv2d(3, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_, self.init_num_filters_ * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.embedding_dim_ // 4, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.features(x) # shape == (num_image, self.embedding, 1, 1)
        x = x.view(x.shape[0], self.embedding_dim_) # shape == (num_image, self.embedding)
        return x


class STL10Decoder(nn.Module):

    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=32):
        super(STL10Decoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim_ // 4, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.init_num_filters_, self.init_num_filters_, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, z): # z.shape == (num_image, self.embedding)
        z = z.view(z.shape[0], self.embedding_dim_ // 4, 2, 2) # shape == (num_image, self.embedding //4, 2, 2)
        z = self.features(z)
        return torch.sigmoid(z)


class STL10Autoencoder(nn.Module):

    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=64*4):
        super(STL10Autoencoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim

        self.encoder = STL10Encoder(init_num_filters, lrelu_slope, embedding_dim)
        self.decoder = STL10Decoder(init_num_filters, lrelu_slope, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def generate(self, z):
        return self.decoder(z)
