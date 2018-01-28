import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class NetG(nn.Module):
    def __init__(self, z_dim=100, ngf=32, num_channels=1, num_classes=3):
        super(NetG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
            # state size. (nc) x 64 x 64
        )
        self.image = nn.Sequential(
            nn.Conv2d(ngf, num_channels, 3, padding=1),
            nn.Tanh()
        )

        self.segmentation = nn.Sequential(
            nn.Conv2d(ngf, num_classes, 1, padding=0)
        )

    def forward(self, input):
        main_output = self.main(input)
        image = self.image(main_output)
        segmentation = self.segmentation(main_output)
        return image, segmentation
