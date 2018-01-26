import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DownModule(nn.Module):
    """
    Downscale module
    """

    def __init__(self, in_dims, out_dims, repeats=1):
        super(DownModule, self).__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        layers = [nn.Conv2d(out_dims, out_dims, 3, padding=1), nn.LeakyReLU(0.01, inplace=True)] * repeats
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return self.convs(x)


class UpModule(nn.Module):
    """
    Upscale module
    """

    def __init__(self, in_dims, out_dims):
        super(UpModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dims, out_dims, 2, stride=2)
        self.conv1x1 = nn.Conv2d(in_dims * 2, in_dims, 3, padding=1)
        self.insnorm = nn.InstanceNorm2d(in_dims, affine=True)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.smoothing = nn.Conv2d(out_dims, out_dims, 3, padding=1)

    def forward(self, x, y):
        x = F.upsample_bilinear(x, y.size()[2:])
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.insnorm(x)
        x = self.relu(self.conv(x))
        return self.relu(self.smoothing(x))


class Unet(nn.Module):
    """
    Deep neural network with skip connections
    """

    def __init__(self, num_z_channels=3, num_classes=3, image_channels=1, k=1, s=1):
        """
        Creates a u-net network
        :param in_dims: input image number of channels
        :param out_dims: number of feature maps
        :param: image_channels: number of channels in output generation image
        :param k: width coefficient
        """
        super(Unet, self).__init__()
        self.conv = nn.Conv2d(num_z_channels, 8 * k, 3, padding=1)
        self.d1 = DownModule(8 * k, 16 * k, s)
        self.d2 = DownModule(16 * k, 32 * k, s+1)
        self.d3 = DownModule(32 * k, 64 * k, s+2)
        self.d4 = DownModule(64 * k, 128 * k, s+3)
        self.u0 = UpModule(128 * k, 64 * k)
        self.u1 = UpModule(64 * k, 32 * k)
        self.u2 = UpModule(32 * k, 16 * k)
        self.u3 = UpModule(16 * k, 8 * k)

        self.conv1x1 = nn.Conv2d(8 * k, num_classes, 1, padding=0)
        self.conv_image = nn.Conv2d(8 * k, image_channels, 3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        a = self.d1(x)
        b = self.d2(a)
        c = self.d3(b)
        cc = self.d4(c)
        dd = self.u0(cc, cc)
        d = self.u1(dd, c)
        e = self.u2(b, d)
        f = self.u3(e, a)

        return F.tanh(self.conv_image(f)), self.conv1x1(f)
