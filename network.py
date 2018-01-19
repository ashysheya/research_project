import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, num_z_channels=1, num_classes=10):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_z_channels, out_channels=64, kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.upconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                         padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, z):
        conv_1_output = F.relu(self.conv1(z))
        conv_2_output = F.relu(self.conv2(self.max_pool(conv_1_output)))
        upconv_output = torch.cat((self.upconv(conv_2_output), conv_1_output), 1)
        conv_3_output = F.relu(self.conv3(upconv_output))

        # image restoration
        conv_4_1_output = F.relu(self.conv4_1(conv_3_output))
        generated_image = F.tanh(self.conv5_1(conv_4_1_output))

        # segmentation
        conv_4_2_output = F.relu(self.conv4_2(conv_3_output))
        generated_segmentation = self.conv5_2(conv_4_2_output)

        return generated_image, generated_segmentation
