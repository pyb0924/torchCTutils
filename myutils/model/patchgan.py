from torch import nn
import torch.nn.functional as F


class ConvWithLeakyRelU(nn.Module):
    """(convolution => LeakyReLU) """

    def __init__(self, in_channels, out_channels):
        self.conv = nn.Conv2d(in_channels, input_channels,
                              kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_blocks=3):
        super(PatchGANDiscriminator, self).__init__()
        input_channels = 64
        self.input = ConvWithLeakyRelU(in_channels, input_channels)
        self.encoder = nn.Sequential(
            *[ConvWithLeakyRelU(input_channels * 2**i, input_channels * 2**(i + 1)) for i in range(num_blocks)])
        self.output = nn.Conv2d(
            input_channels * 2**num_blocks, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.output(x)
        return F.sigmoid(x)
