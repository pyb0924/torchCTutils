from torch import nn


class ConvWithLeakyReLU2d(nn.Sequential):
    """(convolution => LeakyReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvWithLeakyReLU2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.2)


class PatchGANDiscriminator2d(nn.Sequential):
    def __init__(self, in_channels=3, num_blocks=3):
        super(PatchGANDiscriminator2d, self).__init__()
        input_channels = 64
        self.input = ConvWithLeakyReLU2d(in_channels, input_channels)
        self.encoder = nn.Sequential(
            *[
                ConvWithLeakyReLU2d(
                    input_channels * 2**i, input_channels * 2 ** (i + 1)
                )
                for i in range(num_blocks)
            ]
        )
        self.output = nn.Conv2d(
            input_channels * 2**num_blocks, 1, kernel_size=4, stride=1, padding=1
        )
        self.sigmoid = nn.Sigmoid()


class ConvWithLeakyReLU3d(nn.Sequential):
    """(convolution => LeakyReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvWithLeakyReLU3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.2)


class PatchGANDiscriminator3d(nn.Sequential):
    def __init__(self, in_channels=3, num_blocks=3):
        super(PatchGANDiscriminator3d, self).__init__()
        input_channels = 64
        self.input = ConvWithLeakyReLU3d(in_channels, input_channels)
        self.encoder = nn.Sequential(
            *[
                ConvWithLeakyReLU3d(
                    input_channels * 2**i, input_channels * 2 ** (i + 1)
                )
                for i in range(num_blocks)
            ]
        )
        self.output = nn.Conv3d(
            input_channels * 2**num_blocks, 1, kernel_size=4, stride=1, padding=1
        )
        self.sigmoid = nn.Sigmoid()
