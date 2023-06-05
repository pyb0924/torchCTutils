from torch import nn


class ConvWithLeakyReLU2d(nn.Sequential):
    """(convolution => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, use_norm=True):
        super(ConvWithLeakyReLU2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        if use_norm:
            self.norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, True)


class PatchGANDiscriminator2d(nn.Sequential):
    def __init__(self, in_channels=1, num_blocks=3):
        super(PatchGANDiscriminator2d, self).__init__()
        input_channels = 64
        self.input = ConvWithLeakyReLU2d(in_channels, input_channels, use_norm=False)
        self.encoder = nn.Sequential(
            *[
                ConvWithLeakyReLU2d(
                    input_channels * 2**i, input_channels * 2 ** (i + 1)
                )
                for i in range(num_blocks - 1)
            ]
        )
        last_channel = input_channels * 2**num_blocks
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                last_channel // 2,
                last_channel,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(last_channel),
            nn.LeakyReLU(0.2, True),
        )
        self.output = nn.Conv2d(last_channel, 1, kernel_size=4, stride=1, padding=1)


class ConvWithLeakyReLU3d(nn.Sequential):
    """(convolution => LeakyReLU)"""

    def __init__(self, in_channels, out_channels, use_norm=True):
        super(ConvWithLeakyReLU3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        if use_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, True)



class PatchGANDiscriminator3d(nn.Sequential):
    def __init__(self, in_channels=1, num_blocks=3):
        super(PatchGANDiscriminator3d, self).__init__()
        input_channels = 64
        self.input = ConvWithLeakyReLU3d(in_channels, input_channels, use_norm=False)
        self.encoder = nn.Sequential(
            *[
                ConvWithLeakyReLU3d(
                    input_channels * 2**i, input_channels * 2 ** (i + 1)
                )
                for i in range(num_blocks - 1)
            ]
        )
        last_channel = input_channels * 2**num_blocks
        self.last_conv = nn.Sequential(
            nn.Conv3d(
                last_channel // 2,
                last_channel,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm3d(last_channel),
            nn.LeakyReLU(0.2, True),
        )
        self.output = nn.Conv3d(last_channel, 1, kernel_size=4, stride=1, padding=1)
