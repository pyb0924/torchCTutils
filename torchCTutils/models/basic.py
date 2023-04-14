import torch
from torch import nn


class ConvBlock2d(nn.Sequential):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


class ConvBlock3d(nn.Sequential):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)


class BasicEncoder2d(nn.Sequential):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(BasicEncoder2d, self).__init__()
        for i, feature in enumerate(features):
            self.add_module(f"conv{i}", ConvBlock2d(in_channels, feature))
            self.add_module(f"pool{i}", nn.MaxPool2d(2))
            in_channels = feature


class BasicDecoder2d(nn.Sequential):
    def __init__(self, in_channels=512, features=[64, 128, 256, 512]):
        super(BasicDecoder2d, self).__init__()
        features = reversed(features)
        for i, feature in enumerate(features):
            self.add_module(f"up{i}", nn.Upsample(scale_factor=2))
            self.add_module(f"conv{i}", ConvBlock2d(in_channels, feature))
            in_channels = feature


class BasicEncoder3d(nn.Sequential):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(BasicEncoder3d, self).__init__()
        for i, feature in enumerate(features):
            self.add_module(f"conv{i}", ConvBlock3d(in_channels, feature))
            self.add_module(f"pool{i}", nn.MaxPool3d(2))
            in_channels = feature


class BasicDecoder3d(nn.Sequential):
    def __init__(self, in_channels=512, features=[64, 128, 256, 512]):
        super(BasicDecoder3d, self).__init__()
        features = reversed(features)
        for i, feature in enumerate(features):
            self.add_module(f"up{i}", nn.Upsample(scale_factor=2))
            self.add_module(f"conv{i}", ConvBlock3d(in_channels, feature))
            in_channels = feature
