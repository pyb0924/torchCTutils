import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DoubleConv2d(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


class DoubleConv3d(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv3d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm3d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)


class UNetEncoder2d(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], hierarchical=True):
        super(UNetEncoder2d, self).__init__()
        self.hierarchical = hierarchical
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv2d(features[-1], features[-1] * 2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        features.append(x)
        if self.hierarchical:
            return features
        else:
            return features[-1]


class UNetDecoder2d(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetDecoder2d, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2d(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, features: list[Tensor]):
        x = features.pop()
        skip_connections = features[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class UNet2d(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet2d, self).__init__()
        self.encoder = UNetEncoder2d(in_channels, features)
        self.decoder = UNetDecoder2d(out_channels, features)


class UNetEncoder3d(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512],hierarchical=True):
        super(UNetEncoder3d, self).__init__()
        self.hierarchical = hierarchical
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv3d(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        features.append(x)
        if self.hierarchical:
            return features
        else:
            return features[-1]


class UNetDecoder3d(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetDecoder3d, self).__init__()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3d(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, features: list[Tensor]):
        x = features.pop()
        skip_connections = features[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class UNet3d(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3d, self).__init__()
        self.encoder = UNetEncoder3d(in_channels, features)
        self.decoder = UNetDecoder3d(out_channels, features)
