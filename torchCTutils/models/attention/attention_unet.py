import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..unet import DoubleConv2d, DoubleConv3d


class AttentionGate2d(nn.Module):
    def __init__(self, gate_channels, input_channels, attention_channels):
        super(AttentionGate2d, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                attention_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(attention_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                input_channels,
                attention_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(attention_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                attention_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionGate3d(nn.Module):
    def __init__(self, gate_channels, input_channels, attention_channels):
        super(AttentionGate3d, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(
                gate_channels,
                attention_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(attention_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(
                input_channels,
                attention_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(attention_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(
                attention_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNetDecoder2d(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNetDecoder2d, self).__init__()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2d(feature * 2, feature))
            self.attentions.append(AttentionGate2d(feature, feature, feature // 2))

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
            attention_value = self.attentions[idx // 2](skip_connection, x)

            concat_skip = torch.cat((attention_value, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class AttentionUNetDecoder3d(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNetDecoder3d, self).__init__()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3d(feature * 2, feature))
            self.attentions.append(AttentionGate3d(feature, feature, feature // 2))

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
            attention_value = self.attentions[idx // 2](skip_connection, x)

            concat_skip = torch.cat((attention_value, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
