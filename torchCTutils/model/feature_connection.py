from functools import reduce

import torch
from torch import nn, Tensor

from ..odl import odl_FBP_layer


class FeatureConnectionA(nn.Module):
    """2D => Flatten => FC => Dropout => ReLU => Reshape => 3D"""

    def __init__(self, feature_shape, output_size):
        super(FeatureConnectionA, self).__init__()
        self.feature_shape = feature_shape
        self.output_size = output_size
        _, channels, h, w = self.feature_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * h * w, output_size**3)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature) -> Tensor:
        feature = self.relu(self.dropout(self.fc(self.flatten(feature))))
        batch_size = self.feature_shape[0]
        return feature.reshape(
            (
                batch_size,
                1,
                self.output_size,
                self.output_size,
                self.output_size,
            )
        ).expand(-1, self.feature_shape[1], -1, -1, -1)


class FeatureConnectionB(nn.Module):
    """2D => Conv-IN-ReLU(2D) => Expand => Conv-IN-ReLU(3D) => 3D"""

    def __init__(self, input_channels, output_channels):
        super(FeatureConnectionB, self).__init__()
        self.output_channels = output_channels
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        size = feature.shape[-1]
        feature = self.conv2d(feature).unsqueeze(2).expand(-1, -1, size, -1, -1)
        return self.conv3d(feature)


class FeatureConnectionC(nn.Module):
    """3D => Permute => Add => Average => 3D"""

    def __init__(self):
        super(FeatureConnectionC, self).__init__()

    def forward(self, features):
        features = list(map(lambda x: torch.permute(x, (0, 1, 2, 4, 3)), features))
        features = reduce(lambda x, y: x + y, features, 0)
        return features


class FeatureFBPConnection(nn.Module):
    """2D => Combination => (3D) => FP => Interpolation => FBP => 3D"""

    def __init__(self, feature_shape, angles=60, output_size=64):
        super().__init__()
        self.output_size = output_size
        _, channels, h, w = feature_shape

        self.fbp = odl_FBP_layer(output_size, angles, dim=3)

    def forward(self, *features):
        pass
