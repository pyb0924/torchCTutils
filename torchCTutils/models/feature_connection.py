from functools import reduce
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F

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

    def __init__(self, input_channels, output_channels, output_depth):
        super(FeatureConnectionB, self).__init__()
        self.output_depth = output_depth
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
        feature = (
            self.conv2d(feature).unsqueeze(2).expand(-1, -1, self.output_depth, -1, -1)
        )
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
    """2D => Combination => (3D) => Interpolation => FBP => 3D"""

    def __init__(
        self,
        size,
        dim: Literal[2, 3] = 2,
        angles=None,
        detector_shape=None,
        interpolate=False,
    ):
        super().__init__()
        self.angles = angles if angles is not None else size
        self.fbp = odl_FBP_layer(size, dim, self.angles, detector_shape)
        self.interpolate = interpolate

    def circle_interpolate(self, features: tuple[Tensor]):
        projections = torch.cat(
            [f.unsqueeze(2) for f in features] + [features[0].unsqueeze(2)], dim=2
        )
        projections = F.interpolate(
            projections,
            size=(self.angles + 1, projections.shape[-2], projections.shape[-1]),
            mode="trilinear",
            align_corners=True,
        )
        return projections[:, :, :-1, :, :]

    def forward(self, *features: Tensor):
        if self.interpolate:
            projections = self.circle_interpolate(features)
        else:
            projections = torch.cat([f.unsqueeze(2) for f in features], dim=2)
        return self.fbp(projections).permute(0, 1, 4, 2, 3)
        # [batch_size, channels, size, size, size]
