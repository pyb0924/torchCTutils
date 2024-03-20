from functools import reduce
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..odl import odl_FBP_layer
from ..tomo import BPlayer


class FlattenConnection(nn.Module):
    """2D => Flatten => FC => Dropout => ReLU => Reshape => 3D"""

    def __init__(self, feature_shape, output_size):
        super(FlattenConnection, self).__init__()
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


class ExpandConnection(nn.Module):
    """2D => Conv-IN-ReLU(2D) => Expand => Conv-IN-ReLU(3D) => 3D"""

    def __init__(self, input_channels, output_channels, output_depth):
        super(ExpandConnection, self).__init__()
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
        feature = self.conv2d(feature).unsqueeze(2).expand(-1, -1, self.output_depth, -1, -1)
        return self.conv3d(feature)


class TransposeConnection(nn.Module):
    """3D => Permute => Add => Average => 3D"""

    def __init__(self):
        super(TransposeConnection, self).__init__()

    def forward(self, features, transpose_dims):
        features = list(map(lambda x: torch.permute(x, transpose_dims), features))
        features = reduce(lambda x, y: x + y, features, 0)
        return features


class FeatureFBPConnection(nn.Module):
    """2D => Combination => (3D) => Interpolation => FBP => 3D"""

    def __init__(self, size, angles=None, detector_shape=None):
        super().__init__()
        self.angles = angles if angles is not None else size
        self.fbp = BPlayer(size, self.angles, detector_shape)

    def forward(self, *features: Tensor):
        projections = torch.cat([f.unsqueeze(3) for f in features], dim=3)
        return self.fbp(projections)


# class MixConnection(nn.Module):
#     """2D => Combination => (3D) => Interpolation => FBP => 3D"""

#     def __init__(self, size, angles=None,input_channels, output_channels, output_depth, detector_shape=None):
#         super().__init__()
#         self.expand_connection = nn.ModuleList(
#             [ExpandConnection(input_channels, output_channels, output_depth) for i in range(angles)]
#         )
#         self.fbp_connection = FeatureFBPConnection(size, angles, detector_shape)

#     def forward(self, *features: Tensor):
        

#         expanded_feature = [self.expand_connection(feature) for feature in features]
#         fbp_feature = self.fbp_connection(features)
#         mixed_feature = torch.cat([])
#         return
