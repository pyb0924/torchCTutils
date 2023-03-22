from functools import reduce

import torch
from torch import nn

from ..odl import odl_FBP_layer


class FeatureConnectionA(nn.Module):
    """ 2D => Flatten => FC => Dropout => ReLU => Reshape => 3D """

    def __init__(self, feature_shape, output_size=32):
        super(FeatureConnectionA, self).__init__()
        self.feature_shape = feature_shape
        self.output_size = output_size
        _, channels, h, w = self.feature_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * h * w, output_size**3)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.relu(self.dropout(self.fc(self.flatten(feature))))
        batch_size = self.feature_shape[0]
        return feature.reshape((batch_size, 1, self.output_size, self.output_size, self.output_size))


class FeatureConnectionB(nn.Module):
    """ 2D => Conv-IN-ReLU(2D) => Expand => Conv-IN-ReLU(3D) => 3D """

    def __init__(self, feature_shape, output_channels=32):
        super(FeatureConnectionB, self).__init__()
        self.feature_shape = feature_shape
        _, channels, h, w = self.feature_shape
        self.output_channels = output_channels
        self.conv2d = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(output_channels, output_channels,
                      kernel_size=3, padding=1),
            nn.InstanceNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature):
        feature = self.conv2d(feature).unsqueeze(
            1).expand(-1, self.output_channels, -1, -1, -1)


class FeatureConnectionC(nn.Module):
    """ 3D => Permute => Add => Average => 3D """

    def __init__(self):
        super(FeatureConnectionC, self).__init__()

    def forward(self, *features):
        num_features = len(features)
        features = map(lambda x: torch.permute(x, (0, 1, 2, 4, 3)), features)
        features = reduce(lambda x, y: x + y, features, 0)
        return features / num_features


class FeatureFBPConnection(Feature2Dto3D):
    """ 2D => Combination => (3D) => FP => Interpolation => FBP => 3D """

    def __init__(self, feature_shape, angles=60, output_size=64):
        super(FeatureConnectionA, self).__init__(feature_shape)
        self.output_size = output_size
        _, channels, h, w = self.feature_shape

        self.fbp = odl_FBP_layer(output_size, angles, dim=3)

    def forward(self, *features):
        pass
