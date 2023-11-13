import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Union, cast, Dict


def make_vgg_encoder_2d(
    cfg: List[Union[str, int]], batch_norm: bool = False, in_channels=1
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_vgg_decoder_2d(
    cfg: List[Union[str, int]], batch_norm: bool = False, out_channels=1
) -> nn.Sequential:
    layers: List[nn.Module] = []
    cfg = list(reversed(cfg))[1:]
    in_channels = int(cfg[0])
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            up = nn.ConvTranspose2d(in_channels, v, kernel_size=2, stride=2)
            conv = nn.Conv2d(v, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [up, conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [up, conv, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def make_vgg_encoder_3d(
    cfg: List[Union[str, int]], batch_norm: bool = False, in_channels=1
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_vgg_decoder_3d(
    cfg: List[Union[str, int]], batch_norm: bool = False, out_channels=1
) -> nn.Sequential:
    layers: List[nn.Module] = []
    cfg = list(reversed(cfg))[1:]
    in_channels = int(cfg[0])
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            up = nn.ConvTranspose3d(in_channels, v, kernel_size=2, stride=2)
            conv = nn.Conv3d(v, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [up, conv, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [up, conv, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def get_cfgs(feature_channels: List[int]):
    return {
        "vgg11": [
            feature_channels[0],
            "M",
            feature_channels[1],
            "M",
            feature_channels[2],
            feature_channels[2],
            "M",
            feature_channels[3],
            feature_channels[3],
            "M",
            feature_channels[3],
            feature_channels[3],
            "M",
        ],
        "vgg13": [
            feature_channels[0],
            feature_channels[0],
            "M",
            feature_channels[1],
            feature_channels[1],
            "M",
            feature_channels[2],
            feature_channels[2],
            "M",
            feature_channels[3],
            feature_channels[3],
            "M",
            feature_channels[3],
            feature_channels[3],
            "M",
        ],
    }


class VGGEncoder2d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.cfg = cfg
        self.features = make_vgg_encoder_2d(
            get_cfgs(feature_channels)[cfg], batch_norm, in_channels
        )

    def forward(self, x: Tensor):
        return self.features(x)


class VGGEncoder3d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.cfg = cfg
        self.features = make_vgg_encoder_3d(
            get_cfgs(feature_channels)[cfg], batch_norm, in_channels
        )

    def forward(self, x: Tensor):
        return self.features(x)


class VGGDecoder2d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.cfg = cfg
        self.features = make_vgg_decoder_2d(
            get_cfgs(feature_channels)[cfg], batch_norm, in_channels
        )

    def forward(self, x: Tensor):
        return self.features(x)


class VGGDecoder3d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.cfg = cfg
        self.features = make_vgg_decoder_3d(
            get_cfgs(feature_channels)[cfg], batch_norm, in_channels
        )

    def forward(self, x: Tensor):
        return self.features(x)


class VGG2d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.encoder = VGGEncoder2d(feature_channels, batch_norm, in_channels, cfg=cfg)
        self.decoder = VGGDecoder2d(feature_channels, batch_norm, in_channels, cfg=cfg)

    def forward(self, x: Tensor):
        return self.decoder(self.encoder(x))


class VGG3d(nn.Module):
    def __init__(
        self,
        feature_channels: List[int] = [64, 128, 256, 512],
        batch_norm: bool = False,
        in_channels: int = 1,
        cfg: str = "vgg11",
    ):
        super().__init__()
        self.encoder = VGGEncoder3d(feature_channels, batch_norm, in_channels, cfg=cfg)
        self.decoder = VGGDecoder3d(feature_channels, batch_norm, in_channels, cfg=cfg)

    def forward(self, x: Tensor):
        return self.decoder(self.encoder(x))
