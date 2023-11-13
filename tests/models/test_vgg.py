import torch

from torchCTutils.models import (
    VGGEncoder2d,
    VGGDecoder3d,
    VGGDecoder2d,
    VGGEncoder3d,
    VGG2d,
    VGG3d,
)


def test_vgg2d(size):
    model = VGG2d()
    assert model is not None
    data = torch.rand((8, 1, size, size))
    output = model(data)
    assert output.shape == data.shape


def test_vgg3d(size):
    model = VGG3d()
    assert model is not None
    data = torch.rand((8, 1, size, size, size))
    output = model(data)
    assert output.shape == data.shape


def test_fusion_vgg2d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder1 = VGGEncoder2d(encoder_features)
    encoder2 = VGGEncoder2d(encoder_features)
    decoder = VGGDecoder2d()
    assert encoder1 is not None
    assert encoder2 is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size))
    features1 = encoder1(data)
    features2 = encoder2(data)
    features = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(features1, features2)]
    output = decoder(features)
    assert output.shape == data.shape


def test_fusion_vgg3d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder1 = VGGEncoder3d(encoder_features)
    encoder2 = VGGEncoder3d(encoder_features)
    decoder = VGGDecoder3d()
    assert encoder1 is not None
    assert encoder2 is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size, size))
    features1 = encoder1(data)
    features2 = encoder2(data)
    features = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(features1, features2)]
    output = decoder(features)
    assert output.shape == data.shape
