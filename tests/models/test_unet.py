import torch

from torchCTutils.models import (
    UNet2d,
    UNet3d,
    UNetEncoder2d,
    UNetDecoder2d,
    UNetEncoder3d,
    UNetDecoder3d,
    AttentionUNetDecoder2d,
    AttentionUNetDecoder3d,
)


def test_unet2d(size):
    model = UNet2d(1, 1)
    assert model is not None
    data = torch.rand((8, 1, size, size))
    output = model(data)
    assert output.shape == data.shape


def test_unet3d(size):
    model = UNet3d(1, 1)
    assert model is not None
    data = torch.rand((8, 1, size, size, size))
    output = model(data)
    assert output.shape == data.shape


def test_fusion_unet2d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder1 = UNetEncoder2d(1, encoder_features)
    encoder2 = UNetEncoder2d(1, encoder_features)
    decoder = UNetDecoder2d(1)
    assert encoder1 is not None
    assert encoder2 is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size))
    features1 = encoder1(data)
    features2 = encoder2(data)
    features = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(features1, features2)]
    output = decoder(features)
    assert output.shape == data.shape


def test_fusion_unet3d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder1 = UNetEncoder3d(1, encoder_features)
    encoder2 = UNetEncoder3d(1, encoder_features)
    decoder = UNetDecoder3d(1)
    assert encoder1 is not None
    assert encoder2 is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size, size))
    features1 = encoder1(data)
    features2 = encoder2(data)
    features = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(features1, features2)]
    output = decoder(features)
    assert output.shape == data.shape


def test_attention_unet2d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder = UNetEncoder2d(1, encoder_features)
    decoder = AttentionUNetDecoder2d(1, encoder_features)
    assert encoder is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size))
    features = encoder(data)
    output = decoder(features)
    assert output.shape == data.shape


def test_attention_unet3d(size):
    encoder_features = [32 * 2**i for i in range(4)]
    encoder = UNetEncoder3d(1, encoder_features)
    decoder = AttentionUNetDecoder3d(1, encoder_features)
    assert encoder is not None
    assert decoder is not None
    data = torch.rand((8, 1, size, size, size))
    features = encoder(data)
    output = decoder(features)
    assert output.shape == data.shape
