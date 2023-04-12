import torch

from torchCTutils.models import UNet2d,UNet3d


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
