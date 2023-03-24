import torch

from torchCTutils.model import UNet


def test_unet(size):
    model = UNet(1, 1)
    assert model is not None
    data = torch.rand((8, 1, size, size))
    output = model(data)
    assert output.shape == data.shape
