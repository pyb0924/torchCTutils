import torch
from tests.conftest import output_dir

from torchCTutils.models import resnet34_2d, resnet34_3d, resnet50_2d, resnet50_3d


def run_resnet_model_2d(model, size):
    assert model is not None
    data = torch.rand((8, 1, size, size))
    output = model(data)
    assert output.shape == torch.Size([8, 2])


def run_resnet_model_3d(model, size):
    assert model is not None
    data = torch.rand((8, 1, size, size, size))
    output = model(data)
    assert output.shape == torch.Size([8, 2])


def test_resnet_2d(size):
    resnet34 = resnet34_2d()
    run_resnet_model_2d(resnet34, size)
    resnet34_cbam = resnet34_2d(use_attention="CBAM")
    run_resnet_model_2d(resnet34_cbam, size)

    resnet50 = resnet50_2d()
    run_resnet_model_2d(resnet50, size)
    resnet50_cbam = resnet50_2d(use_attention="CBAM")
    run_resnet_model_2d(resnet50_cbam, size)


def test_resnet_3d(size):
    resnet34 = resnet34_3d()
    run_resnet_model_3d(resnet34, size)
    resnet34_cbam = resnet34_3d(use_attention="CBAM")
    run_resnet_model_3d(resnet34_cbam, size)

    resnet50 = resnet50_3d()
    run_resnet_model_3d(resnet50, size)
    resnet50_cbam = resnet50_3d(use_attention="CBAM")
    run_resnet_model_3d(resnet50_cbam, size)
