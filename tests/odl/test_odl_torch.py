import torch

from torchCTutils.odl import get_paired_CT_func, get_paired_CT_layer, odl_FP, odl_FBP


def test_fpfbp_func_2D(phantom_2D_tensor):
    sinogram = odl_FP(phantom_2D_tensor)
    assert type(sinogram) == torch.Tensor

    size = phantom_2D_tensor.shape[-1]
    recon = odl_FBP(sinogram, size)
    assert type(recon) == torch.Tensor
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_func_2D_paired(phantom_2D_tensor):
    size = phantom_2D_tensor.shape[-1]
    fp_func, fbp_func = get_paired_CT_func(size)

    sinogram = fp_func(phantom_2D_tensor)
    assert type(sinogram) == torch.Tensor
    recon = fbp_func(sinogram)
    assert type(recon) == torch.Tensor
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_module_2D(phantom_2D_tensor):
    size = phantom_2D_tensor.shape[-1]
    fp_layer, fbp_layer = get_paired_CT_layer(size)

    sinogram = fp_layer(phantom_2D_tensor)
    assert type(sinogram) == torch.Tensor
    recon = fbp_layer(sinogram)
    assert type(recon) == torch.Tensor
    assert phantom_2D_tensor.shape == recon.shape


def test_fpfbp_func_3D(phantom_3D_tensor):
    sinogram = odl_FP(phantom_3D_tensor, dim=3)
    assert type(sinogram) == torch.Tensor

    size = phantom_3D_tensor.shape[-1]
    recon = odl_FBP(sinogram, size, dim=3)
    assert type(recon) == torch.Tensor
    assert phantom_3D_tensor.shape == recon.shape


def test_fpfbp_func_3D_paired(phantom_3D_tensor):
    size = phantom_3D_tensor.shape[-1]
    fp_func, fbp_func = get_paired_CT_func(size, dim=3)

    sinogram = fp_func(phantom_3D_tensor)
    assert type(sinogram) == torch.Tensor
    recon = fbp_func(sinogram)
    assert type(recon) == torch.Tensor
    assert phantom_3D_tensor.shape == recon.shape


def test_fpfbp_module_3D(phantom_3D_tensor):
    size = phantom_3D_tensor.shape[-1]
    fp_layer, fbp_layer = get_paired_CT_layer(size, dim=3)

    sinogram = fp_layer(phantom_3D_tensor)
    assert type(sinogram) == torch.Tensor
    recon = fbp_layer(sinogram)
    assert type(recon) == torch.Tensor
    assert phantom_3D_tensor.shape == recon.shape
