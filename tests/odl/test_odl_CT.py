import numpy as np
import matplotlib.pyplot as plt

from torchCTutils.odl import get_FP_operator, get_FBP_operator, get_paired_CT_operator


def test_fpfbp_2D(phantom_2D, output_dir, size, angles):
    for mode in ['parallel', 'cone']:
        fp = get_FP_operator(size, angles, mode, 2, 60, 80)
        fbp = get_FBP_operator(size, angles, mode, 2, 60, 80)

        sinogram = fp(phantom_2D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T), 'gray')
        plt.savefig(output_dir / f'sinogram_2D_{mode}.png')

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T), 'gray')
        plt.savefig(output_dir / f'recon_2D_{mode}.png')


def test_fpfbp_paired(phantom_2D, output_dir, size, angles):
    for mode in ['parallel', 'cone']:
        fp, fbp = get_paired_CT_operator(size, angles, mode, 2, 60, 80)

        sinogram = fp(phantom_2D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T), 'gray')
        plt.savefig(output_dir / f'sinogram_2D_{mode}_paired.png')

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T), 'gray')
        plt.savefig(output_dir / f'recon_2D_{mode}_paired.png')


def test_fpfbp_3D(phantom_3D, output_dir, size, angles):
    for mode in ['parallel', 'cone']:
        fp = get_FP_operator(size, angles, mode, 3, 60, 80)
        fbp = get_FBP_operator(size, angles, mode, 3, 60, 80)

        sinogram = fp(phantom_3D)
        plt.figure()
        plt.imshow(np.flipud(sinogram.data.T)[64], 'gray')
        plt.savefig(output_dir / f'sinogram_3D_{mode}.png')

        recon = fbp(sinogram)
        plt.figure()
        plt.imshow(np.flipud(recon.data.T)[64], 'gray')
        plt.savefig(output_dir / f'recon_3D_{mode}.png')
