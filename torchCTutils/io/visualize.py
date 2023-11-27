import matplotlib.pyplot as plt


def show_gray_image(image, ax, vmin=None, vmax=None):
    ax.imshow(image, vmin=vmin, vmax=vmax, cmap="gray")
    ax.axis("off")


def data_range_to_window(vmin, vmax):
    return (vmax + vmin) / 2, vmax - vmin


def window_to_data_range(wl, ww):
    return wl - ww / 2, wl + ww / 2


def visualize_3d_image(image, z_slice=1, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    # print(nodule_array[0].shape)
    for i, ax in enumerate(axes.ravel()):
        if i < image.shape[0] // z_slice:
            show_gray_image(image[i * z_slice, :, :], ax)
            ax.axis("off")

    return fig, axes


def compare_3d_image(image1, image2, z_slice=1, wl=None, ww=None, *args, **kwargs):
    cols = image1.shape[0] // z_slice
    vmin = vmax = None
    if wl is not None and ww is not None:
        vmin, vmax = window_to_data_range(wl, ww)

    fig, axes = plt.subplots(3, cols, *args, **kwargs)
    for i, ax_rows in enumerate(axes):
        for j, ax in enumerate(ax_rows):
            if i == 0:
                show_gray_image(image1[j * z_slice, :, :], ax, vmin, vmax)
            elif i == 1:
                show_gray_image(image2[j * z_slice, :, :], ax, vmin, vmax)
            else:
                residual = image2[j * z_slice, :, :] - image1[j * z_slice, :, :]
                show_gray_image(residual, ax, vmin, vmax)
            ax.axis("off")
