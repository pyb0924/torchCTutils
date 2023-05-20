import matplotlib.pyplot as plt



def visualize_3d_image(image, z_slice=1, method="slice", *args, **kwargs):
    if method == "slice":
        fig, axes = plt.subplots(*args, **kwargs)
        # print(nodule_array[0].shape)
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(image[i * z_slice, :, :], cmap="gray")
            ax.axis("off")
    elif method=="3d":
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.voxels(image, edgecolor='k')
    return fig, axes
