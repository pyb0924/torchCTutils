from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
from torchvision.utils import save_image

def setup_by_config(config):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = config["output_path"]
    folder_name = (
        f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    )
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config["output_path"] = output_path.as_posix()


def save_multichannel_grayscale_image(tensor: torch.Tensor, filenames: List[Union[str, Path]], normalize=False):
    if tensor.shape[1] != len(filenames):
        raise ValueError('Invalid Input!')

    for i in range(tensor.shape[1]):
        tensor_per_channel = tensor[:, i, :, :].unsqueeze(1)
        save_image(tensor_per_channel, filenames[i], normalize=normalize)


def add_circle_mask_to_output_tensor(tensor, ratio=0.9):
    bs, channels, height, width = tensor.shape
    result = torch.zeros_like(tensor)
    for i in range(height):
        for j in range(width):
            if (i - height / 2)**2. + (j - width / 2)**2 <= ratio**2 * height * width / 4:
                result[:, :, i, j] = tensor[:, :, i, j]
    return result
