from pathlib import Path
from typing import Union

import SimpleITK as sitk

from torch import Tensor
from lightning.pytorch.loggers import CSVLogger


from torchCTutils.io import (
    save_dcm_from_output,
    read_series_from_dcm,
    save_image_to_dcm,
)


class DCMLogger(CSVLogger):
    def log_dcm(
        self,
        real: Tensor,
        fake: Tensor,
        path: Union[str, Path],
        stage: str,
        epoch: int,
    ):
        dcm_dir = Path(self.log_dir) / stage / f"epoch_{epoch}"
        real_dir, fake_dir = dcm_dir / "real", dcm_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = str(path).replace("preprocessed", "foryibo")
        raw_path = Path(raw_dir).parent / "1" / "withpin"
        if not raw_path.exists():
            raise ValueError(f"raw_path {raw_path} does not exist!")
        label = read_series_from_dcm(str(raw_path), read_info=True)

        # save_image_to_dcm(label, str(real_dir / "label.dcm"))
        save_dcm_from_output(
            real.squeeze(0).cpu().numpy(), ds=label, output_path=real_dir
        )
        save_dcm_from_output(
            fake.squeeze(0).cpu().numpy(), ds=label, output_path=fake_dir
        )
