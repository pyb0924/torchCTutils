from pathlib import Path
from typing import Union

from torch import Tensor
from lightning.pytorch.loggers import CSVLogger


from ..io import (
    save_dcm_from_output,
    read_series_from_dcm,
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

        raw_path = self.get_raw_path(path)
        label = read_series_from_dcm(str(raw_path), read_info=True)

        # save_image_to_dcm(label, str(real_dir / "label.dcm"))
        save_dcm_from_output(
            real.squeeze(0).cpu().numpy(),
            ds=label,
            output_path=real_dir,
        )
        save_dcm_from_output(
            fake.squeeze(0).cpu().numpy(),
            ds=label,
            output_path=fake_dir,
        )

    def get_raw_path(self, path: Union[str, Path]):
        path_split = list(Path(path).parts)
        path_split[-3] = "raw"
        raw_path = Path("").joinpath(*path_split).parent / "1" / "withpin"
        if not raw_path.exists():
            raise ValueError(f"raw_path {raw_path} does not exist!")
        return raw_path
