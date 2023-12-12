from pathlib import Path
from typing import Literal, Union
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        data_paths: Union[list[Path], list[list[Path]]],
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 8,
        split_mode: Literal["random", "sequential"] = "sequential",
        split_ratio: list[float] = [0.8, 0.2],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_paths = data_paths
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.args = args
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.split_mode == "random":
                dataset_full = self.dataset_cls(
                    self.data_paths, *self.args, **self.kwargs
                )
                train_data_length = int(self.split_ratio[0] * len(dataset_full))
                val_data_length = len(dataset_full) - train_data_length

                self.dataset_train, self.dataset_val = random_split(
                    dataset_full, [train_data_length, val_data_length]
                )
            elif self.split_mode == "sequential":
                if type(self.data_paths[0]) != list or len(self.data_paths) != 2:
                    raise ValueError(
                        "split_ratio must be a list of Path of length 2 when split_mode is 'sequential'"
                    )
                self.dataset_train, self.dataset_val = (
                    self.dataset_cls(self.data_paths[0], *self.args, **self.kwargs),
                    self.dataset_cls(self.data_paths[1], *self.args, **self.kwargs),
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(
                self.data_paths, *self.args, **self.kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
