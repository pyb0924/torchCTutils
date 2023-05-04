from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.args = args
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset_full = self.dataset_cls(*self.args, **self.kwargs)
            train_length, val_length = int(len(dataset_full) * 0.8), int(
                len(dataset_full) * 0.2
            )

            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [train_length, val_length]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(*self.args, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )
