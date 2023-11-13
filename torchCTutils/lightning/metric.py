from typing import Union, Tuple
from torchmetrics import MeanSquaredError


class NormalizedRMSE(MeanSquaredError):
    def __init__(
        self,
        data_range: Union[float, Tuple[float, float], None] = None,
        squared: bool = True,
        dist_sync_on_step=False,
    ):
        super().__init__(squared=squared, dist_sync_on_step=dist_sync_on_step)
        self.data_range = data_range

    def update(self, preds, target):
        if not self.normalize:
            preds = preds / preds.sum(dim=1, keepdim=True)
            target = target / target.sum(dim=1, keepdim=True)
        super().update(preds, target)

    def compute(self):
        if self.squared:
            return super().compute().sqrt()
        else:
            return super().compute()
