import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    MeanSquaredError,
)
from lightning import LightningModule


class BaseReconModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # metrics
        self.ssim = StructuralSimilarityIndexMeasure()
        self.psnr = PeakSignalNoiseRatio()
        self.rmse = MeanSquaredError(squared=False)

    def log_metrics(self, fake, real, stage):
        self.ssim(fake, real)
        self.log(f"{stage}/SSIM", self.ssim, on_step=True, on_epoch=True, logger=True)
        self.psnr(fake, real)
        self.log(f"{stage}/PSNR", self.psnr, on_step=True, on_epoch=True, logger=True)
        self.rmse(fake, real)
        self.log(f"{stage}/RMSE", self.rmse, on_step=True, on_epoch=True, logger=True)

    def log_images(self, fake, real, channels, stage):
        fake_list = list(torch.chunk(fake[0], channels, dim=0))
        real_list = list(torch.chunk(real[0], channels, dim=0))
        self.logger.log_image(
            f"{stage}/reconstruct_images",
            fake_list,
            caption=[f"epoch_{self.current_epoch}_view_{i}" for i in range(channels)],
        )
        self.logger.log_image(
            f"{stage}/real_images",
            real_list,
            caption=[f"epoch_{self.current_epoch}_view_{i}" for i in range(channels)],
        )
