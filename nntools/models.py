from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as pl

from .layers import BlockFFTLayer


class BlockFFTAutoEncoder(pl.LightningModule):
    lr = 1e-3

    @property
    def name(self):
        return self.__class__.__qualname__

    def __init__(
        self,
        num_channels: int,
        block_size: int,
        expand: float,
        compress: float | int,
        color_mix: bool = True,
        spatial_mix: bool = True,
        fourier_mix: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        kwargs = dict(
            num_channels=num_channels,
            block_size=block_size,
            expand=expand,
            compress=compress,
            color_mix=color_mix,
            spatial_mix=spatial_mix,
            fourier_mix=fourier_mix,
            residual=residual,
        )
        self.encoder = BlockFFTLayer(**kwargs, is_encoder=True)
        self.decoder = BlockFFTLayer(**kwargs, is_encoder=False)

        self.block_fft = partial(torch.fft.rfftn, dim=(3, 4), norm="ortho")
        self.block_ifft = partial(torch.fft.irfftn, dim=(3, 4), norm="ortho")

        self.fft = partial(torch.fft.rfftn, dim=(1, 2), norm="ortho")
        self.ifft = partial(torch.fft.irfftn, dim=(1, 2), norm="ortho")

    def forward(self, x: Tensor):
        z = self.encoder.forward(x)
        out = self.decoder.forward(z)

        return z, out

    def training_step(self, data: Tensor, batch_idx: int):
        z, out = self(data)

        # real
        loss = F.mse_loss(out, data)

        # fourier
        with torch.no_grad():
            diff = self.fft(out).sub(self.fft(data))
            loss_fourier = diff.conj().mul(diff).mean().real

        self.log("train/loss", loss)
        self.log("train/loss_fourier", loss_fourier)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer
