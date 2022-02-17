from __future__ import annotations
from typing import Optional

from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .utils import zigzag_weight


class Residual(nn.Sequential):
    "Residual layer with scale parameter."

    def __init__(self, *layers: nn.Module):
        super().__init__(*layers)

        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * super().forward(x)


def linear_block(
    in_size: int,
    out_size: Optional[int] = None,
    expand: float = 1,
    residual: bool = False,
) -> nn.Module:
    """Create a linear-GELU-linear block with
    expansion and residual connection."""

    if out_size is None:
        out_size = in_size

    mid_size = round((in_size + out_size) / 2 * expand)

    assert in_size >= 1
    assert out_size >= 1
    assert mid_size >= 1

    fn = nn.Sequential if not residual else Residual

    return fn(
        nn.LayerNorm(in_size),
        nn.Linear(in_size, mid_size, bias=False),
        nn.LayerNorm(mid_size),
        nn.GELU(),
        nn.LayerNorm(mid_size),
        nn.Linear(mid_size, out_size, bias=False),
        nn.LayerNorm(out_size),
    )


class BlockFFTLayer(nn.Module):
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
        is_encoder: bool = True,
    ):
        super().__init__()

        # parameters
        self.block_size = block_size
        self.is_encoder = is_encoder

        # color-mix
        if color_mix:
            args = (num_channels, num_channels, expand, residual)
            self.color_mix = linear_block(*args)

        # mix spatial+channels features
        self.spatial_shape = (block_size, block_size, num_channels)
        size = np.prod(self.spatial_shape)

        args = (size, size, expand, residual)
        self.spatial_mix = linear_block(*args) if spatial_mix else None

        # mix frequencies+channels features
        self.fourier_shape = (
            block_size,
            block_size // 2 + 1,
            2 * num_channels,
        )
        size = np.prod(self.fourier_shape)

        args = (size, size, expand, residual)
        self.fourier_mix = linear_block(*args) if fourier_mix else None

        # (de)compression
        W = zigzag_weight(self.fourier_shape, compress)
        in_size, out_size = W.shape
        self.latent_size = out_size

        if is_encoder:
            self.decompress = None
            self.compress = nn.Linear(in_size, out_size, bias=False)
            self.compress.weight.data = W.T.clone()
        else:
            self.compress = None
            self.decompress = nn.Linear(out_size, in_size, bias=False)
            self.decompress.weight.data = W.clone()

        # FFTs
        self.block_fft = partial(torch.fft.rfftn, dim=(3, 4), norm="ortho")
        self.block_ifft = partial(torch.fft.irfftn, dim=(3, 4), norm="ortho")

        self.fft = partial(torch.fft.rfftn, dim=(1, 2), norm="ortho")
        self.ifft = partial(torch.fft.irfftn, dim=(1, 2), norm="ortho")

    def blockfy(self, x: Tensor):
        batch, height, width, channels = x.shape
        block = self.block_size

        x = x.unflatten(2, [width // block, block])
        x = x.unflatten(1, [height // block, block])
        # (N, H/b, b, W/b, b, C)

        x = x.permute(0, 1, 3, 2, 4, 5)
        # (N, H/b, W/b, b, b, C)

        return x

    def deblockfy(self, x: Tensor):
        batch, height_block, width_block, block, block, channels = x.shape

        x = x.permute(0, 1, 3, 2, 4, 5)
        # (N, H/b, b, W/b, b, C)

        x = x.flatten(3, 4).flatten(1, 2)
        # (N, H/b*b, W/b*b, C)

        return x

    def quantize(self, x: Tensor):
        with torch.no_grad():
            q = x.half()

        x = q + (x - x.detach())

        return x

    def encode(self, x: Tensor):
        assert self.compress is not None

        if self.color_mix is not None:
            x = self.color_mix(x)

        # separate into blocks
        x = self.blockfy(x)

        # spatial mix
        if self.spatial_mix is not None:
            x = x.flatten(3, 5)
            x = self.spatial_mix(x)
            x = x.unflatten(3, self.spatial_shape)

        # to fourier space
        x = self.block_fft(x)
        x = torch.cat([x.real, x.imag], dim=-1)

        # fourier mix
        if self.fourier_mix is not None:
            x = x.flatten(3, 5)
            x = self.fourier_mix(x)
            x = x.unflatten(3, self.fourier_shape)

        # compress
        x = x.flatten(3, 5)
        x = self.compress(x)

        # quantize
        x = self.quantize(x)

        return x

    def decode(self, x: Tensor):
        assert self.decompress is not None

        # decompress
        x = self.decompress(x)
        x = x.unflatten(3, self.fourier_shape)

        # fourier mix
        if self.fourier_mix is not None:
            x = x.flatten(3, 5)
            x = self.fourier_mix(x)
            x = x.unflatten(3, self.fourier_shape)

        # to real space
        x = torch.complex(*x.chunk(2, dim=-1)).squeeze(3)
        x = self.block_ifft(x)

        # spatial mix
        if self.spatial_mix is not None:
            x = x.flatten(3, 5)
            x = self.spatial_mix(x)
            x = x.unflatten(3, self.spatial_shape)

        # decompose blocks
        x = self.deblockfy(x)

        # color mix
        if self.color_mix is not None:
            x = self.color_mix(x)

        return x.clamp(0, 1)

    def forward(self, x: Tensor):
        if self.is_encoder:
            return self.encode(x)
        return self.decode(x)
