from __future__ import annotations
from typing import Optional

import numpy as np

import torch
import torch.nn as nn


def parameter_count(module: nn.Module | nn.Parameter) -> int:
    "Total number of parameters in module."

    if isinstance(module, nn.Parameter):
        return module.numel() if module.requires_grad else 0

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def zigzag(nx: int, ny: Optional[int] = None, nz: int = 1):
    if ny is None:
        ny = nx

    assert nx >= 1
    assert ny >= 1
    assert nz >= 1

    def sort_point(pt: tuple[int, int, int]):
        (i, j, _k) = pt

        return (i + j, -j if (i + j) % 2 == 0 else j)

    idx = [(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)]
    idx = list(enumerate(idx))
    idx = sorted(idx, key=lambda v: sort_point(v[1]))
    idx = torch.tensor([m for m, _ in idx])

    return idx


def zigzag_weight(shape: tuple[int, int, int], compress: int | float):
    assert len(shape) == 3

    idx = zigzag(*shape)
    in_size = np.prod(shape)

    if isinstance(compress, float):
        assert 0 < compress <= 1
        out_size = round(in_size * compress)
    elif isinstance(compress, int):
        assert compress >= 1
        out_size = compress

    assert in_size >= out_size >= 1

    W = torch.zeros(in_size, out_size)
    W[idx[:out_size], torch.arange(out_size)] = 1

    return W
