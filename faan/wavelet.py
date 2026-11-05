"""Differentiable two-dimensional wavelet decomposition used by FAM."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# db4 analysis filters from PyWavelets.
_DB4_LO = (
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679298587,
    0.7148465705529154,
    0.23037781330885523,
)
_DB4_HI = (
    -0.23037781330885523,
    0.7148465705529154,
    -0.6308807679298587,
    -0.02798376941698385,
    0.18703481171888114,
    0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
)


def _analysis_kernels() -> Tensor:
    # conv2d uses cross-correlation, so the filters are reversed.
    lo = torch.tensor(_DB4_LO, dtype=torch.float32).flip(0)
    hi = torch.tensor(_DB4_HI, dtype=torch.float32).flip(0)
    return torch.stack(
        (
            torch.outer(lo, lo),
            torch.outer(lo, hi),
            torch.outer(hi, lo),
            torch.outer(hi, hi),
        )
    ).unsqueeze(1)


class DWT2D(nn.Module):
    """One-level fixed db4 DWT with gradients to its input.

    Reflection padding and stride two preserve the expected half-resolution
    output for even-sized feature maps. Output order is LL, LH, HL, HH.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("kernels", _analysis_kernels(), persistent=False)
        self.padding = self.kernels.shape[-1] // 2 - 1

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"DWT2D expects BCHW input, got shape {tuple(x.shape)}")
        if min(x.shape[-2:]) <= self.padding:
            raise ValueError("Feature map is too small for a db4 decomposition")

        channels = x.shape[1]
        padded = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        filters = self.kernels.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        bands = F.conv2d(padded, filters, stride=2, groups=channels)
        bands = bands.reshape(x.shape[0], channels, 4, *bands.shape[-2:])
        return bands[:, :, 0], bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]


class TwoLevelDB4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dwt = DWT2D()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        ll1, lh1, hl1, hh1 = self.dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        return ll2, lh2, hl2, hh2, lh1, hl1, hh1
