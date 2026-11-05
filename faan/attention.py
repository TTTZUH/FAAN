"""Frequency fusion and three-branch attention modules for FAAN."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wavelet import TwoLevelDB4


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class EfficientSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int, grid_size: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.grid_size = grid_size
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        grid = (min(height, self.grid_size), min(width, self.grid_size))
        pooled = F.adaptive_avg_pool2d(x, grid)
        qkv = self.qkv(pooled).reshape(
            x.shape[0], 3, self.heads, self.head_dim, grid[0] * grid[1]
        )
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value).transpose(-2, -1)
        output = output.reshape(x.shape[0], self.channels, *grid)
        output = self.output(output)
        if grid != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return output


class AttentionProcessingBlock(nn.Module):
    def __init__(self, channels: int, heads: int, grid_size: int) -> None:
        super().__init__()
        self.attention = EfficientSelfAttention(channels, heads, grid_size)
        self.processing = ConvNormAct(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.processing(x + self.attention(x))


class FrequencyAdaptiveFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 8,
        grid_size: int = 16,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        if wavelet.lower() != "db4":
            raise ValueError(f"Unsupported wavelet: {wavelet!r}")
        if wavelet_levels != 2:
            raise ValueError("FAAN requires a two-level wavelet decomposition")
        self.wavelet = TwoLevelDB4()
        self.frequency_reduce = ConvNormAct(channels * 7, channels, kernel_size=1)
        self.frequency_path = AttentionProcessingBlock(channels, heads, grid_size)
        self.spatial_path = AttentionProcessingBlock(channels, heads, grid_size)
        self.output_path = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1),
            AttentionProcessingBlock(channels, heads, grid_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        bands = [
            F.interpolate(band, size=size, mode="bilinear", align_corners=False)
            for band in self.wavelet(x)
        ]
        frequency = self.frequency_path(self.frequency_reduce(torch.cat(bands, dim=1)))
        spatial = self.spatial_path(x)
        return self.output_path(torch.cat((frequency, spatial), dim=1))


class QKVProjection(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )


class ThreeBranchAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 8,
        grid_size: int = 16,
        layers: tuple[int, int, int] = (4, 4, 2),
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.grid_size = grid_size
        self.layers = layers

        self.global_context = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_context = ConvNormAct(channels, channels, groups=channels)
        self.multiscale_context = nn.Conv2d(channels, channels, 1, bias=False)

        self.qg, self.ks, self.vm = (QKVProjection(channels) for _ in range(3))
        self.qs, self.km, self.vg = (QKVProjection(channels) for _ in range(3))
        self.qm, self.kg, self.vs = (QKVProjection(channels) for _ in range(3))
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _sources(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        global_feature = x + self.global_context(F.adaptive_avg_pool2d(x, 1))
        spatial_feature = self.spatial_context(x)
        multiscale_feature = (
            F.avg_pool2d(x, 3, stride=1, padding=1)
            + F.avg_pool2d(x, 5, stride=1, padding=2)
            + F.avg_pool2d(x, 7, stride=1, padding=3)
        ) / 3.0
        return global_feature, spatial_feature, self.multiscale_context(multiscale_feature)

    def _tokens(self, projection: nn.Module, x: Tensor, grid: tuple[int, int]) -> Tensor:
        projected = projection(F.adaptive_avg_pool2d(x, grid))
        return projected.reshape(
            projected.shape[0], self.heads, self.head_dim, grid[0] * grid[1]
        ).transpose(-2, -1)

    def _restore(self, tokens: Tensor, size: tuple[int, int], grid: tuple[int, int]) -> Tensor:
        feature = tokens.transpose(-2, -1).reshape(tokens.shape[0], self.channels, *grid)
        if grid != size:
            feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        return feature

    @staticmethod
    def _soft_pool2d(x: Tensor, kernel_size: int) -> Tensor:
        padding = kernel_size // 2
        weights = x.exp()
        numerator = F.avg_pool2d(x * weights, kernel_size, stride=1, padding=padding)
        denominator = F.avg_pool2d(weights, kernel_size, stride=1, padding=padding)
        return numerator / denominator.clamp_min(1e-6)

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        normalise: bool = False,
        multiscale_pool: bool = False,
    ) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = scores.softmax(dim=-1)
        if multiscale_pool:
            batch, heads, queries, keys = attention.shape
            maps = attention.reshape(batch * heads, 1, queries, keys)
            attention = (
                F.avg_pool2d(maps, 3, stride=1, padding=1)
                + self._soft_pool2d(maps, 5)
                + F.max_pool2d(maps, 7, stride=1, padding=3)
            ).reshape(batch, heads, queries, keys)
            normalise = True
        if normalise:
            attention = F.layer_norm(attention, (attention.shape[-1],))
        return torch.matmul(attention, value)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        grid = (min(size[0], self.grid_size), min(size[1], self.grid_size))
        global_source, spatial_source, multiscale_source = self._sources(x)

        global_state = global_source
        for _ in range(self.layers[0]):
            output = self._attention(
                self._tokens(self.qg, global_state, grid),
                self._tokens(self.ks, spatial_source, grid),
                self._tokens(self.vm, multiscale_source, grid),
            )
            global_state = self._restore(output, size, grid)

        spatial_state = spatial_source
        for _ in range(self.layers[1]):
            output = self._attention(
                self._tokens(self.qs, spatial_state, grid),
                self._tokens(self.km, multiscale_source, grid),
                self._tokens(self.vg, global_source, grid),
                normalise=True,
            )
            spatial_state = self._restore(output, size, grid)

        multiscale_state = multiscale_source
        for _ in range(self.layers[2]):
            output = self._attention(
                self._tokens(self.qm, multiscale_state, grid),
                self._tokens(self.kg, global_source, grid),
                self._tokens(self.vs, spatial_source, grid),
                multiscale_pool=True,
            )
            multiscale_state = self._restore(output, size, grid)

        return self.fusion(torch.cat((global_state, spatial_state, multiscale_state), dim=1))
