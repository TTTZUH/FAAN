"""FAAN network: shared ResNet-18, FAM, stage-wise TBAM fusion and heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import ConvNormAct, FrequencyAdaptiveFusion, ThreeBranchAttention


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(output + identity)


class ResNet18Encoder(nn.Module):
    channels = (64, 128, 256, 512)

    def __init__(
        self,
        attention_heads: int = 8,
        attention_grid: int = 16,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.fam = FrequencyAdaptiveFusion(
            64,
            attention_heads,
            attention_grid,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
        )
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.apply(_initialise)

    @staticmethod
    def _make_layer(
        in_channels: int, channels: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(in_channels, channels, stride)]
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        stage1 = self.fam(self.layer1(self.stem(image)))
        stage2 = self.layer2(stage1)
        stage3 = self.layer3(stage2)
        stage4 = self.layer4(stage3)
        return stage1, stage2, stage3, stage4


class DownsampleCarry(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PairEncoder(nn.Module):
    def __init__(
        self,
        *,
        attention_heads: int = 8,
        attention_grid: int = 16,
        tbam_layers: tuple[int, int, int] = (4, 4, 2),
        wavelet: str = "db4",
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = ResNet18Encoder(
            attention_heads,
            attention_grid,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
        )
        channels = self.backbone.channels
        self.fuse1 = ConvNormAct(
            channels[0] * 2, channels[0], kernel_size=1, groups=2
        )
        self.fuse2 = ConvNormAct(
            channels[1] * 2, channels[1], kernel_size=1, groups=2
        )
        self.fuse3 = ConvNormAct(
            channels[2] * 2, channels[2], kernel_size=1, groups=2
        )
        self.fuse4 = ConvNormAct(
            channels[3] * 2, channels[3], kernel_size=1, groups=2
        )
        self.carry2 = DownsampleCarry(channels[0], channels[1])
        self.carry3 = DownsampleCarry(channels[1], channels[2])
        self.carry4 = DownsampleCarry(channels[2], channels[3])
        self.tbam2 = ThreeBranchAttention(
            channels[1], attention_heads, attention_grid, tbam_layers
        )
        self.tbam3 = ThreeBranchAttention(
            channels[2], attention_heads, attention_grid, tbam_layers
        )
        self.tbam4 = ThreeBranchAttention(
            channels[3], attention_heads, attention_grid, tbam_layers
        )
        self.apply(_initialise)

    def forward(
        self, pre: Tensor, post: Tensor
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        pre1 = self.backbone.fam(self.backbone.layer1(self.backbone.stem(pre)))
        post1 = self.backbone.fam(self.backbone.layer1(self.backbone.stem(post)))
        fused1 = self.fuse1(torch.cat((pre1, post1), dim=1))

        pre2, post2 = self.backbone.layer2(pre1), self.backbone.layer2(post1)
        fused2 = self.tbam2(
            self.fuse2(torch.cat((pre2, post2), dim=1)) + self.carry2(fused1)
        )

        pre3, post3 = self.backbone.layer3(pre2), self.backbone.layer3(post2)
        fused3 = self.tbam3(
            self.fuse3(torch.cat((pre3, post3), dim=1)) + self.carry3(fused2)
        )

        pre4, post4 = self.backbone.layer4(pre3), self.backbone.layer4(post3)
        fused4 = self.tbam4(
            self.fuse4(torch.cat((pre4, post4), dim=1)) + self.carry4(fused3)
        )
        return (pre1, pre2, pre3, pre4), (post1, post2, post3, post4), (
            fused1,
            fused2,
            fused3,
            fused4,
        )


class FPNDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], output_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            ConvNormAct(channel, output_channels, kernel_size=1) for channel in channels
        )
        self.smooth = nn.ModuleList(
            ConvNormAct(output_channels, output_channels) for _ in range(len(channels) - 1)
        )

    def forward(self, features: tuple[Tensor, ...]) -> Tensor:
        output = self.lateral[-1](features[-1])
        smooth_index = 0
        for index in range(len(features) - 2, -1, -1):
            output = F.interpolate(
                output, size=features[index].shape[-2:], mode="bilinear", align_corners=False
            )
            output = self.smooth[smooth_index](output + self.lateral[index](features[index]))
            smooth_index += 1
        return output


@dataclass(frozen=True)
class FAANOutput:
    localisation_logits: Tensor
    damage_logits: Tensor
    contrast_features: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "localisation_logits": self.localisation_logits,
            "damage_logits": self.damage_logits,
            "contrast_features": self.contrast_features,
        }


class FAAN(nn.Module):
    def __init__(
        self,
        *,
        attention_heads: int = 8,
        attention_grid: int = 16,
        tbam_layers: tuple[int, int, int] = (4, 4, 2),
        decoder_channels: int = 128,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = PairEncoder(
            attention_heads=attention_heads,
            attention_grid=attention_grid,
            tbam_layers=tbam_layers,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
        )
        channels = self.encoder.backbone.channels
        self.localisation_decoder = FPNDecoder(channels, decoder_channels)
        self.damage_decoder = FPNDecoder(channels, decoder_channels)
        self.localisation_head = nn.Conv2d(decoder_channels, 1, 1)
        self.damage_head = nn.Conv2d(decoder_channels, 4, 1)
        self.apply(_initialise)

    def forward(self, pre: Tensor, post: Tensor) -> dict[str, Tensor]:
        if pre.shape != post.shape:
            raise ValueError("Pre- and post-disaster tensors must have identical shapes")
        pre_features, post_features, fused_features = self.encoder(pre, post)
        localisation = self.localisation_head(self.localisation_decoder(pre_features))
        damage = self.damage_head(self.damage_decoder(fused_features))
        localisation = F.interpolate(
            localisation, size=pre.shape[-2:], mode="bilinear", align_corners=False
        )
        damage = F.interpolate(damage, size=pre.shape[-2:], mode="bilinear", align_corners=False)

        contrast = build_contrast_features(pre_features, post_features, fused_features)
        return FAANOutput(localisation, damage, contrast).as_dict()


def build_contrast_features(
    pre_features: tuple[Tensor, ...],
    post_features: tuple[Tensor, ...],
    fused_features: tuple[Tensor, ...],
) -> Tensor:
    return torch.cat(
        (
            pre_features[-1],
            post_features[-1],
            (pre_features[-1] - post_features[-1]).abs(),
            fused_features[-1],
        ),
        dim=1,
    )


class ReconstructionDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], decoder_channels: int = 128) -> None:
        super().__init__()
        self.decoder = FPNDecoder(channels, decoder_channels)
        self.head = nn.Sequential(
            ConvNormAct(decoder_channels, 64),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: tuple[Tensor, ...], output_size: tuple[int, int]) -> Tensor:
        image = self.head(self.decoder(features))
        return F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)


class ReconstructionModel(nn.Module):
    def __init__(
        self,
        encoder: ResNet18Encoder | None = None,
        *,
        attention_heads: int = 8,
        attention_grid: int = 16,
        decoder_channels: int = 128,
        wavelet: str = "db4",
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder or ResNet18Encoder(
            attention_heads,
            attention_grid,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
        )
        self.decoder = ReconstructionDecoder(self.encoder.channels, decoder_channels)
        self.apply(_initialise)

    def forward(self, image: Tensor) -> Tensor:
        return self.decoder(self.encoder(image), image.shape[-2:])


def _initialise(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(config: dict[str, Any]) -> FAAN:
    model_config = config.get("model", config)
    backbone = model_config.get("backbone", "resnet18")
    if backbone.lower() != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    if model_config.get("pretrained", False):
        raise ValueError("Pretrained backbone weights are not supported")
    values = tuple(int(value) for value in model_config.get("tbam_layers", (4, 4, 2)))
    if len(values) != 3:
        raise ValueError("tbam_layers must contain three values")
    layers = (values[0], values[1], values[2])
    return FAAN(
        attention_heads=int(model_config.get("attention_heads", 8)),
        attention_grid=int(model_config.get("attention_grid", 16)),
        tbam_layers=layers,
        decoder_channels=int(model_config.get("decoder_channels", 128)),
        wavelet=str(model_config.get("wavelet", "db4")),
        wavelet_levels=int(model_config.get("wavelet_levels", 2)),
    )
