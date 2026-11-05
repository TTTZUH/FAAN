"""Loss functions used by FAAN."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import PairEncoder, build_contrast_features


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.to(dtype=logits.dtype)
        binary_cross_entropy = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probability = logits.sigmoid()
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = (alpha_t * (1.0 - p_t).pow(self.gamma) * binary_cross_entropy).mean()

        dims = tuple(range(1, logits.ndim))
        intersection = (probability * target).sum(dim=dims)
        denominator = probability.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return focal + dice


class ProjectionHead(nn.Sequential):
    def __init__(self, dimensions: tuple[int, int, int] = (2048, 512, 256)) -> None:
        super().__init__(
            nn.Linear(dimensions[0], dimensions[1]),
            nn.ReLU(inplace=True),
            nn.Linear(dimensions[1], dimensions[2]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(super().forward(x), dim=-1)


class HardSampleContrastiveLoss(nn.Module):
    """Loss-ranked supervised contrastive anchors with a labelled FIFO queue."""

    def __init__(
        self,
        dimensions: tuple[int, int, int] = (2048, 512, 256),
        *,
        hard_fraction: float = 0.3,
        temperature: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if not 0.0 < hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in (0, 1]")
        self.projector = ProjectionHead(dimensions)
        self.momentum_projector = deepcopy(self.projector)
        self.momentum_projector.requires_grad_(False)
        self.hard_fraction = hard_fraction
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        self.chunk_size = chunk_size
        self.register_buffer("queue", torch.zeros(queue_size, dimensions[-1]))
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update_momentum_projector(self) -> None:
        for online, momentum in zip(
            self.projector.parameters(), self.momentum_projector.parameters()
        ):
            momentum.mul_(self.momentum).add_(online.detach(), alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, embeddings: Tensor, labels: Tensor) -> None:
        embeddings = embeddings.detach().to(device=self.queue.device, dtype=self.queue.dtype)
        labels = labels.detach().to(device=self.queue_labels.device, dtype=torch.long)
        if embeddings.shape[0] >= self.queue_size:
            embeddings = embeddings[-self.queue_size :]
            labels = labels[-self.queue_size :]
        count = embeddings.shape[0]
        if count == 0:
            return
        pointer = int(self.queue_pointer.item())
        first = min(count, self.queue_size - pointer)
        self.queue[pointer : pointer + first].copy_(embeddings[:first])
        self.queue_labels[pointer : pointer + first].copy_(labels[:first])
        remainder = count - first
        if remainder:
            self.queue[:remainder].copy_(embeddings[first:])
            self.queue_labels[:remainder].copy_(labels[first:])
        self.queue_pointer.fill_((pointer + count) % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + count))

    def forward(
        self,
        features: Tensor,
        momentum_features: Tensor,
        labels: Tensor,
        classification_losses: Tensor,
    ) -> Tensor:
        if features.ndim != 2:
            raise ValueError("HSCL expects a matrix of valid pixel features")
        if momentum_features.shape != features.shape:
            raise ValueError("Online and momentum features must have identical shapes")
        if not (features.shape[0] == labels.numel() == classification_losses.numel()):
            raise ValueError("Features, labels and losses must have equal first dimensions")
        if features.shape[0] == 0:
            return features.sum() * 0.0

        embeddings = self.projector(features).float()
        labels = labels.long()
        anchor_count = max(1, math.ceil(self.hard_fraction * features.shape[0]))
        anchor_indices = classification_losses.detach().topk(anchor_count).indices

        filled = int(self.queue_filled.item())
        queue_embeddings = self.queue[:filled].detach()
        queue_labels = self.queue_labels[:filled].detach()
        candidates = torch.cat((embeddings, queue_embeddings), dim=0)
        candidate_labels = torch.cat((labels, queue_labels), dim=0)

        loss_sum = embeddings.new_zeros(())
        valid_anchor_count = 0
        for start in range(0, anchor_count, self.chunk_size):
            indices = anchor_indices[start : start + self.chunk_size]
            anchors = embeddings[indices]
            logits = torch.matmul(anchors, candidates.transpose(0, 1)) / self.temperature
            comparison_mask = torch.ones_like(logits, dtype=torch.bool)
            rows = torch.arange(indices.shape[0], device=indices.device)
            comparison_mask[rows, indices] = False
            positive_mask = (
                labels[indices, None] == candidate_labels[None, :]
            ) & comparison_mask
            valid = positive_mask.any(dim=1)
            if valid.any():
                logits = logits.masked_fill(~comparison_mask, float("-inf"))
                log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
                log_probability = logits - log_denominator
                per_anchor = -(
                    log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
                    / positive_mask.sum(dim=1).clamp_min(1)
                )
                loss_sum = loss_sum + per_anchor[valid].sum()
                valid_anchor_count += int(valid.sum().item())

        with torch.no_grad():
            self._update_momentum_projector()
            queue_embeddings_new = self.momentum_projector(momentum_features.detach()).float()
            self._enqueue(queue_embeddings_new, labels)

        if valid_anchor_count == 0:
            return embeddings.sum() * 0.0
        return loss_sum / valid_anchor_count


class FAANCriterion(nn.Module):
    def __init__(
        self,
        encoder: PairEncoder | None = None,
        *,
        localisation_weight: float = 0.4,
        classification_weight: float = 0.6,
        contrastive_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        projection_dims: tuple[int, int, int] = (2048, 512, 256),
        hard_fraction: float = 0.3,
        temperature: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.localisation = FocalDiceLoss(focal_alpha, focal_gamma)
        self.hscl = HardSampleContrastiveLoss(
            projection_dims,
            hard_fraction=hard_fraction,
            temperature=temperature,
            queue_size=queue_size,
            momentum=momentum,
        )
        self.localisation_weight = localisation_weight
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.ignore_index = ignore_index
        self.momentum = momentum
        self.momentum_encoder = deepcopy(encoder) if encoder is not None else None
        if self.momentum_encoder is not None:
            self.momentum_encoder.requires_grad_(False)
            self.momentum_encoder.eval()

    def train(self, mode: bool = True) -> "FAANCriterion":
        super().train(mode)
        if self.momentum_encoder is not None:
            self.momentum_encoder.eval()
        return self

    @torch.no_grad()
    def _update_momentum_encoder(self, online_encoder: PairEncoder) -> None:
        if self.momentum_encoder is None:
            return
        for online, momentum in zip(
            online_encoder.parameters(), self.momentum_encoder.parameters()
        ):
            momentum.mul_(self.momentum).add_(
                online.detach(), alpha=1.0 - self.momentum
            )
        for online, momentum in zip(
            online_encoder.buffers(), self.momentum_encoder.buffers()
        ):
            if momentum.is_floating_point():
                momentum.mul_(self.momentum).add_(
                    online.detach(), alpha=1.0 - self.momentum
                )
            else:
                momentum.copy_(online)

    @torch.no_grad()
    def _encode_momentum(
        self, pre: Tensor, post: Tensor, online_encoder: PairEncoder
    ) -> Tensor:
        if self.momentum_encoder is None:
            raise RuntimeError("Momentum encoder is not configured")
        self._update_momentum_encoder(online_encoder)
        pre_features, post_features, fused_features = self.momentum_encoder(pre, post)
        return build_contrast_features(pre_features, post_features, fused_features)

    def forward(
        self,
        outputs: dict[str, Tensor],
        localisation_target: Tensor,
        damage_target: Tensor,
        pre: Tensor | None = None,
        post: Tensor | None = None,
        online_encoder: PairEncoder | None = None,
    ) -> dict[str, Tensor]:
        localisation_logits = outputs["localisation_logits"]
        damage_logits = outputs["damage_logits"]
        contrast_features = outputs["contrast_features"]
        localisation_target = localisation_target.to(dtype=localisation_logits.dtype)

        localisation_loss = self.localisation(localisation_logits, localisation_target)
        valid_full = damage_target != self.ignore_index
        if valid_full.any():
            classification_loss = F.cross_entropy(
                damage_logits, damage_target.long(), ignore_index=self.ignore_index
            )
        else:
            classification_loss = damage_logits.sum() * 0.0

        contrast_size = contrast_features.shape[-2:]
        small_labels = F.interpolate(
            damage_target[:, None].float(), size=contrast_size, mode="nearest"
        )[:, 0].long()
        small_logits = F.interpolate(
            damage_logits, size=contrast_size, mode="bilinear", align_corners=False
        )
        per_pixel_loss = F.cross_entropy(
            small_logits, small_labels, ignore_index=self.ignore_index, reduction="none"
        )
        valid = small_labels != self.ignore_index
        feature_vectors = contrast_features.permute(0, 2, 3, 1)[valid]
        if self.momentum_encoder is not None:
            if pre is None or post is None or online_encoder is None:
                raise ValueError("Momentum encoder requires pre/post inputs and online encoder")
            momentum_features = self._encode_momentum(pre, post, online_encoder)
        else:
            momentum_features = contrast_features.detach()
        momentum_vectors = momentum_features.permute(0, 2, 3, 1)[valid]
        contrastive_loss = self.hscl(
            feature_vectors,
            momentum_vectors,
            small_labels[valid],
            per_pixel_loss[valid],
        )
        total = (
            self.localisation_weight * localisation_loss
            + self.classification_weight * classification_loss
            + self.contrastive_weight * contrastive_loss
        )
        return {
            "loss": total,
            "localisation_loss": localisation_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }


def build_criterion(config: dict[str, Any], encoder: PairEncoder) -> FAANCriterion:
    train = config.get("train", {})
    model = config.get("model", {})
    values = tuple(int(value) for value in model.get("projection_dims", (2048, 512, 256)))
    if len(values) != 3:
        raise ValueError("projection_dims must contain three values")
    dimensions = (values[0], values[1], values[2])
    return FAANCriterion(
        encoder,
        localisation_weight=float(train.get("localisation_weight", 0.4)),
        classification_weight=float(train.get("classification_weight", 0.6)),
        contrastive_weight=float(train.get("contrastive_weight", 0.1)),
        focal_alpha=float(train.get("focal_alpha", 0.25)),
        focal_gamma=float(train.get("focal_gamma", 2.0)),
        projection_dims=dimensions,
        hard_fraction=float(train.get("hard_fraction", 0.3)),
        temperature=float(train.get("temperature", 0.1)),
        queue_size=int(train.get("queue_size", 65536)),
        momentum=float(train.get("momentum", 0.999)),
    )
