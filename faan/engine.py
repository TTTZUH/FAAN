"""Training and evaluation loops shared by the command-line entry points."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import FAANCriterion
from .metrics import EventMetrics


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    criterion: FAANCriterion,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    grad_clip_norm: float | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    criterion.train()
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        localisation = batch["localisation"].to(device, non_blocking=True)
        damage = batch["damage"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(pre, post)
            losses = criterion(
                outputs,
                localisation,
                damage,
                pre,
                post,
                model.encoder,
            )
        scaler.scale(losses["loss"]).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ],
                grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = pre.shape[0]
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach().item()) * batch_size
        progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")
    return {name: value / max(1, sample_count) for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float = 0.5,
    amp: bool = True,
) -> dict[str, object]:
    model.eval()
    metrics = EventMetrics(threshold)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        damage = batch["damage"].to(device, non_blocking=True)
        with _autocast(device, amp):
            outputs = model(pre, post)
        metrics.update(
            outputs["localisation_logits"],
            outputs["damage_logits"],
            damage,
            batch["event"],
        )
    return metrics.compute()


def pretrain_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    image_count = 0
    progress = tqdm(loader, desc=f"pretrain {epoch:03d}", leave=False)
    for batch in progress:
        inputs = torch.cat((batch["pre"], batch["post"]), dim=0).to(
            device, non_blocking=True
        )
        targets = torch.cat((batch["pre_raw"], batch["post_raw"]), dim=0).to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            reconstruction = model(inputs)
            loss = torch.nn.functional.mse_loss(reconstruction, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item()) * inputs.shape[0]
        image_count += inputs.shape[0]
        progress.set_postfix(mse=f"{float(loss.detach()):.5f}")
    return total_loss / max(1, image_count)
