"""Prepared xBD dataset, paired augmentation and data-loader helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _read_mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.int64).copy()
    return torch.from_numpy(array)


class PairedTransform:
    def __init__(
        self,
        image_size: int = 1024,
        *,
        training: bool,
        rotate90: bool = True,
        flip_probability: float = 0.5,
        crop_scale: tuple[float, float] = (0.8, 1.0),
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.rotate90 = rotate90
        self.flip_probability = flip_probability
        self.crop_scale = crop_scale

    @staticmethod
    def _crop(tensor: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
        return tensor[..., top : top + height, left : left + width]

    def __call__(
        self, pre: Tensor, post: Tensor, localisation: Tensor, damage: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensors = [pre, post, localisation, damage]
        height, width = pre.shape[-2:]
        if self.training:
            scale = random.uniform(*self.crop_scale)
            crop_size = max(1, min(round(min(height, width) * scale), height, width))
            top = random.randint(0, height - crop_size)
            left = random.randint(0, width - crop_size)
            tensors = [self._crop(item, top, left, crop_size, crop_size) for item in tensors]

            if self.rotate90:
                turns = random.randint(0, 3)
                tensors = [torch.rot90(item, turns, dims=(-2, -1)) for item in tensors]
            if random.random() < self.flip_probability:
                tensors = [torch.flip(item, dims=(-1,)) for item in tensors]
            if random.random() < self.flip_probability:
                tensors = [torch.flip(item, dims=(-2,)) for item in tensors]

        pre, post = (
            F.interpolate(
                item[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            for item in tensors[:2]
        )
        localisation, damage = (
            F.interpolate(
                item[None, None].float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            )[0, 0].long()
            for item in tensors[2:]
        )
        return pre, post, localisation, damage


class XBDDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 1024,
        training: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        manifest = self.root / "splits" / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Missing manifest {manifest}. Run tools/prepare_xbd.py first."
            )
        with manifest.open("r", encoding="utf-8") as stream:
            self.samples = [json.loads(line) for line in stream if line.strip()]
        if not self.samples:
            raise ValueError(f"Manifest contains no samples: {manifest}")
        augmentation = augmentation or {}
        crop_values = tuple(
            float(value) for value in augmentation.get("crop_scale", (0.8, 1.0))
        )
        if len(crop_values) != 2:
            raise ValueError("crop_scale must contain two values")
        crop = (crop_values[0], crop_values[1])
        self.transform = PairedTransform(
            image_size,
            training=training,
            rotate90=bool(augmentation.get("rotate90", True)),
            flip_probability=float(augmentation.get("flip_probability", 0.5)),
            crop_scale=crop,
        )
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre = _read_rgb(_resolve(self.root, sample["pre_image"]))
        post = _read_rgb(_resolve(self.root, sample["post_image"]))
        localisation = _read_mask(_resolve(self.root, sample["localisation_mask"]))
        damage = _read_mask(_resolve(self.root, sample["damage_mask"]))
        pre, post, localisation, damage = self.transform(
            pre, post, localisation, damage
        )
        pre_raw, post_raw = pre, post
        pre = (pre - self.mean) / self.std
        post = (post - self.mean) / self.std
        return {
            "pre": pre,
            "post": post,
            "pre_raw": pre_raw,
            "post_raw": post_raw,
            "localisation": localisation[None].float(),
            "damage": damage.long(),
            "sample_id": sample["id"],
            "event": sample.get("event", "unknown"),
        }


def build_dataset(config: dict[str, Any], split: str, *, training: bool) -> XBDDataset:
    data = config["data"]
    return XBDDataset(
        data["root"],
        split,
        image_size=int(data.get("image_size", 1024)),
        training=training,
        mean=tuple(float(value) for value in data.get("mean", (0.485, 0.456, 0.406))),
        std=tuple(float(value) for value in data.get("std", (0.229, 0.224, 0.225))),
        augmentation=data.get("augmentation", {}),
    )


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_loader(
    config: dict[str, Any],
    split: str,
    *,
    training: bool,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    dataset = build_dataset(config, split, training=training)
    data = config["data"]
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=int(data.get("batch_size", 2)),
        shuffle=training,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=training,
        persistent_workers=int(data.get("num_workers", 4)) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
