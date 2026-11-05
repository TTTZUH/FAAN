"""xBD-style localisation, class-wise damage and composite metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor


CLASS_NAMES = ("background", "no_damage", "minor_damage", "major_damage", "destroyed")


def predictions_from_logits(
    localisation_logits: Tensor, damage_logits: Tensor, threshold: float = 0.5
) -> Tensor:
    building = localisation_logits.sigmoid()[:, 0] >= threshold
    damage = damage_logits.argmax(dim=1) + 1
    return torch.where(building, damage, torch.zeros_like(damage))


def targets_to_combined(damage_target: Tensor, ignore_index: int = 255) -> Tensor:
    return torch.where(
        damage_target == ignore_index,
        torch.zeros_like(damage_target),
        damage_target + 1,
    )


class ConfusionMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.confusion = torch.zeros(5, 5, dtype=torch.int64)

    def update(self, localisation_logits: Tensor, damage_logits: Tensor, damage: Tensor) -> None:
        prediction = predictions_from_logits(
            localisation_logits.detach(), damage_logits.detach(), self.threshold
        ).cpu()
        target = targets_to_combined(damage.detach()).cpu()
        indices = target.reshape(-1) * 5 + prediction.reshape(-1)
        self.confusion += torch.bincount(indices, minlength=25).reshape(5, 5)

    def merge(self, other: "ConfusionMetrics") -> None:
        self.confusion += other.confusion

    @staticmethod
    def _f1(tp: Tensor, fp: Tensor, fn: Tensor) -> float:
        denominator = 2 * tp + fp + fn
        return float((2 * tp / denominator).item()) if denominator > 0 else 0.0

    def compute(self) -> dict[str, float]:
        matrix = self.confusion.float()
        foreground_tp = matrix[1:, 1:].sum()
        localisation_tp = matrix.diag()[1:].sum()
        localisation_tp = foreground_tp
        localisation_fp = matrix[0, 1:].sum()
        localisation_fn = matrix[1:, 0].sum()
        localisation_f1 = self._f1(localisation_tp, localisation_fp, localisation_fn)

        class_f1: list[float] = []
        result: dict[str, float] = {"localisation_f1": localisation_f1}
        for class_index, name in enumerate(CLASS_NAMES[1:], start=1):
            tp = matrix[class_index, class_index]
            fp = matrix[:, class_index].sum() - tp
            fn = matrix[class_index, :].sum() - tp
            value = self._f1(tp, fp, fn)
            class_f1.append(value)
            result[f"f1_{name}"] = value
        macro = sum(class_f1) / len(class_f1)
        result["macro_f1"] = macro
        result["composite"] = 0.3 * localisation_f1 + 0.7 * macro

        # Exclude true-background agreement from kappa.
        union = matrix.clone()
        union[0, 0] = 0
        total = union.sum()
        if total > 0:
            observed = union.diag().sum() / total
            expected = (union.sum(dim=0) * union.sum(dim=1)).sum() / total.square()
            denominator = 1.0 - expected
            result["kappa"] = float(((observed - expected) / denominator).item()) if denominator > 0 else 0.0
        else:
            result["kappa"] = 0.0
        return result


class EventMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.overall = ConfusionMetrics(threshold)
        self.events: dict[str, ConfusionMetrics] = defaultdict(
            lambda: ConfusionMetrics(threshold)
        )

    def update(
        self,
        localisation_logits: Tensor,
        damage_logits: Tensor,
        damage: Tensor,
        events: Iterable[str],
    ) -> None:
        self.overall.update(localisation_logits, damage_logits, damage)
        for index, event in enumerate(events):
            self.events[event].update(
                localisation_logits[index : index + 1],
                damage_logits[index : index + 1],
                damage[index : index + 1],
            )

    def compute(self) -> dict[str, object]:
        return {
            "overall": self.overall.compute(),
            "events": {name: metric.compute() for name, metric in sorted(self.events.items())},
        }
