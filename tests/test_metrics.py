import torch

from faan.metrics import ConfusionMetrics


def test_perfect_predictions_score_one() -> None:
    damage = torch.tensor([[[255, 0], [1, 3]]])
    localisation_logits = torch.tensor([[[[-10.0, 10.0], [10.0, 10.0]]]])
    damage_logits = torch.full((1, 4, 2, 2), -10.0)
    damage_logits[0, 0, 0, 1] = 10.0
    damage_logits[0, 1, 1, 0] = 10.0
    damage_logits[0, 3, 1, 1] = 10.0
    metric = ConfusionMetrics()
    metric.update(localisation_logits, damage_logits, damage)
    result = metric.compute()
    assert result["localisation_f1"] == 1.0
    assert result["f1_no_damage"] == 1.0
    assert result["f1_minor_damage"] == 1.0
    assert result["f1_destroyed"] == 1.0

