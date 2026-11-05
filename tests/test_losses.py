import torch

from faan.losses import FAANCriterion, HardSampleContrastiveLoss


def test_hscl_is_finite_and_updates_queue() -> None:
    loss_fn = HardSampleContrastiveLoss(
        (8, 6, 4), hard_fraction=0.3, temperature=0.1, queue_size=32
    )
    features = torch.randn(12, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    losses = torch.linspace(0.0, 1.0, 12)
    loss = loss_fn(features, features.detach(), labels, losses)
    assert torch.isfinite(loss)
    assert loss_fn.queue_filled.item() == 12
    loss.backward()
    assert features.grad is not None


def test_joint_loss_handles_valid_and_ignored_pixels() -> None:
    criterion = FAANCriterion(projection_dims=(8, 6, 4), queue_size=32)
    output = {
        "localisation_logits": torch.randn(1, 1, 8, 8, requires_grad=True),
        "damage_logits": torch.randn(1, 4, 8, 8, requires_grad=True),
        "contrast_features": torch.randn(1, 8, 2, 2, requires_grad=True),
    }
    localisation = torch.zeros(1, 1, 8, 8)
    localisation[:, :, 2:6, 2:6] = 1
    damage = torch.full((1, 8, 8), 255, dtype=torch.long)
    damage[:, 2:6, 2:6] = 1
    result = criterion(output, localisation, damage)
    assert set(result) == {
        "loss",
        "localisation_loss",
        "classification_loss",
        "contrastive_loss",
    }
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


def test_joint_loss_uses_full_momentum_encoder() -> None:
    from faan.model import FAAN

    model = FAAN(attention_grid=4, decoder_channels=16).train()
    criterion = FAANCriterion(model.encoder, queue_size=32).train()
    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    localisation = torch.ones(1, 1, 64, 64)
    damage = torch.randint(0, 4, (1, 64, 64))

    output = model(pre, post)
    result = criterion(
        output, localisation, damage, pre, post, model.encoder
    )
    result["loss"].backward()

    assert criterion.momentum_encoder is not None
    assert not criterion.momentum_encoder.training
    assert criterion.hscl.queue_filled.item() == 4
