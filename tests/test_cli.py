import sys

import pytest

from faan.cli import parse_args


def test_stage2_requires_pretrain_or_resume(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["faan", "train"])
    with pytest.raises(SystemExit):
        parse_args()


def test_stage2_accepts_encoder_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["faan", "train", "--encoder-checkpoint", "stage1.pt"]
    )
    args = parse_args()
    assert args.encoder_checkpoint.name == "stage1.pt"

