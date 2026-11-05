import torch
import pytest

from faan.model import FAAN, build_model
from faan.utils import parameter_count


def test_faan_output_shapes() -> None:
    model = FAAN(attention_grid=4, decoder_channels=112).eval()
    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        output = model(pre, post)
    assert output["localisation_logits"].shape == (1, 1, 64, 64)
    assert output["damage_logits"].shape == (1, 4, 64, 64)
    assert output["contrast_features"].shape == (1, 2048, 2, 2)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_parameter_budget_matches_reference() -> None:
    model = FAAN(attention_grid=4, decoder_channels=112)
    assert 17_690_000 <= parameter_count(model) <= 17_710_000


@pytest.mark.parametrize(
    ("wavelet", "levels"), (("haar", 2), ("db4", 1))
)
def test_wavelet_configuration_is_validated(wavelet: str, levels: int) -> None:
    config = {
        "model": {
            "backbone": "resnet18",
            "pretrained": False,
            "wavelet": wavelet,
            "wavelet_levels": levels,
        }
    }
    with pytest.raises(ValueError):
        build_model(config)
