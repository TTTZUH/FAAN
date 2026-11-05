import torch

from faan.wavelet import TwoLevelDB4


def test_two_level_db4_shapes_and_gradients() -> None:
    feature = torch.randn(2, 3, 64, 48, requires_grad=True)
    bands = TwoLevelDB4()(feature)
    assert len(bands) == 7
    assert [band.shape[-2:] for band in bands] == [
        (16, 12),
        (16, 12),
        (16, 12),
        (16, 12),
        (32, 24),
        (32, 24),
        (32, 24),
    ]
    sum(band.square().mean() for band in bands).backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()

