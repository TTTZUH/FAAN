import csv

import pytest

from tools.prepare_xbd import split_from_file, write_split_file


def _samples() -> list[dict[str, str]]:
    return [
        {"id": "a", "group": "g1"},
        {"id": "b", "group": "g2"},
        {"id": "c", "group": "g3"},
    ]


def test_fixed_split_round_trip(tmp_path) -> None:
    expected = {
        "train": [_samples()[0]],
        "val": [_samples()[1]],
        "test": [_samples()[2]],
    }
    path = tmp_path / "main_split.csv"
    write_split_file(path, expected)
    actual = split_from_file(_samples(), path)
    assert {name: [item["id"] for item in values] for name, values in actual.items()} == {
        "train": ["a"],
        "val": ["b"],
        "test": ["c"],
    }


def test_fixed_split_rejects_group_leakage(tmp_path) -> None:
    samples = _samples()
    samples[1]["group"] = "g1"
    path = tmp_path / "main_split.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "split"))
        writer.writeheader()
        writer.writerows(
            (
                {"id": "a", "split": "train"},
                {"id": "b", "split": "val"},
                {"id": "c", "split": "test"},
            )
        )
    with pytest.raises(ValueError, match="appears in both"):
        split_from_file(samples, path)

