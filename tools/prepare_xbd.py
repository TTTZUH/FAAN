#!/usr/bin/env python
"""Rasterise xBD JSON labels and create deterministic grouped manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST")
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the prepared tree instead of recording absolute paths.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="CSV with id,split columns for the fixed main-experiment split.",
    )
    return parser.parse_args()


def _rings_from_wkt(wkt: str) -> list[list[tuple[float, float]]]:
    """Parse polygon rings used by xBD without requiring GEOS/Shapely."""
    rings: list[list[tuple[float, float]]] = []
    for body in re.findall(r"\(([^()]+)\)", wkt):
        points: list[tuple[float, float]] = []
        for coordinate in body.split(","):
            values = coordinate.strip().split()
            if len(values) >= 2:
                points.append((float(values[0]), float(values[1])))
        if len(points) >= 3:
            rings.append(points)
    return rings


def _features(label_path: Path) -> Iterable[dict]:
    with label_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    features = payload.get("features", payload)
    if isinstance(features, dict):
        # xBD provides geographic and pixel polygons; masks require pixel coordinates.
        if isinstance(features.get("xy"), list):
            yield from features["xy"]
            return
        for value in features.values():
            if isinstance(value, list):
                yield from value
    elif isinstance(features, list):
        yield from features


def rasterise(
    label_path: Path, size: tuple[int, int], *, damage: bool
) -> Image.Image:
    fill_background = 255 if damage else 0
    mask = Image.new("L", size, color=fill_background)
    draw = ImageDraw.Draw(mask)
    for feature in _features(label_path):
        properties = feature.get("properties", {})
        if properties.get("feature_type", "building") != "building":
            continue
        if damage:
            fill = DAMAGE_CLASSES.get(properties.get("subtype", ""))
            if fill is None:
                continue
        else:
            fill = 1
        wkt = feature.get("wkt") or feature.get("geometry", {}).get("wkt", "")
        for ring in _rings_from_wkt(wkt):
            draw.polygon(ring, fill=fill)
    return mask


def event_from_id(sample_id: str) -> str:
    match = re.match(r"(.+?)_\d", sample_id)
    return match.group(1) if match else sample_id.rsplit("_", 1)[0]


def geographic_group(sample_id: str) -> str:
    return sample_id.split("__crop", 1)[0]


def discover(raw_root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for pre_image in sorted(raw_root.rglob("*_pre_disaster.png")):
        sample_id = pre_image.name.removesuffix("_pre_disaster.png")
        post_image = pre_image.with_name(f"{sample_id}_post_disaster.png")
        labels_dir = pre_image.parent.parent / "labels"
        pre_label = labels_dir / f"{sample_id}_pre_disaster.json"
        post_label = labels_dir / f"{sample_id}_post_disaster.json"
        if post_image.is_file() and pre_label.is_file() and post_label.is_file():
            samples.append(
                {
                    "id": sample_id,
                    "event": event_from_id(sample_id),
                    "group": geographic_group(sample_id),
                    "pre_image": str(pre_image.resolve()),
                    "post_image": str(post_image.resolve()),
                    "pre_label": str(pre_label.resolve()),
                    "post_label": str(post_label.resolve()),
                }
            )
    if not samples:
        raise FileNotFoundError(f"No labelled pre/post xBD pairs found under {raw_root}")
    return samples


def split_grouped(
    samples: list[dict[str, str]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[dict[str, str]]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    by_event_group: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        by_event_group[sample["event"]][sample["group"]].append(sample)

    result: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for event, groups in sorted(by_event_group.items()):
        group_names = sorted(groups)
        event_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{event}".encode()).digest()[:8], "little"
        )
        random.Random(event_seed).shuffle(group_names)
        count = len(group_names)
        train_end = round(count * ratios[0])
        val_end = train_end + round(count * ratios[1])
        assignments = {
            "train": group_names[:train_end],
            "val": group_names[train_end:val_end],
            "test": group_names[val_end:],
        }
        for split, names in assignments.items():
            for name in names:
                result[split].extend(groups[name])
    return result


def split_from_file(
    samples: list[dict[str, str]], path: Path
) -> dict[str, list[dict[str, str]]]:
    valid_splits = {"train", "val", "test"}
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "split"}.issubset(reader.fieldnames):
            raise ValueError("Split CSV must contain id and split columns")
        for row in reader:
            sample_id = row["id"].strip()
            split = row["split"].strip().lower()
            if not sample_id:
                raise ValueError("Split CSV contains an empty sample id")
            if split not in valid_splits:
                raise ValueError(f"Invalid split {split!r} for sample {sample_id!r}")
            if sample_id in assignments:
                raise ValueError(f"Duplicate sample id in split CSV: {sample_id}")
            assignments[sample_id] = split

    samples_by_id = {sample["id"]: sample for sample in samples}
    missing = sorted(set(samples_by_id) - set(assignments))
    unknown = sorted(set(assignments) - set(samples_by_id))
    if missing:
        raise ValueError(f"Split CSV is missing {len(missing)} samples; first: {missing[0]}")
    if unknown:
        raise ValueError(f"Split CSV contains {len(unknown)} unknown samples; first: {unknown[0]}")

    group_splits: dict[str, str] = {}
    result: dict[str, list[dict[str, str]]] = {name: [] for name in valid_splits}
    for sample_id, sample in samples_by_id.items():
        split = assignments[sample_id]
        group = sample["group"]
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(f"Group {group!r} appears in both {previous!r} and {split!r}")
        result[split].append(sample)
    if any(not result[name] for name in valid_splits):
        raise ValueError("The fixed split must contain train, val and test samples")
    return result


def write_split_file(path: Path, splits: dict[str, list[dict[str, str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "split"))
        writer.writeheader()
        rows = (
            {"id": sample["id"], "split": split}
            for split, samples in splits.items()
            for sample in samples
        )
        writer.writerows(sorted(rows, key=lambda row: row["id"]))


def write_manifest(path: Path, samples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in sorted(samples, key=lambda item: item["id"]):
            record = {key: value for key, value in sample.items() if key not in {"group", "pre_label", "post_label"}}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    localisation_dir = output / "masks" / "localisation"
    damage_dir = output / "masks" / "damage"
    split_dir = output / "splits"
    for directory in (localisation_dir, damage_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = discover(args.raw_root.resolve())
    if args.split_file:
        splits = split_from_file(samples, args.split_file.resolve())
    else:
        ratios = tuple(float(value) for value in args.ratios)
        splits = split_grouped(samples, ratios, args.seed)
    image_dir = output / "images"
    if args.copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        pre_path, post_path = Path(sample["pre_image"]), Path(sample["post_image"])
        with Image.open(pre_path) as image:
            size = image.size
        localisation_path = localisation_dir / f"{sample['id']}.png"
        damage_path = damage_dir / f"{sample['id']}.png"
        rasterise(Path(sample["pre_label"]), size, damage=False).save(localisation_path)
        rasterise(Path(sample["post_label"]), size, damage=True).save(damage_path)

        if args.copy_images:
            copied_pre = image_dir / pre_path.name
            copied_post = image_dir / post_path.name
            shutil.copy2(pre_path, copied_pre)
            shutil.copy2(post_path, copied_post)
            sample["pre_image"] = str(copied_pre.relative_to(output))
            sample["post_image"] = str(copied_post.relative_to(output))
        sample["localisation_mask"] = str(localisation_path.relative_to(output))
        sample["damage_mask"] = str(damage_path.relative_to(output))
        prepared.append(sample)
        if index % 500 == 0:
            print(f"Prepared {index}/{len(samples)} samples")

    for name, records in splits.items():
        write_manifest(split_dir / f"{name}.jsonl", records)
    write_split_file(split_dir / "main_split.csv", splits)

    counts = {name: len(records) for name, records in splits.items()}
    print(f"Prepared {len(prepared)} pairs at {output}")
    print("Grouped split counts:", counts)


if __name__ == "__main__":
    main()
