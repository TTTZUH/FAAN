"""Command-line interface for FAAN pre-training, training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_loader
from .engine import evaluate, pretrain_one_epoch, train_one_epoch
from .losses import build_criterion
from .metrics import predictions_from_logits
from .model import ReconstructionModel, build_model
from .utils import (
    configure_logging,
    load_config,
    parameter_count,
    resolve_device,
    save_checkpoint,
    set_seed,
    trainable_parameters,
    write_json,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/faan_xbd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default="auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="faan", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Stage 1 reconstruction pre-training")
    _common(pretrain)
    pretrain.add_argument("--train-split", default="train")
    pretrain.add_argument("--resume", type=Path)

    train = subparsers.add_parser("train", help="Stage 2 joint supervised training")
    _common(train)
    initialisation = train.add_mutually_exclusive_group(required=True)
    initialisation.add_argument("--encoder-checkpoint", type=Path)
    initialisation.add_argument("--resume", type=Path)
    train.add_argument("--train-split", default="train")
    train.add_argument("--val-split", default="val")

    evaluation = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint")
    _common(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", default="test")
    evaluation.add_argument("--output", type=Path)

    inference = subparsers.add_parser("infer", help="Infer one pre/post image pair")
    _common(inference)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--pre", type=Path, required=True)
    inference.add_argument("--post", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)

    inspect = subparsers.add_parser("inspect", help="Print architecture and parameter counts")
    _common(inspect)
    return parser.parse_args()


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.set)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "attention_heads": int(model.get("attention_heads", 8)),
        "attention_grid": int(model.get("attention_grid", 16)),
        "decoder_channels": int(model.get("decoder_channels", 128)),
        "wavelet": str(model.get("wavelet", "db4")),
        "wavelet_levels": int(model.get("wavelet_levels", 2)),
    }


def _load_model_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}


def run_pretrain(args: argparse.Namespace) -> None:
    config = _configuration(args)
    device = resolve_device(args.device)
    set_seed(int(config.get("seed", 42)))
    output = Path(config.get("output_dir", "runs/faan_xbd")) / "pretrain"
    logger = configure_logging(output)
    loader = build_loader(config, args.train_split, training=True, device=device)
    model = ReconstructionModel(**_model_config(config)).to(device)
    settings = config["pretrain"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    epochs = int(settings["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(config.get("train", {}).get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint.get("best_mse", best))
    logger.info("Stage 1 | device=%s | parameters=%.3fM", device, parameter_count(model) / 1e6)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch + 1, epochs + 1):
        mse = pretrain_one_epoch(
            model, loader, optimizer, scaler, device, amp=amp, epoch=epoch
        )
        scheduler.step()
        record = {"epoch": epoch, "mse": mse, "lr": scheduler.get_last_lr()[0]}
        history.append(record)
        logger.info("epoch=%d mse=%.6f lr=%.3e", epoch, mse, record["lr"])
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mse": min(best, mse),
            "config": config,
        }
        save_checkpoint(output / "last.pt", payload)
        if mse < best:
            best = mse
            save_checkpoint(output / "best.pt", payload)
        write_json(output / "history.json", history)


def _load_pretrained_encoder(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if "encoder" not in checkpoint:
        raise KeyError("Stage-1 checkpoint does not contain an 'encoder' state")
    model.encoder.backbone.load_state_dict(checkpoint["encoder"], strict=True)


def run_train(args: argparse.Namespace) -> None:
    config = _configuration(args)
    device = resolve_device(args.device)
    set_seed(int(config.get("seed", 42)))
    output = Path(config.get("output_dir", "runs/faan_xbd")) / "train"
    logger = configure_logging(output)
    train_loader = build_loader(config, args.train_split, training=True, device=device)
    val_loader = build_loader(config, args.val_split, training=False, device=device)
    model = build_model(config).to(device)
    if args.encoder_checkpoint:
        _load_pretrained_encoder(model, args.encoder_checkpoint)
        logger.info("Loaded Stage-1 encoder from %s", args.encoder_checkpoint)
    criterion = build_criterion(config, model.encoder).to(device)

    settings = config["train"]
    optimizer = AdamW(
        trainable_parameters(model, criterion),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    epochs = int(settings["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, best = 0, -1.0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["criterion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint.get("best_macro_f1", best))
        history = list(checkpoint.get("history", history))

    logger.info(
        "Stage 2 | device=%s | model=%.3fM | trainable total=%.3fM",
        device,
        parameter_count(model) / 1e6,
        (parameter_count(model) + parameter_count(criterion)) / 1e6,
    )
    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp=amp,
            grad_clip_norm=float(settings.get("grad_clip_norm", 1.0)),
            epoch=epoch,
        )
        scheduler.step()
        validation = evaluate(
            model,
            val_loader,
            device,
            threshold=float(settings.get("localisation_threshold", 0.5)),
            amp=amp,
        )
        overall = validation["overall"]
        macro_f1 = float(overall["macro_f1"])
        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "validation": validation,
        }
        history.append(record)
        logger.info(
            "epoch=%d loss=%.4f val_macro_f1=%.4f val_composite=%.4f",
            epoch,
            train_metrics["loss"],
            macro_f1,
            float(overall["composite"]),
        )
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": max(best, macro_f1),
            "history": history,
            "config": config,
        }
        save_checkpoint(output / "last.pt", payload)
        if macro_f1 > best:
            best = macro_f1
            save_checkpoint(output / "best.pt", payload)
        write_json(output / "history.json", history)


def run_evaluate(args: argparse.Namespace) -> None:
    config = _configuration(args)
    device = resolve_device(args.device)
    logger = configure_logging()
    model = build_model(config).to(device)
    _load_model_checkpoint(model, args.checkpoint)
    loader = build_loader(config, args.split, training=False, device=device)
    metrics = evaluate(
        model,
        loader,
        device,
        threshold=float(config["train"].get("localisation_threshold", 0.5)),
        amp=bool(config["train"].get("amp", True)) and device.type == "cuda",
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    logger.info("Results:\n%s", rendered)
    if args.output:
        write_json(args.output, metrics)


def _load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    with Image.open(path) as source:
        original = source.convert("RGB")
    size = original.size
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor, size, original


@torch.inference_mode()
def run_infer(args: argparse.Namespace) -> None:
    config = _configuration(args)
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    _load_model_checkpoint(model, args.checkpoint)
    model.eval()
    data = config["data"]
    image_size = int(data.get("image_size", 1024))
    pre, original_size, _ = _load_image(args.pre, image_size)
    post, post_size, post_image = _load_image(args.post, image_size)
    if original_size != post_size:
        raise ValueError("Pre- and post-disaster source images must have the same dimensions")
    mean = torch.tensor(data.get("mean", (0.485, 0.456, 0.406)))[:, None, None]
    std = torch.tensor(data.get("std", (0.229, 0.224, 0.225)))[:, None, None]
    pre = ((pre - mean) / std)[None].to(device)
    post = ((post - mean) / std)[None].to(device)
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    context = torch.cuda.amp.autocast(enabled=amp) if device.type == "cuda" else torch.no_grad()
    with context:
        outputs = model(pre, post)
    combined = predictions_from_logits(
        outputs["localisation_logits"],
        outputs["damage_logits"],
        float(config["train"].get("localisation_threshold", 0.5)),
    )[0].byte().cpu().numpy()
    localisation = (combined > 0).astype(np.uint8) * 255
    palette = np.array(
        [[0, 0, 0], [0, 180, 0], [255, 225, 0], [255, 140, 0], [220, 0, 0]],
        dtype=np.uint8,
    )
    colour = palette[combined]
    mask_image = Image.fromarray(colour).resize(original_size, Image.Resampling.NEAREST)
    loc_image = Image.fromarray(localisation).resize(original_size, Image.Resampling.NEAREST)
    overlay = Image.blend(post_image, mask_image, alpha=0.45)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_image.save(args.output_dir / "damage.png")
    loc_image.save(args.output_dir / "localisation.png")
    overlay.save(args.output_dir / "overlay.png")


def run_inspect(args: argparse.Namespace) -> None:
    config = _configuration(args)
    model = build_model(config)
    criterion = build_criterion(config, model.encoder)
    model_parameters = parameter_count(model)
    trainable_total = model_parameters + parameter_count(criterion)
    print(model)
    print(f"Model parameters: {model_parameters:,} ({model_parameters / 1e6:.3f} M)")
    print(f"Model + HSCL trainable parameters: {trainable_total:,} ({trainable_total / 1e6:.3f} M)")
    print("Reference parameter count: 17.700 M")


def main() -> None:
    args = parse_args()
    commands = {
        "pretrain": run_pretrain,
        "train": run_train,
        "evaluate": run_evaluate,
        "infer": run_infer,
        "inspect": run_inspect,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
