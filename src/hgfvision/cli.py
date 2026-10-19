"""Python-only entry points for source training, adaptation, and offline checks."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CleanDataset, FakeCIFAR10, Phase, SourceDataset, StreamDataset, load_cifar10, make_splits
from .engine import AdapterUpdater, METHODS, environment, evaluate, file_hash, json_hash, run_stream, seed_everything, select_device, write_json
from .filters import FilterConfig
from .models import build_model, freeze_for_source, load_source_weights


def read_config(path: str | Path | None) -> dict:
    """Validate the full experiment configuration and reject misspelled fields."""
    config = {
        "rank": 4, "batch_size": 8, "max_observations": 320,
        "adamw": {"learning_rate": 0.001, "weight_decay": 0.0, "loss": "brier"},
        "filter": asdict(FilterConfig()),
        "phases": [
            {"name": "stable", "steps": 50},
            {"name": "persistent_shift", "steps": 50, "brightness": 0.65, "contrast": 0.8},
            {"name": "shift_plus_noise", "steps": 50, "brightness": 0.65, "contrast": 0.8, "noise_std": 0.15},
            {"name": "shift_recovery", "steps": 50, "brightness": 0.65, "contrast": 0.8},
        ],
    }
    if path is not None:
        provided = json.loads(Path(path).read_text())
        if not isinstance(provided, dict) or set(provided) - set(config):
            raise ValueError("Unknown experiment configuration keys")
        for key, value in provided.items():
            if key in {"filter", "adamw"}:
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Unknown {key} configuration keys")
                config[key].update(value)
            else:
                config[key] = value
    for name in ("rank", "batch_size", "max_observations"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["batch_size"] * 10 > config["max_observations"]:
        raise ValueError("batch_size*10 exceeds max_observations")
    FilterConfig(**config["filter"])
    phases = [Phase(**phase) for phase in config["phases"]]
    if not phases or len({phase.name for phase in phases}) != len(phases):
        raise ValueError("Use nonempty phases with unique names for reporting")
    if config["adamw"]["loss"] not in {"brier", "ce"}:
        raise ValueError("AdamW loss must be brier or ce")
    for key in ("learning_rate", "weight_decay"):
        value = config["adamw"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"AdamW {key} must be finite and numeric")
        if value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid AdamW {key}")
    return config


def train_source(args, raw_train=None, fake: bool = False) -> Path:
    """Train on source indices; choose checkpoint using source_val only."""
    device = select_device(args.device)
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if raw_train is None:
        raw_train, _ = load_cifar10(args.data_root, download=args.download)
    splits = make_splits(len(raw_train), args.split_seed)
    train_data = SourceDataset(raw_train, splits["source"], seed=args.seed, augment=True)
    val_data = CleanDataset(raw_train, splits["source_val"])
    if len(train_data) < 2 or not len(val_data):
        raise ValueError("Source and validation splits are too small")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    model = freeze_for_source(build_model(adapters=False)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "dataset": "fake_smoke_only" if fake else "cifar10", "dataset_length": len(raw_train),
        "split_seed": args.split_seed, "training_seed": args.seed,
        "split_hashes": {key: json_hash(value) for key, value in splits.items()},
        "num_classes": 10, "architecture": "cifar_resnet18", "epochs_requested": args.epochs,
        "source_training": {"batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "optimizer": "sgd_momentum_0.9", "scheduler": "cosine", "augmentation": "crop_flip"},
        "source_environment": environment(device),
    }
    write_json(output / "splits.json", splits)
    write_json(output / "manifest.json", {**metadata, "environment": environment(device), "status": "running"})
    best_accuracy = -1.0
    checkpoint = output / "source.pt"
    try:
        with (output / "epochs.jsonl").open("w") as handle:
            for epoch in range(args.epochs):
                train_data.set_epoch(epoch)
                model.train()
                total_loss, count = 0.0, 0
                tick = time.perf_counter()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite source loss")
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(y)
                    count += len(y)
                val = evaluate(model, val_loader, device)
                row = {"epoch": epoch, "training_loss": total_loss / count, "validation": val, "learning_rate": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - tick}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                print(json.dumps(row), flush=True)
                if val["accuracy"] > best_accuracy:
                    best_accuracy = val["accuracy"]
                    torch.save({"model": {key: value.cpu() for key, value in model.state_dict().items()}, "metadata": {**metadata, "selected_epoch": epoch, "validation": val}}, checkpoint)
                scheduler.step()
        write_json(output / "manifest.json", {**metadata, "environment": environment(device), "status": "completed", "checkpoint_sha256": file_hash(checkpoint)})
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json(output / "manifest.json", {**metadata, "environment": environment(device), "status": "failed"})
        raise
    return checkpoint


def adapt(args, raw_train=None, raw_test=None, fake: bool = False, config_override=None) -> list[dict]:
    """Reload identical source weights and stream order for each independent arm."""
    device = select_device(args.device)
    config = config_override if config_override is not None else read_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    info = checkpoint["metadata"]
    expected_dataset = "fake_smoke_only" if fake else "cifar10"
    if info["dataset"] != expected_dataset or info["architecture"] != "cifar_resnet18" or info["num_classes"] != 10:
        raise ValueError("Checkpoint dataset/architecture does not match this experiment")
    if raw_train is None:
        raw_train, raw_test = load_cifar10(args.data_root, download=args.download)
    if len(raw_train) != info["dataset_length"]:
        raise ValueError("Dataset length differs from the source checkpoint")
    splits = make_splits(len(raw_train), info["split_seed"])
    if any(json_hash(value) != info["split_hashes"][key] for key, value in splits.items()):
        raise ValueError("Reconstructed splits differ from source provenance")
    indices = np.random.default_rng(args.order_seed).permutation(splits[args.partition]).tolist()
    phases = [Phase(**phase) for phase in config["phases"]]
    stream = StreamDataset(raw_train, indices, phases, config["batch_size"], args.noise_seed)
    stream_loader = DataLoader(stream, batch_size=config["batch_size"], shuffle=False, num_workers=args.workers)
    if args.partition == "dev":
        clean = CleanDataset(raw_train, splits["source_val"])
        clean_name = "source_val"
    else:
        if raw_test is None:
            raise ValueError("Held-out official test data required for confirmatory stream")
        clean = CleanDataset(raw_test)
        clean_name = "official_test" if not fake else "fake_holdout"
    clean_loader = DataLoader(clean, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "stream_indices.json", list(stream.indices))
    write_json(output / "splits.json", splits)
    summaries = []
    selected = list(METHODS) if args.method == "all" else [args.method]
    if args.method == "all":
        selected.insert(0, "frozen")
    for method in selected:
        seed_everything(args.seed)
        model = build_model(rank=config["rank"])
        load_source_weights(model, checkpoint["model"])
        model.to(device)
        updater = AdapterUpdater(
            model, method, FilterConfig(**config["filter"]),
            learning_rate=config["adamw"]["learning_rate"], weight_decay=config["adamw"]["weight_decay"],
            adamw_loss=config["adamw"]["loss"], max_observations=config["max_observations"],
        )
        metadata = {
            "dataset": expected_dataset, "method": method, "config": config,
            "seed": args.seed, "order_seed": args.order_seed, "noise_seed": args.noise_seed,
            "partition": args.partition, "clean_evaluation_split": clean_name,
            "checkpoint_sha256": file_hash(checkpoint_path), "source_metadata": info,
            "stream_indices_sha256": json_hash(list(stream.indices)),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "observation_model": "Gaussian surrogate on softmax probabilities",
            "proposed_method_status": "HGF-inspired Fisher approximation; not exact generalized-HGF reproduction",
            "tuning_status": "defaults unless explicitly overridden; no tuning claimed",
        }
        summary = run_stream(updater, stream_loader, output / method, clean_loader, metadata)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del updater, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "comparison.json", summaries)
    return summaries


def smoke(args) -> None:
    """Exercise real ResNet code on fake images, without downloading any data."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw_train, raw_test = FakeCIFAR10(200, seed=11), FakeCIFAR10(16, seed=12)
    source_args = argparse.Namespace(
        device=args.device, seed=0, split_seed=0, output=output / "source", data_root="unused", download=False,
        batch_size=20, eval_batch_size=16, workers=0, epochs=1, lr=0.01, weight_decay=0.0005,
    )
    checkpoint = train_source(source_args, raw_train=raw_train, fake=True)
    config = read_config(None)
    config["batch_size"] = 2
    config["phases"] = [
        {"name": "stable", "steps": 2},
        {"name": "shift", "steps": 2, "brightness": 0.7},
        {"name": "noise", "steps": 2, "brightness": 0.7, "noise_std": 0.1},
        {"name": "recovery", "steps": 2, "brightness": 0.7},
    ]
    adapt_args = argparse.Namespace(
        device=args.device, checkpoint=checkpoint, config=None, data_root="unused", download=False,
        partition="dev", order_seed=0, noise_seed=0, workers=0, eval_batch_size=16,
        output=output / "adapt", method="all", seed=0,
    )
    results = adapt(adapt_args, raw_train=raw_train, raw_test=raw_test, fake=True, config_override=config)
    write_json(output / "validation.json", {"kind": "offline_fake_image_smoke_only", "methods_completed": [r["method"] for r in results], "cuda_profiled": select_device(args.device).type == "cuda", "real_data_efficacy_evidence": False})


def positive_int(value: str) -> int:
    """Reject zero and negative CLI dimensions."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    """Dispatch commands without network access unless --download is explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=positive_int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("train-source")
    adaptation = sub.add_parser("adapt")
    smoke_parser = sub.add_parser("smoke")
    synthetic_parser = sub.add_parser("synthetic")
    for command in (source, adaptation, smoke_parser, synthetic_parser):
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto")
    for command in (source, adaptation):
        command.add_argument("--data-root", default="data")
        command.add_argument("--download", action="store_true")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--eval-batch-size", type=positive_int, default=128)
    source.add_argument("--split-seed", type=int, default=0)
    source.add_argument("--epochs", type=positive_int, default=100)
    source.add_argument("--batch-size", type=positive_int, default=128)
    source.add_argument("--lr", type=float, default=0.1)
    source.add_argument("--weight-decay", type=float, default=0.0005)
    adaptation.add_argument("--checkpoint", required=True)
    adaptation.add_argument("--config")
    adaptation.add_argument("--method", choices=["all", "frozen", *METHODS], default="all")
    adaptation.add_argument("--partition", choices=["dev", "stream"], default="dev")
    adaptation.add_argument("--order-seed", type=int, default=0)
    adaptation.add_argument("--noise-seed", type=int, default=0)
    synthetic_parser.add_argument("--steps-per-phase", type=positive_int, default=100)
    synthetic_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(getattr(args, key, 0) < 0 for key in ("seed", "split_seed", "order_seed", "noise_seed", "workers")):
        parser.error("seeds and worker counts must be nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    if args.command == "train-source":
        train_source(args)
    elif args.command == "adapt":
        adapt(args)
    elif args.command == "smoke":
        smoke(args)
    else:
        from .synthetic import run_synthetic
        run_synthetic(args)


if __name__ == "__main__":
    main()
