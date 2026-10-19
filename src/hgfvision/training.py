"""Source training with atomic, reproducible completed-epoch checkpoints."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CleanDataset, SourceDataset, load_cifar10, make_splits
from .engine import environment, evaluate, file_hash, json_hash, seed_everything, select_device, write_json
from .models import build_model, freeze_for_source


def atomic_save(payload: Any, path: str | Path) -> None:
    """Replace a checkpoint only after its temporary file is fully written."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_copy(value: Any) -> Any:
    """Detach checkpoint tensors from live CPU or CUDA training storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    """Encode all random state using weights-only-loadable types."""
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state(),
        "numpy": {
            "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    """Restore the exact next-epoch random streams after constructing the model."""
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["loader_generator"])
    cuda_state = state["torch_cuda"]
    if cuda_state:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise ValueError("CUDA random-state device count differs from the checkpoint")
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    random.setstate(state["python"])


def _dataset_fingerprint(dataset) -> str:
    """Hash image and label contents so equal lengths cannot hide changed data."""
    digest = hashlib.sha256()
    digest.update(f"{type(dataset).__module__}.{type(dataset).__qualname__}:{len(dataset)}".encode())
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        values = (np.asarray(dataset.data), np.asarray(dataset.targets, dtype=np.int64))
        for value in values:
            digest.update(f"{value.dtype}:{value.shape}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    else:
        for image, label in dataset:
            value = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            digest.update(f"{value.dtype}:{value.shape}:{int(label)}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _training_spec(args, device: torch.device, runtime: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments and define the immutable continuation contract."""
    integers = {"epochs": 1, "batch_size": 1, "eval_batch_size": 1, "workers": 0, "seed": 0, "split_seed": 0}
    for name, minimum in integers.items():
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("lr", "weight_decay"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        if value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"Invalid {name}")
    return {
        **{name: getattr(args, name) for name in integers},
        "lr": args.lr, "weight_decay": args.weight_decay,
        "optimizer": "sgd_momentum_0.9", "scheduler": "cosine", "augmentation": "crop_flip",
        "device_type": device.type, "torch": runtime["torch"], "torchvision": runtime["torchvision"],
        "cuda_runtime": runtime["cuda_runtime"] if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
    }


def _validate_resume(state: dict[str, Any], spec: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Reject changed data, schedules, seeds, or malformed epoch boundaries."""
    if state.get("format_version") != 1:
        raise ValueError("Unsupported source resume checkpoint format")
    if state.get("training_spec") != spec:
        raise ValueError("Resume training arguments/runtime differ; keep the original total epochs and scheduler horizon")
    for key in ("dataset", "dataset_length", "dataset_sha256", "split_seed", "training_seed", "split_hashes", "num_classes", "architecture", "epochs_requested"):
        if state.get("metadata", {}).get(key) != metadata[key]:
            raise ValueError(f"Resume checkpoint differs in {key}")
    next_epoch = state.get("next_epoch")
    if type(next_epoch) is not int or not 1 <= next_epoch <= spec["epochs"]:
        raise ValueError("Resume checkpoint has an invalid completed-epoch boundary")
    history = state.get("history", [])
    if len(history) != next_epoch or [row.get("epoch") for row in history] != list(range(next_epoch)):
        raise ValueError("Resume checkpoint history is inconsistent with next_epoch")
    best = state.get("best", {})
    selected = best.get("metadata", {}).get("selected_epoch")
    if type(selected) is not int or not 0 <= selected < next_epoch:
        raise ValueError("Resume checkpoint best epoch is invalid")
    scores = [row["validation"]["accuracy"] for row in history]
    if selected != max(range(next_epoch), key=scores.__getitem__) or state.get("best_accuracy") != scores[selected]:
        raise ValueError("Resume checkpoint best selection disagrees with committed history")
    if state.get("scheduler", {}).get("T_max") != spec["epochs"] or state["scheduler"].get("last_epoch") != next_epoch:
        raise ValueError("Resume scheduler state disagrees with the completed epochs")


def _restore_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Discard uncommitted log rows and preserve all committed rows exactly."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in history:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_source(args, raw_train=None, fake: bool = False) -> Path:
    """Train or resume source learning at the last atomically completed epoch.

    ``args.resume`` defaults to False. True resumes ``args.output/last.pt``;
    all learning settings, including the original total epoch budget, must
    match. A partial epoch is replayed. The best source weights remain in
    ``source.pt`` and are repaired from the committed best snapshot on resume.
    """
    device = select_device(args.device)
    seed_everything(args.seed)
    runtime = environment(device)
    spec = _training_spec(args, device, runtime)
    resume = getattr(args, "resume", False)
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    output = Path(args.output)
    last_path, best_path = output / "last.pt", output / "source.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"No completed-epoch checkpoint to resume: {last_path}")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
    elif output.exists():
        raise FileExistsError(f"Output already exists; use --resume for a completed-epoch checkpoint: {output}")
    else:
        state = None
    if raw_train is None:
        raw_train, _ = load_cifar10(args.data_root, download=args.download)
    splits = make_splits(len(raw_train), args.split_seed)
    train_data = SourceDataset(raw_train, splits["source"], seed=args.seed, augment=True)
    val_data = CleanDataset(raw_train, splits["source_val"])
    if len(train_data) < 2 or not len(val_data):
        raise ValueError("Source and validation splits are too small")
    metadata = {
        "dataset": "fake_smoke_only" if fake else "cifar10", "dataset_length": len(raw_train),
        "dataset_sha256": _dataset_fingerprint(raw_train),
        "split_seed": args.split_seed, "training_seed": args.seed,
        "split_hashes": {key: json_hash(value) for key, value in splits.items()},
        "num_classes": 10, "architecture": "cifar_resnet18", "epochs_requested": args.epochs,
        "source_training": {key: spec[key] for key in ("batch_size", "lr", "weight_decay", "optimizer", "scheduler", "augmentation")},
        "source_environment": runtime,
    }
    if state is not None:
        _validate_resume(state, spec, metadata)
        metadata = state["metadata"]
    output.mkdir(parents=True, exist_ok=resume)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    model = freeze_for_source(build_model(adapters=False)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = [] if state is None else state["history"]
    next_epoch = 0 if state is None else state["next_epoch"]
    best_accuracy = -1.0 if state is None else state["best_accuracy"]
    best = None if state is None else state["best"]
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"], generator)
    write_json(output / "splits.json", splits)
    write_json(output / "manifest.json", {**metadata, "environment": runtime, "status": "running", "completed_epochs": next_epoch, "next_epoch": next_epoch, "resumed": resume})
    try:
        if state is not None:
            atomic_save(best, best_path)
            _restore_history(output / "epochs.jsonl", history)
        with (output / "epochs.jsonl").open("a" if resume else "w") as handle:
            for epoch in range(next_epoch, args.epochs):
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
                os.fsync(handle.fileno())
                print(json.dumps(row), flush=True)
                if val["accuracy"] > best_accuracy:
                    best_accuracy = val["accuracy"]
                    best = {"model": _cpu_copy(model.state_dict()), "metadata": {**metadata, "selected_epoch": epoch, "validation": val}}
                    atomic_save(best, best_path)
                scheduler.step()
                committed_history = [*history, row]
                last = {
                    "format_version": 1, "model": _cpu_copy(model.state_dict()),
                    "optimizer": _cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                    "metadata": metadata, "training_spec": spec, "next_epoch": epoch + 1,
                    "best_accuracy": best_accuracy, "best": best,
                    "rng": _capture_rng(generator), "history": committed_history,
                }
                atomic_save(last, last_path)
                next_epoch, history = epoch + 1, committed_history
                write_json(output / "manifest.json", {**metadata, "environment": runtime, "status": "running", "completed_epochs": next_epoch, "next_epoch": next_epoch, "resumed": resume})
        write_json(output / "manifest.json", {**metadata, "environment": runtime, "status": "completed", "completed_epochs": next_epoch, "next_epoch": next_epoch, "resumed": resume, "checkpoint_sha256": file_hash(best_path), "last_checkpoint_sha256": file_hash(last_path)})
        if (output / "failure.json").exists():
            (output / "failure.json").unlink()
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        checkpoint_error = None
        resume_available = last_path.is_file()
        if resume_available:
            try:
                next_epoch = torch.load(last_path, map_location="cpu", weights_only=True)["next_epoch"]
            except Exception as read_error:
                checkpoint_error = f"{type(read_error).__name__}: {read_error}"
                resume_available = False
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "completed_epochs": next_epoch, "next_epoch": next_epoch, "resume_available": resume_available, "checkpoint_read_error": checkpoint_error})
        write_json(output / "manifest.json", {**metadata, "environment": runtime, "status": status, "completed_epochs": next_epoch, "next_epoch": next_epoch, "resumed": resume})
        raise
    return best_path
