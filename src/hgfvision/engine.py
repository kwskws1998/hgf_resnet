"""Shared prequential evaluation and parameter updates for all experiment arms."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .filters import DiagonalEKF, FilterConfig
from .models import CIFARResNet18, freeze_for_adaptation


METHODS = ("adamw", "zero_q", "fixed_q", "adaptive_qr", "hierarchical_qr")


def seed_everything(seed: int) -> None:
    """Seed independent runs and request deterministic kernels where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def file_hash(path: str | Path) -> str:
    """Hash a checkpoint without loading it onto a GPU."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    """Hash a canonical JSON value for split and order provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write finite JSON without silently serializing NaN metrics."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def environment(device: torch.device) -> dict[str, Any]:
    """Record runtime and source hashes; no credentials or ambient environment."""
    import torchvision
    result = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return result


def select_device(requested: str) -> torch.device:
    """Fail explicitly if a requested CUDA device is unavailable."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This implementation validates CPU and CUDA only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def synchronize(device: torch.device) -> None:
    """Make CUDA timing include queued device work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, int | None]:
    """Return measured allocator peaks, or null when no CUDA measurement exists."""
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Compute metrics on predictions made before any update with these labels."""
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite prediction")
    probs = logits.detach().float().softmax(-1)
    target = nn.functional.one_hot(labels, logits.shape[-1]).float()
    return {
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "cross_entropy": float(nn.functional.cross_entropy(logits.detach().float(), labels)),
        "brier": float((probs - target).square().sum(-1).mean()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> dict[str, float]:
    """Evaluate clean held-out images without updating parameters or BN state."""
    was_training = model.training
    model.eval()
    totals = {"accuracy": 0.0, "cross_entropy": 0.0, "brier": 0.0}
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        values = classification_metrics(model(x), y)
        for name, value in values.items():
            totals[name] += len(y) * value
        count += len(y)
    model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation dataset is empty")
    return {name: value / count for name, value in totals.items()}


class AdapterUpdater:
    """Use only image/label tensors; regime metadata never enters the updater."""

    def __init__(
        self, model: CIFARResNet18, method: str, filter_config: FilterConfig,
        learning_rate: float = 1e-3, weight_decay: float = 0.0,
        adamw_loss: str = "brier", max_observations: int = 320,
    ):
        """Share trainable parameters and a probability observation surrogate.

        The default AdamW loss is half squared probability error per image,
        matching the filter's Gaussian observation surrogate up to scale.
        Cross entropy is available as an explicitly distinct practical baseline.
        """
        if method not in (*METHODS, "frozen"):
            raise ValueError(f"Unknown method {method}")
        if adamw_loss not in {"brier", "ce"}:
            raise ValueError("AdamW loss must be brier or ce")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        self.model, self.method = freeze_for_adaptation(model), method
        self.params = list(model.adapter_parameters())
        self.device = self.params[0].device
        self.adamw_loss, self.max_observations = adamw_loss, max_observations
        self.optimizer = torch.optim.AdamW(self.params, lr=learning_rate, weight_decay=weight_decay) if method == "adamw" else None
        self.filter = None if method in {"adamw", "frozen"} else DiagonalEKF(
            [sum(p.numel() for p in group) for group in model.parameter_groups()],
            method, filter_config, self.device,
        )

    def process(self, images: Tensor, labels: Tensor) -> dict[str, Any]:
        """Predict first, then make exactly one logical minibatch update.

        Jacobian rows are differentiated one at a time to avoid vectorized
        autograd expansion. The complete small [batch*classes, adapters] matrix
        is retained; no weight updates occur between rows or individual images.
        """
        self.model.train()
        if self.method == "frozen":
            with torch.no_grad():
                return classification_metrics(self.model(images), labels)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        metrics = classification_metrics(logits, labels)
        probabilities = logits.float().softmax(-1)
        targets = nn.functional.one_hot(labels, probabilities.shape[-1]).float()
        if self.optimizer is not None:
            if self.adamw_loss == "ce":
                loss = nn.functional.cross_entropy(logits.float(), labels)
            else:
                loss = 0.5 * (probabilities - targets).square().sum(-1).mean()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params):
                raise FloatingPointError("Nonfinite AdamW gradient")
            self.optimizer.step()
            metrics["optimization_loss"] = float(loss.detach())
        else:
            flat = probabilities.reshape(-1)
            if flat.numel() > self.max_observations:
                raise ValueError("batch*classes exceeds max_observations; decrease batch size")
            jacobian = torch.empty((flat.numel(), sum(p.numel() for p in self.params)), device=self.device)
            for row in range(flat.numel()):
                gradients = torch.autograd.grad(flat[row], self.params, retain_graph=row + 1 < flat.numel())
                jacobian[row] = torch.cat([g.detach().reshape(-1) for g in gradients])
            residual = (targets - probabilities.detach()).reshape(-1)
            displacement, diagnostics = self.filter.step(residual.float(), jacobian.float())
            with torch.no_grad():
                offset = 0
                for parameter in self.params:
                    size = parameter.numel()
                    parameter.add_(displacement[offset:offset + size].reshape_as(parameter))
                    offset += size
            metrics.update(diagnostics)
        if not all(torch.isfinite(p).all() for p in self.params):
            raise FloatingPointError("Nonfinite adapter weights")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Include optimizer state without suggesting an unsupported resume path."""
        if self.filter is not None:
            return self.filter.state_dict()
        return {} if self.optimizer is None else self.optimizer.state_dict()


def run_stream(
    updater: AdapterUpdater, loader: DataLoader, output_dir: Path,
    clean_loader: DataLoader, metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist per-step prequential metrics and final clean retention metrics."""
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {**metadata, "status": "running", "environment": environment(updater.device)})
    rows = []
    started = time.perf_counter()
    try:
        before = evaluate(updater.model, clean_loader, updater.device)
        if updater.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(updater.device)
        with (output_dir / "steps.jsonl").open("w") as handle:
            for step, batch in enumerate(loader):
                images, labels = batch["x"].to(updater.device), batch["y"].to(updater.device)
                if len(set(batch["phase"])) != 1 or not (batch["step"] == step).all():
                    raise ValueError("Loader reordered or mixed stream steps")
                synchronize(updater.device)
                tick = time.perf_counter()
                values = updater.process(images, labels)
                synchronize(updater.device)
                row = {
                    "step": step, "phase": batch["phase"][0], "samples": len(labels),
                    "indices": batch["index"].tolist(),
                    "update_seconds": time.perf_counter() - tick, **values,
                }
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
        after = evaluate(updater.model, clean_loader, updater.device)
        count = sum(row["samples"] for row in rows)
        if count == 0:
            raise ValueError("Empty stream")
        phase_metrics = {}
        for phase in dict.fromkeys(row["phase"] for row in rows):
            selected = [row for row in rows if row["phase"] == phase]
            denom = sum(row["samples"] for row in selected)
            phase_metrics[phase] = {
                key: sum(row[key] * row["samples"] for row in selected) / denom
                for key in ("accuracy", "cross_entropy", "brier")
            }
        summary = {
            "status": "completed", "method": updater.method, "steps": len(rows), "samples": count,
            "prequential": {key: sum(row[key] * row["samples"] for row in rows) / count for key in ("accuracy", "cross_entropy", "brier")},
            "phase_metrics": phase_metrics, "clean_before": before, "clean_after": after,
            "clean_accuracy_change": after["accuracy"] - before["accuracy"],
            "wall_seconds": time.perf_counter() - started,
            "mean_update_seconds": sum(row["update_seconds"] for row in rows) / len(rows),
            **memory_stats(updater.device),
        }
        torch.save({"model": updater.model.state_dict(), "updater": updater.state_dict(), "metadata": metadata}, output_dir / "final.pt")
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "manifest.json", {**metadata, "status": "completed", "environment": environment(updater.device)})
        return summary
    except (Exception, KeyboardInterrupt) as error:
        write_json(output_dir / "failure.json", {"type": type(error).__name__, "message": str(error), "completed_steps": len(rows), **memory_stats(updater.device)})
        write_json(output_dir / "manifest.json", {**metadata, "status": "failed", "environment": environment(updater.device)})
        raise
