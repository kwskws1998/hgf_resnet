"""Short device preflight for the exact source and adaptation batch sizes."""

import argparse
from pathlib import Path
import time

import torch
from torch import nn

from .engine import AdapterUpdater, environment, memory_stats, seed_everything, select_device, synchronize, write_json
from .filters import FilterConfig
from .models import build_model, freeze_for_source, load_source_weights


def benchmark(output: Path, device: torch.device, source_batch: int, batch: int, repeats: int = 3) -> dict:
    """Measure generated-image operations; never reuse probe weights in experiments."""
    if min(source_batch, batch, repeats) < 1:
        raise ValueError("Batch sizes and repetitions must be positive")
    output.parent.mkdir(parents=True, exist_ok=False)
    seed_everything(0)
    model = freeze_for_source(build_model(adapters=False)).to(device)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    images = torch.randn(source_batch, 3, 32, 32, device=device)
    labels = torch.arange(source_batch, device=device) % 10
    source_times = []
    for step in range(repeats + 2):
        synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if step >= 2:
            source_times.append(time.perf_counter() - tick)
        if step == 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    source_memory = memory_stats(device)
    model.eval()
    synchronize(device)
    tick = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(images)
    synchronize(device)
    eval_seconds = (time.perf_counter() - tick) / repeats
    del model, optimizer, images, labels, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    adapted = build_model()
    load_source_weights(adapted, original)
    adapted.to(device)
    updater = AdapterUpdater(adapted, "hierarchical_qr", FilterConfig())
    images = torch.randn(batch, 3, 32, 32, device=device)
    labels = torch.arange(batch, device=device) % 10
    times, diagnostics = [], []
    for step in range(repeats + 2):
        synchronize(device)
        tick = time.perf_counter()
        values = updater.process(images, labels)
        synchronize(device)
        if step >= 2:
            times.append(time.perf_counter() - tick)
            diagnostics.append(values)
        if step == 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    result = {
        "status": "completed", "kind": "generated_image_device_preflight_only",
        "environment": environment(device), "source_batch_size": source_batch,
        "adaptation_batch_size": batch, "warmup_steps": 2, "measured_steps": repeats,
        "source_step_seconds": sum(source_times) / len(source_times),
        "source_evaluation_batch_seconds": eval_seconds,
        "source_memory": source_memory,
        "adaptation_step_seconds": sum(times) / len(times),
        "adaptation_memory": memory_stats(device), "adaptation_diagnostics": diagnostics,
        "real_data_efficacy_evidence": False,
    }
    write_json(output, result)
    print(f"Device preflight completed: {output}", flush=True)
    return result


def main() -> None:
    """Exercise both actual model paths before starting a long unattended job."""
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    benchmark(args.output.resolve(), select_device(args.device), args.source_batch_size, args.batch_size, args.repeats)


if __name__ == "__main__":
    main()
