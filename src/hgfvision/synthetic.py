"""Controlled latent-parameter drift versus independent observation-noise probe."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from .engine import environment, seed_everything, select_device, write_json
from .filters import DiagonalEKF, FilterConfig


def run_synthetic(args) -> dict:
    """Measure all filters on identical generated data; do not assert superiority.

    At each step eight independent observations share one four-dimensional
    latent parameter vector. This makes Q induce correlated innovations while
    R induces independent observation noise. Ground truth is never passed to an
    adaptive controller. The explicitly labeled oracle alone receives Q/R.
    """
    device = select_device(args.device)
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(args.seed)
    regimes = [
        ("stable_low_noise", 1e-4, 0.02),
        ("stable_high_noise", 1e-4, 0.3),
        ("drift_low_noise", 0.01, 0.02),
        ("drift_high_noise", 0.01, 0.3),
    ]
    latent = torch.zeros(4)
    samples = []
    for name, q, r in regimes:
        for _ in range(args.steps_per_phase):
            latent = latent + q ** 0.5 * torch.randn(4, generator=generator)
            design = torch.randn(8, 4, generator=generator) / 2
            noiseless = design @ latent
            observed = noiseless + r ** 0.5 * torch.randn(8, generator=generator)
            samples.append((name, q, r, latent.clone().to(device), design.to(device), noiseless.to(device), observed.to(device)))
    config = FilterConfig(initial_q=1e-4, initial_r=0.02)
    results = {}
    for method in ("zero_q", "fixed_q", "adaptive_qr", "hierarchical_qr", "oracle_full_kf"):
        estimate = torch.zeros(4, device=device)
        covariance = torch.eye(4, device=device) * config.initial_p
        filter_state = None if method == "oracle_full_kf" else DiagonalEKF([2, 2], method, config, device)
        rows = []
        for step, (name, true_q, true_r, truth, design, noiseless, observed) in enumerate(samples):
            prediction = design @ estimate
            residual = observed - prediction
            if filter_state is None:
                prior = covariance + true_q * torch.eye(4, device=device)
                innovation = design @ prior @ design.mT + true_r * torch.eye(8, device=device)
                gain = torch.linalg.solve(innovation, design @ prior).mT
                estimate = estimate + gain @ residual
                residual_operator = torch.eye(4, device=device) - gain @ design
                covariance = residual_operator @ prior @ residual_operator.mT + true_r * gain @ gain.mT
                diagnostics = {"q_used": [true_q, true_q], "r_used": true_r}
            else:
                displacement, diagnostics = filter_state.step(residual, design)
                estimate = estimate + displacement
            rows.append({
                "step": step, "phase": name, "true_q": true_q, "true_r": true_r,
                "prequential_noiseless_mse": float((prediction - noiseless).square().mean()),
                "posterior_parameter_mse": float((estimate - truth).square().mean()),
                **diagnostics,
            })
        with (output / f"{method}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        result = {}
        for name, _, _ in regimes:
            phase = [row for row in rows if row["phase"] == name]
            tail = phase[len(phase) // 2:]
            result[name] = {
                "prequential_noiseless_mse": sum(row["prequential_noiseless_mse"] for row in phase) / len(phase),
                "posterior_parameter_mse": sum(row["posterior_parameter_mse"] for row in phase) / len(phase),
                "tail_mean_q": sum(sum(row["q_used"]) / 2 for row in tail) / len(tail),
                "tail_mean_r": sum(row["r_used"] for row in tail) / len(tail),
            }
        results[method] = result
    summary = {
        "kind": "synthetic_mechanism_probe_not_vision_efficacy", "seed": args.seed,
        "steps_per_phase": args.steps_per_phase, "filter_config": asdict(config),
        "environment": environment(device), "results": results,
        "oracle_caveat": "Oracle has true Q/R and a full parameter covariance; it is an informed reference, not a fair competitor.",
        "interpretation": "Defaults are untuned. Inspect separation and tracking across multiple seeds before any efficacy claim.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "methods_completed": list(results)}), flush=True)
    return summary
