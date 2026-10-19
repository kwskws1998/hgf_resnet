# Validation performed

This record distinguishes numerical correctness, executable image code, and
empirical method evidence. The original scaffold results below are historical;
their original code hashes are retained in
[validation/results.json](validation/results.json). The unattended-run revision
is recorded separately in
[validation/unattended_20261019/results.json](validation/unattended_20261019/results.json).

## Numerical and integration checks

Original scaffold: `python -m pytest -q`: **82 passed, 0 failed**, 8.52 seconds on
the local CPU. This is the historical test count, not the current suite size.

The suite includes independent finite-difference derivatives, numerical
Gaussian integration of Fisher information, dense/scalar Kalman reference
calculations, and regression cases where FP32 covariance subtraction loses
accuracy. It also checks shared initial weights/order, frozen backbone and
BatchNorm, pre-update metrics, split isolation, and recorded failed runs.

## Real architecture, generated images

The offline smoke command trained the actual CIFAR ResNet-18 for one short epoch
on generated patterns, then ran **8 updates per arm** for all five adaptation
methods plus the frozen reference. All six paths completed. No CIFAR-10 data
was downloaded or trained on. Pattern-image accuracy is not research evidence.

## Controlled synthetic probe

Three seeds each ran 400 steps: 100 steps in each of four process/noise regimes.
Every method received the same samples within a seed. The table shows mean
prequential squared error against the noiseless output; lower is better.

| Seed | Adaptive Q/R | Hierarchical Q/R |
|---|---:|---:|
| 0 | 0.037357 | 0.032673 |
| 1 | 0.020189 | 0.020363 |
| 2 | 0.030329 | 0.026515 |

The hierarchical arm improved this metric in two seeds and slightly worsened it
in one. All configurations were untuned defaults. This is preliminary evidence
on a tiny linear state model, not proof of an advantage on vision or a general
identification result. In particular, low prediction error does not by itself
prove that Q and R separately recovered their true values.

Hyperstate/parent/parameter bound-hit totals were 1/1/0 for adaptive Q/R and
1/4/2 for hierarchical Q/R across seeds 0/1/2. No precision rescues occurred in
these probe runs; separate numerical regression tests exercise the rescue path.
Raw per-step trajectories and phase-level summaries are retained under
`validation/synthetic_seed0`, `synthetic_seed1`, and `synthetic_seed2`.

## Not yet established

- CUDA compatibility and peak VRAM on the target 24 GB GPU.
- CIFAR-10 source accuracy and adaptation effectiveness.
- Fairly tuned baseline rankings or statistical significance.
- Novelty, robustness across architectures, or publication readiness.

The next empirical step is source training and the development pilot on the
target GPU, recording VRAM and update latency before scaling.

## Unattended-run revision, 2026-10-19

`python -m pytest -q`: **159 passed, 0 failed**, 22.11 seconds on the local CPU.

The current revision adds environment preparation, a finite detached pipeline,
atomic source-training continuation, and guarded reuse of completed stages.
Its machine-readable validation record includes the current test count and
source/test hashes. CPU generated-image smoke logs and a device-preflight
report are retained beside that record. All six smoke arms completed eight
steps each; the benchmark used generated images and CPU batch sizes 4/2.
These timings are not RTX 4090 estimates.

Interrupted two-epoch CPU source training was resumed and compared exactly
with uninterrupted training, including model weights, optimizer, scheduler,
random state, and metric history (excluding elapsed wall time). Separate
subprocess tests exercised inherited locks after worker death. Mocked runner
tests checked corrupted artifacts, immutable settings, failed-arm isolation,
and source/adaptation retry boundaries. Setup tests and a dry run checked
installation commands and launch ordering without installing GPU packages.

No Vast.ai instance was rented or accessed during this validation. CUDA
installation, real CIFAR-10 training, GPU memory fit, and GPU runtime remain
unmeasured. The setup command will perform those device checks on the rented
machine before starting the real-data pilot.
