# ResNet adapter uncertainty pilot

An executable research scaffold for supervised online adaptation on one GPU.
The proposed arm is an **HGF-inspired hierarchical Fisher filter**. It is not
an exact reproduction of generalized HGF and its benefit has not been established.
The mathematical contract and source excerpts are in [METHOD.md](METHOD.md).
Completed checks and their limits are recorded in [VALIDATION.md](VALIDATION.md).

For the one-command Vast.ai setup and finite unattended pilot, use
[VAST_QUICKSTART.md](VAST_QUICKSTART.md). Source training now supports atomic
per-epoch continuation; the runner verifies completed artifacts before skipping
them and retries an interrupted adaptation arm from its original source model.

## Implemented experiment

- CIFAR ResNet-18: 3x3 stride-one stem, no initial max pooling.
- Two identity-initialized 1x1 adapters after the final residual blocks,
  each 512 -> 4 -> 512: **8,192 adaptable weights**.
- Frozen backbone, classifier, and BatchNorm statistics during adaptation.
- Five adaptation arms: AdamW, zero-Q Kalman, fixed-Q Kalman, adaptive Q/R,
  and hierarchical Q/R. A frozen source reference is evaluated as well.
- Predictions are scored before labels are used for an update.
- A minibatch is one filtering time step. Its Jacobian is computed at fixed
  parameters, and all observations enter one simultaneous update.
- Per-step prediction metrics, Q/R, covariance diagnostics, elapsed time,
  source/split/code hashes, and measured CUDA allocator peaks are saved.

## Install and check

Use Python 3.10 or newer with a mutually compatible PyTorch/torchvision pair.
Install the appropriate CUDA build of PyTorch for the target machine first.
From this project directory:

```console
python -m pip install -e ".[test]"
python -m pytest -q
python -m hgfvision --threads 2 smoke --device cpu --output runs/smoke
python -m hgfvision --threads 2 synthetic --device cpu --output runs/synthetic
```

The smoke command uses generated patterns, not CIFAR-10. It performs one short
source-training epoch and executes every adaptation arm. These results verify
code paths only. The synthetic command separately varies true latent-parameter
drift and observation noise, logs inferred Q/R, and includes an explicitly
informed full-covariance Kalman reference. A successful command is not a claim
that the proposed method wins or identifies both noise sources.

New runs require a new output directory. Explicit source-training and pipeline
resume modes reuse verified checkpoints; runs are never silently overwritten.
No network download occurs unless `--download` is supplied.

## Run the initial CIFAR-10 pilot

```console
python -m hgfvision train-source --device cuda --data-root data --download --epochs 100 --output runs/source_seed0
python -m hgfvision adapt --device cuda --data-root data --checkpoint runs/source_seed0/source.pt --config configs/pilot.json --partition dev --method all --output runs/pilot_seed0
```

The source model is trained once and reused by every adaptation arm. Source
training uses SGD, cosine decay, random crop, and horizontal flip. The selected
checkpoint maximizes source-validation accuracy; no official test images enter
source training or checkpoint selection.

The defaults are exploratory starting values, not tuned baselines. The pilot
uses 200 updates at batch 8, covering 1,600 distinct development images. All
methods receive exactly the same initialization, images, labels, ordering, and
corruption. AdamW defaults to half squared probability error, matching the
Gaussian observation surrogate up to scale. Set `adamw.loss` to `ce` in a copied
configuration to add a practical cross-entropy baseline before publication.

Use separate development runs with equal trial budgets to tune each method.
Freeze those configurations before `--partition stream`, which accesses the
disjoint confirmatory stream and uses the official test set for clean retention.
Do not choose hyperparameters from the confirmatory results.

To obtain multiple replications, vary `--seed` (adapter initialization),
`--order-seed` (sample order), and/or `--noise-seed` (corruption realization)
explicitly. These do not vary the shared source checkpoint. The same settings
must be used across competing methods. Multiple order seeds change image order,
not the temporal order of phases; change the config to test a reversed schedule.

## Data boundaries

The 50,000 official training images are partitioned once using `--split-seed`:

| Split | Images | Use |
|---|---:|---|
| source | 30,000 | Initial CNN training |
| source_val | 5,000 | Source checkpoint selection and development retention |
| dev | 5,000 | Method and hyperparameter development |
| stream | 10,000 | Confirmatory adaptation after choices are frozen |

The official 10,000 test images are separate and are never adapted on. In the
confirmatory online stream, a label may be used only after that image has been
scored. This is **supervised online adaptation**, not unlabeled test-time
adaptation. Phase labels are reporting metadata and are never updater inputs.

Image brightness/contrast shifts and Gaussian pixel corruption are performance
stressors, not ground-truth observations of the optimizer's Q/R. In particular,
R is output residual variance under a Gaussian surrogate on class probabilities.
The synthetic experiment is the place to test known process/observation noise.

## 24 GB design and limits

The default parameter covariance is a vector of 8,192 FP32 values (32 KiB).
For batch 8 and 10 classes, the explicit Jacobian is 80 x 8,192 FP32 values
(2.5 MiB), and the innovation covariance is only 80 x 80. These figures exclude
activations, gradients, CUDA workspaces, source weights, and temporary buffers.

The implementation computes Jacobian rows sequentially, then retains the small
full Jacobian for one batch. This controls derivative expansion memory but can
be slow. It is not a claim of arbitrary output-dimension scalability. The
`max_observations` guard prevents accidental large batch-by-class systems.
Filter state and initial image experiments use FP32; AMP is intentionally not
enabled until a numerical parity check exists. Numerically fragile conditioning
can use a reported temporary FP64 rescue.

On CUDA, every adaptation run records `peak_allocated_bytes`,
`peak_reserved_bytes`, and synchronized `mean_update_seconds`. CPU results
record null GPU peaks. **A 24 GB peak has not been verified unless the included
validation record explicitly reports a CUDA run.** No GPU-specific throughput
or completion-time claim follows from CPU smoke tests.

## Files and output

- `src/hgfvision/models.py`: source CNN, adapters, and freezing contracts.
- `src/hgfvision/data.py`: disjoint splits and deterministic transforms.
- `src/hgfvision/filters.py`: diagonal EKF and uncertainty controllers.
- `src/hgfvision/engine.py`: common update/evaluation and logging.
- `src/hgfvision/cli.py`: training, pilot, and smoke commands.
- `src/hgfvision/synthetic.py`: controlled mechanism probe.
- `tests/`: numerical, model, data, and integration checks.

Each adaptation directory contains `manifest.json`, `steps.jsonl`,
`summary.json`, and `final.pt`; failed runs record `failure.json`. Source
checkpoints include training provenance and exact split hashes. The parent
directory contains the actual stream indices and the resolved configuration.
Source training supports `train-source --resume` with the same output directory
and original training arguments. `last.pt` includes optimizer, scheduler, and
random states at the last completed epoch. Adaptation checkpoints remain for
auditing; automatic mid-stream adaptation resume is not implemented. The
unattended runner retries an interrupted arm in a fresh attempt directory.

## Decision before expanding the project

First verify a useful source classifier and a meaningful adaptation stressor.
Then compare hierarchical Q/R against the same lower filter with fixed
hyperstate diffusion, using held-out development streams and equal tuning
budgets. Inspect uncertainty bounds, numerical rescues, adaptation speed,
prequential accuracy/Brier score, and clean retention. Do not expand to larger
models on the basis of an advantage over frozen weights alone.

CIFAR-100, ResNet-50, automated hyperparameter search, and publication-scale
ablations are not implemented in this first version.
