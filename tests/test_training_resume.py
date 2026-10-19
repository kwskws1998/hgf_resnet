"""Verify exact epoch-boundary continuation and atomic checkpoint recovery."""

from argparse import Namespace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from hgfvision import cli, training
from hgfvision.data import FakeCIFAR10, SourceDataset


@pytest.fixture(autouse=True)
def isolated_runtime():
    """Use one CPU thread and restore caller RNGs after training checks."""
    previous_threads = torch.get_num_threads()
    previous_torch = torch.get_rng_state()
    previous_numpy = np.random.get_state()
    previous_python = random.getstate()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)
    torch.set_rng_state(previous_torch)
    np.random.set_state(previous_numpy)
    random.setstate(previous_python)
    torch.use_deterministic_algorithms(previous_determinism)


def source_args(output: Path, **changes) -> Namespace:
    """Build a small, fixed two-epoch source-training contract."""
    values = dict(
        device="cpu", seed=5, split_seed=3, output=output, data_root="unused", download=False,
        batch_size=4, eval_batch_size=2, workers=0, epochs=2, lr=0.01, weight_decay=0.0005,
        resume=False,
    )
    values.update(changes)
    return Namespace(**values)


def assert_same_tree(actual, expected):
    """Compare optimizer, scheduler, RNG, and tensor trees without tolerances."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_same_tree(first, second)
    else:
        assert actual == expected


def test_interrupted_resume_matches_uninterrupted_two_epoch_training(tmp_path, monkeypatch):
    data = FakeCIFAR10(20, seed=11)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    cli.train_source(source_args(full), raw_train=data, fake=True)
    original_save = training.atomic_save

    def interrupt_after_first_commit(payload, path):
        """Stop after atomic commit but before in-memory progress is advanced."""
        original_save(payload, path)
        if Path(path).name == "last.pt" and payload["next_epoch"] == 1:
            raise KeyboardInterrupt("simulated preemption")

    with monkeypatch.context() as context:
        context.setattr(training, "atomic_save", interrupt_after_first_commit)
        with pytest.raises(KeyboardInterrupt, match="simulated preemption"):
            cli.train_source(source_args(resumed), raw_train=data, fake=True)
    interrupted = json.loads((resumed / "manifest.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed_epochs"] == 1
    first = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert first["next_epoch"] == 1
    assert first["scheduler"]["T_max"] == 2
    cli.train_source(source_args(resumed, resume=True), raw_train=data, fake=True)
    expected = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "rng", "best_accuracy", "next_epoch"):
        assert_same_tree(actual[key], expected[key])
    assert_same_tree(actual["best"]["model"], expected["best"]["model"])
    assert actual["best"]["metadata"]["selected_epoch"] == expected["best"]["metadata"]["selected_epoch"]
    rows = [json.loads(line) for line in (resumed / "epochs.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0, 1]
    for observed, baseline in zip(rows, expected["history"]):
        assert {key: value for key, value in observed.items() if key != "seconds"} == {
            key: value for key, value in baseline.items() if key != "seconds"
        }
    best = torch.load(resumed / "source.pt", map_location="cpu", weights_only=True)
    assert_same_tree(best["model"], actual["best"]["model"])
    assert json.loads((resumed / "manifest.json").read_text())["status"] == "completed"
    assert not (resumed / "failure.json").exists()


def test_failed_epoch_commit_repairs_best_and_truncates_uncommitted_history(tmp_path, monkeypatch):
    output = tmp_path / "transaction"
    data = FakeCIFAR10(20, seed=11)
    actual_save = training.atomic_save
    validations = iter([0.1, 0.2])

    def increasing_validation(*args, **kwargs):
        """Guarantee that the uncommitted epoch also writes a new best model."""
        return {"accuracy": next(validations), "cross_entropy": 1.0, "brier": 0.5}

    def fail_last_commit(payload, path):
        """Leave the last durable state behind the best checkpoint and log."""
        if Path(path).name == "last.pt" and payload["next_epoch"] == 2:
            raise OSError("simulated checkpoint failure")
        actual_save(payload, path)

    with monkeypatch.context() as context:
        context.setattr(training, "evaluate", increasing_validation)
        context.setattr(training, "atomic_save", fail_last_commit)
        with pytest.raises(OSError, match="simulated checkpoint failure"):
            training.train_source(source_args(output), raw_train=data, fake=True)
    last_bytes = (output / "last.pt").read_bytes()
    best = torch.load(output / "source.pt", map_location="cpu", weights_only=True)
    assert best["metadata"]["selected_epoch"] == 1
    assert len((output / "epochs.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "manifest.json").read_text())["completed_epochs"] == 1

    def stop_before_replayed_epoch(self, epoch):
        """Inspect repaired committed artifacts before any new update occurs."""
        raise KeyboardInterrupt("inspect repaired boundary")

    with monkeypatch.context() as context:
        context.setattr(SourceDataset, "set_epoch", stop_before_replayed_epoch)
        with pytest.raises(KeyboardInterrupt, match="inspect repaired boundary"):
            training.train_source(source_args(output, resume=True), raw_train=data, fake=True)
    repaired = torch.load(output / "source.pt", map_location="cpu", weights_only=True)
    committed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert repaired["metadata"]["selected_epoch"] == 0
    assert_same_tree(repaired["model"], committed["best"]["model"])
    assert (output / "last.pt").read_bytes() == last_bytes
    rows = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0]


def test_atomic_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    training.atomic_save({"value": torch.tensor([1.0])}, checkpoint)
    previous = checkpoint.read_bytes()

    def incomplete_save(payload, handle):
        """Simulate an I/O failure after writing only part of a temporary file."""
        handle.write(b"partial checkpoint")
        raise OSError("write failed")

    monkeypatch.setattr(training.torch, "save", incomplete_save)
    with pytest.raises(OSError, match="write failed"):
        training.atomic_save({"value": torch.tensor([2.0])}, checkpoint)
    assert checkpoint.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_resume_rejects_changed_schedule_data_and_training_arguments(tmp_path):
    output = tmp_path / "strict"
    data = FakeCIFAR10(20, seed=11)
    training.train_source(source_args(output), raw_train=data, fake=True)
    manifest = (output / "manifest.json").read_bytes()
    changes = [
        {"epochs": 3}, {"seed": 6}, {"split_seed": 4}, {"batch_size": 2},
        {"eval_batch_size": 4}, {"workers": 1}, {"lr": 0.02}, {"weight_decay": 0.01},
    ]
    for change in changes:
        with pytest.raises(ValueError, match="Resume training arguments/runtime differ"):
            training.train_source(source_args(output, resume=True, **change), raw_train=data, fake=True)
        assert (output / "manifest.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="dataset_sha256"):
        training.train_source(source_args(output, resume=True), raw_train=FakeCIFAR10(20, seed=12), fake=True)
    assert (output / "manifest.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="dataset"):
        training.train_source(source_args(output, resume=True), raw_train=data, fake=False)
    with pytest.raises(FileExistsError, match="use --resume"):
        training.train_source(source_args(output), raw_train=data, fake=True)


def test_completed_resume_is_idempotent_and_repairs_missing_best(tmp_path):
    output = tmp_path / "complete"
    data = FakeCIFAR10(20, seed=11)
    training.train_source(source_args(output, epochs=1), raw_train=data, fake=True)
    original_last = (output / "last.pt").read_bytes()
    original_history = (output / "epochs.jsonl").read_text()
    (output / "source.pt").unlink()
    training.train_source(source_args(output, epochs=1, resume=True), raw_train=data, fake=True)
    assert (output / "source.pt").is_file()
    assert (output / "last.pt").read_bytes() == original_last
    assert (output / "epochs.jsonl").read_text() == original_history
    assert json.loads((output / "manifest.json").read_text())["completed_epochs"] == 1


def test_missing_resume_checkpoint_does_not_create_output(tmp_path):
    output = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="No completed-epoch checkpoint"):
        training.train_source(source_args(output, resume=True), raw_train=FakeCIFAR10(20), fake=True)
    assert not output.exists()


def test_cli_exposes_boolean_resume_with_fixed_total_epoch_budget(tmp_path, monkeypatch):
    captured = []

    def capture(args):
        """Inspect CLI parsing without creating a training run."""
        captured.append(args)

    monkeypatch.setattr(cli, "train_source", capture)
    monkeypatch.setattr("sys.argv", ["hgfvision", "train-source", "--output", str(tmp_path), "--epochs", "2", "--resume"])
    cli.main()
    assert captured[0].resume is True
    assert captured[0].epochs == 2
