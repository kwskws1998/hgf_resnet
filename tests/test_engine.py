"""Integration checks for causal metrics, isolated updates, and run provenance."""

from argparse import Namespace
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hgfvision import cli
from hgfvision.data import CleanDataset, FakeCIFAR10, Phase, StreamDataset, make_splits
from hgfvision.engine import (
    METHODS,
    AdapterUpdater,
    classification_metrics,
    json_hash,
    run_stream,
    seed_everything,
)
from hgfvision.filters import FilterConfig
from hgfvision.models import build_model, load_source_weights


@pytest.fixture(autouse=True)
def isolated_random_state():
    """Make CPU integration checks repeatable without changing other tests."""
    previous_threads = torch.get_num_threads()
    previous_torch = torch.get_rng_state()
    previous_numpy = np.random.get_state()
    previous_python = random.getstate()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    seed_everything(23)
    yield
    torch.set_num_threads(previous_threads)
    torch.set_rng_state(previous_torch)
    np.random.set_state(previous_numpy)
    random.setstate(previous_python)
    torch.use_deterministic_algorithms(previous_determinism)


@pytest.fixture
def source_state():
    """Provide an adapter-free checkpoint without dataset downloads."""
    return build_model(adapters=False).state_dict()


def make_updater(source_state, method):
    """Recreate the CLI's identical source and adapter initialization per arm."""
    seed_everything(7)
    model = build_model(rank=4)
    load_source_weights(model, source_state)
    return AdapterUpdater(model, method, FilterConfig(), learning_rate=0.01)


def make_loaders():
    """Return two disjoint fake-data steps and a separate clean reference set."""
    data = FakeCIFAR10(8, seed=3)
    stream = StreamDataset(
        data,
        indices=[0, 1, 2, 3],
        phases=[Phase("stable", 1), Phase("shift", 1, brightness=0.7)],
        batch_size=2,
        seed=4,
    )
    clean = CleanDataset(data, indices=[4, 5, 6, 7])
    return DataLoader(stream, batch_size=2, shuffle=False), DataLoader(clean, batch_size=2)


@pytest.mark.parametrize("method", ["frozen", *METHODS])
def test_two_steps_score_before_update_and_preserve_backbone(source_state, method):
    updater = make_updater(source_state, method)
    loader, _ = make_loaders()
    initial = {key: value.detach().clone() for key, value in updater.model.state_dict().items()}
    for batch in loader:
        updater.model.eval()
        with torch.no_grad():
            expected = classification_metrics(updater.model(batch["x"]), batch["y"])
        actual = updater.process(batch["x"], batch["y"])
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=0, abs=0)
    final = updater.model.state_dict()
    for key, value in initial.items():
        if method == "frozen" or not key.startswith("adapters."):
            torch.testing.assert_close(final[key], value, rtol=0, atol=0)
    if method != "frozen":
        assert any(
            not torch.equal(final[key], value)
            for key, value in initial.items()
            if key.startswith("adapters.")
        )


def test_all_arms_share_exact_adapter_initialization_and_first_predictions(source_state):
    loader, _ = make_loaders()
    batch = next(iter(loader))
    initial_adapters = None
    initial_logits = None
    for method in ["frozen", *METHODS]:
        updater = make_updater(source_state, method)
        adapters = torch.cat([parameter.detach().reshape(-1) for parameter in updater.params])
        updater.model.eval()
        with torch.no_grad():
            logits = updater.model(batch["x"])
        if initial_adapters is None:
            initial_adapters = adapters.clone()
            initial_logits = logits.clone()
        else:
            torch.testing.assert_close(adapters, initial_adapters, rtol=0, atol=0)
            torch.testing.assert_close(logits, initial_logits, rtol=0, atol=0)


def test_cli_reloads_shared_source_initialization_and_stream_order(source_state, tmp_path, monkeypatch):
    raw_train, raw_test = FakeCIFAR10(40, seed=11), FakeCIFAR10(8, seed=12)
    splits = make_splits(len(raw_train), seed=0)
    source_path = tmp_path / "source.pt"
    metadata = {
        "dataset": "fake_smoke_only", "architecture": "cifar_resnet18", "num_classes": 10,
        "dataset_length": len(raw_train), "split_seed": 0,
        "split_hashes": {key: json_hash(value) for key, value in splits.items()},
    }
    torch.save({"model": source_state, "metadata": metadata}, source_path)
    config = cli.read_config(None)
    config["batch_size"] = 2
    config["phases"] = [{"name": "stable", "steps": 1}, {"name": "shift", "steps": 1, "brightness": 0.7}]
    args = Namespace(
        device="cpu", checkpoint=source_path, config=None, data_root="unused", download=False,
        partition="dev", order_seed=3, noise_seed=4, workers=0, eval_batch_size=2,
        output=tmp_path / "comparison", method="all", seed=7,
    )
    observed = []

    def inspect_initial_state(updater, loader, output_dir, clean_loader, metadata):
        """Capture actual CLI initialization before handing control to any arm."""
        batch = next(iter(loader))
        updater.model.eval()
        with torch.no_grad():
            logits = updater.model(batch["x"])
        parameters = torch.cat([parameter.detach().reshape(-1) for parameter in updater.params])
        for key, value in source_state.items():
            torch.testing.assert_close(updater.model.state_dict()[key], value, rtol=0, atol=0)
        observed.append((updater.method, parameters.clone(), logits.clone(), list(loader.dataset.indices)))
        assert metadata["trainable_parameters"] == 8192
        return {"method": updater.method, "status": "completed"}

    monkeypatch.setattr(cli, "run_stream", inspect_initial_state)
    result = cli.adapt(args, raw_train=raw_train, raw_test=raw_test, fake=True, config_override=config)
    assert [entry[0] for entry in observed] == ["frozen", *METHODS]
    for _, parameters, logits, indices in observed[1:]:
        torch.testing.assert_close(parameters, observed[0][1], rtol=0, atol=0)
        torch.testing.assert_close(logits, observed[0][2], rtol=0, atol=0)
        assert indices == observed[0][3]
    assert json.loads((args.output / "comparison.json").read_text()) == result


def test_filter_linearizes_entire_batch_before_one_weight_update(source_state, monkeypatch):
    updater = make_updater(source_state, "fixed_q")
    loader, _ = make_loaders()
    batch = next(iter(loader))
    initial = [parameter.detach().clone() for parameter in updater.params]
    grad_calls = []
    filter_calls = []
    actual_grad = torch.autograd.grad
    actual_step = updater.filter.step

    def checked_grad(*args, **kwargs):
        """Assert that every Jacobian row sees the same parameter values."""
        for parameter, expected in zip(updater.params, initial):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        result = actual_grad(*args, **kwargs)
        grad_calls.append(result)
        return result

    def checked_step(residual, jacobian):
        """Check observation dimensions and the absence of intermediate updates."""
        for parameter, expected in zip(updater.params, initial):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        assert len(grad_calls) == 20
        assert residual.shape == (20,)
        assert jacobian.shape == (20, 8192)
        expected = torch.stack(
            [torch.cat([gradient.reshape(-1) for gradient in row]) for row in grad_calls]
        )
        torch.testing.assert_close(jacobian, expected, rtol=0, atol=0)
        filter_calls.append((residual.shape, jacobian.shape))
        return actual_step(residual, jacobian)

    monkeypatch.setattr(torch.autograd, "grad", checked_grad)
    monkeypatch.setattr(updater.filter, "step", checked_step)
    updater.process(batch["x"], batch["y"])
    assert len(filter_calls) == 1
    assert any(not torch.equal(parameter, expected) for parameter, expected in zip(updater.params, initial))


def test_stream_writes_prequential_metrics_and_loadable_final_state(source_state, tmp_path):
    updater = make_updater(source_state, "fixed_q")
    loader, clean_loader = make_loaders()
    first = next(iter(loader))
    updater.model.eval()
    with torch.no_grad():
        expected_first = classification_metrics(updater.model(first["x"]), first["y"])
    output = tmp_path / "completed"
    metadata = {"dataset": "fake_smoke_only", "method": "fixed_q", "seed": 7}
    summary = run_stream(updater, loader, output, clean_loader, metadata)
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert summary["status"] == "completed"
    assert summary["steps"] == 2
    assert summary["samples"] == 4
    assert summary["peak_allocated_bytes"] is None
    assert [row["indices"] for row in rows] == [[0, 1], [2, 3]]
    assert [row["phase"] for row in rows] == ["stable", "shift"]
    for key, value in expected_first.items():
        assert rows[0][key] == pytest.approx(value, rel=0, abs=0)
        assert summary["prequential"][key] == pytest.approx(
            sum(row[key] * row["samples"] for row in rows) / summary["samples"]
        )
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    assert json.loads((output / "summary.json").read_text()) == summary
    checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metadata"] == metadata
    for key, value in updater.model.state_dict().items():
        torch.testing.assert_close(checkpoint["model"][key], value, rtol=0, atol=0)
    assert checkpoint["updater"]["p"].shape == (8192,)
    assert checkpoint["updater"]["controller"]["method"] == "fixed_q"
    assert not (output / "failure.json").exists()


def test_stream_failure_records_completed_steps_and_never_claims_completion(source_state, tmp_path, monkeypatch):
    updater = make_updater(source_state, "adamw")
    loader, clean_loader = make_loaders()
    actual_process = updater.process
    calls = []

    def fail_second_step(images, labels):
        """Create a deterministic interruption after one successful update."""
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected update failure")
        return actual_process(images, labels)

    monkeypatch.setattr(updater, "process", fail_second_step)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected update failure"):
        run_stream(updater, loader, output, clean_loader, {"dataset": "fake_smoke_only"})
    failure = json.loads((output / "failure.json").read_text())
    assert failure["completed_steps"] == 1
    assert failure["type"] == "RuntimeError"
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert len((output / "steps.jsonl").read_text().splitlines()) == 1
    assert not (output / "summary.json").exists()
    assert not (output / "final.pt").exists()


def test_initial_clean_evaluation_failure_is_recorded(source_state, tmp_path):
    updater = make_updater(source_state, "frozen")
    loader, _ = make_loaders()
    invalid_clean = [(torch.full((2, 3, 32, 32), float("nan")), torch.tensor([0, 1]))]
    output = tmp_path / "bad_initial_evaluation"
    with pytest.raises(FloatingPointError, match="Nonfinite prediction"):
        run_stream(updater, loader, output, invalid_clean, {"dataset": "fake_smoke_only"})
    failure = json.loads((output / "failure.json").read_text())
    assert failure["completed_steps"] == 0
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists()


def test_source_training_failure_is_recorded(tmp_path, monkeypatch):
    output = tmp_path / "bad_source_training"
    args = Namespace(
        device="cpu", seed=0, split_seed=0, output=output, data_root="unused", download=False,
        batch_size=2, eval_batch_size=2, workers=0, epochs=1, lr=0.01, weight_decay=0.0005,
    )

    def nonfinite_loss(*args, **kwargs):
        """Exercise failure reporting before an invalid source update occurs."""
        return torch.tensor(float("nan"))

    monkeypatch.setattr(cli.nn.functional, "cross_entropy", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="Nonfinite source loss"):
        cli.train_source(args, raw_train=FakeCIFAR10(20, seed=11), fake=True)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["type"] == "FloatingPointError"
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "source.pt").exists()
