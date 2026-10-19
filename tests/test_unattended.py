"""Exercise unattended lifecycle and provenance without GPU or cloud access."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_experiments.py"
SPEC = importlib.util.spec_from_file_location("unattended_runner_under_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def options(**values):
    """Leave unspecified settings absent so resume can retain recorded values."""
    return argparse.Namespace(**values)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Create an isolated implementation fingerprint and short experiment plan."""
    root = tmp_path / "project with spaces"
    (root / "src" / "hgfvision").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "runner-fixture"\n')
    (root / "src" / "hgfvision" / "implementation.py").write_text('VALUE = 1\n')
    runner.atomic_json(root / "configs" / "pilot.json", {
        "batch_size": 2, "phases": [{"name": "stable", "steps": 2}],
    })
    monkeypatch.setattr(runner, "PROJECT", root)
    monkeypatch.setattr(runner, "probe_runtime", lambda python, lock_fd=None: runner.runtime_fingerprint())
    return root


@pytest.fixture
def state(project):
    """Use nondefault settings to catch accidental replacement on resume."""
    plan = runner.build_plan(project, options(
        source_epochs=2, source_batch_size=4, workers=0, threads=1,
        seed=3, order_seed=5, noise_seed=7,
    ))
    output = project / "runs" / "pilot with spaces"
    output.mkdir(parents=True)
    runner.atomic_json(output / "config.json", plan["config"])
    result = {
        "schema": 1, "output": str(output), "status": "created", "plan": plan,
        "stages": {name: {"status": "pending", "attempts": []} for name in runner.STAGES},
    }
    runner.save_state(result)
    return result


def write_source(directory, plan):
    """Produce self-consistent source artifacts with a full planned epoch log."""
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source.pt").write_bytes(b"best source checkpoint fixture")
    (directory / "last.pt").write_bytes(b"latest optimizer and RNG state fixture")
    runner.atomic_json(directory / "splits.json", {"source": [0, 1], "source_val": [2]})
    record = {
        "status": "completed", "dataset": "cifar10", "epochs_requested": plan["source_epochs"],
        "training_seed": plan["seed"], "split_seed": plan["source_split_seed"],
        "source_training": {"batch_size": plan["source_batch_size"]},
        "source_environment": {"device": plan["device"]}, "environment": {"device": plan["device"]},
        "checkpoint_sha256": runner.file_sha256(directory / "source.pt"),
    }
    runner.atomic_json(directory / "manifest.json", record)
    (directory / "epochs.jsonl").write_text("".join(
        json.dumps({"epoch": epoch, "validation": {"accuracy": 0.5}}) + "\n"
        for epoch in range(plan["source_epochs"])
    ))


def complete_source(state):
    """Register a reusable source stage through the production validator."""
    directory = Path(state["output"]) / "source_attempt01"
    write_source(directory, state["plan"])
    state["stages"]["source"].update({
        "status": "completed", "directory": str(directory),
        "artifacts": runner.validate_artifacts("source", directory, state["plan"]),
    })
    runner.save_state(state)
    return runner.file_sha256(directory / "source.pt")


def write_adaptation(directory, name, plan, source_hash):
    """Write matching manifests, streamed observations, and final artifact bytes."""
    folder = directory / name
    folder.mkdir(parents=True, exist_ok=False)
    steps = sum(phase["steps"] for phase in plan["config"]["phases"])
    batch_size = plan["config"]["batch_size"]
    indices = list(range(steps * batch_size))
    runner.atomic_json(directory / "config.json", plan["config"])
    runner.atomic_json(directory / "splits.json", {"dev": indices})
    runner.atomic_json(directory / "stream_indices.json", indices)
    manifest = {
        "status": "completed", "dataset": "cifar10", "method": name, "partition": "dev",
        "config": plan["config"], "checkpoint_sha256": source_hash,
        "seed": plan["seed"], "order_seed": plan["order_seed"], "noise_seed": plan["noise_seed"],
        "stream_indices_sha256": hashlib.sha256(json.dumps(indices, sort_keys=True).encode()).hexdigest(),
        "environment": {"device": plan["device"]},
    }
    runner.atomic_json(folder / "manifest.json", manifest)
    runner.atomic_json(folder / "summary.json", {
        "status": "completed", "method": name, "steps": steps, "samples": len(indices),
        "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048,
    })
    (folder / "steps.jsonl").write_text("".join(json.dumps({
        "step": step, "phase": "stable", "samples": batch_size,
        "indices": indices[step * batch_size:(step + 1) * batch_size],
    }) + "\n" for step in range(steps)))
    (folder / "final.pt").write_bytes(b"completed adaptation fixture")


def write_preflight(directory, plan):
    """Provide positive CUDA allocation measurements for the planned batches."""
    runner.atomic_json(directory / "benchmark.json", {
        "status": "completed", "environment": {"device": plan["device"]},
        "source_batch_size": plan["source_batch_size"],
        "adaptation_batch_size": plan["config"]["batch_size"],
        "source_memory": {"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048},
        "adaptation_memory": {"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048},
        "source_step_seconds": 0.02, "adaptation_step_seconds": 0.1,
    })


def edit_json(path, change):
    """Apply one explicit artifact corruption for a validator regression test."""
    value = json.loads(path.read_text())
    change(value)
    runner.atomic_json(path, value)


def test_atomic_json_does_not_replace_good_state_with_nonfinite_data(tmp_path):
    path = tmp_path / "run.json"
    runner.atomic_json(path, {"status": "completed"})
    with pytest.raises(ValueError):
        runner.atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"status": "completed"}


def test_source_digest_tracks_code_but_ignores_results_and_tests(project):
    before = runner.source_digest(project)
    runner.atomic_json(project / "runs" / "result.json", {"result": 1})
    (project / "tests").mkdir()
    (project / "tests" / "test_example.py").write_text("assert True\n")
    assert runner.source_digest(project) == before
    (project / "src" / "hgfvision" / "implementation.py").write_text("VALUE = 2\n")
    assert runner.source_digest(project) != before


def test_default_plan_is_one_finite_ordered_pilot(project):
    plan = runner.build_plan(project, options())
    assert plan["source_epochs"] == 100
    assert plan["seed"] == 0
    assert plan["partition"] == "dev"
    assert plan["arms"] == ["frozen", "adamw", "adaptive_qr", "hierarchical_qr", "zero_q", "fixed_q"]
    assert "max_runtime" not in plan
    assert plan["runtime"] == runner.runtime_fingerprint()
    assert set(plan["runtime"]) == {"python_version", "torch", "torchvision", "numpy", "cuda_build", "cudnn"}


def test_resume_keeps_nondefault_plan_and_rejects_changed_settings(project, state):
    existing = state["plan"]
    assert runner.build_plan(project, options(), existing=existing) == existing
    for override in [{"source_epochs": 3}, {"seed": 99}, {"config": "configs/pilot.json"}]:
        with pytest.raises(ValueError, match="Resume"):
            runner.build_plan(project, options(**override), existing=existing)
    (project / "src" / "hgfvision" / "implementation.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Implementation changed"):
        runner.build_plan(project, options(), existing=existing)


@pytest.mark.parametrize("key", ["python_version", "torch", "torchvision", "numpy", "cuda_build", "cudnn"])
def test_resume_rejects_each_changed_runtime_version(project, state, monkeypatch, key):
    changed = {**state["plan"]["runtime"], key: "changed-version"}
    monkeypatch.setattr(runner, "runtime_fingerprint", lambda: changed)
    with pytest.raises(ValueError, match="Runtime changed in " + key):
        runner.build_plan(project, options(), existing=state["plan"])


def test_resume_rejects_legacy_plan_without_runtime_provenance(project, state):
    legacy = {key: value for key, value in state["plan"].items() if key != "runtime"}
    with pytest.raises(ValueError, match="no runtime fingerprint"):
        runner.build_plan(project, options(), existing=legacy)


def test_runtime_probe_reads_versions_in_a_real_fresh_interpreter():
    assert runner.probe_runtime(sys.executable) == runner.runtime_fingerprint()


def test_runtime_probe_passes_lock_and_reports_import_failure(monkeypatch):
    observed = []

    def failing(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 9, "", "package import failed")

    monkeypatch.setattr(runner.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="code 9.*package import failed"):
        runner.probe_runtime(sys.executable, 456)
    assert observed[0][0][-1] == "--runtime-fingerprint"
    assert observed[0][1]["pass_fds"] == (456,)
    assert observed[0][1]["timeout"] == 60


def test_stage_guard_uses_fresh_runtime_even_when_worker_imports_are_unchanged(project, state, monkeypatch):
    fresh = {**state["plan"]["runtime"], "numpy": "changed-on-disk"}
    monkeypatch.setattr(runner, "probe_runtime", lambda *args: fresh)
    monkeypatch.setattr(runner, "_run_command", lambda *args: pytest.fail("Changed runtime launched a stage"))
    assert runner.runtime_fingerprint() == state["plan"]["runtime"]
    with pytest.raises(ValueError, match="Runtime changed in numpy"):
        runner.execute_stage(state, "preflight", project, 123)
    assert state["stages"]["preflight"]["attempts"] == []


def test_runtime_change_during_stage_prevents_completion(project, state, monkeypatch):
    fresh = state["plan"]["runtime"].copy()

    def command(argv, log_path, lock_fd):
        directory = Path(argv[argv.index("--output") + 1]).parent
        write_preflight(directory, state["plan"])
        fresh["torchvision"] = "changed-during-stage"
        return 0

    monkeypatch.setattr(runner, "probe_runtime", lambda *args: fresh.copy())
    monkeypatch.setattr(runner, "_run_command", command)
    assert not runner.execute_stage(state, "preflight", project, 123)
    record = state["stages"]["preflight"]
    assert record["status"] == "failed"
    assert "Runtime changed in torchvision" in record["attempts"][-1]["error"]
    assert record["attempts"][-1]["returncode"] == 0


def test_worker_rejects_runtime_change_before_any_device_or_stage_work(state, monkeypatch):
    changed = {**state["plan"]["runtime"], "cudnn": "changed-cudnn"}
    monkeypatch.setattr(runner, "probe_runtime", lambda *args: changed)
    monkeypatch.setattr(runner, "check_device", lambda *args: pytest.fail("Device work began with a changed runtime"))
    monkeypatch.setattr(runner, "execute_stage", lambda *args: pytest.fail("Stage work began with a changed runtime"))
    with pytest.raises(ValueError, match="Runtime changed in cudnn"):
        runner.worker(Path(state["output"]), 123)
    saved = json.loads((Path(state["output"]) / "run.json").read_text())
    assert saved["status"] == "failed"
    assert saved["worker_ready"] is False


def test_device_identity_is_diagnostic_and_not_part_of_runtime_fingerprint(state, monkeypatch):
    commands = []
    monkeypatch.setattr(runner, "_run_command", lambda command, *args: commands.append(command) or 0)
    changed_gpu_plan = {**state["plan"], "device": "cuda:1"}
    runner.require_runtime(changed_gpu_plan)
    runner.check_device(changed_gpu_plan, Path(state["output"]), 123)
    code = commands[0][-1]
    assert "cuda:1" in code
    assert "gpu_uuid" in code and "driver_versions" in code
    assert "nvidia-smi" in code
    assert "gpu" not in state["plan"]["runtime"]


@pytest.mark.skipif(os.name != "posix", reason="Unattended worker targets Linux/WSL")
def test_project_lock_excludes_other_run_roots_and_releases_after_close(project):
    path = project / ".unattended.lock"
    with runner.RunLock(path) as first:
        assert os.get_inheritable(first.fd)
        with pytest.raises(RuntimeError, match="holds the project lock"):
            with runner.RunLock(path):
                pytest.fail("Overlapping worker acquired an exclusive project lock")
    with runner.RunLock(path) as second:
        assert second.fd is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX inherited advisory lock check")
def test_killed_worker_cannot_release_lock_while_training_child_survives(tmp_path):
    lock_path = tmp_path / "project.lock"
    marker = tmp_path / "child.pid"
    log_path = tmp_path / "child.log"
    child_code = "import os,time; from pathlib import Path; Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
    worker_code = (
        "import runpy,sys; from pathlib import Path; ns=runpy.run_path(" + repr(str(RUNNER_PATH)) + "); "
        "lock=ns['RunLock'](Path(" + repr(str(lock_path)) + ")); "
        "lock.__enter__(); ns['_run_command']([sys.executable,'-c'," + repr(child_code) + "],Path(" + repr(str(log_path)) + "),lock.fd)"
    )
    process = subprocess.Popen([sys.executable, "-c", worker_code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.is_file() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.is_file(), process.communicate(timeout=1)[1].decode()
        child_pid = int(marker.read_text())
        process.kill()
        process.wait(timeout=5)
        with pytest.raises(RuntimeError, match="holds the project lock"):
            with runner.RunLock(lock_path):
                pytest.fail("Orphan training child lost its inherited GPU lock")
        os.kill(child_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while True:
            try:
                with runner.RunLock(lock_path):
                    break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.stderr is not None:
            process.stderr.close()


def test_stage_commands_keep_paths_as_argv_and_pair_seed_settings(state):
    complete_source(state)
    output = Path(state["output"])
    for name in runner.ARMS:
        command = runner.stage_command(name, state["plan"], output / name, output, False)
        assert command[0] == sys.executable
        assert command[command.index("--method") + 1] == name
        assert command[command.index("--partition") + 1] == "dev"
        assert command[command.index("--seed") + 1] == "3"
        assert command[command.index("--order-seed") + 1] == "5"
        assert command[command.index("--noise-seed") + 1] == "7"
        assert command[command.index("--config") + 1] == str(output / "config.json")
    command = runner.stage_command("source", state["plan"], output / "source", output, True)
    assert "--resume" in command and "--download" in command


@pytest.mark.parametrize("defect", ["status", "cpu", "batch", "missing_memory", "zero_memory", "zero_timing"])
def test_preflight_validator_rejects_incomplete_or_non_gpu_evidence(state, tmp_path, defect):
    directory = tmp_path / "benchmark"
    write_preflight(directory, state["plan"])
    changes = {
        "status": lambda record: record.update(status="failed"),
        "cpu": lambda record: record["environment"].update(device="cpu"),
        "batch": lambda record: record.update(adaptation_batch_size=999),
        "missing_memory": lambda record: record["source_memory"].update(peak_allocated_bytes=None),
        "zero_memory": lambda record: record["source_memory"].update(peak_allocated_bytes=0),
        "zero_timing": lambda record: record.update(source_step_seconds=0),
    }
    edit_json(directory / "benchmark.json", changes[defect])
    with pytest.raises(ValueError):
        runner.validate_artifacts("preflight", directory, state["plan"])


def test_valid_artifacts_can_be_hashed_for_all_stage_types(state, tmp_path):
    preflight, source, adaptation = (tmp_path / name for name in ("preflight", "source", "adaptation"))
    write_preflight(preflight, state["plan"])
    write_source(source, state["plan"])
    source_hash = runner.file_sha256(source / "source.pt")
    write_adaptation(adaptation, "adamw", state["plan"], source_hash)
    for name, directory in [("preflight", preflight), ("source", source), ("adamw", adaptation)]:
        artifacts = runner.validate_artifacts(name, directory, state["plan"], source_hash)
        assert artifacts
        for filename, digest in artifacts.items():
            assert Path(filename).is_absolute()
            assert digest == hashlib.sha256(Path(filename).read_bytes()).hexdigest()


@pytest.mark.parametrize("defect", ["epoch", "dataset", "checkpoint", "seed", "batch", "cpu"])
def test_source_validator_rejects_wrong_or_incomplete_training(state, tmp_path, defect):
    directory = tmp_path / "source"
    write_source(directory, state["plan"])
    if defect == "epoch":
        (directory / "epochs.jsonl").write_text('{"epoch":0}\n')
    elif defect == "checkpoint":
        (directory / "source.pt").write_bytes(b"replaced checkpoint")
    else:
        changes = {
            "dataset": lambda record: record.update(dataset="fake_smoke_only"),
            "seed": lambda record: record.update(training_seed=999),
            "batch": lambda record: record["source_training"].update(batch_size=999),
            "cpu": lambda record: (record["source_environment"].update(device="cpu"), record["environment"].update(device="cpu")),
        }
        edit_json(directory / "manifest.json", changes[defect])
    with pytest.raises(ValueError):
        runner.validate_artifacts("source", directory, state["plan"])


@pytest.mark.parametrize("defect", ["summary", "seed", "partition", "source", "truncated_steps", "reordered_indices"])
def test_adaptation_validator_rejects_wrong_or_partial_stream(state, tmp_path, defect):
    directory = tmp_path / "adaptation"
    write_adaptation(directory, "adamw", state["plan"], "source-hash")
    folder = directory / "adamw"
    if defect == "summary":
        edit_json(folder / "summary.json", lambda record: record.update(steps=1))
    elif defect == "truncated_steps":
        (folder / "steps.jsonl").write_text("")
    elif defect == "reordered_indices":
        runner.atomic_json(directory / "stream_indices.json", [3, 2, 1, 0])
    else:
        changes = {
            "seed": lambda record: record.update(seed=999),
            "partition": lambda record: record.update(partition="stream"),
            "source": lambda record: record.update(checkpoint_sha256="wrong-source"),
        }
        edit_json(folder / "manifest.json", changes[defect])
    with pytest.raises(ValueError):
        runner.validate_artifacts("adamw", directory, state["plan"], "source-hash")


def test_failed_arm_restarts_in_new_directory_and_retains_exit_codes(project, state, monkeypatch):
    source_hash = complete_source(state)
    directories = []

    def command(command, log_path, lock_fd):
        directory = Path(command[command.index("--output") + 1])
        directories.append(directory)
        if len(directories) == 1:
            directory.mkdir()
            (directory / "partial.txt").write_text("preserve failed attempt")
            return 17
        write_adaptation(directory, "adamw", state["plan"], source_hash)
        return 0

    monkeypatch.setattr(runner, "_run_command", command)
    assert not runner.execute_stage(state, "adamw", project, 123)
    assert state["stages"]["adamw"]["status"] == "failed"
    assert state["stages"]["adamw"]["attempts"][0]["returncode"] == 17
    assert runner.execute_stage(state, "adamw", project, 123)
    assert directories[0] != directories[1]
    assert (directories[0] / "partial.txt").read_text() == "preserve failed attempt"
    assert [entry["status"] for entry in state["stages"]["adamw"]["attempts"]] == ["failed", "completed"]


@pytest.mark.parametrize("has_last_checkpoint", [True, False])
def test_source_reuses_directory_only_when_full_training_checkpoint_exists(project, state, monkeypatch, has_last_checkpoint):
    directory = Path(state["output"]) / "source_attempt01"
    write_source(directory, state["plan"])
    if not has_last_checkpoint:
        (directory / "last.pt").unlink()
    previous_hash = runner.file_sha256(directory / "source.pt")
    record = state["stages"]["source"]
    record.update({"status": "interrupted", "directory": str(directory), "attempts": [{"status": "interrupted", "directory": str(directory)}]})
    commands = []

    def command(argv, log_path, lock_fd):
        commands.append(argv)
        destination = Path(argv[argv.index("--output") + 1])
        if not has_last_checkpoint:
            write_source(destination, state["plan"])
        return 0

    monkeypatch.setattr(runner, "_run_command", command)
    assert runner.execute_stage(state, "source", project, 123)
    assert ("--resume" in commands[0]) == has_last_checkpoint
    assert (Path(record["directory"]) == directory) == has_last_checkpoint
    assert record["attempts"][-1]["resumed_source"] == has_last_checkpoint
    assert runner.file_sha256(directory / "source.pt") == previous_hash
    assert record["attempts"][0]["status"] == "interrupted"


def test_completed_stage_is_verified_before_skip_and_tampering_never_relaunches(project, state, monkeypatch):
    source_hash = complete_source(state)
    directory = Path(state["output"]) / "adamw_attempt01"
    write_adaptation(directory, "adamw", state["plan"], source_hash)
    record = state["stages"]["adamw"]
    record.update({"status": "completed", "directory": str(directory), "artifacts": runner.validate_artifacts("adamw", directory, state["plan"], source_hash)})
    monkeypatch.setattr(runner, "_run_command", lambda *args: pytest.fail("A completed stage was relaunched"))
    assert runner.execute_stage(state, "adamw", project, 123)
    (directory / "adamw" / "final.pt").write_bytes(b"changed after completion")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner.execute_stage(state, "adamw", project, 123)


@pytest.mark.parametrize("defect", ["code", "config"])
def test_stage_launch_rejects_edits_during_pipeline(project, state, monkeypatch, defect):
    monkeypatch.setattr(runner, "_run_command", lambda *args: pytest.fail("Changed plan launched a stage"))
    if defect == "code":
        (project / "src" / "hgfvision" / "implementation.py").write_text("VALUE = 9\n")
    else:
        edit_json(Path(state["output"]) / "config.json", lambda record: record.update(batch_size=1))
    with pytest.raises(ValueError, match="changed|modified"):
        runner.execute_stage(state, "preflight", project, 123)


def test_zero_subprocess_exit_is_insufficient_without_completed_artifacts(project, state, monkeypatch):
    monkeypatch.setattr(runner, "_run_command", lambda *args: 0)
    assert not runner.execute_stage(state, "preflight", project, 123)
    assert state["stages"]["preflight"]["status"] == "failed"


@pytest.mark.parametrize("failed_stage,expected", [
    ("preflight", ["preflight"]),
    ("source", ["preflight", "source"]),
    ("adaptive_qr", list(runner.STAGES)),
    (None, list(runner.STAGES)),
])
def test_worker_blocks_on_prerequisites_but_continues_independent_arms(state, monkeypatch, failed_stage, expected):
    calls = []
    device_checks = []

    def execute(current, name, project, lock_fd):
        calls.append(name)
        current["stages"][name]["status"] = "failed" if name == failed_stage else "completed"
        return name != failed_stage

    monkeypatch.setattr(runner, "check_device", lambda *args: device_checks.append(True))
    monkeypatch.setattr(runner, "execute_stage", execute)
    monkeypatch.setattr(runner, "collect_results", lambda *args: None)
    result = runner.worker(Path(state["output"]), 123)
    assert calls == expected
    assert device_checks == [True]
    assert result == (0 if failed_stage is None else 1)
    saved = json.loads((Path(state["output"]) / "run.json").read_text())
    assert saved["status"] == ("completed" if failed_stage is None else "failed")


def test_worker_clears_old_readiness_and_fails_before_stages_when_cuda_check_fails(state, monkeypatch):
    state["worker_ready"] = True
    runner.save_state(state)

    def fail_device(*args):
        raise RuntimeError("No CUDA device")

    monkeypatch.setattr(runner, "check_device", fail_device)
    monkeypatch.setattr(runner, "execute_stage", lambda *args: pytest.fail("Stage launched before CUDA validation"))
    with pytest.raises(RuntimeError, match="No CUDA device"):
        runner.worker(Path(state["output"]), 123)
    saved = json.loads((Path(state["output"]) / "run.json").read_text())
    assert saved["worker_ready"] is False
    assert saved["status"] == "failed"
    assert all(record["status"] == "pending" for record in saved["stages"].values())


def test_worker_marks_stale_running_attempt_interrupted_before_retry(state, monkeypatch):
    state["stages"]["adamw"].update({"status": "running", "attempts": [{"status": "running"}]})
    runner.save_state(state)
    seen = []

    def execute(current, name, project, lock_fd):
        if name == "adamw":
            record = current["stages"][name]
            assert record["status"] == "interrupted"
            assert record["attempts"][0]["status"] == "interrupted"
            seen.append(name)
        current["stages"][name]["status"] = "completed"
        return True

    monkeypatch.setattr(runner, "check_device", lambda *args: None)
    monkeypatch.setattr(runner, "execute_stage", execute)
    monkeypatch.setattr(runner, "collect_results", lambda *args: None)
    assert runner.worker(Path(state["output"]), 123) == 0
    assert seen == ["adamw"]


def test_status_requires_only_stdlib_and_does_not_modify_artifacts(state):
    output = Path(state["output"])
    before = {path: runner.file_sha256(path) for path in output.rglob("*") if path.is_file()}
    result = subprocess.run([sys.executable, "-S", str(RUNNER_PATH), "--status", "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    reported = json.loads(result.stdout)
    assert reported["status"] == "created"
    assert reported["stages"]["source"] == "pending"
    after = {path: runner.file_sha256(path) for path in output.rglob("*") if path.is_file()}
    assert before == after
