"""Run one finite Vast.ai pilot, detached from SSH if requested.

This script does not rent, stop, or destroy cloud instances. It runs one source
model and one seed of the six configured reference/adaptation arms, then exits.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parent
ARMS = ("frozen", "adamw", "adaptive_qr", "hierarchical_qr", "zero_q", "fixed_q")
STAGES = ("preflight", "source", *ARMS)


def now() -> str:
    """Use an explicit UTC timestamp in machine-readable logs."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    """Commit a finite JSON document without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """Hash complete artifacts before treating a stage as reusable."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_digest(project: Path) -> str:
    """Freeze implementation and packaging, excluding data, logs, and tests."""
    paths = sorted([*project.glob("*.py"), *project.glob("src/hgfvision/*.py"), project / "pyproject.toml"])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project)).encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def runtime_fingerprint() -> dict:
    """Import scientific libraries only when checking the executable runtime."""
    import platform
    import numpy
    import torch
    import torchvision
    return {
        "python_version": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(numpy.__version__),
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def probe_runtime(python: str, lock_fd: int | None = None) -> dict:
    """Check a fresh interpreter so cached imports cannot conceal package changes."""
    inherited = {} if lock_fd is None else {"pass_fds": (lock_fd,)}
    result = subprocess.run(
        [python, str(Path(__file__).resolve()), "--runtime-fingerprint"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        check=False, **inherited,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Runtime probe failed with code {result.returncode}: {result.stderr[-2000:]}")
    fingerprint = json.loads(result.stdout)
    if not isinstance(fingerprint, dict):
        raise ValueError("Runtime probe did not return a version fingerprint")
    return fingerprint


def require_runtime(plan: dict, lock_fd: int | None = None, fresh: bool = False) -> None:
    """Reject mixed runtime versions without restricting GPU or driver identity."""
    expected = plan.get("runtime")
    if not isinstance(expected, dict):
        raise ValueError("Run plan has no runtime fingerprint; use a new run directory")
    if plan["python"] != str(Path(sys.executable).absolute()):
        raise ValueError("Resume must use the same Python environment")
    current = probe_runtime(plan["python"], lock_fd) if fresh else runtime_fingerprint()
    changed = sorted(key for key in set(expected) | set(current) if key not in expected or key not in current or expected[key] != current[key])
    if changed:
        raise ValueError("Runtime changed in " + ", ".join(changed) + "; restore the original environment or use a new run directory")


class RunLock:
    def __init__(self, path: Path):
        """Use one project-wide lock to exclude overlapping unattended pipelines."""
        self.path, self.fd = path, None

    def __enter__(self):
        """Acquire before changing run state; the FD can be inherited by children."""
        if os.name != "posix":
            raise RuntimeError("This unattended runner targets Linux/WSL, including Vast.ai")
        import fcntl
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("Another pipeline or its training child still holds the project lock") from None
        os.set_inheritable(self.fd, True)
        return self

    def __exit__(self, *_):
        """Close without explicit unlock, preserving any child's inherited lock."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def resolved_path(value: str | Path, project: Path = PROJECT) -> Path:
    """Anchor relative paths to the project regardless of the caller's cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def build_plan(project: Path, args, existing: dict | None = None) -> dict:
    """Resolve immutable settings, refusing changed settings on resume."""
    defaults = {"source_epochs": 100, "source_batch_size": 128, "workers": 4,
                "threads": 4, "seed": 0, "order_seed": 0, "noise_seed": 0,
                "device": "cuda:0", "data_root": str(project / "data")}
    if existing is not None:
        plan = existing.copy()
        for name in defaults:
            proposed = getattr(args, name, None)
            if proposed is not None:
                if name == "data_root":
                    proposed = str(resolved_path(proposed, project))
                if proposed != plan[name]:
                    raise ValueError(f"Resume would change {name}; use a new run directory")
        if plan["code_sha256"] != source_digest(project):
            raise ValueError("Implementation changed since this run; use a new run directory")
        if plan["python"] != str(Path(sys.executable).absolute()):
            raise ValueError("Resume must use the same Python environment")
        if getattr(args, "config", None) is not None:
            raise ValueError("Resume uses its frozen configuration; omit --config")
        require_runtime(plan)
        return plan
    plan = {name: getattr(args, name, None) if getattr(args, name, None) is not None else default for name, default in defaults.items()}
    plan["data_root"] = str(resolved_path(plan["data_root"], project))
    for name in ("source_epochs", "source_batch_size", "threads"):
        if plan[name] < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("workers", "seed", "order_seed", "noise_seed"):
        if plan[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not plan["device"].startswith("cuda"):
        raise ValueError("The unattended real-data pipeline requires CUDA; use the smoke command for CPU")
    sys.path.insert(0, str(project / "src"))
    from hgfvision.cli import read_config
    config_file = resolved_path(getattr(args, "config", None) or "configs/pilot.json", project)
    config = read_config(config_file)
    plan.update({"config": config, "code_sha256": source_digest(project),
                 "python": str(Path(sys.executable).absolute()), "project": str(project),
                 "partition": "dev", "arms": list(ARMS), "source_split_seed": 0,
                 "pipeline_version": 2, "runtime": runtime_fingerprint()})
    return plan


def _run_command(command: list[str], log_path: Path, lock_fd: int) -> int:
    """Run argv without a shell, preserving the lock through child lifetime."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with log_path.open("a") as log:
        log.write(json.dumps({"launched": now(), "argv": command}) + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def stage_command(name: str, plan: dict, directory: Path, output: Path, resume_source: bool) -> list[str]:
    """Build subprocess arguments without changing shared experimental settings."""
    python = plan["python"]
    if name == "preflight":
        return [python, "-m", "hgfvision.benchmark", "--output", str(directory / "benchmark.json"),
                "--device", plan["device"], "--source-batch-size", str(plan["source_batch_size"]),
                "--batch-size", str(plan["config"]["batch_size"]), "--threads", str(plan["threads"])]
    prefix = [python, "-m", "hgfvision", "--threads", str(plan["threads"])]
    common = ["--device", plan["device"], "--data-root", plan["data_root"],
              "--workers", str(plan["workers"]), "--seed", str(plan["seed"]),
              "--eval-batch-size", "128", "--output", str(directory)]
    if name == "source":
        return prefix + ["train-source", "--download", "--epochs", str(plan["source_epochs"]),
                         "--batch-size", str(plan["source_batch_size"]), "--split-seed", "0",
                         *common, *(["--resume"] if resume_source else [])]
    state = json.loads((output / "run.json").read_text())
    checkpoint = Path(state["stages"]["source"]["directory"]) / "source.pt"
    return prefix + ["adapt", "--checkpoint", str(checkpoint), "--method", name,
                     "--partition", "dev", "--config", str(output / "config.json"),
                     "--order-seed", str(plan["order_seed"]), "--noise-seed", str(plan["noise_seed"]), *common]


def validate_artifacts(name: str, directory: Path, plan: dict, source_hash: str | None = None) -> dict[str, str]:
    """Verify semantic completion, then hash artifacts used to skip work later."""
    if name == "preflight":
        path = directory / "benchmark.json"
        record = json.loads(path.read_text())
        if record.get("status") != "completed" or not record["environment"]["device"].startswith("cuda"):
            raise ValueError("Preflight did not complete on CUDA")
        if record["source_batch_size"] != plan["source_batch_size"] or record["adaptation_batch_size"] != plan["config"]["batch_size"]:
            raise ValueError("Preflight used different batch sizes")
        if any(record[key]["peak_allocated_bytes"] is None for key in ("source_memory", "adaptation_memory")):
            raise ValueError("Preflight omitted measured CUDA memory")
        for key in ("source_memory", "adaptation_memory"):
            allocated, reserved = (record[key][name] for name in ("peak_allocated_bytes", "peak_reserved_bytes"))
            if type(allocated) is not int or type(reserved) is not int or not 0 < allocated <= reserved:
                raise ValueError("Invalid CUDA allocator measurements")
        for key in ("source_step_seconds", "adaptation_step_seconds"):
            if not isinstance(record.get(key), (int, float)) or not math.isfinite(record[key]) or record[key] <= 0:
                raise ValueError("Invalid benchmark timing")
        paths = [path]
    elif name == "source":
        manifest_path = directory / "manifest.json"
        record = json.loads(manifest_path.read_text())
        history = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        if record.get("status") != "completed" or record.get("dataset") != "cifar10":
            raise ValueError("Source training is incomplete or uses generated images")
        if record["epochs_requested"] != plan["source_epochs"] or [row["epoch"] for row in history] != list(range(plan["source_epochs"])):
            raise ValueError("Source history does not cover the full planned epoch budget")
        if record["training_seed"] != plan["seed"] or record["split_seed"] != plan["source_split_seed"]:
            raise ValueError("Source training seed mismatch")
        if record["source_training"]["batch_size"] != plan["source_batch_size"]:
            raise ValueError("Source training batch size mismatch")
        if not record["environment"]["device"].startswith("cuda"):
            raise ValueError("Source training did not run on CUDA")
        if record["checkpoint_sha256"] != file_sha256(directory / "source.pt"):
            raise ValueError("Source checkpoint hash does not match its manifest")
        paths = [manifest_path, directory / "source.pt", directory / "last.pt", directory / "epochs.jsonl", directory / "splits.json"]
    else:
        folder = directory / name
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        expected_steps = sum(phase["steps"] for phase in plan["config"]["phases"])
        if manifest.get("status") != "completed" or summary.get("status") != "completed":
            raise ValueError("Adaptation arm did not complete")
        if manifest.get("dataset") != "cifar10" or manifest.get("method") != name or summary.get("method") != name:
            raise ValueError("Adaptation dataset/method mismatch")
        if manifest["partition"] != "dev" or manifest["config"] != plan["config"]:
            raise ValueError("Adaptation changed data partition or configuration")
        if manifest["checkpoint_sha256"] != source_hash:
            raise ValueError("Adaptation used a different source checkpoint")
        for key in ("seed", "order_seed", "noise_seed"):
            if manifest[key] != plan[key]:
                raise ValueError(f"Adaptation changed {key}")
        if summary["steps"] != expected_steps or summary["samples"] != expected_steps * plan["config"]["batch_size"]:
            raise ValueError("Adaptation did not consume the planned stream")
        rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(expected_steps)):
            raise ValueError("Adaptation log is incomplete or has duplicated steps")
        if any(row["samples"] != plan["config"]["batch_size"] or len(row["indices"]) != row["samples"] for row in rows):
            raise ValueError("Adaptation log has incorrect minibatch sizes")
        observed_indices = [index for row in rows for index in row["indices"]]
        recorded_indices = json.loads((directory / "stream_indices.json").read_text())
        indices_hash = hashlib.sha256(json.dumps(recorded_indices, sort_keys=True).encode()).hexdigest()
        if observed_indices != recorded_indices or len(set(observed_indices)) != len(observed_indices) or indices_hash != manifest["stream_indices_sha256"]:
            raise ValueError("Adaptation stream index provenance mismatch")
        paths = [folder / filename for filename in ("manifest.json", "summary.json", "steps.jsonl", "final.pt")]
        paths += [directory / filename for filename in ("config.json", "splits.json", "stream_indices.json")]
    return {str(path): file_sha256(path) for path in paths}


def verify_saved_stage(record: dict, plan: dict, name: str, source_hash: str | None) -> None:
    """Reject modified or missing artifacts rather than silently skipping them."""
    current = validate_artifacts(name, Path(record["directory"]), plan, source_hash)
    if current != record.get("artifacts"):
        raise ValueError(f"Completed {name} artifacts changed; use a new run directory")


def save_state(state: dict) -> None:
    """Persist updated worker state after each lifecycle transition."""
    state["updated_at"] = now()
    atomic_json(Path(state["output"]) / "run.json", state)


def execute_stage(state: dict, name: str, project: Path, lock_fd: int) -> bool:
    """Resume source epochs or retry a failed arm in a fresh attempt directory."""
    plan, output = state["plan"], Path(state["output"])
    require_runtime(plan, lock_fd, fresh=True)
    if source_digest(project) != plan["code_sha256"]:
        raise ValueError("Source code changed while the pipeline was running")
    if json.loads((output / "config.json").read_text()) != plan["config"]:
        raise ValueError("Frozen run configuration was modified")
    source_hash = None
    if name in ARMS:
        source = state["stages"]["source"]
        verify_saved_stage(source, plan, "source", None)
        source_hash = file_sha256(Path(source["directory"]) / "source.pt")
    record = state["stages"][name]
    if record["status"] == "completed":
        verify_saved_stage(record, plan, name, source_hash)
        return True
    resume_source = name == "source" and "directory" in record and (Path(record["directory"]) / "last.pt").is_file()
    attempt = len(record.get("attempts", [])) + 1
    directory = Path(record["directory"]) if resume_source else output / f"{name}_attempt{attempt:02d}"
    log_path = output / "logs" / f"{name}_attempt{attempt:02d}.log"
    record.setdefault("attempts", []).append({"attempt": attempt, "directory": str(directory), "log": str(log_path), "started_at": now(), "resumed_source": resume_source})
    record.update({"directory": str(directory), "status": "running", "attempt": attempt})
    state["current_stage"] = name
    save_state(state)
    command = stage_command(name, plan, directory, output, resume_source)
    record["attempts"][-1]["command"] = command
    save_state(state)
    try:
        code = _run_command(command, log_path, lock_fd)
        record["attempts"][-1]["returncode"] = code
        if code != 0:
            raise RuntimeError(f"{name} exited with code {code}; inspect {log_path}")
        require_runtime(plan, lock_fd, fresh=True)
        artifacts = validate_artifacts(name, directory, plan, source_hash)
        record.update({"status": "completed", "artifacts": artifacts})
        record["attempts"][-1].update({"status": "completed", "returncode": code, "finished_at": now()})
        save_state(state)
        return True
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["attempts"][-1].update({"status": record["status"], "error": f"{type(error).__name__}: {error}", "finished_at": now()})
        save_state(state)
        if isinstance(error, KeyboardInterrupt):
            raise
        return False


def check_device(plan: dict, output: Path, lock_fd: int) -> None:
    """Record fresh CUDA and optional driver identity without freezing hardware."""
    code = "\n".join([
        "import json,subprocess,torch",
        "d=torch.device(" + repr(plan["device"]) + ")",
        "p=torch.cuda.get_device_properties(d)",
        "x=torch.ones(8,device=d)",
        "record={'device':str(d),'gpu':p.name,'gpu_uuid':str(getattr(p,'uuid','unavailable')),'total_bytes':p.total_memory,'check':float((x*x).sum())}",
        "try:",
        "    result=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=False)",
        "    record['driver_query_returncode']=result.returncode",
        "    record['driver_versions']=sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip())) if result.returncode==0 else []",
        "except (OSError,subprocess.TimeoutExpired) as error:",
        "    record['driver_versions']=[]",
        "    record['driver_query_error']=type(error).__name__",
        "print(json.dumps(record))",
    ])
    command = [plan["python"], "-c", code]
    if _run_command(command, output / "logs" / "device_checks.log", lock_fd) != 0:
        raise RuntimeError("CUDA device check failed; see logs/device_checks.log")


def collect_results(state: dict) -> None:
    """Write a partial comparison even when an independent arm failed."""
    results = []
    for name in ARMS:
        record = state["stages"][name]
        if record["status"] == "completed":
            results.append(json.loads((Path(record["directory"]) / name / "summary.json").read_text()))
    atomic_json(Path(state["output"]) / "comparison.json", results)


def worker(output: Path, lock_fd: int) -> int:
    """Run prerequisites, then independent arms, and release resources on exit."""
    state = json.loads((output / "run.json").read_text())
    state.update({"status": "running", "pid": os.getpid(), "worker_started_at": now(), "worker_ready": False})
    for record in state["stages"].values():
        if record["status"] == "running":
            record["status"] = "interrupted"
            if record.get("attempts"):
                record["attempts"][-1].update({"status": "interrupted", "finished_at": now(), "error": "Previous worker ended before committing stage completion"})
    save_state(state)
    try:
        require_runtime(state["plan"], lock_fd, fresh=True)
        check_device(state["plan"], output, lock_fd)
        state["worker_ready"] = True
        save_state(state)
        for name in ("preflight", "source"):
            if not execute_stage(state, name, PROJECT, lock_fd):
                state["status"] = "failed"
                save_state(state)
                return 1
        all_ok = True
        for name in ARMS:
            all_ok = execute_stage(state, name, PROJECT, lock_fd) and all_ok
            collect_results(state)
        state["status"] = "completed" if all_ok else "failed"
        state["current_stage"] = None
        state["finished_at"] = now()
        save_state(state)
        return 0 if all_ok else 1
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        save_state(state)
        raise


def show_status(output: Path) -> None:
    """Read status without importing torch or changing any file."""
    state = json.loads((output / "run.json").read_text())
    overview = {"output": state["output"], "status": state["status"], "pid": state.get("pid"),
                "updated_at": state.get("updated_at"), "current_stage": state.get("current_stage"),
                "stages": {key: record["status"] for key, record in state["stages"].items()}}
    source = state["stages"]["source"]
    if "directory" in source:
        history = Path(source["directory"]) / "epochs.jsonl"
        if history.is_file():
            complete = []
            for line in history.read_text().splitlines():
                try:
                    complete.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            overview["source_completed_epochs"] = len(complete)
            overview["source_planned_epochs"] = state["plan"]["source_epochs"]
            if complete:
                overview["source_latest_validation"] = complete[-1]["validation"]
    print(json.dumps(overview, indent=2))


def main() -> int:
    """Start or resume a finite pipeline; detached mode waits for worker startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--data-root")
    for name in ("source-epochs", "source-batch-size", "workers", "threads", "seed", "order-seed", "noise-seed"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-fingerprint", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.resume or args.status or args.worker) and not args.output:
        parser.error("--output is required for resume/status")
    output = resolved_path(args.output or f"runs/vast_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    if args.status:
        show_status(output)
        return 0
    if args.runtime_fingerprint:
        print(json.dumps(runtime_fingerprint(), sort_keys=True))
        return 0
    if args.worker:
        if args.lock_fd is None:
            parser.error("Worker requires its inherited lock")
        if os.fstat(args.lock_fd).st_ino != (PROJECT / ".unattended.lock").stat().st_ino:
            raise RuntimeError("Invalid inherited project lock")
        try:
            return worker(output, args.lock_fd)
        finally:
            os.close(args.lock_fd)
    with RunLock(PROJECT / ".unattended.lock") as lock:
        if args.resume:
            state = json.loads((output / "run.json").read_text())
            build_plan(PROJECT, args, existing=state["plan"])
        else:
            plan = build_plan(PROJECT, args)
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(output / "config.json", plan["config"])
            state = {"schema": 1, "output": str(output), "status": "created", "created_at": now(),
                     "plan": plan, "stages": {name: {"status": "pending", "attempts": []} for name in STAGES}}
            save_state(state)
        if not args.detach:
            return worker(output, lock.fd)
        with (output / "launcher.log").open("a") as log:
            command = [sys.executable, str(PROJECT / "run_experiments.py"), "--worker", "--output", str(output), "--lock-fd", str(lock.fd)]
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fd,))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = json.loads((output / "run.json").read_text())
            if current.get("pid") == child.pid and current.get("worker_ready") and current["status"] in {"running", "completed"}:
                print(f"Background worker started: PID {child.pid}\nRun: {output}\nLog: {output / 'launcher.log'}")
                return 0
            if child.poll() is not None:
                raise RuntimeError(f"Worker exited during startup; inspect {output / 'launcher.log'}")
            time.sleep(0.1)
        raise RuntimeError(f"Worker startup was not confirmed; inspect {output / 'run.json'} before retrying")


if __name__ == "__main__":
    raise SystemExit(main())
