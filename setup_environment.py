"""Prepare and verify the NVIDIA environment for this project using Python only.

The default isolated environment pins the CUDA 12.6 wheels documented at
https://pytorch.org/get-started/previous-versions/ (checked 2026-10-19).
Environment preparation never installs drivers or changes system packages.
With --launch-pilot, setup stays in the foreground until a verified environment
hands off to the detached dataset/training pipeline. It does not detach setup.
Use --dry-run to inspect the plan without filesystem or subprocess side effects.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv


TORCH_VERSION = "2.12.1"
VISION_VERSION = "0.27.1"
WHEEL_INDEX = "https://download.pytorch.org/whl/cu126"
OFFICIAL_SOURCE = "https://pytorch.org/get-started/previous-versions/"
REPORT_MARKER = "HGF_ENVIRONMENT_REPORT="
BUILD_MARKER = "HGF_ENVIRONMENT_BUILD="
GPU_PROBE = r'''
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys
import torch

build = {"python": platform.python_version(), "torch": str(torch.__version__), "torchvision": None, "cuda_runtime": torch.version.cuda}
try:
    build["torchvision"] = version("torchvision")
except PackageNotFoundError:
    pass
print("HGF_ENVIRONMENT_BUILD=" + json.dumps(build), flush=True)
import torchvision

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this Python environment. Check the NVIDIA GPU, driver, container GPU access, and CUDA-enabled PyTorch wheel.")
device = torch.device("cuda:0")
torch.manual_seed(0)
a = torch.randn(64, 64, device=device, requires_grad=True)
b = torch.randn(64, 64, device=device, requires_grad=True)
loss = (a @ b).square().mean()
loss.backward()
if not torch.isfinite(loss) or not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
    raise RuntimeError("CUDA matrix multiplication/backward produced nonfinite values")
conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
x = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
conv_loss = conv(x).square().mean()
conv_loss.backward()
if not torch.isfinite(conv_loss) or not torch.isfinite(x.grad).all() or not torch.isfinite(conv.weight.grad).all():
    raise RuntimeError("CUDA convolution/backward produced nonfinite values")
torch.cuda.synchronize(device)
devices = []
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    devices.append({"index": index, "name": torch.cuda.get_device_name(index), "free_bytes": free, "total_bytes": total, "compute_capability": list(torch.cuda.get_device_capability(index))})
print("HGF_ENVIRONMENT_REPORT=" + json.dumps({"python": platform.python_version(), "executable": sys.executable, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "cuda_runtime": torch.version.cuda, "cuda_available": True, "cuda_matmul_backward": "passed", "cuda_conv_backward": "passed", "tested_device": 0, "devices": devices}))
'''


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose explicit installation, validation, and inspection modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-existing-env", action="store_true", help="Preserve current torch/torchvision; validate CUDA and install the project in this interpreter.")
    parser.add_argument("--skip-tests", action="store_true", help="Explicitly omit pytest; CUDA operation checks still run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without installing, checking hardware, or writing files.")
    parser.add_argument("--launch-pilot", action="store_true", help="After CUDA checks and pytest pass, start run_experiments.py --detach using the verified Python.")
    parser.add_argument("--run-output", help="New pilot output directory, relative to this project by default (default: runs/first_pilot).")
    args = parser.parse_args(argv)
    if args.launch_pilot and args.skip_tests:
        parser.error("--launch-pilot requires passing tests; it cannot be combined with --skip-tests")
    if args.run_output is not None and not args.launch_pilot:
        parser.error("--run-output requires --launch-pilot")
    return args


def venv_python(directory: Path) -> Path:
    """Return the platform's interpreter path inside a project-local environment."""
    return directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def install_commands(python: Path, constraints: str, existing: bool) -> list[list[str]]:
    """Use isolated pip settings and explicit constraints to preserve torch pins."""
    prefix = [str(python), "-m", "pip", "--isolated", "install"]
    commands = []
    if not existing:
        commands.append(prefix + [f"torch=={TORCH_VERSION}", f"torchvision=={VISION_VERSION}", "--index-url", WHEEL_INDEX])
    commands.append(prefix + ["--constraint", constraints, "--editable", ".[test]"])
    return commands


def pilot_output(project: Path, args: argparse.Namespace) -> Path | None:
    """Validate a new run destination before hardware checks or large installs."""
    if not args.launch_pilot:
        return None
    if args.skip_tests:
        raise ValueError("Launching an unattended pilot requires the project tests")
    if not (project / "run_experiments.py").is_file():
        raise RuntimeError("--launch-pilot requires run_experiments.py in this checkout")
    if not args.dry_run and platform.system() == "Windows":
        raise RuntimeError("The detached pilot requires Linux/WSL; native Windows supports environment setup only")
    value = Path(args.run_output or "runs/first_pilot").expanduser()
    output = value if value.is_absolute() else project / value
    if output.exists() or output.is_symlink():
        raise ValueError(f"Pilot output already exists: {output}. Inspect or resume that run with run_experiments.py, or choose a new --run-output.")
    for parent in output.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Pilot output parent is not a directory: {parent}")
            break
    return output.resolve()


def pilot_command(python: Path, project: Path, output: Path) -> list[str]:
    """Launch only the finite detached runner, using the environment just tested."""
    return [str(python), str(project / "run_experiments.py"), "--detach", "--output", str(output)]


def dry_run_plan(project: Path, args: argparse.Namespace) -> dict:
    """Describe all work without invoking a process or creating an environment."""
    python = Path(sys.executable) if args.use_existing_env else venv_python(project / ".venv")
    commands = [["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"]]
    if not args.use_existing_env:
        commands.append([sys.executable, "-m", "venv", str(project / ".venv")])
        commands.extend(install_commands(python, "<temporary-constraints.txt>", False)[:1])
    commands.append([str(python), "-c", "<CUDA matrix multiplication and convolution backward probe>"])
    commands.extend(install_commands(python, "<temporary-constraints.txt>", True))
    commands.append([str(python), "-c", "<repeat CUDA probe and verify versions unchanged>"])
    commands.append([str(python), "-c", "import hgfvision; print(hgfvision.__file__)"])
    commands.append([str(python), "-m", "pip", "--isolated", "freeze"])
    if not args.skip_tests:
        commands.append([str(python), "-m", "pytest", "-q"])
    output = pilot_output(project, args)
    if output is not None:
        commands.append(pilot_command(python, project, output))
    return {
        "dry_run": True, "project": str(project), "python": str(python),
        "mode": "existing_environment" if args.use_existing_env else "project_venv",
        "torch_policy": "preserve installed versions" if args.use_existing_env else {"torch": TORCH_VERSION, "torchvision": VISION_VERSION, "wheel_index": WHEEL_INDEX},
        "commands": commands, "reports": [str(project / "environment_setup.json"), str(project / "environment_freeze.txt")],
        "tests": "explicitly_skipped" if args.skip_tests else "required",
        "pilot_output": None if output is None else str(output),
        "setup_runs_in_foreground": True,
    }


def run_command(argv: list[str], project: Path, capture: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run argument vectors without a shell and surface command failures clearly."""
    display = "CUDA verification probe" if len(argv) > 2 and argv[1:3] == ["-c", GPU_PROBE] else shlex.join(argv)
    print(f"Running: {display}", flush=True)
    try:
        return subprocess.run(argv, cwd=project, check=True, text=True, capture_output=capture, timeout=timeout)
    except subprocess.CalledProcessError as error:
        detail = "\n".join(part.strip() for part in (error.stdout, error.stderr) if part and part.strip())
        failure = RuntimeError(f"Command failed (exit {error.returncode}): {display}" + (f"\n{detail[-4000:]}" if detail else ""))
        if len(argv) > 2 and argv[1:3] == ["-c", GPU_PROBE]:
            for line in (error.stdout or "").splitlines():
                if line.startswith(BUILD_MARKER):
                    try:
                        failure.cuda_build = json.loads(line[len(BUILD_MARKER):])
                    except json.JSONDecodeError:
                        pass
        raise failure from error


def check_host(project: Path) -> str:
    """Reject unsupported hosts and missing NVIDIA access before installing anything."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required; choose a suitable rental image.")
    if platform.system() not in {"Linux", "Windows"}:
        raise RuntimeError("This CUDA setup supports NVIDIA Linux/Windows hosts. Run --dry-run locally, then run setup on the GPU rental.")
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi is missing. Use a GPU-enabled rental/container with an installed NVIDIA driver; this script does not install drivers.")
    result = run_command([binary, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"], project, capture=True, timeout=20)
    if not result.stdout.strip():
        raise RuntimeError("nvidia-smi reported no NVIDIA GPUs; verify the rental's GPU access.")
    return result.stdout.strip()


def prepare_venv(project: Path) -> Path:
    """Create or validate only this project's real, non-symlinked .venv directory."""
    directory = project / ".venv"
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked .venv; create a real project-local environment.")
    python = venv_python(directory)
    if directory.exists():
        if not directory.is_dir() or not (directory / "pyvenv.cfg").is_file() or not python.is_file():
            raise RuntimeError("Existing .venv is incomplete or is not a virtual environment. Move it aside explicitly before retrying.")
    else:
        print(f"Creating virtual environment: {directory}", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    result = run_command([str(python), "-c", "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"], project, capture=True, timeout=30)
    identity = json.loads(result.stdout)
    if Path(identity["prefix"]).resolve() != directory.resolve() or identity["prefix"] == identity["base_prefix"]:
        raise RuntimeError("The .venv interpreter does not identify as this project's isolated environment.")
    return python


def gpu_probe(python: Path, project: Path) -> dict:
    """Test real CUDA forwards/backwards and return machine-readable measurements."""
    result = run_command([str(python), "-c", GPU_PROBE], project, capture=True, timeout=180)
    reports = [line[len(REPORT_MARKER):] for line in result.stdout.splitlines() if line.startswith(REPORT_MARKER)]
    if len(reports) != 1:
        raise RuntimeError("CUDA probe did not produce exactly one valid report")
    report = json.loads(reports[0])
    for package, minimum in (("torch", (2, 2)), ("torchvision", (0, 17))):
        match = re.match(r"^(\d+)\.(\d+)", report[package])
        if match is None or tuple(map(int, match.groups())) < minimum:
            raise RuntimeError(f"Existing {package} does not meet this project's minimum version")
    return report


def write_report(project: Path, report: dict) -> None:
    """Atomically retain success or failure evidence without implying unrun tests passed."""
    destination = project / "environment_setup.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def setup(args: argparse.Namespace, project: Path | None = None) -> dict:
    """Install the project, preserve torch versions, test CUDA, and run pytest."""
    project = Path(__file__).resolve().parent if project is None else Path(project).resolve()
    if not (project / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from a complete hgf_resnet checkout with pyproject.toml.")
    output = pilot_output(project, args)
    if args.dry_run:
        plan = dry_run_plan(project, args)
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "existing_environment" if args.use_existing_env else "project_venv",
        "official_wheel_source": OFFICIAL_SOURCE, "wheel_index": None if args.use_existing_env else WHEEL_INDEX,
        "tests": "explicitly_skipped" if args.skip_tests else "not_run",
        "stage": "host_preflight", "project": str(project),
        "setup_status": "running",
        "pilot_launch": {"status": "not_requested" if output is None else "not_started", "output": None if output is None else str(output)},
    }
    try:
        report["nvidia_smi"] = check_host(project)
        report["stage"] = "environment"
        python = Path(sys.executable) if args.use_existing_env else prepare_venv(project)
        report["python_executable"] = str(python)
        if not args.use_existing_env:
            report["stage"] = "pinned_torch_install"
            run_command(install_commands(python, "unused", False)[0], project)
        report["stage"] = "initial_cuda_verification"
        initial = gpu_probe(python, project)
        report["initial_cuda_verification"] = initial
        if not args.use_existing_env:
            if initial["torch"].split("+")[0] != TORCH_VERSION or initial["torchvision"].split("+")[0] != VISION_VERSION or initial["cuda_runtime"] != "12.6":
                raise RuntimeError("Installed wheels do not match the requested torch/torchvision/CUDA 12.6 pins")
        report["preserved_versions"] = {name: initial[name] for name in ("torch", "torchvision")}
        report["stage"] = "project_install"
        with tempfile.TemporaryDirectory(prefix="hgf-environment-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            constraints.write_text("".join(f"{name}=={initial[name]}\n" for name in ("torch", "torchvision")), encoding="utf-8")
            run_command(install_commands(python, str(constraints), True)[0], project)
        report["stage"] = "final_cuda_verification"
        final = gpu_probe(python, project)
        report["cuda_verification"] = final
        if any(final[name] != initial[name] for name in ("torch", "torchvision", "cuda_runtime")):
            raise RuntimeError("Project installation unexpectedly changed the torch/torchvision/CUDA versions")
        run_command([str(python), "-c", "import hgfvision; print(hgfvision.__file__)"], project, capture=True, timeout=30)
        report["stage"] = "dependency_record"
        freeze = run_command([str(python), "-m", "pip", "--isolated", "freeze"], project, capture=True, timeout=60).stdout
        (project / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
        report["pip_freeze"] = freeze.splitlines()
        if not args.skip_tests:
            report["stage"] = "project_tests"
            report["tests"] = "running"
            run_command([str(python), "-m", "pytest", "-q"], project)
            report["tests"] = "passed"
        report["setup_status"] = "completed"
        if output is not None:
            report["stage"] = "pilot_launch"
            report["pilot_launch"]["status"] = "starting"
            report["pilot_launch"]["command"] = pilot_command(python, project, output)
            write_report(project, report)
            print("Environment checks passed. Starting the detached pilot; keep this console open until worker startup is confirmed.", flush=True)
            run_command(pilot_command(python, project, output), project)
            report["pilot_launch"]["status"] = "started"
        report["status"] = "completed"
        report["stage"] = "complete"
        print(json.dumps({"status": report["status"], "tests": report["tests"], "cuda": final, "pilot_launch": report["pilot_launch"], "report": str(project / "environment_setup.json")}, indent=2), flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["status"] = status
        if report["setup_status"] != "completed":
            report["setup_status"] = status
        if report["pilot_launch"]["status"] == "starting":
            report["pilot_launch"]["status"] = status
            report["pilot_launch"]["worker_state"] = "unknown; inspect run.json and launcher.log before retrying"
        if report["tests"] == "running":
            report["tests"] = status
        if hasattr(error, "cuda_build"):
            report["failed_cuda_probe_build"] = error.cuda_build
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(project, report)


def main(argv: list[str] | None = None) -> int:
    """Return a failing exit status so an unattended runner cannot ignore setup errors."""
    try:
        setup(arguments(argv))
    except KeyboardInterrupt:
        print("Environment setup interrupted. Inspect environment_setup.json before retrying; pilot worker state may need checking if launch had begun.", file=sys.stderr, flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Environment setup failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
