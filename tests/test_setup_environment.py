"""Test environment bootstrap control flow without installing packages or using CUDA."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "setup_environment.py"
SPEC = importlib.util.spec_from_file_location("hgf_setup_environment_tested", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class EnvironmentSetupTests(unittest.TestCase):
    """Protect hardware checks, pin preservation, isolated setup, and report honesty."""

    def _project(self, directory: str) -> Path:
        """Create temporary project markers without a runnable experiment worker."""
        project = Path(directory).resolve()
        (project / "pyproject.toml").write_text("[project]\nname='fake-test-project'\n")
        (project / "run_experiments.py").write_text('"""Test placeholder; never executed."""\n')
        return project

    def _probe(self, torch_version: str = "2.12.1+cu126") -> dict:
        """Supply measured-like test data without claiming actual GPU execution."""
        return {"torch": torch_version, "torchvision": "0.27.1+cu126", "cuda_runtime": "12.6", "devices": [{"name": "mock_gpu", "free_bytes": 1, "total_bytes": 2}]}

    def test_dry_run_has_no_process_or_filesystem_side_effects(self):
        """Inspection works on a CPU host and never creates even the report file."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            before = set(project.iterdir())
            with patch.object(bootstrap, "check_host") as host, patch.object(bootstrap, "run_command") as command, patch.object(bootstrap.venv, "EnvBuilder") as builder, redirect_stdout(io.StringIO()):
                result = bootstrap.setup(bootstrap.arguments(["--dry-run"]), project)
            host.assert_not_called()
            command.assert_not_called()
            builder.assert_not_called()
            self.assertEqual(set(project.iterdir()), before)
            self.assertTrue(result["dry_run"])
            self.assertIn(bootstrap.WHEEL_INDEX, str(result["commands"]))

    def test_missing_nvidia_smi_fails_before_install(self):
        """An unavailable GPU cannot trigger a large package download."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            with patch.object(bootstrap.platform, "system", return_value="Linux"), patch.object(bootstrap.shutil, "which", return_value=None), patch.object(bootstrap, "prepare_venv") as prepare, patch.object(bootstrap, "run_command") as command:
                with self.assertRaisesRegex(RuntimeError, "nvidia-smi is missing"):
                    bootstrap.setup(bootstrap.arguments([]), project)
            prepare.assert_not_called()
            command.assert_not_called()
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["tests"], "not_run")

    def test_existing_environment_preserves_versions_and_runs_tests(self):
        """Explicit existing-environment mode installs only the project under pins."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            commands = []

            def command(argv, project, **kwargs):
                """Capture pip constraints while their temporary file still exists."""
                commands.append(argv)
                if "--constraint" in argv:
                    constraints = Path(argv[argv.index("--constraint") + 1]).read_text()
                    self.assertIn("torch==2.12.1+cu126", constraints)
                    self.assertIn("torchvision==0.27.1+cu126", constraints)
                return subprocess.CompletedProcess(argv, 0, stdout="mock-package==1\n", stderr="")

            with patch.object(bootstrap, "check_host", return_value="mock NVIDIA GPU"), patch.object(bootstrap, "prepare_venv") as prepare, patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                result = bootstrap.setup(bootstrap.arguments(["--use-existing-env"]), project)
            prepare.assert_not_called()
            self.assertEqual(result["tests"], "passed")
            self.assertTrue(any("pytest" in argv for argv in commands))
            self.assertFalse(any(bootstrap.WHEEL_INDEX in argv for argv in commands))
            self.assertEqual(result["pip_freeze"], ["mock-package==1"])

    def test_default_mode_installs_pinned_wheels_and_respects_skip_tests(self):
        """The default plan pins CUDA wheels while skipped tests stay explicit."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            with patch.object(bootstrap, "check_host", return_value="mock NVIDIA GPU"), patch.object(bootstrap, "prepare_venv", return_value=project / ".venv/bin/python"), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")) as command, redirect_stdout(io.StringIO()):
                result = bootstrap.setup(bootstrap.arguments(["--skip-tests"]), project)
            commands = [call.args[0] for call in command.call_args_list]
            wheel = next(argv for argv in commands if bootstrap.WHEEL_INDEX in argv)
            self.assertIn("torch==2.12.1", wheel)
            self.assertIn("torchvision==0.27.1", wheel)
            self.assertFalse(any("pytest" in argv for argv in commands))
            self.assertEqual(result["tests"], "explicitly_skipped")

    def test_project_install_cannot_change_torch(self):
        """Version drift is a setup failure rather than a silently accepted upgrade."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            with patch.object(bootstrap, "check_host", return_value="mock NVIDIA GPU"), patch.object(bootstrap, "gpu_probe", side_effect=[self._probe(), self._probe("2.13.0+cu126")]), patch.object(bootstrap, "run_command", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "unexpectedly changed"):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env"]), project)
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["tests"], "not_run")

    def test_test_failure_is_recorded(self):
        """A failed test suite cannot produce a successful setup manifest."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)

            def command(argv, project, **kwargs):
                """Simulate a failing pytest process after successful installation."""
                if "pytest" in argv:
                    raise RuntimeError("pytest failed")
                return subprocess.CompletedProcess(argv, 0, stdout="mock-package==1\n", stderr="")

            with patch.object(bootstrap, "check_host", return_value="mock NVIDIA GPU"), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "pytest failed"):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env"]), project)
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["tests"], "failed")
            self.assertTrue((project / "environment_freeze.txt").is_file())

    def test_existing_non_venv_directory_is_rejected(self):
        """Never overwrite an unrelated existing .venv directory."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            (project / ".venv").mkdir()
            with patch.object(bootstrap.venv, "EnvBuilder") as builder:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    bootstrap.prepare_venv(project)
            builder.assert_not_called()

    def test_interpreter_outside_project_venv_is_rejected(self):
        """An apparent local environment cannot redirect installation into system Python."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            environment = project / ".venv"
            python = bootstrap.venv_python(environment)
            python.parent.mkdir(parents=True)
            python.touch()
            (environment / "pyvenv.cfg").write_text("home = /unrelated/python\n")
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"prefix": "/unrelated/python", "base_prefix": "/unrelated/python"}), stderr="")
            with patch.object(bootstrap, "run_command", return_value=result), patch.object(bootstrap.venv, "EnvBuilder") as builder:
                with self.assertRaisesRegex(RuntimeError, "isolated environment"):
                    bootstrap.prepare_venv(project)
            builder.assert_not_called()

    def test_symlinked_venv_is_rejected(self):
        """Refuse linked environments before spawning their Python interpreter."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            with patch.object(Path, "is_symlink", return_value=True), patch.object(bootstrap, "run_command") as command:
                with self.assertRaisesRegex(RuntimeError, "symlinked"):
                    bootstrap.prepare_venv(project)
            command.assert_not_called()

    def test_launch_dry_run_is_side_effect_free_and_uses_selected_python(self):
        """The launch plan uses the verified interpreter only after the test command."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            before = set(project.iterdir())
            with patch.object(bootstrap, "check_host") as host, patch.object(bootstrap, "run_command") as command, redirect_stdout(io.StringIO()):
                plan = bootstrap.setup(bootstrap.arguments(["--dry-run", "--launch-pilot"]), project)
            self.assertEqual(set(project.iterdir()), before)
            command.assert_not_called()
            host.assert_not_called()
            expected = [str(bootstrap.venv_python(project / ".venv")), str(project / "run_experiments.py"), "--detach", "--output", str(project / "runs/first_pilot")]
            self.assertEqual(plan["commands"][-1], expected)
            self.assertIn("pytest", plan["commands"][-2])
            self.assertTrue(plan["setup_runs_in_foreground"])

    def test_launch_rejects_skipped_tests_and_unattached_output(self):
        """Reject options that could bypass validation or silently ignore a run path."""
        with redirect_stderr(io.StringIO()):
            for argv in (["--launch-pilot", "--skip-tests"], ["--run-output", "runs/example"]):
                with self.assertRaises(SystemExit):
                    bootstrap.arguments(argv)

    def test_existing_pilot_output_rejected_before_install(self):
        """A repeated launch cannot spend installation time before an inevitable conflict."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            (project / "runs/first_pilot").mkdir(parents=True)
            with patch.object(bootstrap, "check_host") as host, patch.object(bootstrap, "run_command") as command:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    bootstrap.setup(bootstrap.arguments(["--launch-pilot"]), project)
            host.assert_not_called()
            command.assert_not_called()
            self.assertFalse((project / "environment_setup.json").exists())

    def test_pilot_launch_follows_successful_tests_in_project_venv(self):
        """The detached worker receives the verified Python and an absolute output path."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            python = bootstrap.venv_python(project / ".venv")
            commands = []

            def command(argv, project, **kwargs):
                """Record ordering and inspect persisted setup success at launch time."""
                if "--detach" in argv:
                    self.assertTrue(any("pytest" in previous for previous in commands))
                    self.assertEqual(argv[0], str(python))
                    self.assertEqual(argv[-1], str(project / "runs/custom_pilot"))
                    report = json.loads((project / "environment_setup.json").read_text())
                    self.assertEqual(report["setup_status"], "completed")
                    self.assertEqual(report["tests"], "passed")
                commands.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="mock-package==1\n", stderr="")

            with patch.object(bootstrap.platform, "system", return_value="Linux"), patch.object(bootstrap, "check_host", return_value="mock GPU"), patch.object(bootstrap, "prepare_venv", return_value=python), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                report = bootstrap.setup(bootstrap.arguments(["--launch-pilot", "--run-output", "runs/custom_pilot"]), project)
            self.assertEqual(report["pilot_launch"]["status"], "started")
            self.assertEqual(report["setup_status"], "completed")
            self.assertEqual(len([argv for argv in commands if "--detach" in argv]), 1)

    def test_pilot_is_not_launched_after_test_failure(self):
        """A failed setup leaves the requested pilot explicitly unstarted."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            commands = []

            def command(argv, project, **kwargs):
                """Fail the validation gate while recording any attempted launches."""
                commands.append(argv)
                if "pytest" in argv:
                    raise RuntimeError("test gate failed")
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

            with patch.object(bootstrap.platform, "system", return_value="Linux"), patch.object(bootstrap, "check_host", return_value="mock GPU"), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "test gate failed"):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env", "--launch-pilot"]), project)
            self.assertFalse(any("--detach" in argv for argv in commands))
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["pilot_launch"]["status"], "not_started")

    def test_launch_failure_preserves_completed_environment_status(self):
        """A startup problem is recorded separately from a passing environment."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)

            def command(argv, project, **kwargs):
                """Simulate an unconfirmed detached worker after the test gate passes."""
                if "--detach" in argv:
                    raise RuntimeError("worker startup not confirmed")
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

            with patch.object(bootstrap.platform, "system", return_value="Linux"), patch.object(bootstrap, "check_host", return_value="mock GPU"), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "startup not confirmed"):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env", "--launch-pilot"]), project)
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "pilot_launch")
            self.assertEqual(report["setup_status"], "completed")
            self.assertEqual(report["tests"], "passed")
            self.assertEqual(report["pilot_launch"]["status"], "failed")

    def test_interrupt_records_an_interrupted_state(self):
        """Keyboard interruption never leaves a finalized report claiming running tests."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)

            def command(argv, project, **kwargs):
                """Interrupt validation after package verification has completed."""
                if "pytest" in argv:
                    raise KeyboardInterrupt()
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

            with patch.object(bootstrap, "check_host", return_value="mock GPU"), patch.object(bootstrap, "gpu_probe", return_value=self._probe()), patch.object(bootstrap, "run_command", side_effect=command), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env"]), project)
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["tests"], "interrupted")
            self.assertEqual(report["setup_status"], "interrupted")

    def test_failed_cuda_probe_retains_build_metadata(self):
        """A CPU wheel's version is retained without asserting any CUDA test passed."""
        with tempfile.TemporaryDirectory() as directory:
            project = self._project(directory)
            build = {"torch": "2.12.1+cpu", "torchvision": "0.27.1+cpu", "cuda_runtime": None}
            failure = subprocess.CalledProcessError(1, ["python", "-c", bootstrap.GPU_PROBE], output=bootstrap.BUILD_MARKER + json.dumps(build), stderr="CUDA is unavailable")
            with patch.object(bootstrap, "check_host", return_value="mock GPU"), patch.object(bootstrap.subprocess, "run", side_effect=failure), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    bootstrap.setup(bootstrap.arguments(["--use-existing-env"]), project)
            report = json.loads((project / "environment_setup.json").read_text())
            self.assertEqual(report["failed_cuda_probe_build"], build)
            self.assertNotIn("cuda_verification", report)
            self.assertEqual(report["tests"], "not_run")


if __name__ == "__main__":
    unittest.main()
