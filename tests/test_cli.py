"""Cheap CLI provenance and split-routing checks without model training."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hgfvision.cli import adapt, read_config
from hgfvision.data import FakeCIFAR10, make_splits
from hgfvision.engine import json_hash


class CLIDataContractTests(unittest.TestCase):
    """Exercise orchestration with mocked model work and real split bookkeeping."""

    def _checkpoint(self, directory: Path, corrupt_hash: bool = False) -> Path:
        """Create a tiny synthetic checkpoint carrying real split metadata."""
        splits = make_splits(100, seed=3)
        hashes = {key: json_hash(value) for key, value in splits.items()}
        if corrupt_hash:
            hashes["stream"] = "invalid"
        metadata = {
            "dataset": "fake_smoke_only",
            "architecture": "cifar_resnet18",
            "num_classes": 10,
            "dataset_length": 100,
            "split_seed": 3,
            "split_hashes": hashes,
        }
        checkpoint = directory / "source.pt"
        torch.save({"model": {}, "metadata": metadata}, checkpoint)
        return checkpoint

    def _args(self, directory: Path, checkpoint: Path, partition: str) -> argparse.Namespace:
        """Supply valid CLI-like arguments while keeping all outputs temporary."""
        return argparse.Namespace(
            device="cpu", checkpoint=checkpoint, config=None,
            data_root=directory / "data", download=False, partition=partition,
            order_seed=4, noise_seed=5, workers=0, eval_batch_size=8,
            output=directory / f"adapt_{partition}", method="frozen", seed=6,
        )

    def test_config_rejects_unknown_keys_and_oversized_observations(self):
        """Catch misspellings and unsupported observation dimensions before data I/O."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for config in ({"rnak": 4}, {"filter": {"not_a_parameter": 1}}, {"batch_size": 33}):
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_fake_checkpoint_rejected_before_real_data_load(self):
        """An offline smoke checkpoint cannot silently enter a CIFAR experiment."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            args = self._args(directory, checkpoint, "stream")
            with patch("hgfvision.cli.load_cifar10") as load_data:
                with self.assertRaisesRegex(ValueError, "Checkpoint dataset"):
                    adapt(args, fake=False)
                load_data.assert_not_called()
            self.assertFalse(Path(args.output).exists())

    def test_split_hash_mismatch_rejected(self):
        """Reject checkpoints whose recorded split assignment cannot be reproduced."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory, corrupt_hash=True)
            args = self._args(directory, checkpoint, "dev")
            with patch("hgfvision.cli.build_model") as model:
                with self.assertRaisesRegex(ValueError, "splits differ"):
                    adapt(args, raw_train=FakeCIFAR10(100), raw_test=FakeCIFAR10(16), fake=True)
                model.assert_not_called()
            self.assertFalse(Path(args.output).exists())

    def test_dev_and_confirmatory_routing(self):
        """Development never evaluates test images; confirmatory retention does."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            raw_train, raw_test = FakeCIFAR10(100, seed=11), FakeCIFAR10(16, seed=12)
            splits = make_splits(100, seed=3)
            config = read_config(None)
            config["batch_size"] = 2
            config["phases"] = [{"name": "clean", "steps": 2}]
            captures = []

            def capture(updater, loader, output, clean_loader, metadata):
                """Inspect constructed datasets without running a model or updater."""
                captures.append((loader.dataset, clean_loader.dataset, metadata))
                return {"status": "completed", "method": metadata["method"]}

            with patch("hgfvision.cli.build_model") as build, patch("hgfvision.cli.load_source_weights"), patch("hgfvision.cli.AdapterUpdater"), patch("hgfvision.cli.run_stream", side_effect=capture), redirect_stdout(io.StringIO()):
                build.return_value.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
                for partition in ("dev", "stream"):
                    adapt(
                        self._args(directory, checkpoint, partition),
                        raw_train=raw_train, raw_test=raw_test, fake=True,
                        config_override=config,
                    )
            for partition, (stream, clean, metadata) in zip(("dev", "stream"), captures):
                self.assertTrue(set(stream.indices) <= set(splits[partition]))
                self.assertFalse(set(stream.indices) & set(splits["source"]))
                self.assertEqual(metadata["stream_indices_sha256"], json_hash(list(stream.indices)))
                self.assertEqual(metadata["dataset"], "fake_smoke_only")
                saved = json.loads((directory / f"adapt_{partition}" / "stream_indices.json").read_text())
                self.assertEqual(saved, list(stream.indices))
                if partition == "dev":
                    self.assertIs(clean.base_dataset, raw_train)
                    self.assertEqual(list(clean.indices), splits["source_val"])
                    self.assertEqual(metadata["clean_evaluation_split"], "source_val")
                else:
                    self.assertIs(clean.base_dataset, raw_test)
                    self.assertEqual(list(clean.indices), list(range(16)))
                    self.assertEqual(metadata["clean_evaluation_split"], "fake_holdout")
            self.assertFalse(set(captures[0][0].indices) & set(captures[1][0].indices))


if __name__ == "__main__":
    unittest.main()
