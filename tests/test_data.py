"""Verify deterministic data isolation, indexed corruption, and offline loading."""

from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from hgfvision.data import (
    CIFAR_MEAN,
    CIFAR_STD,
    CleanDataset,
    FakeCIFAR10,
    Phase,
    SourceDataset,
    StreamDataset,
    load_cifar10,
    make_splits,
)


class DataTests(unittest.TestCase):
    """Check data contracts that prevent evaluation leakage and order effects."""

    def test_default_split_sizes_and_disjointness(self):
        """Default partitions cover every official training index exactly once."""
        splits = make_splits()
        self.assertEqual([len(part) for part in splits.values()], [30_000, 5_000, 10_000, 5_000])
        for left, right in combinations(splits.values(), 2):
            self.assertFalse(set(left) & set(right))
        self.assertEqual(set().union(*map(set, splits.values())), set(range(50_000)))
        self.assertEqual(splits, make_splits())
        self.assertNotEqual(splits, make_splits(seed=1))

    def test_custom_and_small_splits(self):
        """Small fake datasets retain the same isolation guarantees."""
        sizes = {"source": 12, "source_val": 4, "stream": 8, "dev": 6}
        splits = make_splits(n=30, seed=5, sizes=sizes)
        self.assertEqual({key: len(value) for key, value in splits.items()}, sizes)
        self.assertEqual(sum(map(len, make_splits(n=13).values())), 13)
        with self.assertRaises(ValueError):
            make_splits(n=31, sizes=sizes)
        with self.assertRaises(ValueError):
            make_splits(n=30, sizes={"source": 30})

    def test_stream_determinism_across_access_order(self):
        """An indexed sample has identical corruption regardless of access order."""
        base = FakeCIFAR10(size=24, seed=9)
        phases = [Phase("domain", 2, brightness=0.6), Phase("burst", 2, brightness=0.6, noise_std=0.2)]
        stream = StreamDataset(base, list(range(24)), phases, batch_size=3, seed=7)
        first = [stream[index] for index in range(len(stream))]
        for index in reversed(range(len(stream))):
            self.assertTrue(torch.equal(first[index]["x"], stream[index]["x"]))
        same = StreamDataset(base, list(range(24)), phases, batch_size=3, seed=7)
        changed = StreamDataset(base, list(range(24)), phases, batch_size=3, seed=8)
        self.assertTrue(torch.equal(first[7]["x"], same[7]["x"]))
        self.assertFalse(torch.equal(first[7]["x"], changed[7]["x"]))
        self.assertEqual(len(stream), 12)
        self.assertEqual([stream[index]["step"] for index in range(12)], [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
        self.assertEqual([stream[index]["phase"] for index in range(12)], ["domain"] * 6 + ["burst"] * 6)
        self.assertEqual(len({stream[index]["index"] for index in range(12)}), 12)

    def test_source_augmentation_is_indexed_and_epoch_dependent(self):
        """Access order and subset order cannot change a source augmentation."""
        base = FakeCIFAR10(size=16, seed=9)
        dataset = SourceDataset(base, list(range(8)), seed=13)
        original = np.array(base[3][0], copy=True)
        first = [dataset[index][0] for index in range(len(dataset))]
        for index in reversed(range(len(dataset))):
            self.assertTrue(torch.equal(first[index], dataset[index][0]))
            self.assertEqual(tuple(first[index].shape), (3, 32, 32))
            self.assertEqual(first[index].dtype, torch.float32)
            self.assertTrue(torch.isfinite(first[index]).all())
            self.assertTrue(first[index].is_contiguous())
        reordered = SourceDataset(base, [3, 7, 0], seed=13)
        self.assertTrue(torch.equal(first[3], reordered[0][0]))
        dataset.set_epoch(1)
        self.assertTrue(any(not torch.equal(first[index], dataset[index][0]) for index in range(len(dataset))))
        dataset.set_epoch(0)
        self.assertTrue(torch.equal(first[3], dataset[3][0]))
        np.testing.assert_array_equal(np.array(base[3][0]), original)
        with self.assertRaises(ValueError):
            dataset.set_epoch(-1)

    def test_source_augmentation_can_be_disabled(self):
        """Disabling augmentation exactly recovers clean evaluation preprocessing."""
        base = FakeCIFAR10(size=4, seed=3)
        source = SourceDataset(base, [2, 0], seed=17, augment=False)
        clean = CleanDataset(base, [2, 0])
        source.set_epoch(7)
        for index in range(len(source)):
            image, label = source[index]
            clean_image, clean_label = clean[index]
            self.assertTrue(torch.equal(image, clean_image))
            self.assertEqual(label, clean_label)

    def test_stream_validation(self):
        """Fail before adaptation if the schedule would require sample reuse."""
        base = FakeCIFAR10(size=8)
        with self.assertRaisesRegex(ValueError, "requires 12 unique"):
            StreamDataset(base, list(range(8)), [Phase("clean", 3)], batch_size=4)
        with self.assertRaisesRegex(ValueError, "unique"):
            StreamDataset(base, [0, 0], [Phase("clean", 1)], batch_size=2)
        with self.assertRaises(ValueError):
            StreamDataset(base, [8], [Phase("clean", 1)], batch_size=1)
        with self.assertRaises(ValueError):
            StreamDataset(base, [0], [], batch_size=1)
        with self.assertRaises(ValueError):
            Phase("invalid", 1, noise_std=float("nan"))
        with self.assertRaises(ValueError):
            Phase("invalid", 0)

    def test_transforms_do_not_mutate_source(self):
        """PIL, array and tensor sources remain intact after noisy stream access."""
        source_array = np.full((32, 32, 3), 80, dtype=np.uint8)
        sources = [Image.fromarray(source_array), source_array.copy(), torch.full((3, 32, 32), 80, dtype=torch.uint8)]
        for source in sources:
            before = np.array(source, copy=True) if not isinstance(source, torch.Tensor) else source.clone()
            base = [(source, 2)]
            stream = StreamDataset(base, [0], [Phase("burst", 1, brightness=0.5, contrast=1.8, noise_std=0.4)], batch_size=1)
            stream[0]
            clean, _ = CleanDataset(base)[0]
            self.assertEqual(tuple(clean.shape), (3, 32, 32))
            if isinstance(source, torch.Tensor):
                self.assertTrue(torch.equal(source, before))
            else:
                np.testing.assert_array_equal(np.array(source), before)

    def test_clean_normalization_and_stream_identity(self):
        """A clean regime exactly agrees with held-out preprocessing."""
        base = [(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)), 4)]
        clean, label = CleanDataset(base)[0]
        stream = StreamDataset(base, [0], [Phase("clean", 1)], batch_size=1)
        self.assertEqual(label, 4)
        self.assertEqual(clean.dtype, torch.float32)
        self.assertTrue(torch.equal(clean, stream[0]["x"]))
        expected = -torch.tensor(CIFAR_MEAN) / torch.tensor(CIFAR_STD)
        torch.testing.assert_close(clean[:, 0, 0], expected)
        self.assertEqual(set(stream[0]), {"x", "y", "index", "phase", "step"})
        with self.assertRaises(IndexError):
            stream[1]

    def test_fake_data_is_deterministic(self):
        """Smoke-test data is reproducible and explicitly synthetic."""
        first = FakeCIFAR10(size=20, seed=11)
        second = FakeCIFAR10(size=20, seed=11)
        self.assertEqual({first[index][1] for index in range(20)}, set(range(10)))
        np.testing.assert_array_equal(np.array(first[12][0]), np.array(second[12][0]))

    def test_download_requires_explicit_flag(self):
        """The default loader requests local files and retains separate test data."""
        with patch("torchvision.datasets.CIFAR10") as dataset:
            load_cifar10("unused")
            self.assertEqual(dataset.call_count, 2)
            self.assertEqual([call.kwargs["train"] for call in dataset.call_args_list], [True, False])
            self.assertTrue(all(call.kwargs["download"] is False for call in dataset.call_args_list))
            self.assertTrue(all(call.kwargs["transform"] is None for call in dataset.call_args_list))
            dataset.reset_mock()
            load_cifar10("unused", download=True)
            self.assertTrue(all(call.kwargs["download"] is True for call in dataset.call_args_list))


if __name__ == "__main__":
    unittest.main()
