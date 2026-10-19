"""Validate adapter identity, parameter isolation, and source checkpoint loading."""

import unittest

import torch
from torch import nn

from hgfvision.models import (
    build_model,
    freeze_for_adaptation,
    freeze_for_source,
    load_source_weights,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Keep small CPU checks from oversubscribing the shared test host."""
        cls.previous_threads = torch.get_num_threads()
        cls.previous_rng = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(123)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's CPU thread setting after the model checks."""
        torch.set_num_threads(cls.previous_threads)
        torch.set_rng_state(cls.previous_rng)

    def test_initial_adapters_preserve_source_outputs_and_base_keys(self):
        source = build_model(adapters=False).eval()
        adapted = build_model().eval()
        load_source_weights(adapted, source.state_dict())
        base_keys = {key for key in adapted.state_dict() if not key.startswith("adapters.")}
        assert base_keys == set(source.state_dict())
        inputs = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = source(inputs)
            actual = adapted(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (3, 10)
        assert adapted.conv1.kernel_size == (3, 3)
        assert adapted.conv1.stride == (1, 1)
        assert isinstance(adapted.maxpool, nn.Identity)


    def test_adapter_parameter_count_and_groups(self):
        model = freeze_for_adaptation(build_model())
        parameters = list(model.adapter_parameters())
        groups = model.parameter_groups()
        assert len(groups) == 2
        assert [sum(parameter.numel() for parameter in group) for group in groups] == [4096, 4096]
        assert sum(parameter.numel() for parameter in parameters) == 8192
        assert [id(parameter) for group in groups for parameter in group] == [
            id(parameter) for parameter in parameters
        ]
        assert {id(parameter) for parameter in model.parameters() if parameter.requires_grad} == {
            id(parameter) for parameter in parameters
        }
        for adapter in model.adapters:
            assert adapter.down.bias is None
            assert adapter.up.bias is None
            assert not torch.count_nonzero(adapter.up.weight)


    def test_adaptation_preserves_backbone_bn_and_propagates_adapter_gradients(self):
        model = freeze_for_adaptation(build_model())
        model.train()
        frozen_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("adapters.")
        }
        assert all(not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
        optimizer = torch.optim.SGD(model.adapter_parameters(), lr=0.1)
        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        loss = nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()
        for adapter in model.adapters:
            assert adapter.down.weight.grad is not None
            assert torch.count_nonzero(adapter.down.weight.grad) == 0
            assert adapter.up.weight.grad is not None
            assert torch.isfinite(adapter.up.weight.grad).all()
            assert torch.count_nonzero(adapter.up.weight.grad) > 0
        assert all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith("adapters.")
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), labels).backward()
        for adapter in model.adapters:
            assert torch.isfinite(adapter.down.weight.grad).all()
            assert torch.count_nonzero(adapter.down.weight.grad) > 0
        for key, expected in frozen_state.items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
        model.eval()
        assert not model.training
        model.train()
        assert all(not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))


    def test_source_mode_bypasses_adapters_and_restores_bn_training(self):
        source = build_model(adapters=False).eval()
        model = build_model()
        load_source_weights(model, source.state_dict())
        freeze_for_adaptation(model)
        with torch.no_grad():
            for adapter in model.adapters:
                adapter.up.weight.fill_(0.2)
        freeze_for_source(model)
        model.train()
        assert all(module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
        assert all(not parameter.requires_grad for parameter in model.adapter_parameters())
        assert all(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith("adapters.")
        )
        model.eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), source(inputs), rtol=0, atol=0)


    def test_source_loading_rejects_invalid_checkpoints_without_loading(self):
        for defect in ["missing", "unexpected", "shape", "adapter"]:
            with self.subTest(defect=defect):
                source_state = build_model(adapters=False).state_dict()
                model = build_model()
                initial_stem = model.conv1.weight.detach().clone()
                if defect == "missing":
                    del source_state["fc.bias"]
                elif defect == "unexpected":
                    source_state["unknown.weight"] = torch.zeros(1)
                elif defect == "shape":
                    source_state["fc.bias"] = torch.zeros(11)
                else:
                    source_state["adapters.0.up.weight"] = model.adapters[0].up.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "source checkpoint|Source checkpoint"):
                    load_source_weights(model, source_state)
                torch.testing.assert_close(model.conv1.weight, initial_stem, rtol=0, atol=0)


    def test_source_loading_resets_adapter_outputs_and_supports_source_model(self):
        source = build_model(num_classes=3, adapters=False)
        model = build_model(num_classes=3, rank=2)
        with torch.no_grad():
            for adapter in model.adapters:
                adapter.up.weight.fill_(1.0)
        load_source_weights(model, source.state_dict())
        assert all(torch.count_nonzero(adapter.up.weight) == 0 for adapter in model.adapters)
        target = build_model(num_classes=3, adapters=False)
        load_source_weights(target, source.state_dict())
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


    def test_source_only_model_cannot_silently_adapt(self):
        model = build_model(adapters=False)
        assert list(model.adapter_parameters()) == []
        assert model.parameter_groups() == []
        with self.assertRaisesRegex(ValueError, "adapters=True"):
            freeze_for_adaptation(model)


if __name__ == "__main__":
    unittest.main()
