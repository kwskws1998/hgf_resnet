"""CIFAR ResNet-18 with isolated, linear adapters for online adaptation."""

from collections.abc import Iterator, Mapping

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int, rank: int) -> None:
        """Create an identity-initialized, bias-free low-rank 1x1 residual map."""
        super().__init__()
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.up(self.down(inputs))


class CIFARResNet18(nn.Module):
    def __init__(
        self, num_classes: int = 10, rank: int = 4, adapters: bool = True
    ) -> None:
        """Preserve torchvision base keys while adding final-stage adapters.

        The 3x3, stride-one stem and absent max-pooling suit 32x32 CIFAR images.
        Adapters follow the two layer4 residual blocks. Calling
        ``freeze_for_adaptation`` makes subsequent ``train()`` calls retain
        frozen BatchNorm statistics. Construction never downloads weights.
        """
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        backbone = resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        backbone.maxpool = nn.Identity()
        for name, module in backbone.named_children():
            self.add_module(name, module)
        self.adapters = nn.ModuleList(
            [ResidualAdapter(512, rank) for _ in range(2)] if adapters else []
        )
        self._adapters_enabled = adapters
        self._adaptation_mode = False

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        for index, block in enumerate(self.layer4):
            features = block(features)
            if self._adapters_enabled:
                features = self.adapters[index](features)
        return self.fc(torch.flatten(self.avgpool(features), 1))

    def train(self, mode: bool = True) -> "CIFARResNet18":
        """Set training mode without changing frozen BatchNorm statistics."""
        super().train(mode)
        if self._adaptation_mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """Yield adapter parameters in block order, independently of freeze mode."""
        yield from self.adapters.parameters()

    def parameter_groups(self) -> list[list[nn.Parameter]]:
        """Return one parameter list per adapter for layerwise uncertainty states."""
        return [list(adapter.parameters()) for adapter in self.adapters]


def build_model(
    num_classes: int = 10, rank: int = 4, adapters: bool = True
) -> CIFARResNet18:
    """Create a randomly initialized source or adapted CIFAR ResNet-18."""
    return CIFARResNet18(num_classes=num_classes, rank=rank, adapters=adapters)


def freeze_for_adaptation(model: CIFARResNet18) -> CIFARResNet18:
    """Enable and train only adapters; freeze the classifier and all BN state."""
    if not model.adapters:
        raise ValueError("Adaptation requires a model constructed with adapters=True")
    model.requires_grad_(False)
    for parameter in model.adapter_parameters():
        parameter.requires_grad_(True)
    model._adapters_enabled = True
    model._adaptation_mode = True
    model.train(model.training)
    return model


def freeze_for_source(model: CIFARResNet18) -> CIFARResNet18:
    """Train the backbone and classifier while bypassing and freezing adapters.

    Existing adapter weights are preserved but cannot affect source training.
    Prefer ``build_model(adapters=False)`` for adapter-free source checkpoints.
    The current train/eval mode is retained; ``train()`` re-enables BN updates.
    """
    model._adapters_enabled = False
    model._adaptation_mode = False
    model.requires_grad_(True)
    for parameter in model.adapter_parameters():
        parameter.requires_grad_(False)
    model.train(model.training)
    return model


def load_source_weights(
    model: CIFARResNet18, state_dict: Mapping[str, Tensor]
) -> CIFARResNet18:
    """Load a complete adapter-free source state and reset adapter outputs.

    Missing, extra, and incorrectly shaped base tensors raise before loading.
    Adapter-containing checkpoints are deliberately rejected; resuming an
    adaptation checkpoint should use ``load_state_dict`` instead. Loading does
    not change the model's train/eval or parameter-freezing mode.
    """
    target_state = model.state_dict()
    adapter_keys = {key for key in target_state if key.startswith("adapters.")}
    base_keys = set(target_state) - adapter_keys
    missing = sorted(base_keys - set(state_dict))
    unexpected = sorted(set(state_dict) - base_keys)
    if missing or unexpected:
        raise ValueError(
            f"Invalid source checkpoint: missing keys={missing}; unexpected keys={unexpected}"
        )
    for key in sorted(base_keys):
        value = state_dict[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Source checkpoint value {key!r} must be a tensor")
        if value.shape != target_state[key].shape:
            raise ValueError(
                f"Source checkpoint shape mismatch for {key}: "
                f"expected {tuple(target_state[key].shape)}, got {tuple(value.shape)}"
            )
    result = model.load_state_dict(state_dict, strict=False)
    if set(result.missing_keys) != adapter_keys or result.unexpected_keys:
        raise RuntimeError("Source checkpoint did not match the validated model state")
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.up.weight.zero_()
    return model
