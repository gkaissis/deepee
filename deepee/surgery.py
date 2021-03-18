r"""
Several functions in this module reused directly from 
https://github.com/pytorch/opacus/blob/master/opacus/utils/module_modification.py
under Apache-2.0 License terms.
"""
from typing import Callable, Type, Union
import torch
from torch import nn


def _replace_child(
    root: nn.Module, child_name: str, converter: Callable[[nn.Module], nn.Module]
) -> None:
    parent = root
    nameList = child_name.split(".")
    for name in nameList[:-1]:
        parent = parent._modules[name]
    # set to identity
    parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])


def replace_all_modules(
    root: nn.Module,
    target_class: Type[nn.Module],
    converter: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if isinstance(root, target_class):
        return converter(root)
    for name, obj in root.named_modules():
        if isinstance(obj, target_class):
            _replace_child(root, name, converter)
    return root


bn_to_in = {
    nn.BatchNorm1d: nn.InstanceNorm1d,
    nn.BatchNorm2d: nn.InstanceNorm2d,
    nn.BatchNorm3d: nn.InstanceNorm3d,
}

in_to_in = {
    nn.InstanceNorm1d: nn.InstanceNorm1d,
    nn.InstanceNorm2d: nn.InstanceNorm2d,
    nn.InstanceNorm3d: nn.InstanceNorm3d,
}

bn_to_bn_nostats = {
    nn.BatchNorm1d: nn.BatchNorm1d,
    nn.BatchNorm2d: nn.BatchNorm2d,
    nn.BatchNorm3d: nn.BatchNorm3d,
}


class SurgicalProcedures:
    @staticmethod
    def BN_to_IN(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
        """BatchNorm to InstanceNorm"""
        return bn_to_in.get(type(module))(
            module.num_features, track_running_stats=False
        )

    @staticmethod
    def BN_to_BN_nostats(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
        """BatchNorm to BatchNorm without running stats (=InstanceNorm)"""
        return bn_to_bn_nostats.get(type(module))(
            module.num_features, track_running_stats=False
        )

    @staticmethod
    def IN_to_IN_nostats(module: nn.modules.instancenorm._InstanceNorm) -> nn.Module:
        """InstanceNorm to InstanceNorm (without running stats)"""
        return in_to_in.get(type(module))(
            module.num_features, track_running_stats=False
        )

    @staticmethod
    def BN_to_GN(
        module: nn.modules.batchnorm._BatchNorm, num_groups: Union[str, int] = "default"
    ) -> nn.Module:
        """BatchNorm to GroupNorm"""
        if num_groups == "default":
            return nn.GroupNorm(min(32, module.num_features), module.num_features)
        elif isinstance(num_groups, int):
            return nn.GroupNorm(num_groups, module.num_features)
        else:
            raise ValueError(
                "num_groups must either be set to 'default' or the number of groups"
            )

    @staticmethod
    def BN_to_LN(
        module: nn.Module,
        normalized_shape: Union[int, list, torch.Size],
    ) -> nn.Module:
        """BatchNorm to LayerNorm"""
        return nn.LayerNorm(normalized_shape=normalized_shape)


class ModelSurgeon:
    def __init__(self, converter: Callable):
        """Convenience class to replace privacy-incompatible
        normalisation layers (most commonly BatchNorm) in the model.
        Can be used to remedy BadModelError exceptions thrown by the
        ModelSnooper.

        Args:
            converter (Callable): The converter to be used. Options:
            - BN_to_BN_nostats : BatchNorm to BatchNorm without running stats.
            This is equivalent to InstanceNorm.
            - BN_to_IN : BatchNorm to InstanceNorm
            - BN_to_GN : BatchNorm to GroupNorm. By default will use min(32, model.num_features)
            groups. A different value can be bound to the function using functools.partial.
            - BN_to_LN : BatchNorm to LayerNorm. Requires a normalized_shape value to be passed
            using functools partial.
            - IN_to_IN_nostats : InstanceNorm with running stats to InstanceNorm without running stats.

        Example use:
        >> from functools import partial
        >> from deepee.surgery import BN_to_GN
        >> model = ... [Throws BadModuleError when the PrivacyWrapper is attached]
        >> surgeon = ModelSurgeon(partial(BN_to_GN, num_channels=32))
        >> converted_model = surgeon.operate(model)
        """
        self.converter = converter

    def operate(self, model: nn.Module) -> nn.Module:
        """Convert the model based on the defined converter."""
        return replace_all_modules(
            model,
            target_class=nn.modules.batchnorm._BatchNorm,
            converter=self.converter,
        )
