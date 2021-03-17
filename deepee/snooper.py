import torch
from typing import Callable


class BadModuleError(Exception):
    pass


class ModelSnooper:
    def __init__(self) -> None:
        self.validators = [
            Validator(
                "InstanceNorm Running Stats Off",
                IN_running_stats_off,
                "InstanceNorm Layers must have track_running_stats turned off. ",
            ),
            Validator(
                "BatchNorm Running Stats Off",
                BN_running_stats_off,
                "BatchNorm Layers must have track_running_stats turned off, otherwise be replaced with InstanceNorm, LayerNorm or GroupNorm. ",
            ),
        ]

    def snoop(self, model: torch.nn.Module) -> None:
        msg = ""
        for validator in self.validators:
            msg += validator.validate(model)
        if msg != "":
            raise BadModuleError(msg)


class Validator:
    def __init__(self, name: str, val_func: Callable, message: str) -> None:
        self.name = name
        self.val_func = val_func
        self.message = message

    def validate(self, model: torch.nn.Module) -> None:
        if not self.val_func(model):
            return self.message
        return ""


def IN_running_stats_off(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d),
        ):
            if module.track_running_stats:
                return False
    return True


def BN_running_stats_off(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        ):
            if module.track_running_stats:
                return False
    return True


def training_(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if module.train:
            return
    return True