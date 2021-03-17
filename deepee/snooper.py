import torch
from typing import Callable


class BadModuleError(Exception):
    """Informs the user that the module has some kinda problem.
    """
    pass


class ModelSnooper:
    def __init__(self) -> None:
        """The ModelSnooper checks the model for common problems in DP deep learning.
        In general, all layers which maintain state based on multiple samples from a batch
        are not natively supported since the state is calculated non-privately.   
        These include:
        - Having vanilla BatchNorm layers in the model unless the track_running_stats
        attribute is deactivated. In PyTorch, this turns them into InstanceNorm layers.
        - Having InstanceNorm layers which track running statistics.
        """
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
        """Private class for use in the ModelSnooper. Takes a validator function
        and applies it to the model's modules. If the function fails, it returns
        an exception message for the user.

        Args:
            name (str): Human-readable description. Not used elsewhere.
            val_func (Callable): The function which validates the model. Should
            return False if it fails
            message (str): The error message which is passed to the ModelSnooper
            to be shown to the user.
        """
        self.name = name
        self.val_func = val_func
        self.message = message

    def validate(self, model: torch.nn.Module) -> str:
        """Run the validators on the model. Returns an error
        message if one of the validators fails or an empty string otherwise.
        """
        if not self.val_func(model):
            return self.message
        return ""


def IN_running_stats_off(model: torch.nn.Module) -> bool:
    """Module has InstanceNorm layer with running stats
    activated.
    """
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d),
        ):
            if module.track_running_stats:
                return False
    return True


def BN_running_stats_off(model: torch.nn.Module) -> bool:
    """Module contains BatchNorm layer with running stats
    activated.
    """
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        ):
            if module.track_running_stats:
                return False
    return True

