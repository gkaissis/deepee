from deepee import ModelSnooper
from deepee.snooper import BadModuleError
import torch
import pytest


class Model_with_BN(torch.nn.Module):
    """Model contains regular BN"""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm1d(2)
        self.norm2 = torch.nn.BatchNorm2d(2)
        self.norm3 = torch.nn.BatchNorm3d(2)

    def forward(self, x):
        pass


class Model_with_BN_running_stats_off(torch.nn.Module):
    """Model contains BN but with running stats off"""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm1d(2, track_running_stats=False)
        self.norm2 = torch.nn.BatchNorm2d(2, track_running_stats=False)
        self.norm3 = torch.nn.BatchNorm3d(2, track_running_stats=False)

    def forward(self, x):
        pass


class Model_with_IN(torch.nn.Module):
    """Model contains regular InstanceNorm"""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.InstanceNorm1d(2)
        self.norm2 = torch.nn.InstanceNorm2d(2)
        self.norm3 = torch.nn.InstanceNorm3d(2)

    def forward(self, x):
        pass


class Model_with_IN_running_stats_on(torch.nn.Module):
    """Model contains IN but with running stats on"""

    """THIS IS THE OTHER WAY AROUND THAN BATCH NORM!!!"""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.InstanceNorm1d(2, track_running_stats=True)
        self.norm2 = torch.nn.InstanceNorm2d(2, track_running_stats=True)
        self.norm3 = torch.nn.InstanceNorm3d(2, track_running_stats=True)

    def forward(self, x):
        pass


snooper = ModelSnooper()


def test_bn_incompatible():
    with pytest.raises(BadModuleError):
        snooper.snoop(Model_with_BN())


def test_bn_without_running_stats():
    snooper.snoop(Model_with_BN_running_stats_off())


def test_IN_regular():
    snooper.snoop(Model_with_IN())


def test_IN_with_running_stats():
    with pytest.raises(BadModuleError):
        snooper.snoop(Model_with_IN_running_stats_on())
