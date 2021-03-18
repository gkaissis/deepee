from torchvision.models import resnet18
from torch import nn
from deepee import ModelSurgeon, SurgicalProcedures
from functools import partial
from copy import deepcopy


def test_bn_to_bn_nostats():
    model = resnet18()
    surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
    converted_model = surgeon.operate(model)
    assert converted_model.bn1.track_running_stats == False


def test_bn_to_gn_default():
    model = resnet18()
    surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
    converted_model = surgeon.operate(model)
    assert isinstance(converted_model.bn1, nn.modules.normalization.GroupNorm)
    assert converted_model.bn1.num_groups == 32


def test_bn_to_gn_96():
    model = resnet18()
    surgeon = ModelSurgeon(partial(SurgicalProcedures.BN_to_GN, num_groups=96))
    converted_model = surgeon.operate(model)
    assert isinstance(converted_model.bn1, nn.modules.normalization.GroupNorm)
    assert converted_model.bn1.num_groups == 96


def test_bn_to_in():
    model = resnet18()
    surgeon = ModelSurgeon(SurgicalProcedures.BN_to_IN)
    converted_model = surgeon.operate(model)
    assert isinstance(converted_model.bn1, nn.modules.instancenorm._InstanceNorm)


def test_bn_to_ln():
    model = resnet18()
    surgeon = ModelSurgeon(partial(SurgicalProcedures.BN_to_LN, normalized_shape=16))
    converted_model = surgeon.operate(model)
    assert isinstance(converted_model.bn1, nn.modules.normalization.LayerNorm)
    assert converted_model.bn1.normalized_shape == (16,)