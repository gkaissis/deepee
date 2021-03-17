from deepee import PrivacyWrapper
import torch
import pytest


class MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x)


def test_wrap():
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)


def test_forward():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    output = wrapped(data)
    assert output.shape == (2, 1, 1)


def test_clip_accum():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()


def test_noise():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_next_batch():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()
    wrapped.prepare_next_batch()
    main_params = list(wrapped.model.parameters())
    copy_1_params = list(wrapped.models[0].parameters())
    copy_2_params = list(wrapped.models[1].parameters())
    for mp, c1, c2 in zip(main_params, copy_1_params, copy_2_params):
        assert (mp == c1).all() and (mp == c2).all() and (c1 == c2).all()


def test_raises_param_error():
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    with pytest.raises(ValueError):
        params = wrapped.parameters()
