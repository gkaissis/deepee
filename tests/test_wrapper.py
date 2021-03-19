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


def test_noise_insecure():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0, secure_rng=False, seed=None)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_noise_insecure_seed():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0, secure_rng=False, seed=42)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_noise_secure():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0, secure_rng=True)
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
    main_params = list(wrapped.wrapped_model.parameters())
    copy_1_params = list(wrapped.models[0].parameters())
    copy_2_params = list(wrapped.models[1].parameters())
    for mp, c1, c2 in zip(main_params, copy_1_params, copy_2_params):
        assert (mp == c1).all() and (mp == c2).all() and (c1 == c2).all()


def test_verification_and_steps_taken():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    assert wrapped._steps_taken == 0
    assert (
        wrapped._forward_succesful
        == wrapped._noise_succesful
        == wrapped._clip_succesful
        == False
    )
    output = wrapped(data)
    assert wrapped._forward_succesful == True
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    assert wrapped._clip_succesful == True
    wrapped.noise_gradient()
    assert wrapped._noise_succesful == True
    wrapped.prepare_next_batch()
    assert wrapped._steps_taken == 1
    with pytest.raises(RuntimeError):
        wrapped.prepare_next_batch()  # call a second time to raise error
    assert wrapped._steps_taken == 1  # steps should still be 1


def test_steps_taken():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    for _ in range(5):
        output = wrapped(data)
        loss = output.mean()
        loss.backward()
        wrapped.clip_and_accumulate()
        wrapped.noise_gradient()
        wrapped.prepare_next_batch()
    assert wrapped._steps_taken == 5


def test_raises_param_error():
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0)
    with pytest.raises(ValueError):
        params = wrapped.parameters()


def test_check_device():
    wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0).to("cpu")
    assert wrapped.device == "cpu"
    for model in wrapped.models:
        assert list(model.parameters())[0].device.type == "cpu"
    if torch.cuda.is_available():
        wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0).to("cuda")
        assert "cuda" in wrapped.device
        for model in wrapped.models:
            assert "cuda" in list(model.parameters())[0].device.type


def test_raises_rng_collision():
    with pytest.raises(ValueError):
        wrapped = PrivacyWrapper(MiniModel, 2, 1.0, 1.0, secure_rng=True, seed=42)
