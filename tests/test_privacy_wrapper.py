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
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)


def test_forward():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    output = wrapped(data)
    assert output.shape == (2, 1, 1)


def test_clip_accum():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()


def test_noise_insecure():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0, secure_rng=False, seed=None)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_noise_insecure_seed():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0, secure_rng=False, seed=42)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_noise_secure():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0, secure_rng=True)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    wrapped.noise_gradient()


def test_noise_mean():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1000, 1e-12, secure_rng=True)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    accumulated_grads_pre = torch.cat(
        [
            param.accumulated_gradients.mean(dim=0).flatten()
            for param in wrapped.wrapped_model.parameters()
        ]
    )
    wrapped.noise_gradient(reduce="mean")
    grads_post = torch.cat(
        [param.grad.flatten() for param in wrapped.wrapped_model.parameters()]
    )
    assert torch.allclose(accumulated_grads_pre, grads_post)


def test_noise_sum():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1000, 1e-12, secure_rng=True)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    accumulated_grads_pre = torch.cat(
        [
            param.accumulated_gradients.sum(dim=0).flatten()
            for param in wrapped.wrapped_model.parameters()
        ]
    )
    wrapped.noise_gradient(reduce="sum")
    grads_post = torch.cat(
        [param.grad.flatten() for param in wrapped.wrapped_model.parameters()]
    )
    assert torch.allclose(accumulated_grads_pre, grads_post)


def test_raise_reduce_error():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1000, 1e-12, secure_rng=True)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    with pytest.raises(ValueError):
        wrapped.noise_gradient(reduce="foo")


def test_next_batch():
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
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
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
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
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    for _ in range(5):
        output = wrapped(data)
        loss = output.mean()
        loss.backward()
        wrapped.clip_and_accumulate()
        wrapped.noise_gradient()
        wrapped.prepare_next_batch()
    assert wrapped._steps_taken == 5


def test_in_order():
    """Case 1: forward not called before clip"""
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    with pytest.raises(RuntimeError):
        wrapped.clip_and_accumulate()

    """Case 2: clip not called before noise"""
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    output = wrapped(data)
    with pytest.raises(RuntimeError):
        wrapped.noise_gradient()

    """Case 3: noise not called before prepare """
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    output = wrapped(data)
    loss = output.mean()
    loss.backward()
    wrapped.clip_and_accumulate()
    with pytest.raises(RuntimeError):
        wrapped.prepare_next_batch()


def test_raises_param_error():
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0)
    with pytest.raises(ValueError):
        params = wrapped.parameters()


def test_check_device_cpu():
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0).to("cpu")
    assert (
        next(
            iter(
                set([param.device.type for param in wrapped.wrapped_model.parameters()])
            )
        )
        == "cpu"
    )
    for model in wrapped.models:
        assert (
            next(iter(set([param.device.type for param in model.parameters()])))
            == "cpu"
        )


def test_check_device_gpu():
    if torch.cuda.is_available():
        wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0).to("cuda")
        assert "cuda" in next(
            iter(
                set([param.device.type for param in wrapped.wrapped_model.parameters()])
            )
        )
        for model in wrapped.models:
            assert "cuda" in next(
                iter(set([param.device.type for param in model.parameters()]))
            )
    else:
        pass


def test_raises_rng_collision():
    with pytest.raises(ValueError):
        wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0, secure_rng=True, seed=42)