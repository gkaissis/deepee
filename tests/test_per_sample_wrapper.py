from deepee import PerSampleGradientWrapper
import torch
import pytest


class MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x)


def test_wrap():
    wrapped = PerSampleGradientWrapper(MiniModel, 2)


def test_forward():
    data = torch.randn(2, 1, 10)
    wrapped = PerSampleGradientWrapper(MiniModel, 2)
    output = wrapped(data)
    assert output.shape == (2, 1, 1)


def test_raises_param_error():
    wrapped = PerSampleGradientWrapper(MiniModel, 2)
    with pytest.raises(ValueError):
        params = wrapped.parameters()


def test_check_device_cpu():
    wrapped = PerSampleGradientWrapper(MiniModel, 2).to("cpu")
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
        wrapped = PerSampleGradientWrapper(MiniModel, 2).to("cuda")
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


def test_per_sample_grads():
    torch.manual_seed(42)
    data = torch.randn(2, 1, 10)
    torch.manual_seed(42)
    wrapped = PerSampleGradientWrapper(MiniModel, 2)
    torch.manual_seed(42)
    model = MiniModel()  # single copy
    output_single = model(data)
    output_wrapped = wrapped(data)
    loss_single = output_single.mean()
    loss_wrapped = output_wrapped.mean()

    loss_single.backward()
    loss_wrapped.backward()
    wrapped.calculate_per_sample_gradients()
    single_grads = torch.cat([param.grad.flatten() for param in model.parameters()])
    accumulated_grads = torch.cat(
        [
            param.accumulated_gradients.sum(dim=0).flatten()
            for param in wrapped.wrapped_model.parameters()
        ]
    )
    assert torch.allclose(single_grads, accumulated_grads)
