import torch
from torch import nn
from copy import deepcopy
from typing import Optional, Any
from .snooper import ModelSnooper


class PrivacyWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_replicas: int,
        L2_clip: float,
        noise_multiplier: float,
        **kwargs: Optional[Any],
    ) -> None:
        """Factory class which wraps any model and returns a model suitable for DP-SGD learning.
        The class will replicate the model in a memory-efficient way and run the forward and
        backward passes in parallel over the inputs.

        Args:
            base_model (nn.Module): The model to wrap. Must be a class name, not a model instance.
            num_replicas (int): How many times to replicate the model. Must be set equal to the batch size.
            L2_clip (float): Clipping norm for the DP-SGD procedure.
            noise_multiplier (float): Noise multiplier for the DP-SGD procedure.

        For more information on L2_clip and noise_multiplier, see Abadi et al., 2016.

        The model is compatible with any first-order optimizer (SGD, Adam, etc.) without any modifications
        to the optimizer itself.

        Sample use:

        >> model = DPWrapper(resnet18, num_replicas=64, L2_clip=1., noise_multiplier=1.)
        >> optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)
        >> y_pred = model(data)
        >> loss = criterion(y_pred, y_true)
        >> loss.backward()
        >> model.clip_and_accumulate()
        >> model.noise_gradient()
        >> optimizer.step()
        >> model.prepare_next_batch()
        >> ...(repeat)
        """
        super().__init__()
        self.L2_clip = L2_clip
        self.noise_multiplier = noise_multiplier
        self.num_replicas = num_replicas
        self.wrapped_model = base_model(**kwargs)
        self.snooper = ModelSnooper()
        self.snooper.snoop(self.wrapped_model)
        del self.snooper  # snooped enough
        self.input_size = getattr(self.wrapped_model, "input_size", None)
        self.models = self._clone_model(self.wrapped_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.wrapped_model(x)
        else:
            if not self.num_replicas == x.shape[0]:
                raise ValueError(
                    f"num_replicas ({self.num_replicas}) must be equal to the batch size ({x.shape[0]})."
                )
            y_pred = torch.nn.parallel.parallel_apply(
                self.models, torch.stack(x.split(1))  # type: ignore
            )
            return torch.cat(y_pred)

    @torch.no_grad()
    def clip_and_accumulate(self) -> None:
        """Clips and averages the per-sample gradients.
        Raises:
            RuntimeError: If no gradients have been calculated yet.
        """

        for model in self.models:
            for param in model.parameters():
                if param.requires_grad and param.grad is None:
                    raise RuntimeError(
                        "No gradients have been calculated yet! This method should be called after .backward() was called on the loss."
                    )

        # Each model has seen one sample, hence we are "per-sample clipping" here.
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.L2_clip)

        model_grads = [
            [param.grad for param in m.parameters() if param.requires_grad]
            for m in self.models
        ]  # a list of length = self.replicas which holds lists of parameter gradients

        for param, gradient_source in zip(
            self.wrapped_model.parameters(), zip(*model_grads)
        ):
            if param.requires_grad:
                setattr(
                    param,
                    "accumulated_gradients",
                    torch.stack([grad for grad in gradient_source]),
                )

    @torch.no_grad()
    def noise_gradient(self) -> None:
        """Applies noise to the gradient before the optimizer step."""
        for param in self.wrapped_model.parameters():
            if param.requires_grad and hasattr(param, "accumulated_gradients"):
                aggregated_gradient = torch.mean(param.accumulated_gradients, dim=0)
                noise = torch.randn_like(aggregated_gradient) * (
                    self.L2_clip
                    * self.noise_multiplier
                    / self.num_replicas  # i.e. batch size
                )
                param.grad = aggregated_gradient + noise
                param.accumulated_gradients = None  # free memory

    @torch.no_grad()
    def prepare_next_batch(self) -> None:
        """Prepare model for the next batch by re-initializing the model replicas with
        the updated weights.
        """
        for model in self.models:
            model.load_state_dict(self.wrapped_model.state_dict())

    @torch.no_grad()
    def _clone_model(self, model):
        models = []
        for _ in range(self.num_replicas):
            models.append(deepcopy(model))
        return nn.ModuleList(models)

    @property
    def parameters(self):
        raise ValueError(
            "The DPWrapper instance has no own parameters. Please use <Instance>.model.parameters()"
        )
