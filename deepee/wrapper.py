import torch
from torch import nn
from copy import deepcopy
from typing import Optional, Any, Union

from .snooper import ModelSnooper
from .watchdog import PrivacyWatchdog


class PrivacyWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_replicas: int,
        L2_clip: float,
        noise_multiplier: float,
        secure_rng: bool = False,
        seed: Optional[int] = None,
        watchdog: Optional[PrivacyWatchdog] = None,
    ) -> None:
        """Factory class which wraps any model and returns a model suitable for
        DP-SGD learning.
        The class will replicate the model in a memory-efficient way and run the
        forward and backward passes in parallel over the inputs.

        Args:
            base_model (nn.Module): The model instalce to wrap.
            num_replicas (int): How many times to replicate the model. Must be set equal
            to the batch size.
            L2_clip (float): Clipping norm for the DP-SGD procedure.
            noise_multiplier (float): Noise multiplier for the DP-SGD procedure.
            secure_rng (bool): Whether to use a cryptographically secure random number
            generator to produce the noise. Compare Mironov, or Gazeau et al. Models trained
            without secure RNG are not suitable for anything except experimentation. The
            secure RNG has a significant performance overhead from collecting entropy at
            each step.
            seed (optional int): The seed for the (insecure) random number generator.
            This is incompatible with the cryptographic RNG and will raise an error if
            both are set.
            watchdog (optional PrivacyWatchDog): A PrivacyWatchdog instance to attach
            to the PrivacyWrapper.

        For more information on L2_clip and noise_multiplier, see Abadi et al., 2016.

        The wrapped model is compatible with any first-order optimizer (SGD, Adam, etc.)
        without any modifications to the optimizer itself.

        The wrapper includes a sanity check to make sure that the model doesn't include
        any layers incompatible with the notion of "per-sample" gradient calculation,
        such as BatchNorm layers.
        If it throws an error, look to the ModelSurgeon to remedy these issues.

        Sample use:

        >> model = PrivacyWrapper(resnet18, num_replicas=64, L2_clip=1., noise_multiplier=1.)
        >> optimizer = torch.optim.SGD(model.wrapped_model.parameters(), lr=0.1)
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
        self.wrapped_model = base_model
        self.snooper = ModelSnooper()
        self.snooper.snoop(self.wrapped_model)
        del self.snooper  # snooped enough
        self.watchdog = watchdog
        if self.watchdog:
            setattr(self.watchdog, "wrapper", self)
        self.input_size = getattr(self.wrapped_model, "input_size", None)
        self.models = self._clone_model(self.wrapped_model)
        self.seed = seed
        self.secure_rng = secure_rng
        if self.seed and self.secure_rng:
            raise ValueError(
                "Setting a seed is incompatible with the secure_rng option."
            )
        if self.secure_rng:
            try:
                import torchcsprng as rng

                self.noise_gen = rng.create_random_device_generator("/dev/urandom")
            except ImportError as e:
                raise ImportError(
                    "To use the secure RNG, torchcsprng must be installed."
                ) from e

        self._steps_taken = 0
        self._forward_succesful = False
        self._clip_succesful = False
        self._noise_succesful = False
        self._privacy_spent = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.wrapped_model(x)
        else:  # in training mode
            if not self.num_replicas == x.shape[0]:
                raise ValueError(
                    f"num_replicas ({self.num_replicas}) must be equal to the"
                    " batch size ({x.shape[0]})."
                )
            try:
                y_pred = torch.nn.parallel.parallel_apply(
                    self.models, torch.stack(x.split(1))  # type: ignore
                )
                self._forward_succesful = True
                return torch.cat(y_pred)
            except ValueError as e:
                if "Expected more than 1 value per channel when training" in str(e):
                    raise RuntimeError(
                        "An error occured during the forward pass. "
                        " This is typical if using BatchNorm with a small image input size."
                        " If this is the case, please switch to GroupNorm."
                    ) from e
                else:
                    raise

    @torch.no_grad()
    def clip_and_accumulate(self, reduce: str = "mean") -> None:
        """Clips and averages the per-sample gradients.

        Args:
            reduce (str): How to reduce the accumulated gradients. As per the Abadi paper
            this defaults to "mean". Alternatively, "sum" sums the per-sample gradients.

        Raises:
            RuntimeError: If no gradients have been calculated yet
            or the method was not called in order.

        """
        if reduce == "mean":
            reduction = torch.mean
        elif reduce == "sum":
            reduction = torch.sum
        else:
            raise ValueError(
                f"'reduce' must be one of 'mean' or 'sum', got '{reduce}''."
            )

        if not self._forward_succesful:
            raise RuntimeError(
                "An error occured during model training. Please ascertain that the"
                "model.forward(), model.clip_and_accumulate() and model.noise_gradient()"
                " methods are called successfuly and in this order."
            )

        for model in self.models:
            for param in model.parameters():
                if param.requires_grad and param.grad is None:
                    raise RuntimeError(
                        "No gradients have been calculated yet! This method should be"
                        " called after .backward() was called on the loss."
                    )

        # Each model has seen one sample, hence we are "per-sample clipping" here.
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.L2_clip)

        model_grads = zip(
            *[
                [param.grad for param in m.parameters() if param.requires_grad]
                for m in self.models
            ]
        )

        for param, gradient_source in zip(self.wrapped_model.parameters(), model_grads):
            if param.requires_grad:
                param.grad = reduction(torch.stack(gradient_source), dim=0)

        self._clip_succesful = True

    @torch.no_grad()
    def noise_gradient(self) -> None:
        """Applies noise before the optimizer step."""

        if not self._clip_succesful:
            raise RuntimeError(
                "An error occured during model training. Please ascertain that the"
                "model.forward(), model.clip_and_accumulate() and model.noise_gradient()"
                " methods are called successfuly and in order."
            )

        for param in self.wrapped_model.parameters():
            if param.requires_grad:
                if not self.secure_rng:
                    if self.seed:
                        torch.manual_seed(self.seed)
                    noise = torch.randn_like(param.grad) * (
                        self.L2_clip
                        * self.noise_multiplier
                        / self.num_replicas  # i.e. batch size
                    )
                else:
                    noise = torch.normal(
                        mean=0,
                        std=(self.L2_clip * self.noise_multiplier / self.num_replicas),
                        size=param.grad.shape,
                        generator=self.noise_gen,
                    )
                param.grad.add_(noise)
                self._noise_succesful = True

    @torch.no_grad()
    def prepare_next_batch(
        self, return_privacy_spent: Optional[bool] = False
    ) -> Union[None, float]:
        """Prepare model for the next batch by re-initializing the model replicas with
        the updated weights and informing the PrivacyWatchdog about the state of the
        training.

        Args:
            return_privacy_spent (Optional[bool], optional): When set to True and a"
            "PrivacyWatchDog is attached, it returns the epsilon (privacy spent) at"
            "the specific report interval. Defaults to False.

        Raises:
            RuntimeError: If the method has not been called in proper order.
        """
        if not (
            self._forward_succesful and self._clip_succesful and self._noise_succesful
        ):
            raise RuntimeError(
                "An error occured during model training. Please ascertain that the"
                "model.forward(), model.clip_and_accumulate() and model.noise_gradient()"
                " methods were called successfuly and in order."
            )
        for model in self.models:
            for target_param, source_param in zip(
                model.parameters(), self.wrapped_model.parameters()
            ):
                target_param.data = source_param.data
        self._steps_taken += 1
        self._forward_succesful = self._clip_succesful = self._noise_succesful = False
        if self.watchdog:
            self.watchdog.inform(self._steps_taken)

        if return_privacy_spent:
            return self._privacy_spent
        return None  # just for you MyPy...

    @torch.no_grad()
    def _clone_model(self, model):
        models = []
        for _ in range(self.num_replicas):
            models.append(deepcopy(model))
            for target_param, source_param in zip(
                models[-1].parameters(), self.wrapped_model.parameters()
            ):
                target_param.data = source_param.data
        return nn.ModuleList(models)

    @property
    def parameters(self):
        raise ValueError(
            "The PrivacyWrapper instance has no own parameters."
            " Please use <Instance>.model.parameters()."
        )


class PerSampleGradientWrapper(nn.Module):
    def __init__(
        self, base_model: nn.Module, num_replicas: int, **kwargs: Optional[Any]
    ) -> None:
        """Factory class which wraps a PyTorch model to provide access to per-sample
        gradients. It will replicate the base model and peform the forward and backward
        passes in parallel. Contrary to the PrivacyWrapper, this class doesn't offer any
        additional features except calculating per-sample gradients. The wrapper includes
        a sanity check to make sure that the model doesn't include any layers incompatible
        with the notion of "per-sample" gradient calculation, such as BatchNorm layers.
        If it throws an error, look to the ModelSurgeon to remedy these issues.

        Args:
            base_model (nn.Module): The model instance to wrap.
            num_replicas (int): How many times to replicate the model. Must be set equal
            to the batch size.

        Sample use:
        >> model = PrivacyWrapper(resnet18, num_replicas=64)
        >> optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)
        >> y_pred = model(data)
        >> loss = criterion(y_pred, y_true)
        >> loss.backward()
        >> model.calculate_per_sample_gradients()
        """
        super().__init__()
        self.num_replicas = num_replicas
        self.wrapped_model = base_model
        self.snooper = ModelSnooper()
        self.snooper.snoop(self.wrapped_model)
        del self.snooper
        self.input_size = getattr(self.wrapped_model, "input_size", None)
        self.models = self._clone_model(self.wrapped_model)
        self._forward_succesful = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.wrapped_model(x)
        else:  # in training mode
            if not self.num_replicas == x.shape[0]:
                raise ValueError(
                    f"num_replicas ({self.num_replicas}) must be equal to the"
                    " batch size ({x.shape[0]})."
                )
            y_pred = torch.nn.parallel.parallel_apply(
                self.models, torch.stack(x.split(1))  # type: ignore
            )
            self._forward_succesful = True
            return torch.cat(y_pred)

    def calculate_per_sample_gradients(self):
        """Calculates per-sample gradients for each sample in the minibatch and stores
        them in <param>.accumulated_gradients. These can then be used to calculate e.g.
        per-sample gradient norms, gradient variance etc.
        """
        if not self._forward_succesful:
            raise RuntimeError(
                "An error occured during model training. Please ascertain that the "
                "model.forward() method was called before calculating per sample gradients"
            )

        for model in self.models:
            for param in model.parameters():
                if param.requires_grad and param.grad is None:
                    raise RuntimeError(
                        "No gradients have been calculated yet! This method should be"
                        " called after .backward() was called on the loss."
                    )

        model_grads = zip(
            *[
                [param.grad for param in m.parameters() if param.requires_grad]
                for m in self.models
            ]
        )  # a list of length = self.replicas which holds lists of parameter gradients

        for param, gradient_source in zip(self.wrapped_model.parameters(), model_grads):
            if param.requires_grad:
                setattr(
                    param,
                    "accumulated_gradients",
                    torch.stack(gradient_source),
                )

    @torch.no_grad()
    def _clone_model(self, model):
        models = []
        for _ in range(self.num_replicas):
            models.append(deepcopy(model))
            for target_param, source_param in zip(
                models[-1].parameters(), self.wrapped_model.parameters()
            ):
                target_param.data = source_param.data
        return nn.ModuleList(models)

    @property
    def parameters(self):
        raise ValueError(
            "The PerSampleGradientWrapper instance has no own parameters."
            " Please use <Instance>.model.parameters()."
        )