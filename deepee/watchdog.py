from typing import Optional, Union
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from .privacy_accounting import (
    compute_eps_uniform,
    compute_rdp,
    get_privacy_spent as rdp_privacy_spent,
)
from .dataloader import UniformDataLoader, UniformWORSubsampler

logging.basicConfig(level=logging.INFO)


class PrivacyBudgetExhausted(Exception):
    pass


class PrivacyWatchdog:
    def __init__(
        self,
        dataloader: Union[DataLoader, UniformDataLoader],
        target_epsilon: Optional[float],
        target_delta: Optional[float],
        report_every_n_steps: int = 100,
        abort: Optional[bool] = False,
        save: Optional[bool] = False,
        path: Optional[Union[str, Path]] = None,
        fallback_to_rdp: bool = False,
    ) -> None:
        """The PrivacyWatchdog can be attached to a model and dataloader and supervises
        the training process. If an epsilon and delta are provided, it can either track
        the privacy spent during model training or abort the training when the privacy
        budget is exhausted. Optionally, it can preserve the final model weights before
        aborting.

        Args:
            target_epsilon (Optional[float]): The target epsilon to warn or abort
            the training at.
            target_delta (Optional[float]): The corresponding delta value.
            report_every_n_steps (int, optional): Outputs the privacy spent to STDERR
            every n steps. Defaults to 100.
            abort (Optional[bool], optional): Whether to abort the training at the set
            epsilon level. Defaults to False.
            save (Optional[bool], optional): Whether to save the last model before
            aborting training. Ignored if abort is set to False. Defaults to False.
            path (Optional[Union[str, Path]], optional): The path to save the final
            model state dictionary before aborting training. Ignored if abort or save
            are set to False. Defaults to None.
            fallback_to_rdp (Optional[bool], optional): Whether to fall back to Renyi
            DP accounting in case Gaussian DP accounting (default) fails. This is for
            convenience and will likely return a worse privacy guarantee which may also
            be incorrect. It should not be used for mission-critical work. If False,
            the PrivacyWatchdog will raise an error if privacy cannot be calculated with
            the given settings.

        """
        self.dataloader = dataloader
        if not (
            isinstance(self.dataloader, UniformDataLoader)
            or isinstance(self.dataloader.batch_sampler, UniformWORSubsampler)
        ):
            logging.critical(
                "Privacy accounting is only correct when using the UniformDataLoader or"
                " a custom DataLoader with a batch_sampler implementing uniform sampling"
                " without replacement."
            )
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.report_every_n_steps = report_every_n_steps
        self.abort = abort
        self.save = save
        self.path = path
        self.wrapper = None
        self.fallback_to_rdp = fallback_to_rdp

        if self.fallback_to_rdp:
            logging.critical(
                "Privacy accounting is set to fall back to RDP if privacy "
                " accounting using GDP fails. This estimate is potentially inaccurate and"
                " should not be used for mission-critical work!",
            )

        if (not self.abort) and (self.save or self.path):
            logging.warning(
                "When setting 'save' or 'path' without setting 'abort=True', the"
                " settings are ignored."
            )

        if (self.abort and self.save) and not self.path:
            raise ValueError("When setting 'save', a path to save to must be provided.")

    def inform(self, steps_taken: int) -> None:
        if not self.wrapper:
            raise RuntimeError("WatchDog must be attached to a PrivacyWrapper.")
        batch_size = (
            self.dataloader.batch_size or self.dataloader.batch_sampler.batch_size  # type: ignore
        )
        epoch = (steps_taken * batch_size) / len(  # type: ignore
            self.dataloader.dataset  # type: ignore
        )
        try:
            spent = compute_eps_uniform(
                epoch=epoch,
                noise_multi=self.wrapper.noise_multiplier,  # type: ignore
                n=len(self.dataloader.dataset),  # type: ignore
                batch_size=batch_size,  # type: ignore
                delta=self.target_delta,  # type: ignore
            )
        except ValueError as e:
            if self.fallback_to_rdp:
                approx_sample_rate = batch_size / len(self.dataloader.dataset)
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = compute_rdp(
                    q=approx_sample_rate,
                    noise_multiplier=self.wrapper.noise_multiplier,
                    steps=steps_taken,
                    orders=orders,
                )
                spent, _ = rdp_privacy_spent(
                    orders=orders, rdp=rdp, delta=self.target_delta
                )
            else:
                raise RuntimeError(
                    "Epsilon could not be determined, likely because of implausible values"
                    " for the L2 clip ratio and/or the noise multiplier. Try decreasing the "
                    " L2 clip ratio or increasing the noise multiplier. If you are trying to"
                    " 'disable' DP by setting these values, errors can be avoided by not"
                    " attaching a WatchDog to your PrivacyWrapper."
                ) from e

        if steps_taken % self.report_every_n_steps == 0:
            logging.info(f"Privacy spent at {steps_taken} steps: {spent:.2f}")
            setattr(self.wrapper, "_privacy_spent", spent)

        if spent >= self.target_epsilon:  # type: ignore
            if self.abort:  # type: ignore
                self.abort_training(epsilon_spent=spent, save=self.save, path=self.path)
            else:
                logging.warning(
                    f"Privacy budget exhausted. Epsilon spent is {spent}, epsilon"
                    "allowed is {self.target_epsilon:.2f} at delta {self.target_delta:.2e}"
                )

    def abort_training(
        self,
        epsilon_spent: float,
        save: Optional[bool] = False,
        path: Optional[Union[str, Path]] = None,
    ):
        error_message = f"Privacy budget exhausted. Epsilon spent is {epsilon_spent},"
        "epsilon allowed is {self.target_epsilon:.2f} at delta {self.target_delta:e}"
        if not save:
            raise PrivacyBudgetExhausted(error_message)
        else:
            torch.save(self.wrapper.wrapped_model, path)  # type: ignore
            raise PrivacyBudgetExhausted(error_message)
