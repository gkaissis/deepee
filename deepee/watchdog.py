from typing import Optional, Union
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from .privacy_accounting import compute_eps_uniform
from .dataloader import UniformDataLoader

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
    ) -> None:
        """The PrivacyWatchdog can be attached to a model and dataloader and supervises the
        training process. If an epsilon and delta are provided, it can either track the privacy
        spent during model training or abort the training when the privacy budget is
        exhausted. Optionally, it can preserve the final model weights before aborting.

        Args:
            target_epsilon (Optional[float]): The target epsilon to warn or abort
            the training at.
            target_delta (Optional[float]): The corresponding delta value.
            report_every_n_steps (int, optional): Outputs the privacy spent to STDERR every
            n steps. Defaults to 100.
            abort (Optional[bool], optional): Whether to abort the training at the set epsilon
            level. Defaults to False.
            save (Optional[bool], optional): Whether to save the last model before aborting training.
            Ignored if abort is set to False. Defaults to False.
            path (Optional[Union[str, Path]], optional): The path to save the final model state
            dictionary before aborting training. Ignored if abort or save are set to False.
            Defaults to None.
        """
        self.dataloader = dataloader
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.report_every_n_steps = report_every_n_steps
        self.abort = abort
        self.save = save
        self.path = path
        self.wrapper = None

        if not self.abort and (self.save or self.path):
            logging.warning(
                "When setting 'save' or 'path' without setting 'abort=True', the settings are ignored"
            )

        if self.save and not self.path:
            raise ValueError("When setting 'save', a path to save to must be provided.")

    def inform(self, steps_taken: int) -> None:
        epoch = (steps_taken * self.dataloader.batch_size) / len(  # type: ignore
            self.dataloader.dataset  # type: ignore
        )
        spent = compute_eps_uniform(
            epoch=epoch,
            noise_multi=self.wrapper.noise_multiplier,  # type: ignore
            n=len(self.dataloader.dataset),  # type: ignore
            batch_size=self.dataloader.batch_size,  # type: ignore
            delta=self.target_delta,  # type: ignore
        )
        if steps_taken % self.report_every_n_steps == 0:
            logging.info(f"Privacy spent at {steps_taken} steps: {spent:.2f}")

        if (spent >= self.target_epsilon) and self.abort:  # type: ignore
            self.abort_training(epsilon_spent=spent, save=self.save, path=self.path)
        else:
            logging.warning(
                f"Privacy budget exhausted. Epsilon spent is {spent}, epsilon allowed is {self.target_epsilon:.2f} at delta {self.target_delta:.2f}"
            )

    def abort_training(
        self,
        epsilon_spent: float,
        save: Optional[bool] = False,
        path: Optional[Union[str, Path]] = None,
    ):
        if not save:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted. Epsilon spent is {epsilon_spent}, epsilon allowed is {self.target_epsilon:.2f} at delta {self.target_delta:.2f}"
            )
        else:
            torch.save(self.wrapper.wrapped_model, path)  # type: ignore
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted. Epsilon spent is {epsilon_spent}, epsilon allowed is {self.target_epsilon:.2f} at delta {self.target_delta:.2f}"
            )