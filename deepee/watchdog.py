from typing import Optional


class PrivacyWatchdog:
    def __init__(self, target_epsilon: Optional[float], target_delta: Optional[float]):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def watch(self, model, target_eps, save_last, path):
        """Watch model and abort training at target epsilon.
        Optionally save model parameters to path"""
        pass

    def observe(self, model):
        """Just observe model training and report epsilon spent."""
        pass