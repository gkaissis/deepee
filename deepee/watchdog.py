class PrivacyWatchdog:
    def __init__(self):
        pass

    def watch(self, model, target_eps, save_last, path):
        """Watch model and interrupt training at target epsilon. Optionally save model parameters to path"""
        pass

    def observe(self, model):
        """Just observe model training and report epsilon spent."""
        pass