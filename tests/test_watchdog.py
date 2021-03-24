from deepee.watchdog import PrivacyBudgetExhausted
from deepee import PrivacyWatchdog, UniformDataLoader, PrivacyWrapper
from deepee.dataloader import UniformWORSubsampler
from torch.utils.data import DataLoader, Dataset
import torch
import pytest
from testfixtures import LogCapture


class DS(Dataset):
    def __getitem__(self, idx):
        return torch.rand(
            1,
        )

    def __len__(self):
        return 5


dl = DataLoader(DS())
udl = UniformDataLoader(DS(), 1)
bsdl = DataLoader(DS(), batch_sampler=UniformWORSubsampler(DS(), 5))


def test_uniform_dl():
    with LogCapture() as l:
        watchdog = PrivacyWatchdog(udl, target_delta=1e-5, target_epsilon=1.0)
        watchdog2 = PrivacyWatchdog(bsdl, target_delta=1e-5, target_epsilon=1.0)
        watchdog2 = PrivacyWatchdog(dl, target_delta=1e-5, target_epsilon=1.0)
        assert "CRITICAL" and "replacement" in str(l)


def test_epsilon_delta_positive():
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(udl, target_delta=1e-5, target_epsilon=None)
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(udl, target_delta=None, target_epsilon=1.0)
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(udl, target_delta=1.2, target_epsilon=1.0)
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(udl, target_delta=1.2, target_epsilon=1.0)
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(udl, target_delta=1e-5, target_epsilon=-4)


def test_warn_without_save_or_path():
    with LogCapture() as l:
        watchdog = PrivacyWatchdog(
            udl,
            target_delta=1e-5,
            target_epsilon=1.0,
            abort=False,
            save=True,
        )
        assert "WARNING" and "ignored" in str(l)

    with LogCapture() as l:
        watchdog = PrivacyWatchdog(
            udl,
            target_delta=1e-5,
            target_epsilon=1.0,
            abort=False,
            save=False,
            path="somepath",
        )
        assert "WARNING" and "ignored" in str(l)


def test_save_fails_without_path():
    """User asked for save without specifying path"""
    with pytest.raises(ValueError):
        watchdog = PrivacyWatchdog(
            udl, target_delta=1e-5, target_epsilon=1.0, abort=True, save=True, path=None
        )


def test_inform():
    """Test reporting of epsilon"""

    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl,
        report_every_n_steps=1,
        target_delta=1e-5,
        target_epsilon=1.0,
    )

    class FakeWrapper:
        noise_multiplier = 1.0

    watchdog.wrapper = FakeWrapper
    with LogCapture() as l:
        watchdog.inform(1)
        assert "Privacy spent at 1 steps" in str(l)


def test_orphan_watchdog():
    """Watchdog not attached"""
    dl = UniformDataLoader(udl, batch_size=200)
    watchdog = PrivacyWatchdog(
        dl,
        report_every_n_steps=1,
        target_delta=1e-5,
        target_epsilon=1.0,
    )
    with pytest.raises(RuntimeError):
        watchdog.inform(1)


def test_abort_training():
    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl, report_every_n_steps=1, target_delta=1e-5, target_epsilon=1.0, abort=True
    )

    class FakeWrapper:
        noise_multiplier = 1.0

    watchdog.wrapper = FakeWrapper
    with pytest.raises(PrivacyBudgetExhausted):
        watchdog.inform(50000)


def test_log_exhausted():
    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl, report_every_n_steps=1, target_delta=1e-5, target_epsilon=1.0, abort=False
    )

    class FakeWrapper:
        noise_multiplier = 1.0

    watchdog.wrapper = FakeWrapper
    with LogCapture() as l:
        watchdog.inform(50000)
        assert "WARNING" and "exhausted" in str(l)


def test_wrapper_returns_epsilon():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl, report_every_n_steps=1, target_delta=1e-5, target_epsilon=1.0, abort=False
    )

    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 1.0, 1.0, watchdog=watchdog)
    epsila = []  # this one's for you @a1302z
    for _ in range(5):
        output = wrapped(data)
        loss = output.mean()
        loss.backward()
        wrapped.clip_and_accumulate()
        wrapped.noise_gradient()
        epsilon = wrapped.prepare_next_batch(return_privacy_spent=True)
        epsila.append(epsilon)
    assert len(epsila) == 5


def test_fallback_warning():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    with LogCapture() as l:
        watchdog = PrivacyWatchdog(
            dl,
            report_every_n_steps=1,
            target_delta=1e-5,
            target_epsilon=1.0,
            abort=False,
            fallback_to_rdp=True,
        )
        assert "CRITICAL" and "RDP" in str(l)


def test_fallback_works():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl,
        report_every_n_steps=1,
        target_delta=1e-5,
        target_epsilon=1.0,
        abort=False,
        fallback_to_rdp=True,
    )
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 10, 0.001, watchdog=watchdog)
    epsila = []  # this one's for you @a1302z
    for _ in range(5):
        output = wrapped(data)
        loss = output.mean()
        loss.backward()
        wrapped.clip_and_accumulate()
        wrapped.noise_gradient()
        epsilon = wrapped.prepare_next_batch(return_privacy_spent=True)
        epsila.append(epsilon)
    assert len(epsila) == 5


def test_no_fallback_crashes():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.lin(x)

    class BigDS(Dataset):
        def __getitem__(self, idx):
            return torch.rand(
                1,
            )

        def __len__(self):
            return 50_000

    dl = UniformDataLoader(BigDS(), batch_size=200)
    watchdog = PrivacyWatchdog(
        dl,
        report_every_n_steps=1,
        target_delta=1e-5,
        target_epsilon=1.0,
        abort=False,
        fallback_to_rdp=False,
    )
    data = torch.randn(2, 1, 10)
    wrapped = PrivacyWrapper(MiniModel(), 2, 10, 0.001, watchdog=watchdog)
    with pytest.raises(RuntimeError):
        output = wrapped(data)
        loss = output.mean()
        loss.backward()
        wrapped.clip_and_accumulate()
        wrapped.noise_gradient()
        wrapped.prepare_next_batch(return_privacy_spent=False)