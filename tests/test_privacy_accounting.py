from deepee.privacy_accounting import compute_eps_uniform
import pytest


def test_epsilon_uniform():
    eps = compute_eps_uniform(10, 1.0, 50_000, 200, 1e-5)
    assert 1.29 < eps < 1.31


def test_epsilon_float():
    eps = compute_eps_uniform(10.0, 1.0, 50_000, 200, 1e-5)
    assert 1.29 < eps < 1.31


def test_implausible_values():
    with pytest.raises(ValueError):
        compute_eps_uniform(1, 0.001, 60_000, 200, 1e-5)