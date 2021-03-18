from deepee.privacy_accounting import compute_eps_uniform


def test_epsilon_uniform():
    eps = compute_eps_uniform(10, 1.0, 50_000, 200, 1e-5)
    assert 1.29 < eps < 1.31
