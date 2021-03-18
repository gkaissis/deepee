# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
r"""Implements privacy accounting for Gaussian Differential Privacy.
Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).

Most of this code is straight outta TensorFlow Privacy
https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py
Documentation improvements. The Poisson subsampling is removed since the PrivacyWrapper
needs a fixed batch size. We implement a slightly nicer interface.
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(epoch, noise_multi, n, batch_size):
    """Compute mu from uniform subsampling."""

    t = epoch * n / batch_size
    c = batch_size * np.sqrt(t) / n
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multi ** (-2)) * norm.cdf(1.5 / noise_multi)
            + 3 * norm.cdf(-0.5 / noise_multi)
            - 2
        )
    )


def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
    """Compute epsilon from mu given delta via inverse dual."""

    def f(x):
        """Reversely solve dual by matching delta."""
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root


def compute_eps_uniform(
    epoch: int, noise_multi: float, n: int, batch_size: int, delta: float
) -> float:
    """Computes the epsilon value for a given delta of a subsampled gaussian mechanism
    with uniform subsampling using Gaussian Differential Privacy.
    For more information see Dong et al. (https://arxiv.org/abs/1905.02383).

    Args:
        epoch (int): How many epochs the model has been trained for.
        noise_multi (float): Noise multiplier of the PrivacyWrapper.
        n (int): Total training dataset size.
        batch_size (int): Batch size used for training. Same as num_replicas
        in the PrivacyWrapper.
        delta (float): Delta value for the DP guarantee.

    Returns:
        epsilon (float): The epsilon value.
    """

    return eps_from_mu(compute_mu_uniform(epoch, noise_multi, n, batch_size), delta)
