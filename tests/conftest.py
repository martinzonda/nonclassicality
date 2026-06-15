"""Shared pytest fixtures and path setup for the test suite."""

import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless backend for boundary-extraction plotting

# Make the project importable without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.numerics import inverse_factorials


@pytest.fixture
def indx4():
    """Consecutive photon-number indices [0, 1, 2, 3]."""
    return np.array([0, 1, 2, 3])


@pytest.fixture
def fact4():
    """Inverse factorials 1/k! for k = 0..3."""
    return inverse_factorials(3)


@pytest.fixture
def lambda_bounds4():
    """Default per-component bounds for a 4-component coefficient vector."""
    return [(-1.0, 1.0)] * 4


def poisson_vector(mu, n):
    """Reference Poisson distribution P_k = e^-mu mu^k / k! for k = 0..n-1."""
    from scipy.special import factorial
    k = np.arange(n)
    return np.exp(-mu) * mu**k / factorial(k)


@pytest.fixture
def classical_probs():
    """A classical distribution: a single Poisson point restricted to k=0..3.

    Because ``P_k = p_k(mu)`` for a fixed ``mu``, we have
    ``lam . P = lam . p(mu) <= max_mu lam . p(mu)`` for every ``lam``, so the
    witness ``f(lam) >= 0`` everywhere and the optimum is 0 (classical). The
    vector is intentionally left un-renormalised to preserve this property.
    """
    return poisson_vector(0.5, 4)


@pytest.fixture
def nonclassical_probs():
    """A clearly nonclassical distribution: (truncated) Fock state |2>."""
    return np.array([0.0, 0.0, 1.0, 0.0])
