"""Tests for src.solvers (LP cutting-plane and Nelder-Mead solvers)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.solvers.lp_solver import (
    pvec_mu,
    max_inner_value,
    max_inner_value_fast,
    true_objective,
    solve_master_lp,
    solve_semi_infinite_lp,
)
from src.solvers.minimize_solver import max_threshold_value, minimize_witness
from src.utils.numerics import inverse_factorials


def _brute_inner_max(indx, lam, fact, mu_hi=50.0, n=200_001):
    """Reference: max(0, max_{mu>=0} lam . p(mu)) on a dense grid."""
    mu = np.linspace(0.0, mu_hi, n)
    vals = np.array([np.dot(lam, np.exp(-m) * m**np.asarray(indx) * np.asarray(fact)[indx])
                     for m in mu])
    return max(0.0, float(vals.max()))


# --------------------------------------------------------------------- pvec_mu
def test_pvec_mu_matches_poisson(indx4, fact4):
    mu = 1.3
    expected = np.exp(-mu) * mu**indx4 / np.array([1, 1, 2, 6])
    assert_allclose(pvec_mu(mu, indx4, fact4), expected, rtol=1e-12)


def test_pvec_mu_zero_mu(indx4, fact4):
    # At mu = 0 only P_0 = 1, the rest vanish.
    assert_allclose(pvec_mu(0.0, indx4, fact4), [1.0, 0.0, 0.0, 0.0], atol=1e-15)


# -------------------------------------------------------------- max_inner_value
@pytest.mark.parametrize("lam", [
    [1.0, -1.0, 1.0, -1.0],
    [0.0, 0.5, -0.3, 0.2],
    [-0.2, -0.1, 0.4, 0.1],
    [1.0, 1.0, 1.0, 1.0],
])
def test_max_inner_value_matches_brute_force(indx4, fact4, lam):
    lam = np.array(lam)
    worst, _ = max_inner_value(indx4, lam, fact4)
    brute = _brute_inner_max(indx4, lam, fact4)
    assert_allclose(worst, brute, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("lam", [
    [1.0, -1.0, 1.0, -1.0],
    [0.0, 0.5, -0.3, 0.2],
    [-0.2, -0.1, 0.4, 0.1],
])
def test_lp_and_minimize_inner_agree(indx4, fact4, lam):
    """The two independent solvers must compute the same inner supremum."""
    lam = np.array(lam)
    w_lp, _ = max_inner_value(indx4, lam, fact4)
    w_mz = max_threshold_value(indx4, lam, fact4)
    assert_allclose(w_lp, w_mz, rtol=1e-6, atol=1e-8)


def test_max_inner_value_fast_agrees(indx4, fact4):
    # Interior maximum (g(mu) = e^-mu * mu peaks at mu = 1); the bounded fast
    # search and the exact polynomial method must agree here. The fast method
    # is documented to be less robust for boundary maxima at mu = 0.
    lam = np.array([0.0, 1.0, 0.0, 0.0])
    w_exact, _ = max_inner_value(indx4, lam, fact4)
    w_fast, _ = max_inner_value_fast(indx4, lam, fact4)
    assert_allclose(w_exact, w_fast, rtol=1e-4, atol=1e-6)


def test_max_inner_value_empty_indx():
    assert max_inner_value(np.array([], dtype=int), np.array([]), np.array([1.0])) == (0.0, 0.0)


def test_max_inner_value_length_mismatch(fact4):
    with pytest.raises(ValueError):
        max_inner_value(np.array([0, 1, 2, 3]), np.array([1.0, 2.0]), fact4)


# ------------------------------------------------------------------ true_objective
def test_true_objective_consistency(indx4, fact4, nonclassical_probs):
    lam = np.array([0.1, -0.2, 0.5, -0.1])
    obj, worst, _ = true_objective(nonclassical_probs, indx4, fact4, lam)
    assert_allclose(obj, worst - np.dot(lam, nonclassical_probs), rtol=1e-12)


# ----------------------------------------------------------------- solve_master_lp
def test_solve_master_lp_length_mismatch(fact4, lambda_bounds4):
    with pytest.raises(ValueError):
        solve_master_lp(np.array([0.5, 0.5]), np.array([0, 1, 2, 3]), fact4,
                        lambda_bounds4, mu_list=[0.0])


def test_solve_master_lp_returns_solution(classical_probs, indx4, fact4, lambda_bounds4):
    lam, t, res = solve_master_lp(classical_probs, indx4, fact4, lambda_bounds4, mu_list=[0.0, 1.0])
    assert res.success
    assert lam.shape == (4,)
    assert t >= -1e-9


# ----------------------------------------------------------- solve_semi_infinite_lp
def test_classical_point_not_flagged(classical_probs, indx4, fact4, lambda_bounds4):
    res = solve_semi_infinite_lp(classical_probs, indx4, fact4, lambda_bounds4)
    # A Poisson point is classical: optimum f >= 0, so num = -f <= ~0.
    assert res["num"] <= 1e-6


def test_nonclassical_point_flagged(nonclassical_probs, indx4, fact4, lambda_bounds4):
    res = solve_semi_infinite_lp(nonclassical_probs, indx4, fact4, lambda_bounds4)
    # The Fock state |2> is strongly nonclassical.
    assert res["num"] > 1e-3
    assert_allclose(res["obj"], -res["num"], rtol=1e-12)


def test_regression_known_objective(indx4, fact4, lambda_bounds4):
    """Lock in the witness value for a fixed probability vector."""
    probs = np.array([0.6, 0.1, 0.2, 0.05])
    res = solve_semi_infinite_lp(probs, indx4, fact4, lambda_bounds4)
    assert_allclose(res["num"], 0.1133973836, rtol=0, atol=1e-7)


def test_minimize_witness_detects_nonclassical(nonclassical_probs, indx4, fact4, lambda_bounds4):
    value, _ = minimize_witness(
        probs=nonclassical_probs, indx=indx4, fact=fact4,
        init_a=np.zeros(4), init_b=np.ones(4), init_c=0.1 * np.ones(4),
        bounds=lambda_bounds4,
    )
    assert value > 1e-3
