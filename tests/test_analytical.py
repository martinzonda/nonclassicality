"""Tests for src.analytical (criteria and majorization machinery)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.analytical.criteria_k import analytical_k, pN_0fun
from src.analytical.criteria_majorization import (
    analytical_m,
    all_pairs_for_list_of_P,
    load_precomputed_pairs,
    build_reverse_index,
    nonclass_from_pairs,
)
from src.analytical.criteria_classic import (
    klyshko,
    taylored,
    log01rule,
    analcrit2D,
    generalized_klyshko,
)
from src.analytical.jaynes_cummings import Pjaycum
from src.utils.numerics import inverse_factorials


@pytest.fixture
def jc_grid():
    """A small Jaynes-Cummings probability grid with 6 components."""
    tau = np.linspace(0.01, 10.0, 40)
    nu = np.linspace(0.01, 6.0, 40)
    xtau, ynu = np.meshgrid(tau, nu)
    NP = 6
    return np.stack([Pjaycum(n=n, nu=ynu, tau=xtau, pe=1.0 / 3.0) for n in range(NP)], axis=-1)


# ----------------------------------------------------------------- build_reverse_index
def test_build_reverse_index():
    rev = build_reverse_index(np.array([0, 2, 5]))
    assert rev[0] == 0 and rev[2] == 1 and rev[5] == 2


# --------------------------------------------------------------------- analytical_k
def test_analytical_k_shapes(jc_grid):
    fact = inverse_factorials(5)
    nonclass, ncaf = analytical_k(jc_grid, fact, indx=np.arange(6))
    assert nonclass.shape == jc_grid.shape[:2]
    assert nonclass.dtype == bool
    assert 0.0 <= ncaf <= 1.0


def test_analytical_k_length_mismatch():
    fact = inverse_factorials(3)
    with pytest.raises(ValueError):
        analytical_k(np.ones((5, 5, 4)), fact, indx=np.array([0, 1, 2]))


def test_analytical_k_non_increasing_indx():
    fact = inverse_factorials(3)
    with pytest.raises(ValueError):
        analytical_k(np.ones((5, 5, 3)), fact, indx=np.array([0, 2, 1]))


# ----------------------------------------------------------- load_precomputed_pairs
@pytest.mark.parametrize("indx,expected_none", [
    (np.array([0, 1, 2]), True),       # N_P = 3 -> file_idx = -1, no file
    (np.array([0, 1, 3]), True),       # non-consecutive
    (np.arange(20), True),             # out of range
    (np.array([0, 1, 2, 3]), False),   # N_P = 4 -> available
])
def test_load_precomputed_pairs(indx, expected_none):
    result = load_precomputed_pairs(indx)
    assert (result is None) == expected_none


def test_precomputed_count_NP4():
    pairs = load_precomputed_pairs(np.array([0, 1, 2, 3]))
    assert len(pairs) == 7


# --------------------------------------------------------------------- analytical_m
def test_analytical_m_length_mismatch():
    with pytest.raises(ValueError):
        analytical_m(np.ones((5, 5, 4)), indx=np.array([0, 1, 2]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytical_m_equals_generalized_klyshko(seed):
    """analytical_m (precomputed) and generalized_klyshko (on-the-fly) agree."""
    rng = np.random.default_rng(seed)
    Pall = rng.random((8, 9, 4))
    indx = np.array([0, 1, 2, 3])
    nc_m, ncaf_m, n_m = analytical_m(Pall, indx)
    nc_g, ncaf_g, n_g = generalized_klyshko(Pall, indx)
    assert np.array_equal(nc_m, nc_g)
    assert_allclose(ncaf_m, ncaf_g, rtol=0, atol=0)
    assert n_m == n_g


def test_nonclass_from_pairs_matches_analytical_m():
    rng = np.random.default_rng(3)
    Pall = rng.random((6, 6, 4))
    indx = np.array([0, 1, 2, 3])
    pairs = load_precomputed_pairs(indx)
    rev = build_reverse_index(indx)
    nc_direct = nonclass_from_pairs(Pall, pairs, rev)
    nc_m, _, _ = analytical_m(Pall, indx)
    assert np.array_equal(nc_direct, nc_m)


# ----------------------------------------------------------- all_pairs_for_list_of_P
def test_all_pairs_nonempty():
    pairs = all_pairs_for_list_of_P(indx=np.array([0, 1, 2, 3]))
    assert len(pairs) == 7  # matches the precomputed archive for N_P = 4


# --------------------------------------------------------------- classic criteria
@pytest.mark.parametrize("criterion", [klyshko, taylored, log01rule, analcrit2D])
def test_classic_criteria_shapes(jc_grid, criterion):
    nonclass, ncaf = criterion(jc_grid)
    assert nonclass.shape == jc_grid.shape[:2]
    assert nonclass.dtype == bool
    assert 0.0 <= ncaf <= 1.0


def test_log01rule_known_point():
    # Construct P_0, P_1 with P_1 + P_0 ln P_0 > 0 (nonclassical) vs < 0.
    Pall = np.zeros((1, 2, 4))
    # Point 0: P0=0.5, P1=0.5 -> 0.5 + 0.5 ln 0.5 = 0.5 - 0.3466 > 0 (flagged).
    Pall[0, 0] = [0.5, 0.5, 0.0, 0.0]
    # Point 1: P0=0.9, P1=0.01 -> 0.01 + 0.9 ln 0.9 = 0.01 - 0.0948 < 0 (not).
    Pall[0, 1] = [0.9, 0.01, 0.0, 0.0]
    nonclass, _ = log01rule(Pall)
    assert nonclass[0, 0]
    assert not nonclass[0, 1]


def test_analcrit2D_is_union(jc_grid):
    nc_k, _ = klyshko(jc_grid)
    nc_t, _ = taylored(jc_grid)
    nc_l, _ = log01rule(jc_grid)
    nc_all, _ = analcrit2D(jc_grid)
    assert np.array_equal(nc_all, nc_k | nc_t | nc_l)


# --------------------------------------------------------------------------- pN_0fun
def test_pN_0fun_finite():
    x = np.linspace(0.01, 0.99, 50)
    y = pN_0fun(x, 2)
    assert np.all(np.isfinite(y))
