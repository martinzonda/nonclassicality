"""Tests for src.comparison (the 2D numerical-vs-analytical scan)."""

import os

import numpy as np
import pytest

from src.comparison.scan_2d import run2D


@pytest.fixture
def small_scan(tmp_path):
    """Run a tiny scan and return its outputs plus the archive path."""
    out = os.path.join(tmp_path, "scan.npy")
    result = run2D(
        N0=12, N1=12,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10, uPA=0.9, dPB=1e-10, uPB=0.3,
        Prest=np.array([0.05, 0.05]),
        fi=out,
    )
    return result, out


def test_run2d_returns_four_boundary_lists(small_scan):
    result, _ = small_scan
    assert len(result) == 4
    for curves in result:
        assert isinstance(curves, list)


def test_run2d_writes_readable_archive(small_scan):
    _, path = small_scan
    assert os.path.exists(path)
    with open(path, "rb") as f:
        PA_all = np.load(f)
        PB_all = np.load(f)
        Prest = np.load(f)
        _ = np.load(f, allow_pickle=True)  # boundaries
        _ = np.load(f, allow_pickle=True)
        _ = np.load(f, allow_pickle=True)
        _ = np.load(f, allow_pickle=True)
        nonclass_num = np.load(f)
        nonclass_ana = np.load(f)
    assert PA_all.shape == (12,)
    assert PB_all.shape == (12,)
    assert Prest.shape == (2,)
    assert nonclass_num.shape == (12, 12)
    assert nonclass_ana.shape == (12, 12)


def test_run2d_numerical_superset_of_analytical(small_scan):
    """Every analytically-nonclassical point must also be flagged numerically.

    The analytical criteria are sufficient conditions, so the (exact) LP
    classification should cover at least the analytical region.
    """
    _, path = small_scan
    with open(path, "rb") as f:
        for _ in range(7):
            np.load(f, allow_pickle=True)
        nonclass_num = np.load(f)
        nonclass_ana = np.load(f)
    # numerical >= analytical everywhere (no analytical-only false positives).
    assert np.all(nonclass_num.astype(bool) | ~nonclass_ana.astype(bool))
