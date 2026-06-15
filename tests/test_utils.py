"""Tests for src.utils (numerics and I/O helpers)."""

import os
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.utils.numerics import inverse_factorials, collect_lambda
from src.utils.io import load_pickle, save_arrays


# --------------------------------------------------------------- inverse_factorials
@pytest.mark.parametrize("max_index,expected", [
    (0, [1.0]),
    (1, [1.0, 1.0]),
    (3, [1.0, 1.0, 0.5, 1.0 / 6.0]),
    (5, [1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120]),
])
def test_inverse_factorials_values(max_index, expected):
    assert_allclose(inverse_factorials(max_index), expected, rtol=0, atol=1e-15)


def test_inverse_factorials_length():
    assert inverse_factorials(7).shape == (8,)


def test_inverse_factorials_negative_raises():
    with pytest.raises(ValueError):
        inverse_factorials(-1)


# ------------------------------------------------------------------- collect_lambda
def test_collect_lambda_filters_and_orders():
    # 2x2 loss map; only entries above the threshold are collected.
    loss = np.array([[0.0, 2.0],
                     [3.0, 0.0]])
    lam = np.zeros((2, 2, 2))
    lam[0, 1] = [10.0, 11.0]
    lam[1, 0] = [20.0, 21.0]
    x = np.array([100.0, 200.0])
    y = np.array([300.0, 400.0])

    out = collect_lambda(loss, lam, x, y, threshold=1.0)
    assert out.shape == (2, 4)
    # collect_lambda preserves the original notebook's index convention: for a
    # point at (row, col) it stores [x[col], y[row], lam[col, row]]. So the two
    # above-threshold entries map as follows:
    #   loss[0, 1] -> (x[1], y[0], lam[1, 0]) = (200, 300, 20, 21)
    #   loss[1, 0] -> (x[0], y[1], lam[0, 1]) = (100, 400, 10, 11)
    rows = {tuple(np.round(r, 3)) for r in out}
    assert (200.0, 300.0, 20.0, 21.0) in rows
    assert (100.0, 400.0, 10.0, 11.0) in rows


def test_collect_lambda_empty_when_below_threshold():
    loss = np.zeros((3, 3))
    lam = np.zeros((3, 3, 2))
    out = collect_lambda(loss, lam, np.arange(3), np.arange(3), threshold=1e-6)
    assert out.shape == (0, 4)


# -------------------------------------------------------------------------- io
def test_save_and_load_arrays_roundtrip(tmp_path):
    a = np.arange(5.0)
    b = np.eye(3)
    path = os.path.join(tmp_path, "arrays.npy")
    save_arrays(path, a, b)
    with open(path, "rb") as f:
        ra = np.load(f)
        rb = np.load(f)
    assert_allclose(ra, a)
    assert_allclose(rb, b)


def test_save_arrays_object_roundtrip(tmp_path):
    ragged = np.array([np.arange(2), np.arange(3)], dtype=object)
    path = os.path.join(tmp_path, "obj.npy")
    save_arrays(path, ragged, allow_pickle=True)
    with open(path, "rb") as f:
        loaded = np.load(f, allow_pickle=True)
    assert len(loaded) == 2


def test_load_pickle_roundtrip(tmp_path):
    obj = {"a": [1, 2, 3], "b": np.arange(4)}
    path = os.path.join(tmp_path, "obj.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    loaded = load_pickle(path)
    assert loaded["a"] == [1, 2, 3]


def test_load_pickle_missing_file_raises():
    with pytest.raises(OSError):
        load_pickle("/nonexistent/path/file.pkl")
