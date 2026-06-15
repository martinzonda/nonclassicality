"""
File I/O helpers for pickled majorization-pair archives and .npy result files.

These wrap the raw ``open``/``pickle.load``/``np.save`` idioms scattered across
the original notebooks so that path handling and error reporting live in one
place.

NOTE: type hints, docstrings and logging are completed in Phase 3.
"""

import pickle

import numpy as np


def load_pickle(filename):
    """Load and return a pickled object from ``filename``."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_arrays(filename, *arrays, allow_pickle=False):
    """
    Save a sequence of arrays to ``filename`` as a concatenated .npy stream,
    in the same order they are passed. Mirrors the repeated ``np.save`` blocks
    used by the 2D scan code.
    """
    with open(filename, "wb") as f:
        for arr in arrays:
            np.save(f, arr, allow_pickle=allow_pickle)
