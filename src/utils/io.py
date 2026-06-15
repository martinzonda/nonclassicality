"""File I/O helpers for pickled pair archives and ``.npy`` result files."""

from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_pickle(filename: str) -> Any:
    """Load and return a pickled object from ``filename``.

    Parameters
    ----------
    filename : str
        Path to the pickle file.

    Returns
    -------
    Any
        The unpickled object.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    pickle.UnpicklingError
        If the file contents cannot be unpickled.
    """
    try:
        with open(filename, "rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        logger.exception("Failed to load pickle file %s", filename)
        raise


def save_arrays(filename: str, *arrays: np.ndarray, allow_pickle: bool = False) -> None:
    """Save arrays to ``filename`` as a concatenated ``.npy`` stream.

    The arrays are written in the order given, to be read back with the same
    number of sequential :func:`numpy.load` calls. Mirrors the repeated
    ``np.save`` blocks used by the 2D scan code.

    Parameters
    ----------
    filename : str
        Output path.
    *arrays : numpy.ndarray
        Arrays to write, in order.
    allow_pickle : bool, optional
        Forwarded to :func:`numpy.save`; required for object arrays such as
        ragged boundary lists. Default ``False``.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    try:
        with open(filename, "wb") as handle:
            for array in arrays:
                np.save(handle, array, allow_pickle=allow_pickle)
    except OSError:
        logger.exception("Failed to write arrays to %s", filename)
        raise
