"""Shared numerical helpers used across solvers, criteria and scans."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial


def inverse_factorials(max_index: int) -> NDArray[np.float64]:
    r"""Return the vector of inverse factorials ``fact[k] = 1 / k!``.

    Many of the Poisson-probability expressions in this project depend on the
    weights :math:`1/k!`, where :math:`p_k(\mu) = e^{-\mu}\,\mu^k / k!`. This
    helper builds the dense vector ``[1/0!, 1/1!, ..., 1/max_index!]`` once so
    callers do not re-derive it inline.

    Parameters
    ----------
    max_index : int
        Largest factorial index ``k`` to include. The returned array has
        length ``max_index + 1``.

    Returns
    -------
    numpy.ndarray
        Float array ``fact`` of shape ``(max_index + 1,)`` with
        ``fact[k] = 1 / k!``.

    Raises
    ------
    ValueError
        If ``max_index`` is negative.
    """
    if max_index < 0:
        raise ValueError(f"max_index must be non-negative, got {max_index}")
    return np.array([1.0 / factorial(k) for k in range(max_index + 1)], dtype=float)


def collect_lambda(
    loss_map: ArrayLike,
    lambda_map: ArrayLike,
    x_coords: ArrayLike,
    y_coords: ArrayLike,
    threshold: float = 1e-6,
) -> NDArray[np.float64]:
    """Collect witness coefficient vectors at grid points above a threshold.

    Flattens a 2D scan into a table whose rows are
    ``[x, y, lambda_0, ..., lambda_{Np-1}]`` for every grid point whose loss
    (witness violation) exceeds ``threshold``.

    Parameters
    ----------
    loss_map : array_like, shape (Ny, Nx)
        Witness violation at each grid point.
    lambda_map : array_like, shape (Ny, Nx, Np)
        Witness coefficient vector stored at each grid point.
    x_coords : array_like, shape (Nx,)
        Coordinates along the second (column) axis.
    y_coords : array_like, shape (Ny,)
        Coordinates along the first (row) axis.
    threshold : float, optional
        Minimum loss for a grid point to be included. Default ``1e-6``.

    Returns
    -------
    numpy.ndarray, shape (n_selected, 2 + Np)
        Table with the point coordinates in the first two columns and the
        corresponding ``lambda`` vector in the remaining columns.

    Notes
    -----
    Logic preserved verbatim from the original notebook implementation,
    including the ``(i, j)`` index convention used to index ``lambda_map``.
    """
    loss_map = np.asarray(loss_map, dtype=float)
    lambda_map = np.asarray(lambda_map, dtype=float)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)

    n_components = lambda_map.shape[2]

    indices = np.where(loss_map > threshold)
    collected = np.empty((len(indices[0]), 2 + n_components), dtype=float)

    for row in range(len(indices[0])):
        i, j = indices[1][row], indices[0][row]
        collected[row, :2] = [x_coords[i], y_coords[j]]  # point coordinates
        collected[row, 2:] = lambda_map[i, j, :]  # coefficient vector

    return collected


# Backwards-compatible alias for the original notebook name.
collect_lam = collect_lambda
