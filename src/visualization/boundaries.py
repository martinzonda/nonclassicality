"""Boundary extraction and plotting helpers for 2D nonclassicality maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.config import VIZ

# Contour level separating the classical (0) and nonclassical (1) regions.
_BOUNDARY_LEVEL = VIZ.boundary_level


def extract_binary_boundary(
    P0_all: ArrayLike,
    P1_all: ArrayLike,
    nonclass_num: ArrayLike,
) -> list[NDArray[np.float64]]:
    """Extract the curves separating the 0 and 1 regions of a binary map.

    Uses a matplotlib contour at level 0.5 to recover the ordered boundary
    polylines between the classical and nonclassical regions.

    Parameters
    ----------
    P0_all : array_like, shape (Nx,)
        Coordinates along the first probability axis.
    P1_all : array_like, shape (Ny,)
        Coordinates along the second probability axis.
    nonclass_num : array_like, shape (Ny, Nx)
        Binary nonclassicality map (0 = classical, 1 = nonclassical).

    Returns
    -------
    list of numpy.ndarray
        One ``(M, 2)`` array of ``[P0, P1]`` vertices per boundary curve.
        Degenerate single-point curves are dropped.
    """
    nonclass_num = np.asarray(nonclass_num, dtype=float)
    finite = np.isfinite(nonclass_num)
    if not np.any(finite):
        return []

    finite_values = nonclass_num[finite]
    if finite_values.min() > _BOUNDARY_LEVEL or finite_values.max() < _BOUNDARY_LEVEL:
        return []

    masked_map = np.ma.masked_invalid(nonclass_num)

    fig, ax = plt.subplots()

    cs = ax.contour(P0_all, P1_all, masked_map, levels=[_BOUNDARY_LEVEL])

    boundaries = []
    for path in cs.get_paths():
        vertices = path.vertices
        if len(vertices) > 1:
            boundaries.append(vertices.copy())

    plt.close(fig)
    return boundaries
