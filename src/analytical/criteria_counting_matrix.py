r"""Photon-counting matrix nonclassicality criteria.

This module implements the ideal photoelectric counting matrix from
Krishnaswamy et al., arXiv:2603.06237v1, Sec. II,

.. math::

    C^{(I)} = \left[(k + l)! P_{k+l}\right]_{k,l \in I},

where ``I`` is the row/column index set from Eqs. (8)--(12). Classical states
require every admissible matrix ``C`` to be positive semidefinite, so a
negative minimum eigenvalue certifies nonclassicality.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_indx(indx: ArrayLike, n_probs: int | None = None) -> NDArray[np.int_]:
    """Return a validated one-dimensional photon-index array."""
    indx = np.asarray(indx, dtype=int).reshape(-1)

    if n_probs is not None and len(indx) != n_probs:
        raise ValueError(f"len(indx)={len(indx)} but probability axis has length {n_probs}")

    if len(indx) == 0:
        raise ValueError("indx must contain at least one index")

    if np.any(indx < 0):
        raise ValueError("indx must contain non-negative photon indices")

    if not np.all(np.diff(indx) > 0):
        raise ValueError("indx must be strictly increasing")

    return indx


def _maximal_cliques(vertices: set[int], adjacency: dict[int, set[int]]) -> list[tuple[int, ...]]:
    """Find all maximal cliques using the Bron-Kerbosch algorithm."""
    cliques: list[tuple[int, ...]] = []

    def bronk(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            cliques.append(tuple(sorted(R)))
            return

        while P:
            v = P.pop()
            bronk(R | {v}, P & adjacency[v], X & adjacency[v])
            X.add(v)

    bronk(set(), set(vertices), set())
    return cliques


def admissible_counting_sets(
    indx: ArrayLike,
    min_size: int = 2,
) -> dict[str, list[tuple[float, ...]]]:
    r"""Construct maximal admissible integer and half-integer index sets.

    A set ``I`` is admissible when ``k + l`` is an available probability index
    for every ``k, l in I``. Internally, ``q = 2*k`` is used to represent
    half-integers exactly. It is sufficient to test maximal admissible sets:
    any smaller admissible set is a principal submatrix of one of them.

    Parameters
    ----------
    indx : array_like of int
        Available probability indices, e.g. ``[0, 1, 2, 3, 4]``.
    min_size : int, optional
        Minimum matrix dimension to retain. Default is 2, since 1x1 matrices
        cannot certify nonclassicality for non-negative probabilities.

    Returns
    -------
    dict
        ``{"integer": [...], "half_integer": [...]}``, where each entry is a
        tuple of counting-matrix indices ``k``.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")

    indx = _validate_indx(indx)
    available = set(indx.tolist())
    result: dict[str, list[tuple[float, ...]]] = {
        "integer": [],
        "half_integer": [],
    }

    for parity, name in [(0, "integer"), (1, "half_integer")]:
        # q = 2*k. The diagonal element requires P_{2k} = P_q.
        q_vertices = {int(q) for q in indx if q % 2 == parity}
        adjacency = {q: set() for q in q_vertices}

        for q1 in q_vertices:
            for q2 in q_vertices:
                if q1 == q2:
                    continue

                m = (q1 + q2) // 2
                if m in available:
                    adjacency[q1].add(q2)

        for clique in _maximal_cliques(q_vertices, adjacency):
            if len(clique) >= min_size:
                result[name].append(tuple(q / 2 for q in clique))

    return result


def counting_matrix(
    probs: ArrayLike,
    indx: ArrayLike,
    I: ArrayLike,
) -> NDArray[np.float64]:
    r"""Construct ``C^{(I)} = [(k+l)! P_{k+l}]_{k,l in I}`` for one vector."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    indx = _validate_indx(indx, n_probs=len(probs))

    pmap = {int(k): float(p) for k, p in zip(indx, probs)}
    I = np.asarray(I, dtype=float).reshape(-1)

    C = np.empty((len(I), len(I)), dtype=float)
    for i, k in enumerate(I):
        for j, l in enumerate(I):
            m_float = k + l
            m = int(round(m_float))

            if not np.isclose(m_float, m):
                raise ValueError("Invalid index set: k+l is not an integer")

            if m not in pmap:
                raise ValueError(f"P_{m} is required but is not available")

            C[i, j] = factorial(m) * pmap[m]

    return C


def _single_point_counting_result(
    probs: NDArray[np.float64],
    indx: NDArray[np.int_],
    sets: dict[str, list[tuple[float, ...]]],
    tol: float,
) -> tuple[np.bool_, dict[str, Any]]:
    """Evaluate the strongest counting-matrix violation for one point."""
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or np.sum(probs) > 1.0 + tol:
        return np.bool_(False), {"valid": False}

    best_eig = np.inf
    best_I: tuple[float, ...] | None = None
    best_family: str | None = None
    best_C: NDArray[np.float64] | None = None

    for family, all_sets in sets.items():
        for I in all_sets:
            C = counting_matrix(probs, indx, I)
            eigmin = np.linalg.eigvalsh(C)[0]

            if eigmin < best_eig:
                best_eig = float(eigmin)
                best_I = I
                best_family = family
                best_C = C

    if best_I is None:
        nonclassical = np.bool_(False)
        best_eig = np.nan
    else:
        nonclassical = np.bool_(best_eig < -tol)

    return nonclassical, {
        "valid": True,
        "min_eig": best_eig,
        "family": best_family,
        "I": best_I,
        "matrix": best_C,
    }


def analytical_c(
    Pall: ArrayLike,
    indx: ArrayLike | None = None,
    tol: float = 1e-12,
    return_details: bool = False,
) -> tuple[NDArray[np.bool_], float] | tuple[NDArray[np.bool_], float, dict[str, Any]]:
    r"""Evaluate photon-counting matrix criteria over a probability grid.

    This is the ideal photoelectric counting-matrix criterion from
    Krishnaswamy et al., arXiv:2603.06237v1, applied to photon-number
    probabilities.

    Parameters
    ----------
    Pall : array_like
        Probability array with shape ``(..., Np)``. The last axis contains the
        probabilities whose physical indices are given by ``indx``.
    indx : array_like of int, optional
        Physical photon-number indices in ``Pall``. If omitted, assumes
        ``[0, 1, ..., Np - 1]``.
    tol : float, optional
        Numerical tolerance. A point is nonclassical when
        ``lambda_min(C) < -tol`` for at least one admissible counting matrix.
    return_details : bool, optional
        If ``True``, also return strongest-violation diagnostics for every
        point. The detail arrays have shape ``Pall.shape[:-1]``.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Boolean nonclassicality map with shape ``Pall.shape[:-1]``.
    ncaf : float
        Fraction of nonclassical grid points.
    details : dict, optional
        Returned when ``return_details=True``.
    """
    Pall = np.asarray(Pall, dtype=float)
    if Pall.ndim < 2:
        raise ValueError(f"Pall must have shape (..., Np), got {Pall.shape}")

    n_probs = Pall.shape[-1]
    if indx is None:
        indx = np.arange(n_probs, dtype=int)
    indx = _validate_indx(indx, n_probs=n_probs)

    grid_shape = Pall.shape[:-1]
    nonclass = np.zeros(grid_shape, dtype=bool)
    sets = admissible_counting_sets(indx)

    if return_details:
        min_eig = np.full(grid_shape, np.nan, dtype=float)
        valid = np.zeros(grid_shape, dtype=bool)
        family = np.empty(grid_shape, dtype=object)
        best_I = np.empty(grid_shape, dtype=object)
        matrix = np.empty(grid_shape, dtype=object)

    for idx in np.ndindex(grid_shape):
        point_nonclass, point_details = _single_point_counting_result(
            Pall[idx],
            indx,
            sets,
            tol,
        )
        nonclass[idx] = point_nonclass

        if return_details:
            valid[idx] = bool(point_details["valid"])
            min_eig[idx] = point_details.get("min_eig", np.nan)
            family[idx] = point_details.get("family")
            best_I[idx] = point_details.get("I")
            matrix[idx] = point_details.get("matrix")

    ncaf = np.sum(nonclass.astype(int)) / np.prod(grid_shape)
    if not return_details:
        return nonclass, float(ncaf)

    details = {
        "valid": valid,
        "min_eig": min_eig,
        "family": family,
        "I": best_I,
        "matrix": matrix,
        "sets": sets,
    }
    return nonclass, float(ncaf), details


def analytical_c_point(
    probs: ArrayLike,
    indx: ArrayLike,
    tol: float = 1e-12,
    return_details: bool = False,
) -> np.bool_ | tuple[np.bool_, dict[str, Any]]:
    """Evaluate the arXiv:2603.06237v1 counting-matrix criterion for one vector."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    indx = _validate_indx(indx, n_probs=len(probs))
    sets = admissible_counting_sets(indx)
    nonclass, details = _single_point_counting_result(probs, indx, sets, tol)

    if return_details:
        details = {**details, "sets": sets}
        return nonclass, details
    return nonclass
