r"""Alternative nonclassicality solver based on direct minimization.

This module evaluates the same witness as :mod:`src.solvers.lp_solver`,

.. math::

    f(\boldsymbol\lambda)
        = \max\!\Big(0, \max_{\mu\ge0}\boldsymbol\lambda\cdot\mathbf p(\mu)\Big)
          - \boldsymbol\lambda\cdot\mathbf P,

but minimizes :math:`f` directly with Nelder-Mead instead of the cutting-plane
LP. It is an independent cross-check of the primary LP solver. The inner
supremum is found from the stationary-point polynomial in
:func:`max_threshold_value` (the analogue of ``lp_solver.max_inner_value``;
kept separate on purpose so the two solvers share no inner code).

Function names map to the original notebook as follows::

    max_threshold_value      <- maxPrahfun
    witness_objective        <- fun_full
    witness_objective_fixed  <- fun_fix_one
    minimize_witness         <- objf_no_fixation
    minimize_witness_fixed   <- objf_fixation
    scan_2d_minimize         <- numcrit2D
    scan_2d_minimize_fixed   <- numcrit2D_fix
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from tqdm import tqdm

from src.utils.numerics import inverse_factorials

logger = logging.getLogger(__name__)

# Default search box for the witness coefficients lambda.
_LAMBDA_BOUND = (-1.0, 1.0)


def max_threshold_value(
    indx: ArrayLike,
    lam: ArrayLike,
    fact: ArrayLike,
) -> float:
    r"""Clipped supremum :math:`\max(0, \max_{\mu\ge0}\boldsymbol\lambda\cdot\mathbf p(\mu))`.

    Evaluates the objective at the positive real roots of the stationary-point
    polynomial and the boundary ``mu = 0``, then clips at 0.

    Parameters
    ----------
    indx : array_like of int
        Selected photon-number indices.
    lam : array_like of float
        Coefficient vector, aligned with ``indx``.
    fact : array_like of float
        Inverse factorials ``fact[k] = 1/k!``.

    Returns
    -------
    float
        The clipped supremum value.
    """
    indx = np.asarray(indx, dtype=int).reshape(-1)
    max_index = np.amax(indx)

    lam_aux = np.zeros(max_index + 2)
    lam_aux[indx] = lam

    coef = (lam_aux[1:max_index + 2] - lam_aux[0:max_index + 1]) * fact
    stationary = Polynomial(coef)
    all_roots = stationary.roots()

    real_roots = np.real(all_roots)
    real_roots = np.append(real_roots[real_roots > 0], 0.0)

    coef = lam_aux[0:max_index + 1] * fact
    objective = Polynomial(coef)

    val = np.amax(objective(real_roots) * np.exp(-real_roots))
    return float(np.maximum(0.0, val))


def witness_objective(
    lam: ArrayLike,
    probs: ArrayLike,
    indx: ArrayLike,
    fact: ArrayLike,
) -> float:
    """Witness value ``max_threshold_value(lam) - lam . probs`` (to be minimized)."""
    threshold = max_threshold_value(indx=indx, lam=lam, fact=fact)
    return threshold - np.dot(lam, probs)


def witness_objective_fixed(
    lam: ArrayLike,
    probs: ArrayLike,
    indx: ArrayLike,
    fact: ArrayLike,
    lam_fix: float = 1.0,
    i_fix: int = 0,
) -> float:
    """Witness value with one coefficient pinned to ``lam_fix`` at ``i_fix``.

    The free vector ``lam`` has length ``N_P - 1``; the fixed component is
    inserted before evaluation.
    """
    lam_all = np.insert(lam, i_fix, lam_fix)
    threshold = max_threshold_value(indx=indx, lam=lam_all, fact=fact)
    return threshold - np.dot(lam_all, probs)


def minimize_witness(
    probs: ArrayLike,
    indx: ArrayLike,
    fact: ArrayLike,
    init_a: ArrayLike,
    init_b: ArrayLike,
    init_c: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    tol: float = 1e-6,
) -> tuple[float, NDArray[np.float64]]:
    """Maximize the witness via Nelder-Mead from up to three starting points.

    Returns the most nonclassical (largest ``-f``) result found. The second and
    third starts are only tried if the previous one did not already yield a
    negative objective, matching the original cascade.

    Parameters
    ----------
    probs : array_like, shape (N_P,)
        Target probability vector.
    indx : array_like of int
        Selected photon-number indices.
    fact : array_like of float
        Inverse factorials ``fact[k] = 1/k!``.
    init_a, init_b, init_c : array_like, shape (N_P,)
        Starting coefficient vectors.
    bounds : sequence of (float, float)
        Per-component bounds passed to Nelder-Mead.
    tol : float, optional
        Optimizer tolerance. Default ``1e-6``.

    Returns
    -------
    value : float
        The witness value ``-f`` (positive => nonclassical).
    argmin : numpy.ndarray
        Coefficient vector achieving it.
    """
    objf = lambda x: witness_objective(x, probs=probs, indx=indx, fact=fact)

    ret_a = minimize(objf, init_a, method="Nelder-Mead", bounds=bounds, tol=tol)

    if ret_a.fun >= 0:
        ret_b = minimize(objf, init_b, method="Nelder-Mead", bounds=bounds, tol=tol)

        if ret_b.fun >= 0:
            ret_c = minimize(objf, init_c, method="Nelder-Mead", bounds=bounds, tol=tol)

            results = np.array([ret_a.fun, ret_b.fun, ret_c.fun])
            positions = [ret_a.x, ret_b.x, ret_c.x]

            best = np.argmin(results)
            value = results[best]
            argmin = positions[best]
        else:
            value = ret_b.fun
            argmin = ret_b.x
    else:
        value = ret_a.fun
        argmin = ret_a.x

    return -value, argmin


def minimize_witness_fixed(
    probs: ArrayLike,
    indx: ArrayLike,
    fact: ArrayLike,
    init_a: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    tol: float = 1e-6,
    lam_fix: float = 1.0,
    i_fix: int = 0,
) -> tuple[float, NDArray[np.float64]]:
    """Maximize the witness with one coefficient pinned (single Nelder-Mead run).

    Parameters
    ----------
    probs : array_like, shape (N_P,)
        Target probability vector.
    indx : array_like of int
        Selected photon-number indices.
    fact : array_like of float
        Inverse factorials ``fact[k] = 1/k!``.
    init_a : array_like, shape (N_P - 1,)
        Starting vector for the free coefficients.
    bounds : sequence of (float, float)
        Per-component bounds (length ``N_P - 1``).
    tol : float, optional
        Optimizer tolerance. Default ``1e-6``.
    lam_fix : float, optional
        Value of the pinned coefficient. Default ``1.0``.
    i_fix : int, optional
        Index of the pinned coefficient. Default ``0``.

    Returns
    -------
    value : float
        The witness value ``-f`` (positive => nonclassical).
    argmin : numpy.ndarray
        Free-coefficient vector achieving it.
    """
    objf = lambda x: witness_objective_fixed(
        x, probs=probs, indx=indx, fact=fact, lam_fix=lam_fix, i_fix=i_fix
    )

    ret_a = minimize(objf, init_a, method="Nelder-Mead", bounds=bounds, tol=tol)

    return -ret_a.fun, ret_a.x


def scan_2d_minimize(
    nc_a: NDArray[np.bool_],
    NP: int,
    Pall: NDArray[np.float64],
    bias: float = 1e-7,
    tol: float = 1e-7,
    seed: int | None = None,
) -> tuple[NDArray[np.bool_], float, NDArray[np.float64], NDArray[np.float64]]:
    """Refine an analytical nonclassicality map with the minimize solver.

    For every grid point still classified as classical (``nc_a == False``),
    maximize the witness via :func:`minimize_witness` and flag the point if the
    witness exceeds ``bias``. Warm-starts the next point with the previous
    solution, mirroring the original notebook heuristic.

    Parameters
    ----------
    nc_a : numpy.ndarray of bool, shape (Ny, Nx)
        Input nonclassicality map (e.g. from analytical criteria).
    NP : int
        Number of probability components.
    Pall : numpy.ndarray, shape (Ny, Nx, NP)
        Probability grid.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-7``.
    tol : float, optional
        Optimizer tolerance. Default ``1e-7``.
    seed : int or None, optional
        Seed for the random starting points (reproducibility). Default ``None``.

    Returns
    -------
    nc_n : numpy.ndarray of bool
        Updated nonclassicality map.
    ncnf : float
        Fraction of nonclassical points.
    loss_mem : numpy.ndarray, shape (Ny, Nx)
        Witness value stored at each refined point.
    lam_mem : numpy.ndarray, shape (Ny, Nx, NP)
        Coefficient vector stored at each flagged point.
    """
    rng = np.random.default_rng(seed)

    nc_n = np.copy(nc_a)

    classical_idx = np.array(np.where(nc_n == False))
    n_classical = classical_idx.shape[1]
    logger.debug("scan_2d_minimize: %d classical points to refine", n_classical)

    lam_mem = np.zeros((*nc_n.shape, NP))
    loss_mem = np.zeros(nc_n.shape)

    init_a = np.zeros(NP)
    init_b = np.ones(NP)
    init_c = 0.10 - 0.2 * rng.random(NP)

    n_points = np.prod(np.array([*nc_n.shape]))

    bounds = [_LAMBDA_BOUND for _ in range(NP)]

    indx_all = np.array(range(NP))
    fact = inverse_factorials(int(np.max(indx_all)))

    for point in tqdm(range(n_classical)):
        i, j = classical_idx[:, point]
        probs = Pall[i, j, :]

        loss, init_a = minimize_witness(
            probs=probs, indx=indx_all, fact=fact,
            init_a=init_a, init_b=init_b, init_c=init_c,
            bounds=bounds, tol=tol,
        )

        if loss > bias:
            nc_n[i, j] = True
            init_a = init_a / np.max(np.abs(init_a))
            lam_mem[i, j, :] = init_a
            init_b = np.copy(init_a)
        else:
            init_a = np.copy(init_b)
            init_b = -init_b

        loss_mem[i, j] = loss

    ncnf = np.sum(nc_n.astype(int)) / n_points
    return nc_n, ncnf, loss_mem, lam_mem


def scan_2d_minimize_fixed(
    nc_a: NDArray[np.bool_],
    NP: int,
    Pall: NDArray[np.float64],
    bias: float = 1e-7,
    tol: float = 1e-7,
    seed: int | None = None,
) -> tuple[NDArray[np.bool_], float, NDArray[np.float64], NDArray[np.float64]]:
    """Refine an analytical map with the pinned-coefficient minimize solver.

    For every classical grid point, tries pinning each coefficient to
    ``+1`` then ``-1`` (via :func:`minimize_witness_fixed`) and flags the point
    as soon as the witness exceeds ``bias``.

    Parameters
    ----------
    nc_a : numpy.ndarray of bool, shape (Ny, Nx)
        Input nonclassicality map.
    NP : int
        Number of probability components.
    Pall : numpy.ndarray, shape (Ny, Nx, NP)
        Probability grid.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-7``.
    tol : float, optional
        Optimizer tolerance. Default ``1e-7``.
    seed : int or None, optional
        Seed for the random starting points. Default ``None``.

    Returns
    -------
    nc_n : numpy.ndarray of bool
        Updated nonclassicality map.
    ncnf : float
        Fraction of nonclassical points.
    loss_mem : numpy.ndarray, shape (Ny, Nx)
        Witness value stored at each refined point.
    lam_mem : numpy.ndarray, shape (Ny, Nx, NP)
        Coefficient vector stored at each flagged point.
    """
    rng = np.random.default_rng(seed)

    nc_n = np.copy(nc_a)

    classical_idx = np.array(np.where(nc_n == False))
    n_classical = classical_idx.shape[1]
    logger.debug("scan_2d_minimize_fixed: %d classical points to refine", n_classical)

    lam_mem = np.zeros((*nc_n.shape, NP))
    loss_mem = np.zeros(nc_n.shape)

    n_points = np.prod(np.array([*nc_n.shape]))

    bounds = [_LAMBDA_BOUND for _ in range(NP - 1)]

    indx_all = np.array(range(NP))
    fact = inverse_factorials(int(np.max(indx_all)))

    for point in tqdm(range(n_classical)):

        init_a = 0.1 * (0.5 - rng.random(NP - 1))

        i, j = classical_idx[:, point]
        probs = Pall[i, j, :]

        for i_fix in range(NP):

            loss, init_b = minimize_witness_fixed(
                probs=probs, indx=indx_all, fact=fact,
                init_a=init_a, bounds=bounds, tol=tol,
                lam_fix=1.0, i_fix=i_fix,
            )
            if loss > bias:
                nc_n[i, j] = True
                lam_mem[i, j, :] = np.insert(init_b, i_fix, 1.0)
                loss_mem[i, j] = loss
                break
            else:
                loss, init_b = minimize_witness_fixed(
                    probs=probs, indx=indx_all, fact=fact,
                    init_a=init_a, bounds=bounds, tol=tol,
                    lam_fix=-1.0, i_fix=i_fix,
                )

                if loss > bias:
                    nc_n[i, j] = True
                    lam_mem[i, j, :] = np.insert(init_b, i_fix, -1.0)
                    loss_mem[i, j] = loss
                    break

    ncnf = np.sum(nc_n.astype(int)) / n_points
    return nc_n, ncnf, loss_mem, lam_mem
