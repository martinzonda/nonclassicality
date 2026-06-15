r"""Classic analytical nonclassicality criteria on a 2D probability grid.

Collected criteria:

- :func:`klyshko`   -- three-index Klyshko inequality
  :math:`(k+1)P_{k-1}P_{k+1} - k P_k^2 < 0`.
- :func:`taylored`  -- Taylor-expanded generalized Klyshko bound
  :math:`\mathcal K_{\infty, N_P}`.
- :func:`log01rule` -- :math:`P_1 + P_0 \ln P_0 > 0`.
- :func:`generalized_klyshko` and its precomputed-pair variants -- the
  majorization criterion (see :mod:`src.analytical.criteria_majorization`).
- :func:`analcrit2D` -- union of klyshko, taylored and log01rule.

The majorization variants reuse the shared pair machinery rather than
re-implementing the inner product loop.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial

from src.analytical.criteria_majorization import (
    all_pairs_for_list_of_P,
    build_reverse_index,
    list_of_files,
    load_precomputed_pairs,
    nonclass_from_pairs,
)
from src.utils.io import load_pickle

logger = logging.getLogger(__name__)


def generalized_klyshko(
    Pall: NDArray[np.float64],
    indx: ArrayLike | None = None,
) -> tuple[NDArray[np.bool_], float, int]:
    """Majorization criterion with pairs generated on the fly.

    Equivalent to :func:`src.analytical.criteria_majorization.analytical_m` but
    always generates the majorization pairs combinatorially (no precomputed
    file lookup).

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.
    indx : array_like of int, optional
        Physical photon-number indices. Defaults to ``[0, 1, 2]``.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    n_pairs : int
        Number of pairs evaluated.

    Raises
    ------
    ValueError
        If ``len(indx)`` does not match ``Pall.shape[-1]``.
    """
    if indx is None:
        indx = np.array([0, 1, 2], dtype=int)
    indx = np.asarray(indx, dtype=int).reshape(-1)

    Psh = Pall.shape
    if Psh[-1] != len(indx):
        raise ValueError(
            f"Pall.shape[-1]={Psh[-1]} but len(indx)={len(indx)}"
        )

    rev_index = build_reverse_index(indx)
    all_pairs = all_pairs_for_list_of_P(indx=indx)
    nonclass = nonclass_from_pairs(Pall, all_pairs, rev_index)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def generalized_klyshko_first_Np(
    Pall: NDArray[np.float64],
    lof: Sequence[str] = list_of_files,
) -> tuple[NDArray[np.bool_], float, int]:
    """Majorization criterion using precomputed pairs for the first ``N_P``.

    Assumes consecutive indices ``[0, 1, ..., N_P - 1]`` and loads the matching
    precomputed pair file.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.
    lof : sequence of str, optional
        Precomputed-pair file paths. Defaults to :data:`list_of_files`.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    n_pairs : int
        Number of pairs evaluated.
    """
    Psh = Pall.shape
    n_probs = Psh[-1]
    indx = np.arange(0, n_probs, 1, dtype=int)
    logger.debug("generalized_klyshko_first_Np: indx=%s", indx)

    rev_index = build_reverse_index(indx)
    all_pairs = load_pickle(lof[n_probs - 4])
    nonclass = nonclass_from_pairs(Pall, all_pairs, rev_index)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def generalized_klyshko_first_Np_ch(
    Pall: NDArray[np.float64],
    lof: Sequence[str] = list_of_files,
    ipair: ArrayLike | None = None,
) -> tuple[NDArray[np.bool_], float, int]:
    """Majorization criterion using only selected precomputed pairs.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.
    lof : sequence of str, optional
        Precomputed-pair file paths. Defaults to :data:`list_of_files`.
    ipair : array_like of int, optional
        Indices into the loaded pair list to evaluate. Defaults to
        ``[1, 2, 3]``.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    n_pairs : int
        Total number of pairs in the loaded file (not the number selected),
        preserved from the original implementation.
    """
    if ipair is None:
        ipair = np.array([1, 2, 3])
    ipair = np.asarray(ipair, dtype=int).reshape(-1)

    Psh = Pall.shape
    n_probs = Psh[-1]
    indx = np.arange(0, n_probs, 1, dtype=int)
    logger.debug("generalized_klyshko_first_Np_ch: indx=%s", indx)

    rev_index = build_reverse_index(indx)
    all_pairs = load_pickle(lof[n_probs - 4])
    selected = [all_pairs[i0] for i0 in ipair]
    nonclass = nonclass_from_pairs(Pall, selected, rev_index)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def klyshko(Pall: NDArray[np.float64]) -> tuple[NDArray[np.bool_], float]:
    r"""Three-index Klyshko criterion.

    Flags grid points where
    :math:`(k+1)P_{k-1}P_{k+1} - k P_k^2 < 0` for some ``k``.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    """
    Psh = Pall.shape
    n_points = np.array(np.prod(Psh[0:-1]))

    n_probs = Psh[-1]
    n_max = n_probs - 1

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    for k in range(1, n_max):
        violation = (k + 1) * Pall[:, :, k - 1] * Pall[:, :, k + 1] - k * Pall[:, :, k]**2
        nonclass = np.where(violation < 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / n_points
    return nonclass, ncaf


def taylored(Pall: NDArray[np.float64]) -> tuple[NDArray[np.bool_], float]:
    r"""Taylor-expanded generalized Klyshko bound :math:`\mathcal K_{\infty,N_P}`.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    """
    Psh = Pall.shape
    n_points = np.array(np.prod(Psh[0:-1]))

    n_probs = Psh[-1]
    n_max = n_probs - 1

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    cumulative = np.zeros((Psh[0], Psh[1], n_probs))  # running sum of P_0..P_m
    series = np.zeros((Psh[0], Psh[1]))

    cumulative[:, :, 0] = Pall[:, :, 0]
    for m in range(1, n_max):
        cumulative[:, :, m] = cumulative[:, :, m - 1] + Pall[:, :, m]

    for nn in range(2, n_max + 1):

        series[:, :] = 1.0
        for k in range(1, nn - 1):
            series += ((nn * Pall[:, :, nn] / Pall[:, :, nn - 1])**k) / factorial(k)

        violation = nn * Pall[:, :, nn]
        violation -= Pall[:, :, nn - 1] * np.log(
            (nn**nn / factorial(nn))
            * ((Pall[:, :, nn]**(nn - 1)) / (Pall[:, :, nn - 1]**nn))
            * (1.0 - cumulative[:, :, nn - 2]) + series
        )

        nonclass = np.where(violation > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / n_points
    return nonclass, ncaf


def log01rule(Pall: NDArray[np.float64]) -> tuple[NDArray[np.bool_], float]:
    r"""Vacuum/one-photon criterion :math:`P_1 + P_0 \ln P_0 > 0`.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid (uses ``P_0`` and ``P_1``).

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical points.
    """
    Psh = Pall.shape
    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    violation = Pall[:, :, 1] + Pall[:, :, 0] * np.log(Pall[:, :, 0])
    nonclass = np.where(violation > 0, True, nonclass)
    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf


def analcrit2D(Pall: NDArray[np.float64]) -> tuple[NDArray[np.bool_], float]:
    """Union of the klyshko, taylored and log01rule criteria.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid.

    Returns
    -------
    nonclass : numpy.ndarray of bool
        Combined nonclassicality map (logical OR of the three criteria).
    ncaf : float
        Fraction of nonclassical points.
    """
    Psh = Pall.shape
    n_points = np.array(np.prod(Psh[0:-1]))

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    klyshko_map, _ = klyshko(Pall=Pall)
    nonclass = np.where(klyshko_map > 0, True, nonclass)

    taylored_map, _ = taylored(Pall=Pall)
    nonclass = np.where(taylored_map > 0, True, nonclass)

    log_map, _ = log01rule(Pall=Pall)
    nonclass = np.where(log_map > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / n_points
    return nonclass, ncaf
