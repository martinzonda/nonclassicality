r"""Majorization-based generalized Klyshko nonclassicality criterion.

A classical photon-number distribution must satisfy a family of polynomial
inequalities indexed by pairs of integer tuples :math:`(T^{(0)}, T^{(1)})` with
:math:`T^{(0)} \succcurlyeq T^{(1)}` (majorization). For each such pair the
classical bound reads

.. math::

    \prod_{i \in T^{(1)}} P_i \ \le\
        \Big(\prod_{i\in T^{(0)}} P_i\Big)\,
        \prod \frac{(T^{(0)})!}{(T^{(1)})!},

and a violation certifies nonclassicality. The admissible pairs are generated
combinatorially (:func:`all_pairs_for_list_of_P`) or loaded from a precomputed
archive (:func:`load_precomputed_pairs`).
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import Counter
from functools import reduce
from itertools import combinations_with_replacement
from math import gcd
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial

logger = logging.getLogger(__name__)

# Precomputed generalized Klyshko / majorization pairs for consecutive
# indices indx = [0, 1, ..., N_P - 1]. File genKlyshko_up_toM.pkl contains
# the pairs for N_P = M + 1, so list_of_files[N_P - 4] is the matching file.
# Paths are resolved relative to the repository root (this file lives in
# src/analytical/), so loading works regardless of the current directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GENK_DIR = os.path.join(_REPO_ROOT, "data", "all_genK")
# Offset between N_P and the index into ``list_of_files`` (up_to3 -> N_P = 4).
_FILE_INDEX_OFFSET = 4
list_of_files = [os.path.join(_GENK_DIR, "genKlyshko_up_to3.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to4.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to5.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to6.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to7.pkl"),
                 os.path.join(_GENK_DIR, "genKlyshko_up_to8.pkl")]


def load_precomputed_pairs(
    indx: ArrayLike,
    lof: Sequence[str] = list_of_files,
) -> list | None:
    """Load precomputed majorization pairs for an index set, if available.

    Precomputed pairs only exist for consecutive indices
    ``indx = [0, 1, ..., N_P - 1]`` within the range covered by ``lof``.

    Parameters
    ----------
    indx : array_like of int
        Photon-number indices present in the distribution.
    lof : sequence of str, optional
        Candidate pickle paths, ordered so that ``lof[N_P - 4]`` matches a
        distribution with ``N_P`` indices. Defaults to :data:`list_of_files`.

    Returns
    -------
    list or None
        The loaded list of pairs, or ``None`` if no precomputed file applies
        (non-consecutive indices, ``N_P`` out of range, or missing file).
    """
    indx = np.asarray(indx, dtype=int).reshape(-1)
    n_probs = len(indx)

    # Precomputed pairs assume consecutive indices starting from 0.
    if not np.array_equal(indx, np.arange(n_probs, dtype=int)):
        return None

    file_idx = n_probs - _FILE_INDEX_OFFSET
    if file_idx < 0 or file_idx >= len(lof):
        return None

    filename = lof[file_idx]
    if not os.path.exists(filename):
        logger.warning("Precomputed pair file not found: %s", filename)
        return None

    try:
        with open(filename, "rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        logger.exception("Failed to load precomputed pairs from %s", filename)
        raise


def generate_arrays(
    n_terms: int,
    total: int,
    indx: ArrayLike | None = None,
) -> NDArray[np.int_]:
    """Enumerate non-increasing integer tuples of fixed length and sum.

    Parameters
    ----------
    n_terms : int
        Number of entries in each tuple.
    total : int
        Required sum of the entries.
    indx : array_like of int, optional
        Allowed entry values. Defaults to ``[0, 1, 2, 3]``.

    Returns
    -------
    numpy.ndarray
        Array of the qualifying tuples, each sorted in descending order.
    """
    if indx is None:
        indx = np.array([0, 1, 2, 3], dtype=int)
    result = []
    for comb in combinations_with_replacement(indx, n_terms):
        if sum(comb) == total:
            sorted_comb = tuple(sorted(comb, reverse=True))
            result.append(sorted_comb)
    return np.array(result)


def majorization_rule(array1: ArrayLike, array2: ArrayLike) -> bool:
    """Return ``True`` if ``array1`` majorizes ``array2`` (cumulative sums)."""
    return bool(np.all(np.cumsum(array1) >= np.cumsum(array2)))


def generate_majorization_pairs(arrays: Sequence[ArrayLike]) -> list[tuple]:
    """Return all ordered pairs ``(a, b)`` with ``a`` majorizing ``b``."""
    pairs = []
    for i in range(len(arrays)):
        for j in range(len(arrays)):
            if majorization_rule(arrays[i], arrays[j]):
                pairs.append((arrays[i], arrays[j]))
    return pairs


def filter_non_overlapping_pairs(pairs: Iterable[tuple]) -> list[tuple]:
    """Keep only pairs whose two tuples share no common index value."""
    filtered_pairs = []
    for pair in pairs:
        array1, array2 = set(pair[0]), set(pair[1])
        if array1.isdisjoint(array2):
            filtered_pairs.append(pair)
    return filtered_pairs


def gcd_of_list(values: Sequence[int]) -> int:
    """Greatest common divisor of a list of integers."""
    return reduce(gcd, values)


def multiplicities_gcd(array: ArrayLike) -> int:
    """GCD of the value multiplicities (counts) of ``array``."""
    counts = Counter(array)
    return gcd_of_list(list(counts.values()))


def filter_pairs_with_common_factors(pairs: Iterable[tuple]) -> list[tuple]:
    """Drop pairs that are integer multiples of a smaller pair.

    A pair is kept only if the GCD of the combined value multiplicities is 1,
    i.e. it is primitive and not a repeated copy of a lower-order inequality.
    """
    final_pairs = []
    for pair in pairs:
        aux = np.append(pair[0], pair[1])
        if multiplicities_gcd(aux) == 1:
            final_pairs.append(pair)
    return final_pairs


def all_pairs_for_list_of_P(indx: ArrayLike | None = None) -> list[tuple]:
    """Generate all primitive majorization pairs for the given indices.

    Sweeps tuple length and tuple sum, building majorization pairs and applying
    the non-overlap and primitivity filters. This is the expensive combinatorial
    construction that :func:`load_precomputed_pairs` caches to disk.

    Parameters
    ----------
    indx : array_like of int, optional
        Allowed photon-number indices. Defaults to ``[0, 1, 2, 3]``.

    Returns
    -------
    list of tuple
        The admissible ``(T0, T1)`` majorization pairs.
    """
    if indx is None:
        indx = np.array([0, 1, 2, 3])

    total_max = np.sum(range(np.max(indx) + 1), dtype=int)
    n_terms_max = total_max

    all_pairs = []
    for n_terms in range(n_terms_max + 1):
        for total in range(1, total_max + 1):
            arrays = generate_arrays(n_terms, total, indx=indx)
            pairs = generate_majorization_pairs(arrays)
            pairs = filter_non_overlapping_pairs(pairs)
            pairs = filter_pairs_with_common_factors(pairs)
            all_pairs.extend(pairs)
    return all_pairs


def build_reverse_index(indx: ArrayLike) -> NDArray[np.int_]:
    """Map a physical photon number to its column position in ``Pall``.

    Parameters
    ----------
    indx : array_like of int
        Physical indices stored along the last axis of ``Pall``.

    Returns
    -------
    numpy.ndarray
        Array ``rev`` with ``rev[n] = j`` where ``indx[j] == n``.
    """
    indx = np.asarray(indx, dtype=int).reshape(-1)
    rev_index = np.zeros((np.max(indx) + 1), dtype=int)
    for position, physical in enumerate(indx):
        rev_index[physical] = position
    return rev_index


def nonclass_from_pairs(
    Pall: NDArray[np.float64],
    pairs: Iterable[tuple],
    rev_index: NDArray[np.int_],
) -> NDArray[np.bool_]:
    r"""Boolean nonclassicality map from a set of majorization pairs.

    For each pair ``(T0, T1)`` flags the grid points violating the classical
    bound :math:`\prod_{i\in T1} P_i > c\,\prod_{i\in T0} P_i` with
    :math:`c = \prod (T0)!/(T1)!`, accumulating the union over all pairs. Shared
    by :func:`analytical_m` and the ``generalized_klyshko*`` criteria.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid; last axis indexed via ``rev_index``.
    pairs : iterable of tuple
        Majorization pairs ``(T0, T1)``.
    rev_index : numpy.ndarray
        Output of :func:`build_reverse_index`.

    Returns
    -------
    numpy.ndarray of bool, shape (Ny, Nx)
        ``True`` where at least one pair's classical bound is violated.
    """
    grid_shape = Pall.shape[:2]
    nonclass = np.zeros(grid_shape, dtype=bool)

    for pair in pairs:
        fact0 = factorial(pair[0])
        fact1 = factorial(pair[1])
        prefactor = np.prod(fact0 / fact1)

        product0 = prefactor * np.ones(grid_shape)
        product1 = np.ones(grid_shape)

        for i in pair[1]:
            product1 *= Pall[:, :, rev_index[i]]
        for i in pair[0]:
            product0 *= Pall[:, :, rev_index[i]]

        violation = product1 - product0
        nonclass = np.where(violation > 0, True, nonclass)

    return nonclass


def analytical_m(
    Pall: NDArray[np.float64],
    indx: ArrayLike | None = None,
    lof: Sequence[str] = list_of_files,
) -> tuple[NDArray[np.bool_], float, int]:
    """Evaluate the majorization (generalized Klyshko) criterion over a grid.

    Uses precomputed pairs when available (consecutive indices in range),
    otherwise generates them on the fly via :func:`all_pairs_for_list_of_P`.

    Parameters
    ----------
    Pall : numpy.ndarray, shape (Ny, Nx, N_P)
        Probability grid. The last axis holds ``N_P`` probabilities.
    indx : array_like of int, optional
        Physical photon-number indices in ``Pall``. Defaults to
        ``[0, 1, ..., N_P - 1]`` derived from ``Pall``.
    lof : sequence of str, optional
        Precomputed-pair file paths. Defaults to :data:`list_of_files`.

    Returns
    -------
    nonclass : numpy.ndarray of bool, shape (Ny, Nx)
        Nonclassicality map.
    ncaf : float
        Fraction of nonclassical grid points.
    n_pairs : int
        Number of majorization pairs evaluated.

    Raises
    ------
    ValueError
        If ``len(indx)`` does not match ``Pall.shape[-1]``.
    """
    Psh = Pall.shape
    n_probs = Psh[-1]

    if indx is None:
        indx = np.arange(n_probs, dtype=int)
    indx = np.asarray(indx, dtype=int).reshape(-1)

    if n_probs != len(indx):
        raise ValueError(
            f"Pall.shape[-1]={n_probs} but len(indx)={len(indx)}"
        )

    rev_index = build_reverse_index(indx)

    # Use precomputed majorization pairs when available, otherwise generate.
    all_pairs = load_precomputed_pairs(indx, lof=lof)
    if all_pairs is None:
        all_pairs = all_pairs_for_list_of_P(indx=indx)
    logger.debug("analytical_m evaluating %d majorization pairs", len(all_pairs))

    nonclass = nonclass_from_pairs(Pall, all_pairs, rev_index)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)
