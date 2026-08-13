r"""2D parameter-scan comparison of the LP solver vs analytical criteria.

:func:`analyze_2d` is the main entry point: pass it any already-built
probability grid and choose which analytical criteria to evaluate.  :func:`run2D`
is only a convenience wrapper for the common probability-plane scan where the
two coordinates are themselves two probabilities and all remaining
probabilities are fixed.
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from src.solvers.lp_solver import (
    max_inner_value,
    physical_probability_mask,
    pvec_mu,
    solve_semi_infinite_lp,
)
from src.analytical.criteria_k import analytical_k
from src.analytical.criteria_majorization import analytical_m
from src.analytical.criteria_counting_matrix import analytical_c
from src.visualization.boundaries import extract_binary_boundary
from src.config import SCAN
from src.utils.numerics import inverse_factorials

logger = logging.getLogger(__name__)

# Default per-component bounds on the witness coefficients lambda.
_LAMBDA_BOUND = SCAN.lambda_bound

ScanResult = dict[str, object]


def _analytical_order(
    Pall: NDArray[np.float64],
    indx: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Return a photon-number-sorted view for analytical criteria.

    ``run2D`` uses the input order to decide which two probabilities are swept,
    while analytical criteria expect columns ordered by physical photon number.
    """
    order = np.argsort(indx)
    return Pall[..., order], indx[order]


def _as_unique_index(indx: ArrayLike | None, n_probs: int) -> NDArray[np.int_]:
    """Normalize and validate photon-number indices."""
    if indx is None:
        indx = np.arange(n_probs, dtype=int)
    indx = np.asarray(indx, dtype=int).reshape(-1)

    if len(indx) != n_probs:
        raise ValueError(f"len(indx)={len(indx)} but Pall.shape[-1]={n_probs}")
    if len(np.unique(indx)) != n_probs:
        raise ValueError("indx entries must be unique")
    if np.any(indx < 0):
        raise ValueError("indx entries must be non-negative")

    return indx


def _criterion_map(
    nonclass: ArrayLike,
    physical_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return a float map with non-physical points represented by NaN."""
    nonclass_map = np.asarray(nonclass, dtype=float)
    nonclass_map[~physical_mask] = np.nan
    return nonclass_map


def _empty_criterion_map(
    shape: tuple[int, int],
    physical_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return a disabled-criterion map: zero on valid points, NaN otherwise."""
    nonclass = np.zeros(shape, dtype=float)
    nonclass[~physical_mask] = np.nan
    return nonclass


def _combine_analytical_maps(
    maps: list[NDArray[np.float64]],
    physical_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Union enabled analytical maps into one nonclassicality map."""
    combined = np.zeros(physical_mask.shape, dtype=float)
    if maps:
        valid_stack = np.stack([np.nan_to_num(m, nan=0.0) > 0.5 for m in maps])
        combined = np.any(valid_stack, axis=0).astype(float)
    combined[~physical_mask] = np.nan
    return combined


def _ncaf(nonclass: NDArray[np.float64], physical_mask: NDArray[np.bool_]) -> float:
    """Fraction of valid grid points classified as nonclassical."""
    valid_count = int(np.count_nonzero(physical_mask))
    if valid_count == 0:
        return 0.0
    return float(np.nansum(nonclass > 0.5) / valid_count)


def _as_boundary_array(boundaries: list) -> NDArray[np.object_]:
    """Represent ragged boundary lists as a NumPy object array for archives."""
    array = np.empty(len(boundaries), dtype=object)
    array[:] = list(boundaries)
    return array


def _boundary_list(value: object) -> list:
    """Convert a saved object-array boundary back to a plain Python list."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def save_scan_result(filename: str, result: ScanResult) -> None:
    """Save a 2D scan result as a named-array archive.

    The file is written to exactly ``filename``. Use a ``.npz`` extension for
    clarity, because the contents are a key-addressable NumPy archive.
    """
    numerical = result["numerical"]
    analytical = result["analytical"]
    ana_k = result["ana_k"]
    ana_m = result["ana_m"]
    counting_matrix = result["counting_matrix"]

    archive = {
        "x": result["x"],
        "y": result["y"],
        "indx": result["indx"],
        "fixed": result["fixed"],
        "physical_mask": result["physical_mask"],
        "numerical_nonclass": numerical["nonclass"],
        "numerical_witness": numerical["witness"],
        "numerical_lambda": numerical["lambda"],
        "numerical_boundaries": _as_boundary_array(numerical["boundaries"]),
        "analytical_nonclass": analytical["nonclass"],
        "analytical_boundaries": _as_boundary_array(analytical["boundaries"]),
        "analytical_criteria": np.array(analytical["criteria"], dtype=object),
        "ana_k_enabled": np.array(ana_k["enabled"], dtype=bool),
        "ana_k_nonclass": ana_k["nonclass"],
        "ana_k_ncaf": np.array(ana_k["ncaf"], dtype=float),
        "ana_k_boundaries": _as_boundary_array(ana_k["boundaries"]),
        "ana_m_enabled": np.array(ana_m["enabled"], dtype=bool),
        "ana_m_nonclass": ana_m["nonclass"],
        "ana_m_ncaf": np.array(ana_m["ncaf"], dtype=float),
        "ana_m_n_pairs": np.array(ana_m["n_pairs"], dtype=int),
        "ana_m_boundaries": _as_boundary_array(ana_m["boundaries"]),
        "counting_matrix_enabled": np.array(counting_matrix["enabled"], dtype=bool),
        "counting_matrix_nonclass": counting_matrix["nonclass"],
        "counting_matrix_ncaf": np.array(counting_matrix["ncaf"], dtype=float),
        "counting_matrix_boundaries": _as_boundary_array(counting_matrix["boundaries"]),
    }

    with open(filename, "wb") as handle:
        np.savez_compressed(handle, **archive)


def load_scan_result(filename: str) -> ScanResult:
    """Load a named-array archive produced by :func:`save_scan_result`."""
    with np.load(filename, allow_pickle=True) as data:
        result: ScanResult = {
            "x": data["x"],
            "y": data["y"],
            "indx": data["indx"],
            "fixed": data["fixed"],
            "physical_mask": data["physical_mask"].astype(bool),
            "numerical": {
                "nonclass": data["numerical_nonclass"],
                "witness": data["numerical_witness"],
                "lambda": data["numerical_lambda"],
                "boundaries": _boundary_list(data["numerical_boundaries"]),
            },
            "ana_k": {
                "enabled": bool(data["ana_k_enabled"].item()),
                "nonclass": data["ana_k_nonclass"],
                "ncaf": float(data["ana_k_ncaf"].item()),
                "boundaries": _boundary_list(data["ana_k_boundaries"]),
            },
            "ana_m": {
                "enabled": bool(data["ana_m_enabled"].item()),
                "nonclass": data["ana_m_nonclass"],
                "ncaf": float(data["ana_m_ncaf"].item()),
                "n_pairs": int(data["ana_m_n_pairs"].item()),
                "boundaries": _boundary_list(data["ana_m_boundaries"]),
            },
            "counting_matrix": {
                "enabled": bool(data["counting_matrix_enabled"].item()),
                "nonclass": data["counting_matrix_nonclass"],
                "ncaf": float(data["counting_matrix_ncaf"].item()),
                "boundaries": _boundary_list(data["counting_matrix_boundaries"]),
            },
            "analytical": {
                "nonclass": data["analytical_nonclass"],
                "boundaries": _boundary_list(data["analytical_boundaries"]),
                "criteria": tuple(data["analytical_criteria"].tolist()),
            },
            "archive_path": filename,
        }

    # Convenient aliases for older plotting snippets and interactive use.
    result["boundaries"] = result["numerical"]["boundaries"]
    result["boundaries_ana"] = result["analytical"]["boundaries"]
    result["boundaries_anaK"] = result["ana_k"]["boundaries"]
    result["boundaries_anaM"] = result["ana_m"]["boundaries"]
    result["boundaries_cm"] = result["counting_matrix"]["boundaries"]
    result["nonclass_num"] = result["numerical"]["nonclass"]
    result["nonclass_ana"] = result["analytical"]["nonclass"]
    result["nonclass_cm"] = result["counting_matrix"]["nonclass"]
    result["ncaf_cm"] = result["counting_matrix"]["ncaf"]
    return result


def analyze_2d(
    Pall: ArrayLike,
    x_values: ArrayLike,
    y_values: ArrayLike,
    indx: ArrayLike | None = None,
    fi: str = "aux.npz",
    fixed_values: ArrayLike | None = None,
    tol: float = SCAN.tol,
    bias: float = SCAN.bias,
    max_iter: int = SCAN.max_iter,
    imtol: float = SCAN.imtol,
    use_analytical_m: bool | None = None,
    use_ana_k: bool = True,
    use_ana_m: bool = True,
    use_cm: bool = True,
    y_scan_direction: str = "increasing",
) -> ScanResult:
    r"""Analyze a 2D grid of probability vectors.

    ``Pall[i_y, i_x, :]`` is the photon-number distribution at coordinate
    ``(x_values[i_x], y_values[i_y])``. This is the general entry point for
    physical model scans, for example a Jaynes-Cummings grid over
    ``(tau, nu)``. The last axis of ``Pall`` must be aligned with ``indx`` but
    ``indx`` may be in any order; analytical criteria are evaluated on an
    internally sorted view.

    Parameters
    ----------
    Pall : array_like, shape (N_y, N_x, N_P)
        Probability grid to classify.
    x_values, y_values : array_like
        Coordinates for the second and first axes of ``Pall``. Boundary curves
        are returned in these coordinates.
    indx : array_like of int, optional
        Photon-number indices corresponding to ``Pall[..., :]``. Defaults to
        ``[0, 1, ..., N_P - 1]``.
    fi : str, optional
        Output named NumPy archive path. Default ``"aux.npz"``.
    fixed_values : array_like, optional
        Optional metadata saved in the archive. For the probability-plane
        wrapper this is ``Prest``; for physical-model scans it can be omitted or
        used for model parameters.
    tol : float, optional
        LP convergence tolerance. Default ``1e-10``.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-8``.
    max_iter : int, optional
        Maximum cutting-plane iterations per point. Default ``300``.
    imtol : float, optional
        Imaginary-part tolerance for the inner maximisation. Default ``1e-8``.
    use_analytical_m : bool, optional
        Deprecated alias for ``use_ana_m``.
    use_ana_k, use_ana_m, use_cm : bool, optional
        Select which analytical criteria to evaluate. ``use_ana_k`` evaluates
        :func:`analytical_k`, ``use_ana_m`` evaluates the majorization criterion
        :func:`analytical_m`, and ``use_cm`` evaluates the ideal photoelectric
        photon-counting matrix criterion :func:`analytical_c` from
        Krishnaswamy et al., arXiv:2603.06237v1. The combined analytical map is
        the union of all enabled analytical criteria.
    y_scan_direction : {"increasing", "decreasing"}, optional
        Direction used for the warm-started LP sweep along the first axis of
        ``Pall``. The saved archive and returned boundaries keep the original
        ``y_values`` order; only the order of LP evaluation changes. Default
        ``"increasing"``.

    Returns
    -------
    dict
        Nested dictionary with coordinates, numerical results, each analytical
        criterion, and the combined analytical result. The most useful keys are
        ``result["numerical"]["nonclass"]``,
        ``result["ana_k"]["nonclass"]``, ``result["ana_m"]["nonclass"]``,
        ``result["counting_matrix"]["nonclass"]``, and
        ``result["analytical"]["nonclass"]``.
    """
    if use_analytical_m is not None:
        use_ana_m = bool(use_analytical_m)

    Pall = np.asarray(Pall, dtype=float)
    if Pall.ndim != 3:
        raise ValueError(f"Pall must have shape (N_y, N_x, N_P), got {Pall.shape}")

    N1, N0, N_P = Pall.shape
    if N_P < 1:
        raise ValueError("Pall must contain at least one probability component")
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    if len(x_values) != N0:
        raise ValueError(f"len(x_values)={len(x_values)} but Pall.shape[1]={N0}")
    if len(y_values) != N1:
        raise ValueError(f"len(y_values)={len(y_values)} but Pall.shape[0]={N1}")
    if y_scan_direction not in {"increasing", "decreasing"}:
        raise ValueError("y_scan_direction must be 'increasing' or 'decreasing'")

    indx = _as_unique_index(indx, N_P)
    if fixed_values is None:
        fixed_values = np.array([])
    fixed_values = np.asarray(fixed_values, dtype=float).reshape(-1)

    Pall_ana, indx_ana = _analytical_order(Pall, indx)

    fact = inverse_factorials(int(np.max(indx)))

    physical_mask = physical_probability_mask(Pall)
    invalid_count = int(np.size(physical_mask) - np.count_nonzero(physical_mask))

    nonclass_num = np.full((N1, N0), np.nan, dtype=float)
    nonclass_num[physical_mask] = 0.0
    nonclass_c = np.full((N1, N0), np.nan, dtype=float)
    all_lambda = np.full((*nonclass_num.shape, N_P), np.nan, dtype=float)

    bounds = [_LAMBDA_BOUND] * N_P

    mu_star = 0.0

    y_order = range(N1) if y_scan_direction == "increasing" else range(N1 - 1, -1, -1)

    for i0 in tqdm(range(N0)):
        lam_aux = np.zeros(N_P)  # warm start carried down each column
        for i1 in y_order:
            if not physical_mask[i1, i0]:
                continue

            Pin = Pall[i1, i0, :]

            # Cheap check first: does the previous column's lambda already
            # certify nonclassicality at this point?
            w, mu_star = max_inner_value(indx, lam_aux, fact)
            true_obj = lam_aux @ Pin - w
            if true_obj > bias:
                nonclass_num[i1, i0] = 1.0
                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = lam_aux
            else:
                # Otherwise solve the full LP at this point.
                result = solve_semi_infinite_lp(
                    Pin, indx, fact, bounds,
                    mu_init=(0,),
                    tol=tol, max_iter=max_iter, imtol=imtol,
                    lp_failed_continue=True,
                    warn_nonphysical=False,
                )

                w, mu_star = max_inner_value(indx, result["lam"], fact)
                true_obj = result["lam"] @ Pin - w

                if true_obj > bias:
                    nonclass_num[i1, i0] = 1.0
                    lam_aux = result["lam"]

                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = result["lam"]

    boundaries = extract_binary_boundary(x_values, y_values, nonclass_num)

    # Analytical criteria expect physical probability vectors. Evaluate them on
    # a harmless classical replacement at invalid points, then restore NaNs.
    Pall_ana_eval = Pall_ana.copy()
    if invalid_count:
        Pall_ana_eval[~physical_mask] = pvec_mu(1.0, indx_ana, fact)

    analytical_maps: list[NDArray[np.float64]] = []
    analytical_criteria: list[str] = []

    if use_ana_k:
        nonclass_anaK_bool, _ = analytical_k(Pall_ana_eval, fact, indx=indx_ana)
        nonclass_anaK = _criterion_map(nonclass_anaK_bool, physical_mask)
        boundaries_anaK = extract_binary_boundary(x_values, y_values, nonclass_anaK)
        analytical_maps.append(nonclass_anaK)
        analytical_criteria.append("ana_k")
    else:
        nonclass_anaK = _empty_criterion_map((N1, N0), physical_mask)
        boundaries_anaK = []
        logger.info("analytical_k skipped")

    if use_ana_m:
        nonclass_anaM, ncaf_anaM, nagM = analytical_m(
            Pall_ana_eval, indx=indx_ana
        )
        nonclass_anaM = _criterion_map(nonclass_anaM, physical_mask)
        logger.info("analytical_m evaluated %d pairs", nagM)
        boundaries_anaM = extract_binary_boundary(x_values, y_values, nonclass_anaM)
        analytical_maps.append(nonclass_anaM)
        analytical_criteria.append("ana_m")
    else:
        nonclass_anaM = _empty_criterion_map((N1, N0), physical_mask)
        ncaf_anaM = 0.0
        nagM = 0
        boundaries_anaM = []
        logger.info("analytical_m skipped")

    if use_cm:
        nonclass_cm_bool, ncaf_cm = analytical_c(
            Pall_ana_eval,
            indx=indx_ana,
        )
        nonclass_cm = _criterion_map(nonclass_cm_bool, physical_mask)
        boundaries_cm = extract_binary_boundary(x_values, y_values, nonclass_cm)
        analytical_maps.append(nonclass_cm)
        analytical_criteria.append("counting_matrix")
    else:
        nonclass_cm = _empty_criterion_map((N1, N0), physical_mask)
        ncaf_cm = 0.0
        boundaries_cm = []
        logger.info("counting matrix skipped")

    nonclass_ana = _combine_analytical_maps(analytical_maps, physical_mask)
    boundaries_ana = extract_binary_boundary(x_values, y_values, nonclass_ana)

    if invalid_count:
        warnings.warn(
            f"{invalid_count} non-physical probability vector(s) were skipped "
            "during the 2D scan and stored as NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    result: ScanResult = {
        "x": x_values,
        "y": y_values,
        "indx": indx,
        "fixed": fixed_values,
        "physical_mask": physical_mask,
        "numerical": {
            "nonclass": nonclass_num,
            "witness": nonclass_c,
            "lambda": all_lambda,
            "boundaries": boundaries,
        },
        "ana_k": {
            "enabled": bool(use_ana_k),
            "nonclass": nonclass_anaK,
            "ncaf": _ncaf(nonclass_anaK, physical_mask),
            "boundaries": boundaries_anaK,
        },
        "ana_m": {
            "enabled": bool(use_ana_m),
            "nonclass": nonclass_anaM,
            "ncaf": _ncaf(nonclass_anaM, physical_mask) if use_ana_m else ncaf_anaM,
            "n_pairs": int(nagM),
            "boundaries": boundaries_anaM,
        },
        "counting_matrix": {
            "enabled": bool(use_cm),
            "nonclass": nonclass_cm,
            "ncaf": _ncaf(nonclass_cm, physical_mask) if use_cm else ncaf_cm,
            "boundaries": boundaries_cm,
        },
        "analytical": {
            "nonclass": nonclass_ana,
            "boundaries": boundaries_ana,
            "criteria": tuple(analytical_criteria),
        },
    }

    result["boundaries"] = boundaries
    result["boundaries_ana"] = boundaries_ana
    result["boundaries_anaK"] = boundaries_anaK
    result["boundaries_anaM"] = boundaries_anaM
    result["boundaries_cm"] = boundaries_cm
    result["nonclass_num"] = nonclass_num
    result["nonclass_ana"] = nonclass_ana
    result["nonclass_cm"] = nonclass_cm
    result["ncaf_cm"] = result["counting_matrix"]["ncaf"]

    if fi:
        save_scan_result(fi, result)
        result["archive_path"] = fi

    return result


def run2D(
    N0: int = SCAN.n0,
    N1: int = SCAN.n1,
    indx: ArrayLike | None = None,
    dPA: float = SCAN.dPA,
    uPA: float = SCAN.uPA,
    dPB: float = SCAN.dPB,
    uPB: float = SCAN.uPB,
    Prest: ArrayLike | None = None,
    fi: str = "aux.npz",
    tol: float = SCAN.tol,
    bias: float = SCAN.bias,
    max_iter: int = SCAN.max_iter,
    imtol: float = SCAN.imtol,
    use_analytical_m: bool | None = None,
    use_ana_k: bool = True,
    use_ana_m: bool = True,
    use_cm: bool = True,
    y_scan_direction: str = "increasing",
) -> ScanResult:
    r"""Build and analyze a 2D probability-plane scan.

    The first two probability components are swept over ``[dPA, uPA]`` and
    ``[dPB, uPB]``; the remaining components are held fixed at ``Prest``. Each
    grid point is classified by the LP solver (warm-started along columns) and
    by the analytical criteria, and the resulting decision boundaries are saved.

    Parameters
    ----------
    N0, N1 : int, optional
        Grid resolution along the two swept axes. Default ``512`` each.
    indx : array_like of int, optional
        Photon-number indices. The first two entries are swept by the scan; any
        remaining entries are fixed to ``Prest``. Defaults to ``[0, 1, 2, 3]``.
    dPA, uPA : float, optional
        Range of the first swept probability. Default ``[0.0, 1.0]``.
    dPB, uPB : float, optional
        Range of the second swept probability. Default ``[0.0, 0.4]``.
    Prest : array_like, optional
        Fixed values of the remaining probability components. Default
        ``[0.05, 0.05]``.
    fi : str, optional
        Output named NumPy archive path. Default ``"aux.npz"``.
    tol : float, optional
        LP convergence tolerance. Default ``1e-10``.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-8``.
    max_iter : int, optional
        Maximum cutting-plane iterations per point. Default ``300``.
    imtol : float, optional
        Imaginary-part tolerance for the inner maximisation. Default ``1e-8``.
    use_analytical_m : bool, optional
        Deprecated alias for ``use_ana_m``.
    use_ana_k, use_ana_m, use_cm : bool, optional
        Select which analytical criteria are included in the scan. ``use_cm``
        enables the ideal photoelectric photon-counting matrix criterion from
        Krishnaswamy et al., arXiv:2603.06237v1. The combined analytical map is
        the union of all enabled analytical criteria.
    y_scan_direction : {"increasing", "decreasing"}, optional
        Direction used for the warm-started LP sweep over the second swept
        probability. The output grid remains ordered by increasing ``PB_all``.
        Default ``"increasing"``.

    Returns
    -------
    dict
        Same structured result dictionary returned by :func:`analyze_2d`.
    """
    if indx is None:
        indx = np.array([0, 1, 2, 3], dtype=np.int16)
    indx = np.asarray(indx, dtype=int).reshape(-1)
    N_P = len(indx)

    if Prest is None:
        default_prest = SCAN.prest if len(SCAN.prest) == N_P - 2 else ()
        Prest = np.array(default_prest)
    Prest = np.asarray(Prest, dtype=float).reshape(-1)

    if N_P < 2:
        raise ValueError("indx must contain at least two entries to scan")
    _as_unique_index(indx, N_P)
    if len(Prest) != N_P - 2:
        raise ValueError(
            f"Prest must have length len(indx)-2={N_P - 2}; got {len(Prest)}"
        )

    PA_all = np.linspace(dPA, uPA, num=N0, endpoint=True)
    PB_all = np.linspace(dPB, uPB, num=N1, endpoint=True)

    Pall = np.empty((N1, N0, N_P))
    Pall[..., 0] = PA_all[None, :]
    Pall[..., 1] = PB_all[:, None]
    Pall[..., 2:] = Prest[None, None, :]

    return analyze_2d(
        Pall,
        PA_all,
        PB_all,
        indx=indx,
        fi=fi,
        fixed_values=Prest,
        tol=tol,
        bias=bias,
        max_iter=max_iter,
        imtol=imtol,
        use_analytical_m=use_analytical_m,
        use_ana_k=use_ana_k,
        use_ana_m=use_ana_m,
        use_cm=use_cm,
        y_scan_direction=y_scan_direction,
    )
