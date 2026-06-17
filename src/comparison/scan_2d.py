r"""2D parameter-scan comparison of the LP solver vs analytical criteria.

:func:`analyze_2d` classifies an already-built 2D grid of probability vectors.
:func:`run2D` is a convenience wrapper for the special case where the two scan
coordinates are themselves two probabilities and all remaining probabilities
are fixed.
"""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from src.solvers.lp_solver import max_inner_value, solve_semi_infinite_lp
from src.analytical.criteria_k import analytical_k
from src.analytical.criteria_majorization import analytical_m
from src.visualization.boundaries import extract_binary_boundary
from src.config import SCAN
from src.utils.numerics import inverse_factorials
from src.utils.io import save_arrays

logger = logging.getLogger(__name__)

# Default per-component bounds on the witness coefficients lambda.
_LAMBDA_BOUND = SCAN.lambda_bound


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


def analyze_2d(
    Pall: ArrayLike,
    x_values: ArrayLike,
    y_values: ArrayLike,
    indx: ArrayLike | None = None,
    fi: str = "aux.npy",
    fixed_values: ArrayLike | None = None,
    tol: float = SCAN.tol,
    bias: float = SCAN.bias,
    max_iter: int = SCAN.max_iter,
    imtol: float = SCAN.imtol,
    use_analytical_m: bool = True,
    y_scan_direction: str = "increasing",
) -> tuple[list, list, list, list]:
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
        Output ``.npy`` archive path. Default ``"aux.npy"``.
    fixed_values : array_like, optional
        Optional metadata saved in the archive as the third array. For the
        probability-plane wrapper this is ``Prest``; for physical-model scans it
        can be omitted or used for model parameters.
    tol : float, optional
        LP convergence tolerance. Default ``1e-10``.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-8``.
    max_iter : int, optional
        Maximum cutting-plane iterations per point. Default ``300``.
    imtol : float, optional
        Imaginary-part tolerance for the inner maximisation. Default ``1e-8``.
    use_analytical_m : bool, optional
        Whether to evaluate the majorization/generalized-Klyshko criterion.
        Default ``True``. Set to ``False`` to skip the expensive ``analytical_m``
        pass; then the combined analytical map contains only ``analytical_k``
        and ``boundaries_anaM`` is empty.
    y_scan_direction : {"increasing", "decreasing"}, optional
        Direction used for the warm-started LP sweep along the first axis of
        ``Pall``. The saved arrays and returned boundaries keep the original
        ``y_values`` order; only the order of LP evaluation changes. Default
        ``"increasing"``.

    Returns
    -------
    boundaries : list
        LP-solver decision boundaries.
    boundaries_ana : list
        Combined analytical (K + M) boundaries.
    boundaries_anaK : list
        ``analytical_k`` boundaries.
    boundaries_anaM : list
        ``analytical_m`` (majorization) boundaries.
    """
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

    nonclass_num = np.zeros((N1, N0), dtype=np.int8)
    nonclass_c = np.zeros((N1, N0))
    all_lambda = np.zeros((*nonclass_num.shape, N_P))

    bounds = [_LAMBDA_BOUND] * N_P

    mu_star = 0.0

    y_order = range(N1) if y_scan_direction == "increasing" else range(N1 - 1, -1, -1)

    for i0 in tqdm(range(N0)):
        lam_aux = np.zeros(N_P)  # warm start carried down each column
        for i1 in y_order:
            Pin = Pall[i1, i0, :]

            # Cheap check first: does the previous column's lambda already
            # certify nonclassicality at this point?
            w, mu_star = max_inner_value(indx, lam_aux, fact)
            true_obj = lam_aux @ Pin - w
            if true_obj > bias:
                nonclass_num[i1, i0] = 1
                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = lam_aux
            else:
                # Otherwise solve the full LP at this point.
                result = solve_semi_infinite_lp(
                    Pin, indx, fact, bounds,
                    mu_init=(0,),
                    tol=tol, max_iter=max_iter, imtol=imtol,
                    lp_failed_continue=True,
                )

                w, mu_star = max_inner_value(indx, result["lam"], fact)
                true_obj = result["lam"] @ Pin - w

                if true_obj > bias:
                    nonclass_num[i1, i0] = 1
                    lam_aux = result["lam"]

                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = result["lam"]

    boundaries = extract_binary_boundary(x_values, y_values, nonclass_num)

    # Analytical "K" criteria.
    nonclass_anaK, ncaf_anaK = analytical_k(Pall_ana, fact, indx=indx_ana)
    boundaries_anaK = extract_binary_boundary(x_values, y_values, nonclass_anaK)

    # Generalized Klyshko (majorization) criterion.
    if use_analytical_m:
        nonclass_anaM, ncaf_anaM, nagM = analytical_m(Pall_ana, indx=indx_ana)
        logger.info("analytical_m evaluated %d pairs", nagM)
        boundaries_anaM = extract_binary_boundary(x_values, y_values, nonclass_anaM)
    else:
        nonclass_anaM = np.zeros_like(nonclass_anaK, dtype=bool)
        boundaries_anaM = []
        logger.info("analytical_m skipped")

    nonclass_ana = np.where(nonclass_anaK + nonclass_anaM > 0.5, 1, 0)
    boundaries_ana = extract_binary_boundary(x_values, y_values, nonclass_ana)

    save_arrays(
        fi,
        x_values,
        y_values,
        fixed_values,
        np.array(boundaries, dtype=object),
        np.array(boundaries_ana, dtype=object),
        np.array(boundaries_anaK, dtype=object),
        np.array(boundaries_anaM, dtype=object),
        nonclass_num,
        nonclass_ana,
        allow_pickle=True,
    )

    return boundaries, boundaries_ana, boundaries_anaK, boundaries_anaM


def run2D(
    N0: int = SCAN.n0,
    N1: int = SCAN.n1,
    indx: ArrayLike | None = None,
    dPA: float = SCAN.dPA,
    uPA: float = SCAN.uPA,
    dPB: float = SCAN.dPB,
    uPB: float = SCAN.uPB,
    Prest: ArrayLike | None = None,
    fi: str = "aux.npy",
    tol: float = SCAN.tol,
    bias: float = SCAN.bias,
    max_iter: int = SCAN.max_iter,
    imtol: float = SCAN.imtol,
    use_analytical_m: bool = True,
    y_scan_direction: str = "increasing",
) -> tuple[list, list, list, list]:
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
        Output ``.npy`` archive path. Default ``"aux.npy"``.
    tol : float, optional
        LP convergence tolerance. Default ``1e-10``.
    bias : float, optional
        Minimum witness value to declare a point nonclassical. Default ``1e-8``.
    max_iter : int, optional
        Maximum cutting-plane iterations per point. Default ``300``.
    imtol : float, optional
        Imaginary-part tolerance for the inner maximisation. Default ``1e-8``.
    use_analytical_m : bool, optional
        Whether to evaluate the majorization/generalized-Klyshko criterion.
        Default ``True``. Set to ``False`` to skip the expensive ``analytical_m``
        pass; then the combined analytical map contains only ``analytical_k``
        and ``boundaries_anaM`` is empty.
    y_scan_direction : {"increasing", "decreasing"}, optional
        Direction used for the warm-started LP sweep over the second swept
        probability. The output grid remains ordered by increasing ``PB_all``.
        Default ``"increasing"``.

    Returns
    -------
    boundaries : list
        LP-solver decision boundaries.
    boundaries_ana : list
        Combined analytical (K + M) boundaries.
    boundaries_anaK : list
        ``analytical_k`` boundaries.
    boundaries_anaM : list
        ``analytical_m`` (majorization) boundaries.
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
        y_scan_direction=y_scan_direction,
    )
