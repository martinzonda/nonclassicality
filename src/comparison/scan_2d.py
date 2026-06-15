r"""2D parameter-scan comparison of the LP solver vs analytical criteria.

:func:`run2D` sweeps a 2D grid of probability vectors, classifies each point as
classical or nonclassical with both the LP cutting-plane solver and the
analytical criteria (``analytical_k`` and ``analytical_m``), extracts the
decision boundaries, and writes everything to a ``.npy`` archive.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from src.solvers.lp_solver import max_inner_value, solve_semi_infinite_lp
from src.analytical.criteria_k import analytical_k
from src.analytical.criteria_majorization import analytical_m
from src.visualization.boundaries import extract_binary_boundary
from src.utils.numerics import inverse_factorials
from src.utils.io import save_arrays

logger = logging.getLogger(__name__)

# Default per-component bounds on the witness coefficients lambda.
_LAMBDA_BOUND = (-1.0, 1.0)


def run2D(
    N0: int = 512,
    N1: int = 512,
    indx: ArrayLike | None = None,
    dPA: float = 0.0,
    uPA: float = 1.0,
    dPB: float = 0.0,
    uPB: float = 0.4,
    Prest: ArrayLike | None = None,
    fi: str = "aux.npy",
    tol: float = 1e-10,
    bias: float = 1e-8,
    max_iter: int = 300,
    imtol: float = 1e-8,
) -> tuple[list, list, list, list]:
    r"""Scan a 2D probability grid and compare numerical and analytical maps.

    The first two probability components are swept over ``[dPA, uPA]`` and
    ``[dPB, uPB]``; the remaining components are held fixed at ``Prest``. Each
    grid point is classified by the LP solver (warm-started along columns) and
    by the analytical criteria, and the resulting decision boundaries are saved.

    Parameters
    ----------
    N0, N1 : int, optional
        Grid resolution along the two swept axes. Default ``512`` each.
    indx : array_like of int, optional
        Photon-number indices. Defaults to ``[0, 1, 2, 3]``.
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
    indx = np.asarray(indx)
    if Prest is None:
        Prest = np.array([0.05, 0.05])
    Prest = np.asarray(Prest, dtype=float)

    N_P = len(indx)

    PA_all = np.linspace(dPA, uPA, num=N0, endpoint=True)
    PB_all = np.linspace(dPB, uPB, num=N1, endpoint=True)

    Pall = np.empty((N1, N0, N_P))
    Pall[..., 0] = PA_all[None, :]
    Pall[..., 1] = PB_all[:, None]
    Pall[..., 2:] = Prest[None, None, :]

    fact = inverse_factorials(int(np.max(indx)))

    nonclass_num = np.zeros((N1, N0), dtype=np.int8)
    nonclass_c = np.zeros((N1, N0))
    all_lambda = np.zeros((*nonclass_num.shape, N_P))

    bounds = [_LAMBDA_BOUND] * N_P

    mu_star = 0.0

    for i0 in tqdm(range(N0)):
        lam_aux = np.zeros(N_P)  # warm start carried down each column
        for i1 in range(N1):
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

    boundaries = extract_binary_boundary(PA_all, PB_all, nonclass_num)

    # Analytical "K" criteria.
    nonclass_anaK, ncaf_anaK = analytical_k(Pall, fact, indx=indx)
    boundaries_anaK = extract_binary_boundary(PA_all, PB_all, nonclass_anaK)

    # Generalized Klyshko (majorization) criterion.
    nonclass_anaM, ncaf_anaM, nagM = analytical_m(Pall, indx=indx)
    logger.info("analytical_m evaluated %d pairs", nagM)
    boundaries_anaM = extract_binary_boundary(PA_all, PB_all, nonclass_anaM)

    nonclass_ana = np.where(nonclass_anaK + nonclass_anaM > 0.5, 1, 0)
    boundaries_ana = extract_binary_boundary(PA_all, PB_all, nonclass_ana)

    save_arrays(
        fi,
        PA_all,
        PB_all,
        Prest,
        np.array(boundaries, dtype=object),
        np.array(boundaries_ana, dtype=object),
        np.array(boundaries_anaK, dtype=object),
        np.array(boundaries_anaM, dtype=object),
        nonclass_num,
        nonclass_ana,
        allow_pickle=True,
    )

    return boundaries, boundaries_ana, boundaries_anaK, boundaries_anaM
