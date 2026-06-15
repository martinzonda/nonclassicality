"""
2D parameter-scan comparison of the numerical LP solver against the analytical
nonclassicality criteria.

``run2D`` is extracted verbatim (Phase 2) from ``optimization_reworked.ipynb``.
It sweeps a 2D grid of probability vectors, classifies each point as classical
or nonclassical with (a) the LP cutting-plane solver and (b) the analytical
criteria (analytical_k and analytical_m), extracts the decision boundaries, and
writes everything to a .npy archive.

NOTE: naming, type hints, docstrings, logging and config-driven tolerances are
handled in Phase 3 / Phase 4.
"""

import numpy as np
from scipy.special import factorial
from tqdm import tqdm

from src.solvers.lp_solver import max_inner_value, solve_semi_infinite_lp
from src.analytical.criteria_k import analytical_k
from src.analytical.criteria_majorization import analytical_m
from src.visualization.boundaries import extract_binary_boundary


def run2D(N0=512, N1=512, indx=np.array([0, 1, 2, 3], dtype=np.int16),
          dPA=0.0, uPA=1.0, dPB=0.0, uPB=0.4,
          Prest=np.array([0.05, 0.05]), fi="aux.npy"):
    N_P = len(indx)

    PA_all = np.linspace(dPA, uPA, num=N0, endpoint=True)
    PB_all = np.linspace(dPB, uPB, num=N1, endpoint=True)

    Pall = np.empty((N1, N0, N_P))

    Pall[..., 0] = PA_all[None, :]
    Pall[..., 1] = PB_all[:, None]
    Pall[..., 2:] = Prest[None, None, :]

    fact = np.array([1.0 / factorial(i1) for i1 in range(np.max(indx) + 1)])

    nonclass_num = np.zeros((N1, N0), dtype=np.int8)
    nonclass_c = np.zeros((N1, N0))
    all_lambda = np.zeros((*nonclass_num.shape, N_P))
    tol = 1e-10
    bias = 1e-8

    bounds = [(-1.0, 1.0)] * N_P

    mu_star = 0.0

    for i0 in tqdm(range(N0)):
        lam_aux = np.zeros(N_P)
        for i1 in range(N1):
            Pin = Pall[i1, i0, :]

            w, mu_star = max_inner_value(indx, lam_aux, fact)
            true_obj = lam_aux @ Pin - w
            # true_obj = 0
            if (true_obj > bias):
                nonclass_num[i1, i0] = 1
                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = lam_aux

            else:
                result = solve_semi_infinite_lp(Pin, indx, fact, bounds,
                                                mu_init=(0,),
                                                tol=tol, max_iter=300, imtol=1e-8, lp_failed_continue=True)

                w, mu_star = max_inner_value(indx, result["lam"], fact)
                true_obj = result["lam"] @ Pin - w

                if (true_obj > bias):
                    nonclass_num[i1, i0] = 1
                    lam_aux = result["lam"]

                nonclass_c[i1, i0] = true_obj
                all_lambda[i1, i0, :] = result["lam"]

    boundaries = extract_binary_boundary(PA_all, PB_all, nonclass_num)

    # Analytical
    nonclass_anaK, ncaf_anaK = analytical_k(Pall, fact, indx=indx)
    boundaries_anaK = extract_binary_boundary(PA_all, PB_all, nonclass_anaK)

    # Gen Klyshko
    nonclass_anaM, ncaf_anaM, nagM = analytical_m(Pall, indx=indx)
    print("Number=", nagM)
    boundaries_anaM = extract_binary_boundary(PA_all, PB_all, nonclass_anaM)

    nonclass_ana = np.where(nonclass_anaK + nonclass_anaM > 0.5, 1, 0)
    boundaries_ana = extract_binary_boundary(PA_all, PB_all, nonclass_ana)

    with open(fi, 'wb') as f:
        np.save(f, PA_all)
        np.save(f, PB_all)
        np.save(f, Prest)
        np.save(f, np.array(boundaries, dtype=object), allow_pickle=True)
        np.save(f, np.array(boundaries_ana, dtype=object), allow_pickle=True)
        np.save(f, np.array(boundaries_anaK, dtype=object), allow_pickle=True)
        np.save(f, np.array(boundaries_anaM, dtype=object), allow_pickle=True)
        np.save(f, nonclass_num)
        np.save(f, nonclass_ana)
        return boundaries, boundaries_ana, boundaries_anaK, boundaries_anaM
