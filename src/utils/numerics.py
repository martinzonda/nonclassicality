"""
Shared numerical helpers.

``collect_lam`` is extracted verbatim (Phase 2) from
``mayorisation_functions.ipynb``. ``inverse_factorials`` factors out the
``np.array([1.0/factorial(i) for i in range(...)])`` idiom that is repeated
across the solver and scan code (the ``fact`` vector, where ``fact[k] = 1/k!``).

NOTE: type hints, docstrings and de-duplication of ``fact`` construction across
modules are completed in Phase 3.
"""

import numpy as np
from scipy.special import factorial


def inverse_factorials(max_index):
    """Return the vector fact[k] = 1/k! for k = 0, ..., max_index."""
    return np.array([1.0 / factorial(i1) for i1 in range(max_index + 1)])


def collect_lam(loss_mem, lam_mem, xp, yp, delta=1e-6):

    Nx, Ny, Np = lam_mem.shape[0], lam_mem.shape[1], lam_mem.shape[2]

    indices = np.where(loss_mem > delta)
    lam = np.empty((len(indices[0]), 2 + Np))

    for idx in range(len(indices[0])):
        i, j = indices[1][idx], indices[0][idx]
        lam[idx, :2] = [xp[i], yp[j]]  # store the indices
        lam[idx, 2:] = lam_mem[i, j, :]  # store the vector

    return lam
