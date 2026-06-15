r"""Jaynes-Cummings photon-number statistics (no QuTiP dependency).

Separated from :mod:`src.analytical.models` so that the analytical
Jaynes-Cummings probabilities can be used without importing QuTiP (which the
density-matrix generators in ``models`` require).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def Pjaycum(
    n: int,
    nu: ArrayLike,
    tau: ArrayLike,
    pe: float,
) -> NDArray[np.float64]:
    r"""Jaynes-Cummings photon-number probability :math:`P_n(\nu, \tau)`.

    Analytical occupation of Fock level ``n`` for a thermal field of mean
    photon number ``nu`` interacting with a two-level atom (excited-state
    probability ``pe``) for scaled time ``tau``.

    Parameters
    ----------
    n : int
        Photon number (Fock level).
    nu : array_like
        Mean photon number :math:`\bar n` of the thermal field.
    tau : array_like
        Scaled interaction time :math:`\tau = g t`.
    pe : float
        Initial excited-state probability of the atom, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        :math:`P_n` broadcast over ``nu`` and ``tau``.
    """
    nuf1 = nu**n / (1 + nu)**(1 + n)
    nuf2 = (1 + nu) / nu
    nuf3 = nu / (1 + nu)
    resA = (np.cos(tau * np.sqrt(n + 1)))**2 + nuf2 * (np.sin(tau * np.sqrt(n)))**2
    resB = (np.cos(tau * np.sqrt(n)))**2 + nuf3 * (np.sin(tau * np.sqrt(n + 1)))**2
    res = nuf1 * (pe * resA + (1.0 - pe) * resB)
    return res
