r"""QuTiP density-matrix photon-statistics models used as test inputs.

QuTiP density-matrix generators for thermal, coherent and photon-(boson-)added
states, and helpers to read off their photon-number probabilities.

QuTiP is required for this module; it is declared as an optional extra
(``pip install .[physics]``). Importing this module without QuTiP installed
will fail at the ``import qutip`` line below. The QuTiP-free Jaynes-Cummings
model lives in :mod:`src.analytical.jaynes_cummings`.

NOTE: function names follow the original research code (domain conventions);
their meaning is documented in the docstrings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import factorial

import qutip as qt

# Re-exported for backwards compatibility; the canonical home is
# src.analytical.jaynes_cummings (no QuTiP dependency).
from src.analytical.jaynes_cummings import Pjaycum  # noqa: F401


# Truncated Fock-space dimension and annihilation operator. NOTE: the module
# operator ``a`` is built once at import time from ``Ns``; the QuTiP generators
# below accept an ``Ns`` argument but reuse this fixed operator (preserved from
# the original notebook code -- logic unchanged).
Ns = 100

a = qt.destroy(Ns)


def th_dm_th(Ns: int, mu: float) -> "qt.Qobj":
    """Thermal-state density matrix with mean photon number ``mu``."""
    return qt.thermal_dm(Ns, mu)


def th_dm_co(Ns: int, mu: float) -> "qt.Qobj":
    """Coherent-state density matrix with amplitude ``mu``."""
    return qt.coherent_dm(Ns, mu)


def fs(Ns: int, n: int) -> "qt.Qobj":
    """Fock state ``|n>`` in a Hilbert space of dimension ``Ns``."""
    return qt.states.fock(Ns, n)


def state_dm_th(Ns: int, mu: float, p: float, nad: int = 1) -> "qt.Qobj":
    r"""Photon-added thermal state, ``nad`` probabilistic additions.

    Each addition applies the map :math:`\rho \mapsto p\,a^\dagger\rho a +
    (1-p)\rho`, then the state is renormalized.
    """
    st = (p * a.dag() * th_dm_th(Ns, mu) * a + (1 - p) * th_dm_th(Ns, mu))
    if nad > 1:
        for i in range(nad - 1):
            st = (p * a.dag() * st * a + (1 - p) * st)
    st = st.unit()
    return st


def state_dm_co(Ns: int, mu: float, p: float, nad: int = 1) -> "qt.Qobj":
    """Photon-added coherent state, ``nad`` probabilistic additions."""
    st = (p * a.dag() * th_dm_co(Ns, mu) * a + (1 - p) * th_dm_co(Ns, mu))
    if nad > 1:
        for i in range(nad - 1):
            st = (p * a.dag() * st * a + (1 - p) * st)
    st = st.unit()
    return st


def state_dm_pba_bacs(Ns: int, alpha: float, p: float, l: float, nad: int = 1) -> "qt.Qobj":
    """Photon-added "bacs" state.

    TODO: clarify -- ``state_bacs`` is referenced but not defined in the
    original notebook, so this function is non-functional until supplied.
    """
    rho = state_bacs(Ns, alpha, l) * state_bacs(Ns, alpha, l).dag()  # noqa: F821
    rho = p * a.dag() * rho * a + (1 - p) * rho
    for i in range(nad - 1):
        rho = (p * a.dag() * rho * a + (1 - p) * rho)

    rho = rho.unit()
    return rho


def state_dm_pba_co(Ns: int, alpha: float, p: float, nad: int = 1) -> "qt.Qobj":
    """Photon-added coherent state with per-step renormalized addition."""
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * rho * a).unit() + (1 - p) * rho
    for i in range(nad - 1):
        rho = (p * (a.dag() * rho * a).unit() + (1 - p) * rho)

    rho = rho.unit()
    return rho


def state_dm_pba_co_2_3(Ns: int, alpha: float, p: float, nad: int = 1) -> "qt.Qobj":
    """Coherent state mixing triple- and double-photon addition."""
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * a.dag() * a.dag() * rho * a * a * a).unit() + (1 - p) * (a.dag() * a.dag() * rho * a * a).unit()

    rho = rho.unit()
    return rho


def state_dm_pba_co_2_0(Ns: int, alpha: float, p: float, nad: int = 1) -> "qt.Qobj":
    """Coherent state mixing double- and single-photon addition."""
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * a.dag() * rho * a * a).unit() + (1 - p) * (a.dag() * rho * a).unit()

    rho = rho.unit()
    return rho


def test_P_pba(mu: float, p: float, l: float, k: int) -> float:
    """Closed-form test probability ``P_k`` for a photon-added model."""
    return (np.exp(-mu) / factorial(k)) * ((mu**(k - 1) * p * k**2 / (mu + 1)) + (1 - p) * mu**k)


def pba_Pk_bacs(Ns: int, mu: float, p: float, l: float, k: int, nad: int = 1) -> float:
    """Photon-number probability ``P_k`` of the photon-added bacs state."""
    alpha = np.sqrt(mu)
    st = state_dm_pba_bacs(Ns, alpha, p, l, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co(Ns: int, mu: float, p: float, k: int, nad: int = 1) -> float:
    """Photon-number probability ``P_k`` of the photon-added coherent state."""
    alpha = np.sqrt(mu)
    st = state_dm_pba_co(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co_2_3(Ns: int, mu: float, p: float, k: int, nad: int = 1) -> float:
    """``P_k`` for the 2-/3-photon-added coherent state (alpha is a guess)."""
    alpha = mu / np.sqrt(2)  # This is not tested, it is a guess
    st = state_dm_pba_co_2_3(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co_2_0(Ns: int, mu: float, p: float, k: int, nad: int = 1) -> float:
    """``P_k`` for the 2-/0-photon-added coherent state (alpha is a guess)."""
    alpha = mu / np.sqrt(2)  # This is not tested, it is a guess
    st = state_dm_pba_co_2_0(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)
