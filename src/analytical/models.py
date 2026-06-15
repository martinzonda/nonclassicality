"""
Physical photon-statistics models used as test inputs for the nonclassicality
criteria and solvers.

Extracted verbatim (Phase 2) from ``optimization_reworked.ipynb`` (Pjaycum) and
``mayorisation_functions.ipynb`` (QuTiP density-matrix generators for thermal,
coherent and photon-(boson-)added states).

QuTiP is only required for the density-matrix generators below; it is declared
as an optional extra. ``Pjaycum`` has no QuTiP dependency.

NOTE: naming, type hints, docstrings and logging are handled in Phase 3.
"""

import numpy as np
from scipy.special import factorial

import qutip as qt


def Pjaycum(n, nu, tau, pe):
    nuf1 = nu**n / (1 + nu)**(1 + n)
    nuf2 = (1 + nu) / nu
    nuf3 = nu / (1 + nu)
    resA = (np.cos(tau * np.sqrt(n + 1)))**2 + nuf2 * (np.sin(tau * np.sqrt(n)))**2
    resB = (np.cos(tau * np.sqrt(n)))**2 + nuf3 * (np.sin(tau * np.sqrt(n + 1)))**2
    res = nuf1 * (pe * resA + (1.0 - pe) * resB)
    return res


# Truncated Fock-space dimension and annihilation operator. NOTE: the module
# operator ``a`` is built once at import time from ``Ns``; the QuTiP generators
# below accept an ``Ns`` argument but reuse this fixed operator (preserved from
# the original notebook code -- logic unchanged in Phase 2).
Ns = 100

a = qt.destroy(Ns)


def th_dm_th(Ns, mu):
    return qt.thermal_dm(Ns, mu)


def th_dm_co(Ns, mu):
    return qt.coherent_dm(Ns, mu)


def fs(Ns, n):
    return qt.states.fock(Ns, n)


def state_dm_th(Ns, mu, p, nad=1):
    st = (p * a.dag() * th_dm_th(Ns, mu) * a + (1 - p) * th_dm_th(Ns, mu))
    if nad > 1:
        for i in range(nad - 1):
            st = (p * a.dag() * st * a + (1 - p) * st)
    st = st.unit()
    return st


def state_dm_co(Ns, mu, p, nad=1):
    st = (p * a.dag() * th_dm_co(Ns, mu) * a + (1 - p) * th_dm_co(Ns, mu))
    if nad > 1:
        for i in range(nad - 1):
            st = (p * a.dag() * st * a + (1 - p) * st)
    st = st.unit()
    return st


def state_dm_pba_bacs(Ns, alpha, p, l, nad=1):
    # TODO: clarify -- `state_bacs` is referenced but not defined in the
    # original notebook. This function is non-functional until it is supplied.
    rho = state_bacs(Ns, alpha, l) * state_bacs(Ns, alpha, l).dag()  # noqa: F821
    rho = p * a.dag() * rho * a + (1 - p) * rho
    for i in range(nad - 1):
        rho = (p * a.dag() * rho * a + (1 - p) * rho)

    rho = rho.unit()
    return rho


def state_dm_pba_co(Ns, alpha, p, nad=1):
    # cst = qt.coherent(Ns,alpha)
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * rho * a).unit() + (1 - p) * rho
    for i in range(nad - 1):
        rho = (p * (a.dag() * rho * a).unit() + (1 - p) * rho)

    rho = rho.unit()
    return rho


def state_dm_pba_co_2_3(Ns, alpha, p, nad=1):
    # cst = qt.coherent(Ns,alpha)
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * a.dag() * a.dag() * rho * a * a * a).unit() + (1 - p) * (a.dag() * a.dag() * rho * a * a).unit()
    # for i in range(nad-1):
    #    rho = (p*(a.dag()*rho*a).unit() + (1-p)*rho)

    rho = rho.unit()
    return rho


def state_dm_pba_co_2_0(Ns, alpha, p, nad=1):
    # cst = qt.coherent(Ns,alpha)
    rho = qt.coherent_dm(Ns, alpha)
    rho = p * (a.dag() * a.dag() * rho * a * a).unit() + (1 - p) * (a.dag() * rho * a).unit()

    rho = rho.unit()
    return rho


def test_P_pba(mu, p, l, k):
    return (np.exp(-mu) / factorial(k)) * ((mu**(k - 1) * p * k**2 / (mu + 1)) + (1 - p) * mu**k)


def pba_Pk_bacs(Ns, mu, p, l, k, nad=1):
    alpha = np.sqrt(mu)
    st = state_dm_pba_bacs(Ns, alpha, p, l, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co(Ns, mu, p, k, nad=1):
    alpha = np.sqrt(mu)
    st = state_dm_pba_co(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co_2_3(Ns, mu, p, k, nad=1):
    alpha = mu / np.sqrt(2)  # This is not tested, it is a guess
    st = state_dm_pba_co_2_3(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)


def pba_Pk_co_2_0(Ns, mu, p, k, nad=1):
    alpha = mu / np.sqrt(2)  # This is not tested, it is a guess
    st = state_dm_pba_co_2_0(Ns, alpha, p, nad=nad)
    Pk = (fs(Ns, k).dag() * st * fs(Ns, k))
    return np.real(Pk)
