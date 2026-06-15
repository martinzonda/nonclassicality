import numpy as np
import scipy as sp
from scipy.special import factorial
from math import gcd
from functools import reduce
from collections import Counter

def pN_0fun(x,n):
    xc=n+sp.special.lambertw(-np.exp(-n)*n)
    Pxc=np.exp(-xc)
    Cxc=xc**(n-1)*Pxc/factorial(n-1)
    y=np.where(x<Pxc,x*(-np.log(x))**n/factorial(n),Cxc*(1-x))
    return y


def analytical_k(Pall, fact, indx=None):
    """
    Analytical nonclassicality criteria for possibly non-consecutive
    probability indices.

    Parameters
    ----------
    Pall : ndarray
        Array with shape (N_nu, N_tau, Np). The last axis contains probabilities.
    fact : ndarray
        fact[n] = 1/n!
    indx : ndarray or None
        Physical probability indices contained in Pall.
        Example: indx = np.array([0, 3, 5]).
        If None, assumes indx = [0, 1, ..., Np-1].

    Returns
    -------
    nonclass : ndarray of bool
        Boolean nonclassicality map.
    ncaf : float
        Fraction of nonclassical grid points.
    """

    Pall = np.asarray(Pall, dtype=float)
    fact = np.asarray(fact, dtype=float)

    Psh = Pall.shape
    Ngrid = np.prod(Psh[:-1])
    Np = Psh[-1]

    if indx is None:
        indx = np.arange(Np, dtype=int)
    else:
        indx = np.asarray(indx, dtype=int)

    if len(indx) != Np:
        raise ValueError(f"len(indx)={len(indx)} but Pall.shape[-1]={Np}")

    if not np.all(np.diff(indx) > 0):
        raise ValueError("indx must be strictly increasing")

    max_indx = int(np.max(indx))
    if len(fact) <= max_indx:
        raise ValueError("fact must contain entries up to max(indx)")

    # map physical probability index n -> position in Pall[..., :]
    pos = {int(n): j for j, n in enumerate(indx)}

    nonclass = np.zeros(Psh[:-1], dtype=bool)

    def has(*indices):
        return all(int(i) in pos for i in indices)

    def P(n):
        return Pall[..., pos[int(n)]]

    eps = 1e-300

    # ------------------------------------------------------------
    # Criterion W_{01}: requires P_0 and P_1
    # P_1 + P_0 log(P_0) > 0
    # ------------------------------------------------------------
    if has(0, 1):
        P0 = np.maximum(P(0), eps)
        Paux = P(1) + P0 * np.log(P0)
        nonclass |= (Paux > 0)

    # ------------------------------------------------------------
    # Three-consecutive-index criterion:
    # (n+1) P_{n-1} P_{n+1} - n P_n^2 < 0
    # Requires indices n-1, n, n+1.
    # ------------------------------------------------------------
    for n in indx:
        n = int(n)
        if n >= 1 and has(n - 1, n, n + 1):
            Paux = (n + 1) * P(n - 1) * P(n + 1) - n * P(n)**2
            nonclass |= (Paux < 0)

    # ------------------------------------------------------------
    # Criterion P_n - pN_0fun(P_0,n) > 0
    # Requires P_0 and P_n.
    # Applies for n > 1, following your original code.
    # ------------------------------------------------------------
    if has(0):
        P0 = P(0)
        for n in indx:
            n = int(n)
            if n > 1 and has(n):
                Paux = P(n) - pN_0fun(P0, n)
                nonclass |= (Paux > 0)

    # ------------------------------------------------------------
    # Final criterion:
    # Requires all probabilities P_0, ..., P_n.
    # ------------------------------------------------------------
    for n in range(2, max_indx + 1):

        # Need complete block 0,1,...,n
        if not has(*range(0, n + 1)):
            continue

        Pn = P(n)
        Pnm1 = np.maximum(P(n - 1), eps)

        # SumA[:,:,n-2] = sum_{m=0}^{n-2} P_m
        SumA = np.zeros(Psh[:-1])
        for m in range(0, n - 1):
            SumA += P(m)

        # SumB = 1 + sum_{k=1}^{n-2} ((n P_n / P_{n-1})^k)/k!
        ratio = n * Pn / Pnm1
        SumB = np.ones(Psh[:-1])
        for k in range(1, n - 1):
            SumB += (ratio**k) * fact[k]

        log_arg = (
            (n**n * fact[n])
            * ((np.maximum(Pn, eps)**(n - 1)) / (Pnm1**n))
            * (1.0 - SumA)
            + SumB
        )

        log_arg = np.maximum(log_arg, eps)

        Paux = n * Pn - Pnm1 * np.log(log_arg)
        nonclass |= (Paux > 0)

    ncaf = np.sum(nonclass.astype(int)) / Ngrid
    return nonclass, ncaf