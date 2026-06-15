"""
Classic analytical nonclassicality criteria evaluated on a 2D probability grid.

Extracted verbatim (Phase 2) from ``mayorisation_functions.ipynb``:
- ``klyshko``        : three-index Klyshko criterion
- ``taylored``       : Taylor-expanded generalized Klyshko bound K_{inf,Np}
- ``log01rule``      : P_1 + P_0 log P_0 > 0
- ``generalized_klyshko``          : majorization criterion, pairs generated on the fly
- ``generalized_klyshko_first_Np`` : same, loading precomputed pairs from file
- ``generalized_klyshko_first_Np_ch`` : selected precomputed pairs only
- ``analcrit2D``     : union of klyshko + taylored + log01rule

TODO: clarify overlap with criteria_majorization.analytical_m /
load_precomputed_pairs and de-duplicate in Phase 3.

NOTE: naming, type hints, docstrings and logging are handled in Phase 3.
"""

import pickle

import numpy as np
from scipy.special import factorial

from src.analytical.criteria_majorization import (
    all_pairs_for_list_of_P,
    list_of_files,
)


def generalized_klyshko(Pall=np.ones((10, 10, 3)), indx=np.array([0, 1, 2], dtype=int)):

    Psh = Pall.shape

    NP = Psh[-1]
    if (NP != len(indx)):
        print("The array of probabilities and the index of probailities are different!")
        exit(0)

    rev_index = np.zeros((np.max(indx) + 1), dtype=int)
    for i in range(NP):
        rev_index[indx[i]] = i

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    all_pairs = all_pairs_for_list_of_P(indx=indx)

    for pair in all_pairs:

        fact0 = factorial(pair[0])
        fact1 = factorial(pair[1])
        fact = np.prod(fact0 / fact1)

        Paux0 = fact * np.ones((Psh[0], Psh[1]))
        Paux1 = np.ones((Psh[0], Psh[1]))

        for i in pair[1]:
            Paux1 *= Pall[:, :, rev_index[i]]

        for i in pair[0]:
            Paux0 *= Pall[:, :, rev_index[i]]

        Paux = Paux1 - Paux0

        nonclass = np.where(Paux > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def generalized_klyshko_first_Np(Pall=np.ones((10, 10, 3)), lof=list_of_files):

    Psh = Pall.shape

    NP = Psh[-1]

    indx = np.arange(0, NP, 1, dtype=int)
    print(indx)

    rev_index = np.zeros((np.max(indx) + 1), dtype=int)
    for i in range(NP):
        rev_index[indx[i]] = i

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    with open(lof[NP - 4], 'rb') as f:
        all_pairs = pickle.load(f)

    for pair in all_pairs:

        fact0 = factorial(pair[0])
        fact1 = factorial(pair[1])
        fact = np.prod(fact0 / fact1)

        Paux0 = fact * np.ones((Psh[0], Psh[1]))
        Paux1 = np.ones((Psh[0], Psh[1]))

        for i in pair[1]:
            Paux1 *= Pall[:, :, rev_index[i]]

        for i in pair[0]:
            Paux0 *= Pall[:, :, rev_index[i]]

        Paux = Paux1 - Paux0

        nonclass = np.where(Paux > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def generalized_klyshko_first_Np_ch(Pall=np.ones((10, 10, 3)), lof=list_of_files, ipair=np.array([1, 2, 3])):

    Psh = Pall.shape

    NP = Psh[-1]

    indx = np.arange(0, NP, 1, dtype=int)
    print(indx)

    rev_index = np.zeros((np.max(indx) + 1), dtype=int)
    for i in range(NP):
        rev_index[indx[i]] = i

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    with open(lof[NP - 4], 'rb') as f:
        all_pairs = pickle.load(f)

    for i0 in ipair:

        pair = all_pairs[i0]
        fact0 = factorial(pair[0])
        fact1 = factorial(pair[1])
        fact = np.prod(fact0 / fact1)

        Paux0 = fact * np.ones((Psh[0], Psh[1]))
        Paux1 = np.ones((Psh[0], Psh[1]))

        for i in pair[1]:
            Paux1 *= Pall[:, :, rev_index[i]]

        for i in pair[0]:
            Paux0 *= Pall[:, :, rev_index[i]]

        Paux = Paux1 - Paux0

        nonclass = np.where(Paux > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf, len(all_pairs)


def klyshko(Pall=np.ones((10, 10, 3))):

    Psh = Pall.shape
    Ns = np.array(np.prod(Psh[0:-1]))

    NP = Psh[-1]
    Nm = NP - 1

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    for i1 in range(1, Nm):
        Paux = (i1 + 1) * Pall[:, :, i1 - 1] * Pall[:, :, i1 + 1] - i1 * Pall[:, :, i1]**2
        nonclass = np.where(Paux < 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / Ns
    return nonclass, ncaf


def taylored(Pall=np.ones((10, 10, 3))):

    Psh = Pall.shape
    Ns = np.array(np.prod(Psh[0:-1]))

    NP = Psh[-1]
    Nm = NP - 1

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    SumA = np.zeros((Psh[0], Psh[1], NP))
    SumB = np.zeros((Psh[0], Psh[1]))

    SumA[:, :, 0] = Pall[:, :, 0]

    for i1 in range(1, Nm):
        SumA[:, :, i1] = SumA[:, :, i1 - 1] + Pall[:, :, i1]

    for nn in range(2, Nm + 1):

        SumB[:, :] = 1.0
        for k in range(1, nn - 1):
            SumB += ((nn * Pall[:, :, nn] / Pall[:, :, nn - 1])**k) / factorial(k)

        Paux = nn * Pall[:, :, nn]
        Paux -= Pall[:, :, nn - 1] * np.log((nn**nn / factorial(nn)) *
                                            ((Pall[:, :, nn]**(nn - 1)) / (Pall[:, :, nn - 1]**nn)) *
                                            (1.0 - SumA[:, :, nn - 2]) + SumB)

        nonclass = np.where(Paux > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / Ns
    return nonclass, ncaf


def log01rule(Pall=np.ones((10, 10, 3))):
    Psh = Pall.shape
    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    Paux = Pall[:, :, 1] + Pall[:, :, 0] * np.log(Pall[:, :, 0])
    nonclass = np.where(Paux > 0, True, nonclass)
    ncaf = np.sum(nonclass.astype(int)) / (Psh[0] * Psh[1])
    return nonclass, ncaf


def analcrit2D(Pall=np.ones((10, 10, 3))):

    Psh = Pall.shape
    Ns = np.array(np.prod(Psh[0:-1]))

    Np = Psh[-1]
    Nm = Np - 1

    nonclass = np.zeros((Psh[0], Psh[1]), dtype=bool)

    ac_kl, ac_kl_frac = klyshko(Pall=Pall)
    nonclass = np.where(ac_kl > 0, True, nonclass)

    ac_tay, ac_tay_frac = taylored(Pall=Pall)

    nonclass = np.where(ac_tay > 0, True, nonclass)

    ac_log, ac_log_frac = log01rule(Pall=Pall)

    nonclass = np.where(ac_log > 0, True, nonclass)

    ncaf = np.sum(nonclass.astype(int)) / Ns
    return nonclass, ncaf
