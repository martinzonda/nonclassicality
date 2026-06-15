"""
Alternative numerical nonclassicality solver based on direct minimization.

This module contains the scipy.optimize.minimize-based approach to evaluating
the nonclassicality witness, extracted verbatim from
``mayorisation_functions.ipynb``. The linear-programming cutting-plane method
in :mod:`src.solvers.lp_solver` is the primary solver; this one is kept as an
independent cross-check.

NOTE: code moved verbatim in Phase 2. Naming, type hints, docstrings, logging
and de-duplication are handled in Phase 3.
"""

import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.polynomial import Polynomial
from scipy.special import factorial
from tqdm import tqdm


def maxPrahfun(indx=np.array([]), lam=np.array([]), fact=np.array([])):

    mxin = np.amax(indx)

    lam_aux = np.zeros(mxin + 2)
    lam_aux[indx] = lam

    coef = (lam_aux[1:mxin + 2] - lam_aux[0:mxin + 1]) * fact

    p = Polynomial(coef)
    all_roots = p.roots()

    real_roots = np.real(all_roots)
    real_roots = np.append(real_roots[real_roots > 0], 0.0)

    coef = lam_aux[0:mxin + 1] * fact
    p2 = Polynomial(coef)

    val = np.amax(p2(real_roots) * np.exp(-real_roots))
    prah = np.maximum(0.0, val)

    return prah


def fun_full(lam, probs=np.array([]),
             indx=np.array([], dtype=int), fact=np.array([])):

    prah = maxPrahfun(indx=indx, lam=lam, fact=fact)

    ret = prah - np.dot(lam, probs)

    return ret


def fun_fix_one(lam, probs=np.array([]),
                indx=np.array([], dtype=int),
                fact=np.array([]),
                lam_fix=1.0, i_fix=0):

    lam_all = np.insert(lam, i_fix, lam_fix)
    prah = maxPrahfun(indx=indx, lam=lam_all, fact=fact)

    ret = prah - np.dot(lam_all, probs)

    return ret


def objf_no_fixation(probs=np.array([]), indx=np.array([], dtype=int),
                     fact=np.array([]),
                     initA=np.array([]),
                     initB=np.array([]),
                     initC=np.array([]),
                     bounds=[], bias=1e-8, tol=1e-6):

    # method='TNC'
    # method='Nelder-Mead'
    # method='SLSQP' #ok
    # method='trust-constr'
    objf = lambda x: fun_full(x, probs=probs, indx=indx, fact=fact)

    retA = minimize(objf, initA, method='Nelder-Mead', bounds=bounds, tol=tol)

    if (retA.fun >= 0):
        retB = minimize(objf, initB, method='Nelder-Mead', bounds=bounds, tol=tol)

        if (retB.fun >= 0):
            retC = minimize(objf, initC, method='Nelder-Mead', bounds=bounds, tol=tol)

            res = np.array([retA.fun, retB.fun, retC.fun])
            pos = [retA.x, retB.x, retC.x]

            imin = np.argmin(res)

            ret = res[imin]
            init = pos[imin]

        else:
            ret = retB.fun
            init = retB.x

    else:
        ret = retA.fun
        init = retA.x

    return -ret, init


def objf_fixation(probs=np.array([]), indx=np.array([], dtype=int),
                  fact=np.array([]),
                  initA=np.array([]),
                  bounds=[], bias=1e-8, tol=1e-6,
                  lam_fix=1.0, i_fix=0):

    objf = lambda x: fun_fix_one(x, probs=probs, indx=indx,
                                 fact=fact, lam_fix=lam_fix, i_fix=i_fix)

    retA = minimize(objf, initA, method='Nelder-Mead', bounds=bounds, tol=tol)

    ret = retA.fun
    init = retA.x

    return -ret, init


def numcrit2D(nc_a=np.array([]), NP=None, bias=1e-7, tol=1e-7, Pall=np.ones((10, 10, 3))):

    nc_n = np.copy(nc_a)

    ineg = np.array(np.where(nc_n == False))

    Nn = ineg.shape[1]
    print(Nn, ineg.shape)

    lam_mem = np.zeros((*nc_n.shape, NP))
    loss_mem = np.zeros(nc_n.shape)

    initA = np.zeros(NP)
    initB = np.ones(NP)
    initC = 0.10 - 0.2 * np.random.rand(NP)

    Ns = np.prod(np.array([*nc_n.shape]))

    bound = (-1.0, 1.0)
    bounds = []
    for i in range(NP):
        bounds.append(bound)

    indx_all = np.array(range(NP))
    fact = np.array([1.0 / factorial(i1) for i1 in range(np.max(indx_all) + 1)])

    for iN in tqdm(range(Nn)):
        iNN = ineg[:, iN]
        # for iN in tqdm(range(Ns)):
        Pin = Pall[iNN[0], iNN[1], :]

        loss, initA = objf_no_fixation(probs=Pin, indx=indx_all,
                                       fact=fact,
                                       initA=initA,
                                       initB=initB,
                                       initC=initC,
                                       bounds=bounds, bias=bias, tol=tol)

        if (loss > bias):
            nc_n[iNN[0], iNN[1]] = True
            initA = initA / np.max(np.abs(initA))
            lam_mem[iNN[0], iNN[1], :] = initA
            initB = np.copy(initA)
        else:
            initA = np.copy(initB)
            initB = -initB

        loss_mem[iNN[0], iNN[1]] = loss

    ncnf = np.sum(nc_n.astype(int)) / Ns
    return nc_n, ncnf, loss_mem, lam_mem


def numcrit2D_fix(nc_a=np.array([]), NP=None, bias=1e-7, tol=1e-7, Pall=np.ones((10, 10, 3))):

    nc_n = np.copy(nc_a)

    ineg = np.array(np.where(nc_n == False))

    Nn = ineg.shape[1]
    print(Nn, ineg.shape)

    lam_mem = np.zeros((*nc_n.shape, NP))
    loss_mem = np.zeros(nc_n.shape)

    Ns = np.prod(np.array([*nc_n.shape]))

    bound = (-1.0, 1.0)
    bounds = []
    for i in range(NP - 1):
        bounds.append(bound)

    indx_all = np.array(range(NP))

    fact = np.array([1.0 / factorial(i1) for i1 in range(np.max(indx_all) + 1)])

    for iN in tqdm(range(Nn)):

        initA = 0.1 * (0.5 - np.random.rand(NP - 1))

        iNN = ineg[:, iN]
        Pin = Pall[iNN[0], iNN[1], :]

        for i_fix in range(NP):

            loss, initB = objf_fixation(probs=Pin, indx=indx_all,
                                        fact=fact,
                                        initA=initA,
                                        bounds=bounds, bias=bias, tol=tol,
                                        lam_fix=1.0, i_fix=i_fix)
            if (loss > bias):
                nc_n[iNN[0], iNN[1]] = True
                lam_mem[iNN[0], iNN[1], :] = np.insert(initB, i_fix, 1.0)
                loss_mem[iNN[0], iNN[1]] = loss
                break
            else:
                loss, initB = objf_fixation(probs=Pin, indx=indx_all,
                                            fact=fact,
                                            initA=initA,
                                            bounds=bounds, bias=bias, tol=tol,
                                            lam_fix=-1.0, i_fix=i_fix)

                if (loss > bias):
                    nc_n[iNN[0], iNN[1]] = True
                    lam_mem[iNN[0], iNN[1], :] = np.insert(initB, i_fix, -1.0)
                    loss_mem[iNN[0], iNN[1]] = loss
                    break
    ncnf = np.sum(nc_n.astype(int)) / Ns
    return nc_n, ncnf, loss_mem, lam_mem
