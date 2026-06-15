#!/usr/bin/env python3
"""Evaluate the nonclassicality witness for a single probability vector.

Runs both solvers (the primary LP cutting-plane method and the Nelder-Mead
cross-check) on one photon-number distribution and prints the witness value.
A negative objective (``num > 0``) certifies nonclassicality.

Usage
-----
    python examples/single_point_witness.py
"""

import logging
import os
import sys

# Allow running directly (python examples/...) without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import SOLVER
from src.solvers.lp_solver import solve_semi_infinite_lp, max_inner_value
from src.solvers.minimize_solver import minimize_witness
from src.utils.numerics import inverse_factorials


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # A photon-number distribution P_0..P_3 to test.
    indx = np.array([0, 1, 2, 3])
    probs = np.array([0.6, 0.1, 0.2, 0.05])
    fact = inverse_factorials(int(np.max(indx)))
    bounds = [SOLVER.lambda_bound] * len(indx)

    # Primary LP solver.
    lp = solve_semi_infinite_lp(probs, indx, fact, bounds)
    print(f"LP        : num = {lp['num']:.10f}  (nonclassical: {lp['num'] > 0})")
    print(f"            lambda = {np.round(lp['lam'], 4)}")

    # Nelder-Mead cross-check from three starting points.
    value, lam = minimize_witness(
        probs=probs, indx=indx, fact=fact,
        init_a=np.zeros(len(indx)),
        init_b=np.ones(len(indx)),
        init_c=0.1 * np.ones(len(indx)),
        bounds=bounds,
    )
    print(f"Nelder-Mead: num = {value:.10f}  (nonclassical: {value > 0})")


if __name__ == "__main__":
    main()
