#!/usr/bin/env python3
"""Run a small Jaynes-Cummings nonclassicality scan and save the result.

Builds a coarse Jaynes-Cummings probability grid over ``(tau, nu)`` and runs
``analyze_2d``, which classifies every grid point with the LP solver and the
analytical criteria. The output is written to ``results/jcm_scan_demo.npz``.

Usage
-----
    python examples/run_jcm_scan.py
"""

import logging
import os
import sys

# Allow running directly (python examples/...) without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.analytical.jaynes_cummings import Pjaycum
from src.comparison.scan_2d import analyze_2d
from src.config import PATHS


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    os.makedirs(PATHS.results_dir, exist_ok=True)
    out_path = os.path.join(PATHS.results_dir, "jcm_scan_demo.npz")

    # Coarse grid for a quick demo (raise the sizes for production runs).
    tau = np.linspace(0.001, 10.001, 40)
    nu = np.linspace(0.001, 6.001, 40)
    pe = 0.3
    indx = np.arange(6)

    xtau, ynu = np.meshgrid(tau, nu)
    Pall = np.empty((len(nu), len(tau), len(indx)))
    for j, n in enumerate(indx):
        Pall[..., j] = Pjaycum(n=int(n), nu=ynu, tau=xtau, pe=pe)

    result = analyze_2d(
        Pall,
        x_values=tau,
        y_values=nu,
        indx=indx,
        fixed_values=np.array([pe]),
        fi=out_path,
        use_ana_m=False,
        use_cm=True,
    )

    print(f"Saved Jaynes-Cummings scan to {out_path}")
    print(f"  grid                       : {Pall.shape[:2]}")
    print(f"  LP boundary curves         : {len(result['numerical']['boundaries'])}")
    print(f"  analytical-K curves        : {len(result['ana_k']['boundaries'])}")
    print(f"  analytical-M curves        : {len(result['ana_m']['boundaries'])}")
    print(f"  counting-matrix curves     : {len(result['counting_matrix']['boundaries'])}")
    print(f"  combined analytical curves : {len(result['analytical']['boundaries'])}")


if __name__ == "__main__":
    main()
