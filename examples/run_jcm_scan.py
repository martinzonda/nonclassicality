#!/usr/bin/env python3
"""Run a small Jaynes-Cummings nonclassicality scan and save the result.

Builds a coarse Jaynes-Cummings probability grid and runs :func:`run2D`, which
classifies every grid point with both the LP solver and the analytical criteria
and writes the boundaries to a ``.npy`` archive under ``results/``.

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

from src.config import PATHS
from src.comparison.scan_2d import run2D


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    os.makedirs(PATHS.results_dir, exist_ok=True)
    out_path = os.path.join(PATHS.results_dir, "jcm_scan_demo.npy")

    # Coarse grid for a quick demo (raise N0/N1 for production runs).
    boundaries, boundaries_ana, boundaries_anaK, boundaries_anaM = run2D(
        N0=80,
        N1=80,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10,
        uPA=0.9,
        dPB=1e-10,
        uPB=0.3,
        Prest=np.array([0.05, 0.05]),
        fi=out_path,
    )

    print(f"Saved scan to {out_path}")
    print(f"  LP boundary curves        : {len(boundaries)}")
    print(f"  analytical-K curves       : {len(boundaries_anaK)}")
    print(f"  analytical-M curves       : {len(boundaries_anaM)}")
    print(f"  combined analytical curves: {len(boundaries_ana)}")


if __name__ == "__main__":
    main()
