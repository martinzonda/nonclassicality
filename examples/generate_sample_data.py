#!/usr/bin/env python3
"""Regenerate small sample datasets so the figure notebook is runnable.

The original paper figures were produced from large precomputed archives kept
outside this repository (``../Data/Numerical/...``). This script produces a
small stand-in archive in the same on-disk layout as :func:`run2D`, so
``notebooks/figures_paper.ipynb`` can load and plot something end-to-end. The
sample is coarse and does NOT reproduce the published figures.

Usage
-----
    python examples/generate_sample_data.py
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

    sample_dir = os.path.join(PATHS.data_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    out_path = os.path.join(sample_dir, "lp_sample_NP4.npy")

    run2D(
        N0=60,
        N1=60,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10,
        uPA=0.9,
        dPB=1e-10,
        uPB=0.3,
        Prest=np.array([0.05, 0.05]),
        fi=out_path,
    )
    print(f"Wrote sample dataset to {out_path}")


if __name__ == "__main__":
    main()
