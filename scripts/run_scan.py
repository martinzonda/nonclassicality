#!/usr/bin/env python3
"""Command-line entry point for a 2D nonclassicality scan.

Thin wrapper over :func:`src.comparison.scan_2d.run2D`. Defaults come from the
central configuration (overridable via config.yaml); command-line flags take
precedence over both.

Examples
--------
    python scripts/run_scan.py --n0 256 --n1 256 --out results/scan.npy
    python scripts/run_scan.py --indx 0 1 2 3 --upa 0.9 --upb 0.3
"""

import argparse
import logging
import os
import sys

# Allow running directly without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import SCAN, PATHS
from src.comparison.scan_2d import run2D


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n0", type=int, default=SCAN.n0, help="grid resolution, axis 0")
    parser.add_argument("--n1", type=int, default=SCAN.n1, help="grid resolution, axis 1")
    parser.add_argument("--indx", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="photon-number indices")
    parser.add_argument("--dpa", type=float, default=SCAN.dPA, help="lower bound, P_0")
    parser.add_argument("--upa", type=float, default=SCAN.uPA, help="upper bound, P_0")
    parser.add_argument("--dpb", type=float, default=SCAN.dPB, help="lower bound, P_1")
    parser.add_argument("--upb", type=float, default=SCAN.uPB, help="upper bound, P_1")
    parser.add_argument("--prest", type=float, nargs="+", default=list(SCAN.prest),
                        help="fixed remaining probabilities")
    parser.add_argument("--out", type=str,
                        default=os.path.join(PATHS.results_dir, "scan.npy"),
                        help="output .npy archive path")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    boundaries, *_ = run2D(
        N0=args.n0,
        N1=args.n1,
        indx=np.array(args.indx, dtype=np.int16),
        dPA=args.dpa, uPA=args.upa, dPB=args.dpb, uPB=args.upb,
        Prest=np.array(args.prest),
        fi=args.out,
    )
    print(f"Saved scan to {args.out} ({len(boundaries)} LP boundary curves)")


if __name__ == "__main__":
    main()
