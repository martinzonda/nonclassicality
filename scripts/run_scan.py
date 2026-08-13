#!/usr/bin/env python3
"""Command-line entry point for a 2D nonclassicality scan.

Thin wrapper over :func:`src.comparison.scan_2d.run2D`. Defaults come from the
central configuration (overridable via config.yaml); command-line flags take
precedence over both.

Examples
--------
    python scripts/run_scan.py --n0 256 --n1 256 --out results/scan.npz
    python scripts/run_scan.py --indx 0 1 2 3 --upa 0.9 --upb 0.3
    python scripts/run_scan.py --no-analytical-m --no-counting-matrix
"""

import argparse
import logging
import os
import sys

# Allow running directly without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SCAN, PATHS, apply_thread_limits

apply_thread_limits()

import numpy as np

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
                        default=os.path.join(PATHS.results_dir, "scan.npz"),
                        help="output .npz archive path")
    parser.add_argument("--no-analytical-m", action="store_true",
                        help="skip the expensive majorization criterion")
    parser.add_argument("--no-analytical-k", action="store_true",
                        help="skip the analytical K criterion")
    parser.add_argument("--no-counting-matrix", action="store_true",
                        help="skip the photon-counting matrix criterion")
    parser.add_argument("--y-scan-direction", choices=["increasing", "decreasing"],
                        default="increasing",
                        help="warm-start direction along the y/P_1 axis")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    result = run2D(
        N0=args.n0,
        N1=args.n1,
        indx=np.array(args.indx, dtype=np.int16),
        dPA=args.dpa, uPA=args.upa, dPB=args.dpb, uPB=args.upb,
        Prest=np.array(args.prest),
        fi=args.out,
        use_ana_k=not args.no_analytical_k,
        use_ana_m=not args.no_analytical_m,
        use_cm=not args.no_counting_matrix,
        y_scan_direction=args.y_scan_direction,
    )
    print(
        f"Saved scan to {args.out} "
        f"({len(result['numerical']['boundaries'])} LP boundary curves, "
        f"{len(result['analytical']['boundaries'])} analytical curves)"
    )


if __name__ == "__main__":
    main()
