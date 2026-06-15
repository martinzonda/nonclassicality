"""Numerical nonclassicality witnesses.

``lp_solver`` is the primary solver (semi-infinite LP via cutting planes).
``minimize_solver`` is an independent scipy.optimize.minimize-based cross-check.
"""

from src.solvers.lp_solver import (
    solve_semi_infinite_lp,
    solve_master_lp,
    max_inner_value,
    max_inner_value_fast,
    true_objective,
)

__all__ = [
    "solve_semi_infinite_lp",
    "solve_master_lp",
    "max_inner_value",
    "max_inner_value_fast",
    "true_objective",
]
