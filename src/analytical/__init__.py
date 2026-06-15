"""Analytical nonclassicality criteria and physical photon-statistics models.

- ``criteria_k``            : analytical_k (W01, 3-index + generalized Klyshko)
- ``criteria_majorization`` : analytical_m and majorization-pair machinery
- ``criteria_classic``      : klyshko, taylored, log01rule, generalized_klyshko*
- ``models``                : Pjaycum + QuTiP state generators (optional QuTiP)
"""

from src.analytical.criteria_k import analytical_k, pN_0fun
from src.analytical.criteria_majorization import (
    analytical_m,
    all_pairs_for_list_of_P,
    load_precomputed_pairs,
)

__all__ = [
    "analytical_k",
    "pN_0fun",
    "analytical_m",
    "all_pairs_for_list_of_P",
    "load_precomputed_pairs",
]
