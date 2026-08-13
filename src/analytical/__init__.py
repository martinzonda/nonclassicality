"""Analytical nonclassicality criteria and maintained demo models.

- ``criteria_k``            : analytical_k (W01, 3-index + generalized Klyshko)
- ``criteria_counting_matrix`` : analytical_c (ideal photoelectric
  counting matrices from Krishnaswamy et al., arXiv:2603.06237v1)
- ``criteria_majorization`` : analytical_m and majorization-pair machinery
- ``criteria_classic``      : klyshko, taylored, log01rule, generalized_klyshko*
- ``jaynes_cummings``       : Pjaycum (QuTiP-free)
- ``models``                : probabilistic boson-addition demo helpers
"""

from src.analytical.criteria_k import analytical_k, pN_0fun
from src.analytical.criteria_counting_matrix import (
    admissible_counting_sets,
    analytical_c,
    analytical_c_point,
    counting_matrix,
)
from src.analytical.jaynes_cummings import Pjaycum
from src.analytical.criteria_majorization import (
    analytical_m,
    all_pairs_for_list_of_P,
    load_precomputed_pairs,
    build_reverse_index,
    nonclass_from_pairs,
)

__all__ = [
    "analytical_k",
    "pN_0fun",
    "analytical_c",
    "analytical_c_point",
    "admissible_counting_sets",
    "counting_matrix",
    "Pjaycum",
    "analytical_m",
    "all_pairs_for_list_of_P",
    "load_precomputed_pairs",
    "build_reverse_index",
    "nonclass_from_pairs",
]
