# Nonclassicality detection via linear programming

Certify nonclassicality from finite sets of occupation probabilities using an exact linear-programming witness, and compare the results against analytical criteria: Klyshko-type witnesses, majorization witnesses, and photon-counting matrix criteria.

## Problem statement

Given a finite set of occupation probabilities, the code tests whether they are compatible with a classical state, i.e., with a mixture of Poisson distributions,

$$
P_k = \int \mathrm{d}\mu\, \rho(\mu)\, p_k(\mu),
\qquad \rho(\mu)\ge0,
$$

where

$$
p_k(\mu)=e^{-\mu}\frac{\mu^k}{k!}.
$$

Any incompatibility with this structure certifies nonclassicality.

Nonclassicality is certified by a linear witness $\boldsymbol\lambda$ for which

$$
f({\boldsymbol \lambda}) =
\max\!\Bigl(0,\max_{\mu\ge0}{\boldsymbol\lambda}\!\cdot\!\mathbf p(\mu)\Bigr)
{\boldsymbol\lambda}\!\cdot\!\mathbf P
$$

becomes negative. Minimizing $f$ over ${\boldsymbol\lambda}$ (with bounded components) leads to a semi-infinite linear program: infinitely many constraints, one for each $\mu\ge0$. The problem is solved using a cutting-plane (exchange) method, which repeatedly solves a finite master linear program and adds the most violated constraint until convergence.

The reported quantity

$$
\mathrm{num}=-\min_{\boldsymbol\lambda} f
$$

is positive whenever the supplied occupation probabilities certify nonclassicality. If `num = 0`, no nonclassicality can be certified from the provided probabilities, although the underlying quantum state may still be nonclassical.

Input probability vectors are checked before classification. If any selected
probability is negative or non-finite, or if the selected probabilities already
sum to more than one, the point is treated as non-physical: single-point solvers
return `num = NaN`, and 2D scans store `NaN` at that grid point and draw it as
white in the demo maps.

## Analytical criteria

The LP solver is the primary numerical criterion. The package also implements
several faster analytical checks that can be compared against the LP boundary
on the same probability grid:

- `analytical_k`: Klyshko-type criteria, including the two-probability W01
  test and generalized Klyshko inequalities.
- `analytical_m`: majorization-pair criteria. Precomputed pair archives for
  small probability sets live in `data/all_genK/`.
- `analytical_c`: photon-counting matrix criteria.

The combined analytical map returned by `analyze_2d` and `run2D` is the union
of all enabled analytical criteria. Individual maps and boundaries are also
stored separately under `result["ana_k"]`, `result["ana_m"]`, and
`result["counting_matrix"]`.

### Photon-counting matrix criterion

The counting-matrix implementation follows the ideal photoelectric-counting
form of the observable-witness construction discussed in Sec. II of
[Krishnaswamy, Jung, Ares, Gärttner, and Sperling, *Observable
nonclassicality witnesses for multiplexed detection systems*,
arXiv:2603.06237v1](https://arxiv.org/abs/2603.06237). For a selected set of
available photon-number probabilities, it forms matrices

$$
C^{(I)} = \left[(k+l)! P_{k+l}\right]_{k,l\in I},
$$

where the admissible index set `I` may contain integers or half-integers, but
every sum `k+l` must correspond to an available probability `P_{k+l}`. Classical
states require every admissible matrix to be positive semidefinite. Therefore,
a negative minimum eigenvalue of any such matrix certifies nonclassicality.

In this repository, `admissible_counting_sets` builds the maximal admissible
integer and half-integer index sets, `counting_matrix` constructs a single
matrix, `analytical_c_point` evaluates one probability vector, and
`analytical_c` evaluates the criterion over an arbitrary probability grid.

## Installation

```bash
# pip (editable, with development/notebook tools)
python -m pip install -e ".[dev]"

# optional: include QuTiP for the two-boson PBA demo
python -m pip install -e ".[dev,physics]"

# or pinned runtime deps only
python -m pip install -r requirements.txt

# or conda (full reproducibility)
conda env create -f environment.yml
conda activate nonclassicality-lp
```

Python ≥ 3.10 is required. **QuTiP is optional** — only the coherent-state probabilistic-boson-addition helper used in `notebooks/demo_4_probabilistic_boson_addition.ipynb` requires it. All solvers, analytical criteria, Jaynes--Cummings examples, and the closed-form one-boson PBA demo work without QuTiP.

## Quick start

```python
import numpy as np
from src.solvers.lp_solver import solve_semi_infinite_lp
from src.utils.numerics import inverse_factorials
from src.config import SCAN

indx  = np.array([0, 1, 2, 3])          # occupation probabilities P_0..P_3
probs = np.array([0.6, 0.1, 0.2, 0.05]) # probabilities to test
fact  = inverse_factorials(int(indx.max()))   # 1/k!

res = solve_semi_infinite_lp(probs, indx, fact, bounds=[(-1, 1)] * 4)
print(res["num"], "-> nonclassical:", res["num"] > SCAN.bias)
```

Runnable scripts:

```bash
python examples/single_point_witness.py
python examples/run_jcm_scan.py
python scripts/run_scan.py --n0 256 --n1 256 --out results/scan.npz
python scripts/run_scan.py --no-analytical-m --y-scan-direction decreasing
```

Demo notebooks:

- `notebooks/demo_1_lp_solver.ipynb`: single-point LP witness, Nelder-Mead comparison, and a single Jaynes--Cummings point.
- `notebooks/demo_2_analytical_vs_numerical.ipynb`: probability-plane scans comparing LP and analytical criteria.
- `notebooks/demo_3_jaynes_cummings.ipynb`: Jaynes--Cummings nonclassicality maps.
- `notebooks/demo_4_probabilistic_boson_addition.ipynb`: probabilistic boson addition, reversed y-scan direction, high-precision LP settings, probability-scale diagnostics, and CSV save/load.
- `notebooks/demo_5_counting_matrix_vs_numerics.ipynb`: probability-plane comparison of LP, K, M, and photon-counting matrix boundaries, including reload-from-archive plotting.
- `notebooks/demo_6_counting_matrix_jc.ipynb`: Jaynes--Cummings scans emphasizing photon-counting matrix criteria and their comparison with LP and other analytical boundaries.

## Directory structure

```text
.
├── config.yaml             # optional local configuration overrides
├── environment.yml         # conda environment specification
├── pyproject.toml          # editable install metadata and optional extras
├── requirements.txt        # pinned runtime dependencies
├── data/
│   ├── README.md
│   └── all_genK/           # tracked precomputed majorization-pair archives
├── examples/
│   ├── run_jcm_scan.py     # small Jaynes--Cummings 2D scan
│   └── single_point_witness.py
├── notebooks/
│   ├── demo_1_lp_solver.ipynb
│   ├── demo_2_analytical_vs_numerical.ipynb
│   ├── demo_3_jaynes_cummings.ipynb
│   ├── demo_4_probabilistic_boson_addition.ipynb
│   ├── demo_5_counting_matrix_vs_numerics.ipynb
│   └── demo_6_counting_matrix_jc.ipynb
├── results/                # generated outputs; ignored except .gitkeep
├── scripts/
│   └── run_scan.py         # command-line probability-plane scan
├── src/
│   ├── config.py           # tolerances, bounds, seeds, paths
│   ├── analytical/         # K, M, counting-matrix criteria and demo models
│   ├── comparison/         # 2D numerical-vs-analytical scan helpers
│   ├── solvers/            # LP and Nelder-Mead solvers
│   ├── utils/              # numerical and I/O helpers
│   └── visualization/      # boundary extraction and plotting helpers
└── tests/                  # pytest suite
```

## Key functions

| Function | Module | Purpose |
|----------|--------|---------|
| `solve_semi_infinite_lp` | `solvers.lp_solver` | Primary nonclassicality solver based on semi-infinite linear programming. |
| `max_inner_value` | `solvers.lp_solver` | Exact evaluation of $\max_{\mu\ge0}\boldsymbol\lambda\cdot\mathbf p(\mu)$. |
| `minimize_witness` / `scan_2d_minimize` | `solvers.minimize_solver` | Independent Nelder-Mead cross-check. |
| `analytical_k` | `analytical.criteria_k` | Klyshko and generalized Klyshko criteria. |
| `analytical_m` | `analytical.criteria_majorization` | Majorization-based criteria. |
| `analytical_c` | `analytical.criteria_counting_matrix` | Photon-counting matrix criterion over a probability grid. |
| `analytical_c_point` | `analytical.criteria_counting_matrix` | Photon-counting matrix criterion for one probability vector, optionally with diagnostics. |
| `admissible_counting_sets` | `analytical.criteria_counting_matrix` | Build maximal integer and half-integer index sets for counting matrices. |
| `counting_matrix` | `analytical.criteria_counting_matrix` | Construct one matrix `C^(I)` from selected probabilities. |
| `klyshko`, `taylored`, `log01rule`, `analcrit2D` | `analytical.criteria_classic` | Legacy individual analytical witnesses and their union. |
| `Pjaycum` | `analytical.jaynes_cummings` | Jaynes--Cummings occupation probabilities. |
| `test_P_pba`, `pba_Pk_co` | `analytical.models` | Probabilistic boson-addition examples. |
| `analyze_2d` | `comparison.scan_2d` | Analyze an existing probability grid with LP, K, M, and/or counting-matrix criteria; returns a structured result dictionary. |
| `run2D` | `comparison.scan_2d` | Convenience wrapper for probability-plane scans, returning the same structured result dictionary. |
| `load_scan_result` / `save_scan_result` | `comparison.scan_2d` | Read and write named 2D scan archives. |
| `extract_binary_boundary` | `visualization.boundaries` | Extract decision boundaries from binary maps. |

## Running the tests

```bash
pytest tests/ -v
```

The test suite covers the LP solver, the inner maximization routine, agreement between the LP and Nelder-Mead implementations, analytical criteria, configuration handling, I/O utilities, and two-dimensional scans.

## Dependencies and Python version

- **Python** ≥ 3.10
- **Runtime**: numpy, scipy, matplotlib, tqdm, PyYAML
- **Optional** (`[physics]`): qutip (only for the coherent-state PBA helper)
- **Development** (`[dev]`): pytest, ipykernel, nbconvert

Exact tested versions are listed in `requirements.txt` and `environment.yml`.

## Reproducibility

- **Configuration** is centralized in `src/config.py` and can be overridden through `config.yaml`.
- **Tolerances** (defaults): LP `tol=1e-10`, `max_iter=100`, `imtol=1e-9`; 2D scans `tol=1e-10`, `bias=1e-8`, `max_iter=300`, `imtol=1e-8`; Nelder-Mead `bias=1e-7`, `tol=1e-7`.
- **Random seeds** can be specified for stochastic scans through the `seed` argument.
- **Thread limits** can be controlled via `config.apply_thread_limits()`.
- **Data**: precomputed majorization archives in `data/all_genK/` are version-controlled inputs, while generated outputs are written to `results/`.

## References

- Suchitra Krishnaswamy, Martina Jung, Laura Ares, Martin Gärttner, and Jan Sperling, [*Observable nonclassicality witnesses for multiplexed detection systems*](https://arxiv.org/abs/2603.06237), arXiv:2603.06237v1, 2026.
