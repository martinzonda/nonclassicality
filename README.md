# Nonclassicality detection via linear programming

Detect whether a photon-number distribution $\mathbf P = (P_0, P_1, \dots)$ is
**nonclassical** — i.e. *not* expressible as a classical mixture of coherent
(Poissonian) states — using an exact linear-programming witness, and compare it
against a battery of fast analytical criteria (Klyshko, generalized Klyshko,
majorization).

## Problem statement

A distribution is **classical** iff it is a mixture of Poissonians,
$P_k = \int \mathrm{d}\mu\, \rho(\mu)\, p_k(\mu)$ with $\rho \ge 0$, where

$$ p_k(\mu) = e^{-\mu}\,\frac{\mu^k}{k!}. $$

Nonclassicality is certified by a linear witness $\boldsymbol\lambda$ for which

$$ f(\boldsymbol\lambda) =
   \max\!\Big(0,\ \max_{\mu \ge 0} \boldsymbol\lambda\cdot\mathbf p(\mu)\Big)
   - \boldsymbol\lambda\cdot\mathbf P $$

becomes negative. Minimising $f$ over $\boldsymbol\lambda$ (with bounded
components) is a **semi-infinite linear program**: infinitely many constraints,
one per $\mu \ge 0$. It is solved by a cutting-plane / exchange method —
repeatedly solving a finite master LP and adding the most-violated $\mu$ (found
analytically from the stationary-point polynomial) until convergence. The
reported quantity `num` $= -\min_{\boldsymbol\lambda} f$ is positive exactly
when $\mathbf P$ is nonclassical.

## Installation

```bash
# pip (editable, with dev + physics extras)
python -m pip install -e ".[dev,physics]"

# or pinned runtime deps only
python -m pip install -r requirements.txt

# or conda (full reproducibility)
conda env create -f environment.yml
conda activate nonclassicality-lp
```

Python ≥ 3.10 is required. **QuTiP is optional** — only the density-matrix state
generators in `src/analytical/models.py` need it; everything else (solvers,
analytical criteria, the Jaynes-Cummings model) works without it.

## Quick start

```python
import numpy as np
from src.solvers.lp_solver import solve_semi_infinite_lp
from src.utils.numerics import inverse_factorials

indx  = np.array([0, 1, 2, 3])          # photon numbers P_0..P_3
probs = np.array([0.6, 0.1, 0.2, 0.05]) # the distribution to test
fact  = inverse_factorials(int(indx.max()))   # 1/k!

res = solve_semi_infinite_lp(probs, indx, fact, bounds=[(-1, 1)] * 4)
print(res["num"], "-> nonclassical:", res["num"] > 0)   # 0.1134 -> True
```

Runnable scripts:

```bash
python examples/single_point_witness.py     # one point, both solvers
python examples/run_jcm_scan.py             # small Jaynes-Cummings grid scan
python examples/generate_sample_data.py     # sample data for the figures notebook
python scripts/run_scan.py --n0 256 --n1 256 --out results/scan.npy   # CLI
```

## Directory structure

```
src/
├── config.py            # all tolerances, bounds, seeds, paths (dataclasses)
├── solvers/
│   ├── lp_solver.py        # PRIMARY: semi-infinite LP, cutting-plane method
│   └── minimize_solver.py  # independent Nelder-Mead cross-check
├── analytical/
│   ├── criteria_k.py            # analytical_k (W01, 3-index + generalized Klyshko)
│   ├── criteria_majorization.py # analytical_m + majorization-pair machinery
│   ├── criteria_classic.py      # klyshko, taylored, log01rule, generalized_klyshko*
│   ├── jaynes_cummings.py       # Pjaycum (QuTiP-free)
│   └── models.py                # QuTiP state generators (optional)
├── comparison/scan_2d.py    # run2D: numerical-vs-analytical 2D scan
├── visualization/           # boundary extraction + plotting helpers
└── utils/                   # inverse_factorials, collect_lambda, I/O helpers
notebooks/        # presentation notebooks (import from src); legacy/ = originals
examples/         # minimal runnable scripts
scripts/          # argparse CLI (run_scan.py)
tests/            # pytest suite mirroring src/
data/             # all_genK/ precomputed majorization pairs (tracked); rest ignored
results/          # generated outputs (gitignored)
config.yaml       # optional config overrides
```

## Key functions

| Function | Module | Purpose |
|----------|--------|---------|
| `solve_semi_infinite_lp` | `solvers.lp_solver` | Primary witness solver (cutting-plane LP). |
| `max_inner_value` | `solvers.lp_solver` | Exact $\max_{\mu\ge0}\boldsymbol\lambda\cdot\mathbf p(\mu)$ via roots. |
| `minimize_witness` / `scan_2d_minimize` | `solvers.minimize_solver` | Nelder-Mead cross-check solver. |
| `analytical_k` | `analytical.criteria_k` | Batch analytical criteria (W01, Klyshko, generalized). |
| `analytical_m` | `analytical.criteria_majorization` | Majorization (generalized Klyshko) criterion. |
| `klyshko`, `taylored`, `log01rule`, `analcrit2D` | `analytical.criteria_classic` | Individual analytical criteria + their union. |
| `Pjaycum` | `analytical.jaynes_cummings` | Jaynes-Cummings photon statistics $P_n(\nu,\tau)$. |
| `run2D` | `comparison.scan_2d` | 2D grid scan comparing numerical and analytical maps. |
| `extract_binary_boundary` | `visualization.boundaries` | Decision-boundary curves from a binary map. |

## Running the tests

```bash
pytest tests/ -v
```

64 tests cover the solvers (including a brute-force check of the inner
maximisation and an LP-vs-minimize agreement test), the analytical criteria,
the configuration loader, the I/O helpers, and the 2D scan.

## Dependencies and Python version

- **Python** ≥ 3.10
- **Runtime**: numpy, scipy, matplotlib, tqdm, PyYAML
- **Optional** (`[physics]`): qutip
- **Dev** (`[dev]`): pytest, ipykernel, nbconvert

Exact tested versions are pinned in `requirements.txt` and `environment.yml`.

## Reproducibility

- **Configuration** lives entirely in `src/config.py` (dataclasses). No
  tolerances or magic numbers are hard-coded in the algorithms. Override any
  value via a `config.yaml` at the repository root, or point
  `NONCLASSICALITY_CONFIG` at a different file. Only the keys you set are
  applied; unknown keys raise an error.
- **Tolerances** (defaults): LP `tol=1e-10`, `max_iter=100`, `imtol=1e-9`;
  2D scan `tol=1e-10`, `bias=1e-8`, `max_iter=300`, `imtol=1e-8`; Nelder-Mead
  `bias=1e-7`, `tol=1e-7`.
- **Random seeds**: the stochastic Nelder-Mead scans accept a `seed` argument
  (`MINIMIZE.seed`, default `None`) feeding `numpy.random.default_rng`, so runs
  are reproducible when a seed is set.
- **Thread limits**: `config.apply_thread_limits()` sets the BLAS/OpenMP
  environment variables (call before importing NumPy/SciPy).
- **Data**: the precomputed majorization-pair archives in `data/all_genK/` are
  version-controlled; large/generated artifacts are gitignored and regenerated
  with `examples/generate_sample_data.py`.
