# Data

This directory holds input datasets.

- `all_genK/` — precomputed generalized-Klyshko / majorization pairs (`genKlyshko_up_to{3..8}.pkl`), loaded by `src.analytical.criteria_majorization.load_precomputed_pairs`.
- `aux.npy` — auxiliary scan output.

Large generated data is gitignored (see Phase 7); regenerate samples with `examples/generate_sample_data.py`.
