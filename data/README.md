# Data

This directory holds input datasets.

- `all_genK/` — precomputed generalized-Klyshko / majorization pairs (`genKlyshko_up_to{3..7}.pkl`), loaded by `src.analytical.criteria_majorization.load_precomputed_pairs`.

Generated examples, notebooks, and command-line scans write to `results/`, which is gitignored except for its `.gitkeep` placeholder.

The tracked archives cover consecutive photon indices starting at zero for `N_P=4` through `8`. The optional `genKlyshko_up_to8.pkl` (`N_P=9`) is ignored by Git. Missing archives and nonconsecutive indices fall back to generating pairs at runtime.
