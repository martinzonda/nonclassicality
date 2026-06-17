"""Centralized configuration: tolerances, solver settings, paths and seeds.

All numerical constants, solver parameters and filesystem paths live here as
dataclasses, so nothing is hard-coded inline in the rest of ``src/``. Values may
be overridden at process start via a ``config.yaml`` file at the repository root
(or the path given by the ``NONCLASSICALITY_CONFIG`` environment variable).

The module exposes ready-made singletons consumed throughout the package::

    from src.config import SOLVER, MINIMIZE, SCAN, VIZ, PATHS, THREADS, CONFIG

Because the singletons are built when this module is first imported (before any
other ``src`` module), overrides in ``config.yaml`` are picked up by the default
arguments of the solver/scan functions.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Repository root: this file is src/config.py, so the root is one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SolverConfig:
    """Parameters for the LP cutting-plane solver (:mod:`src.solvers.lp_solver`)."""

    tol: float = 1e-10            # convergence tolerance on the cut violation
    max_iter: int = 100           # maximum cutting-plane iterations
    imtol: float = 1e-9           # imaginary-part tolerance for real roots
    row_tol: float = 1e-12        # relative tolerance for redundant cuts/rows
    zero_row_atol: float = 1e-14  # magnitude below which a row is "zero"
    lambda_lower: float = -1.0    # lower bound on each lambda component
    lambda_upper: float = 1.0     # upper bound on each lambda component
    mu_init: tuple[float, ...] = (0.0,)  # initial cut location(s)

    @property
    def lambda_bound(self) -> tuple[float, float]:
        """Per-component bound ``(lower, upper)`` for the coefficients."""
        return (self.lambda_lower, self.lambda_upper)


@dataclass
class MinimizeConfig:
    """Parameters for the Nelder-Mead solver (:mod:`src.solvers.minimize_solver`)."""

    bias: float = 1e-7            # minimum witness value to flag nonclassicality
    tol: float = 1e-7             # optimizer tolerance
    lambda_lower: float = -1.0
    lambda_upper: float = 1.0
    seed: int | None = None       # RNG seed for random starting points

    @property
    def lambda_bound(self) -> tuple[float, float]:
        return (self.lambda_lower, self.lambda_upper)


@dataclass
class ScanConfig:
    """Parameters for the 2D parameter scan (:func:`src.comparison.scan_2d.run2D`)."""

    n0: int = 512                 # grid resolution along the first axis
    n1: int = 512                 # grid resolution along the second axis
    tol: float = 1e-10            # LP convergence tolerance
    bias: float = 1e-8            # minimum witness value to flag nonclassicality
    max_iter: int = 300           # maximum cutting-plane iterations per point
    imtol: float = 1e-8           # imaginary-part tolerance for real roots
    lambda_lower: float = -1.0
    lambda_upper: float = 1.0
    dPA: float = 1e-10            # range of the first swept probability
    uPA: float = 1.0
    dPB: float = 1e-10            # range of the second swept probability
    uPB: float = 0.4
    prest: tuple[float, ...] = (0.05, 0.05)  # fixed remaining probabilities

    @property
    def lambda_bound(self) -> tuple[float, float]:
        return (self.lambda_lower, self.lambda_upper)


@dataclass
class VizConfig:
    """Visualization parameters (:mod:`src.visualization`)."""

    boundary_level: float = 0.5   # contour level separating classical/nonclassical


@dataclass
class ThreadConfig:
    """BLAS / OpenMP thread limits, applied via :func:`apply_thread_limits`."""

    omp: int = 4
    openblas: int = 4
    mkl: int = 4
    veclib: int = 4
    numexpr: int = 4


@dataclass
class PathConfig:
    """Filesystem paths, all resolved relative to the repository root."""

    repo_root: str = REPO_ROOT
    file_index_offset: int = 4    # up_to3.pkl matches N_P = 4 -> lof[N_P - 4]
    genk_filenames: tuple[str, ...] = (
        "genKlyshko_up_to3.pkl",
        "genKlyshko_up_to4.pkl",
        "genKlyshko_up_to5.pkl",
        "genKlyshko_up_to6.pkl",
        "genKlyshko_up_to7.pkl",
        "genKlyshko_up_to8.pkl",
    )

    @property
    def data_dir(self) -> str:
        return os.path.join(self.repo_root, "data")

    @property
    def genk_dir(self) -> str:
        return os.path.join(self.data_dir, "all_genK")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.repo_root, "results")

    @property
    def genk_files(self) -> list[str]:
        """Full paths to the precomputed majorization-pair archives."""
        return [os.path.join(self.genk_dir, name) for name in self.genk_filenames]


@dataclass
class Config:
    """Aggregate configuration object."""

    seed: int | None = None
    solver: SolverConfig = field(default_factory=SolverConfig)
    minimize: MinimizeConfig = field(default_factory=MinimizeConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    threads: ThreadConfig = field(default_factory=ThreadConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply a (possibly nested) dict of overrides onto ``obj``.

    Parameters
    ----------
    obj : dataclass instance
        Target configuration object, mutated in place.
    overrides : dict
        Mapping of field names to new values; nested dicts recurse into nested
        dataclass fields.

    Raises
    ------
    KeyError
        If an override key does not correspond to a field of ``obj``.
    """
    valid = {f.name for f in fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        elif isinstance(value, list):
            # YAML maps tuples to lists; restore tuple for tuple-typed fields.
            setattr(obj, key, tuple(value) if isinstance(current, tuple) else value)
        else:
            setattr(obj, key, value)


def find_config_file() -> str | None:
    """Locate a ``config.yaml`` override file, if one exists.

    Returns
    -------
    str or None
        Path from ``$NONCLASSICALITY_CONFIG`` if set, otherwise
        ``<repo_root>/config.yaml`` if present, otherwise ``None``.
    """
    env_path = os.environ.get("NONCLASSICALITY_CONFIG")
    if env_path:
        return env_path
    default_path = os.path.join(REPO_ROOT, "config.yaml")
    return default_path if os.path.exists(default_path) else None


def load_config(path: str | None = None) -> Config:
    """Build a :class:`Config`, applying overrides from a YAML file if available.

    Parameters
    ----------
    path : str or None, optional
        Explicit path to a YAML override file. If ``None``, falls back to
        :func:`find_config_file`. A missing file or missing PyYAML simply
        yields the built-in defaults.

    Returns
    -------
    Config
        The (possibly overridden) configuration.
    """
    config = Config()

    if path is None:
        path = find_config_file()
    if not path:
        return config

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; ignoring %s and using defaults.", path)
        return config

    try:
        with open(path, "r") as handle:
            overrides = yaml.safe_load(handle) or {}
    except OSError:
        logger.exception("Could not read config file %s; using defaults.", path)
        return config

    _apply_overrides(config, overrides)
    logger.info("Loaded configuration overrides from %s", path)
    return config


def apply_thread_limits(threads: "ThreadConfig | None" = None) -> None:
    """Set BLAS/OpenMP thread-count environment variables.

    Must be called *before* importing NumPy/SciPy to take effect. Provided as
    an explicit opt-in (scripts/notebooks call it) rather than a side effect of
    import.

    Parameters
    ----------
    threads : ThreadConfig, optional
        Thread limits to apply. Defaults to the module singleton ``THREADS``.
    """
    if threads is None:
        threads = THREADS
    os.environ["OMP_NUM_THREADS"] = str(threads.omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads.openblas)
    os.environ["MKL_NUM_THREADS"] = str(threads.mkl)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads.veclib)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads.numexpr)


# Singletons consumed across the package. Built once, at first import.
CONFIG = load_config()
SOLVER = CONFIG.solver
MINIMIZE = CONFIG.minimize
SCAN = CONFIG.scan
VIZ = CONFIG.viz
THREADS = CONFIG.threads
PATHS = CONFIG.paths
