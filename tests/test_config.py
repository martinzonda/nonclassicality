"""Tests for src.config (dataclasses and YAML override loading)."""

import os

import pytest

from src.config import (
    Config,
    SolverConfig,
    load_config,
    _apply_overrides,
    SOLVER,
    SCAN,
    PATHS,
)


def test_default_values_unchanged():
    # Lock in the documented defaults (also guards the "no logic change" promise).
    assert SOLVER.tol == 1e-10
    assert SOLVER.max_iter == 100
    assert SOLVER.imtol == 1e-9
    assert SCAN.bias == 1e-8
    assert SCAN.max_iter == 300


def test_lambda_bound_property():
    assert SolverConfig().lambda_bound == (-1.0, 1.0)


def test_genk_files_paths():
    files = PATHS.genk_files
    assert len(files) == 6
    assert files[0].endswith(os.path.join("data", "all_genK", "genKlyshko_up_to3.pkl"))


def test_apply_overrides_nested():
    cfg = Config()
    _apply_overrides(cfg, {"solver": {"tol": 1e-5, "max_iter": 9}, "scan": {"n0": 32}})
    assert cfg.solver.tol == 1e-5
    assert cfg.solver.max_iter == 9
    assert cfg.scan.n0 == 32


def test_apply_overrides_tuple_from_list():
    cfg = Config()
    _apply_overrides(cfg, {"scan": {"prest": [0.1, 0.2, 0.3]}})
    assert cfg.scan.prest == (0.1, 0.2, 0.3)


def test_apply_overrides_unknown_key_raises():
    with pytest.raises(KeyError):
        _apply_overrides(Config(), {"solver": {"does_not_exist": 1}})


def test_load_config_from_yaml(tmp_path):
    path = os.path.join(tmp_path, "c.yaml")
    with open(path, "w") as f:
        f.write("seed: 42\nsolver:\n  tol: 1.0e-06\n")
    cfg = load_config(path)
    assert cfg.seed == 42
    assert cfg.solver.tol == 1e-6


def test_load_config_missing_file_uses_defaults():
    cfg = load_config("/nonexistent/config.yaml")
    assert cfg.solver.tol == 1e-10
