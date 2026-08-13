"""Tests for src.comparison (the 2D numerical-vs-analytical scan)."""

import os

import numpy as np
import pytest

import src.comparison.scan_2d as scan_2d
from src.comparison.scan_2d import analyze_2d, load_scan_result, run2D


@pytest.fixture
def small_scan(tmp_path):
    """Run a tiny scan and return its outputs plus the archive path."""
    out = os.path.join(tmp_path, "scan.npz")
    result = run2D(
        N0=12, N1=12,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10, uPA=0.8, dPB=1e-10, uPB=0.1,
        Prest=np.array([0.05, 0.05]),
        fi=out,
    )
    return result, out


def test_run2d_returns_structured_result(small_scan):
    result, _ = small_scan
    assert set(["x", "y", "numerical", "ana_k", "ana_m", "counting_matrix", "analytical"]) <= set(result)
    assert isinstance(result["numerical"]["boundaries"], list)
    assert isinstance(result["ana_k"]["boundaries"], list)
    assert isinstance(result["ana_m"]["boundaries"], list)
    assert isinstance(result["counting_matrix"]["boundaries"], list)
    assert result["analytical"]["criteria"] == ("ana_k", "ana_m", "counting_matrix")


def test_run2d_writes_readable_named_archive(small_scan):
    _, path = small_scan
    assert os.path.exists(path)
    with np.load(path, allow_pickle=True) as archive:
        assert "x" in archive.files
        assert "numerical_nonclass" in archive.files
        assert "ana_k_nonclass" in archive.files
        assert "ana_m_nonclass" in archive.files
        assert "counting_matrix_nonclass" in archive.files
        assert "analytical_nonclass" in archive.files

    loaded = load_scan_result(path)
    assert loaded["x"].shape == (12,)
    assert loaded["y"].shape == (12,)
    assert loaded["fixed"].shape == (2,)
    assert loaded["numerical"]["nonclass"].shape == (12, 12)
    assert loaded["analytical"]["nonclass"].shape == (12, 12)
    assert loaded["counting_matrix"]["nonclass"].shape == (12, 12)


def test_run2d_numerical_superset_of_analytical(small_scan):
    """Every analytically-nonclassical point must also be flagged numerically.

    The analytical criteria are sufficient conditions, so the (exact) LP
    classification should cover at least the analytical region.
    """
    result, _ = small_scan
    nonclass_num = result["numerical"]["nonclass"]
    nonclass_ana = result["analytical"]["nonclass"]
    valid = np.isfinite(nonclass_num) & np.isfinite(nonclass_ana)
    # numerical >= analytical everywhere valid (no analytical-only false positives).
    assert np.all(nonclass_num[valid].astype(bool) | ~nonclass_ana[valid].astype(bool))


def test_run2d_can_skip_analytical_m(monkeypatch, tmp_path):
    """The optional analytical_m pass should not be evaluated when disabled."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError("analytical_m should have been skipped")

    monkeypatch.setattr(scan_2d, "analytical_m", fail_if_called)

    out = os.path.join(tmp_path, "scan_no_m.npz")
    result = run2D(
        N0=4, N1=4,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10, uPA=0.8, dPB=1e-10, uPB=0.1,
        Prest=np.array([0.05, 0.05]),
        fi=out,
        use_ana_m=False,
    )

    assert isinstance(result["numerical"]["boundaries"], list)
    assert isinstance(result["analytical"]["boundaries"], list)
    assert isinstance(result["ana_k"]["boundaries"], list)
    assert result["ana_m"]["boundaries"] == []
    assert result["analytical"]["criteria"] == ("ana_k", "counting_matrix")


def test_run2d_can_select_counting_matrix(monkeypatch, tmp_path):
    """The counting-matrix pass should be optional and named in the result."""
    calls = []

    def fake_analytical_c(Pall, *args, **kwargs):
        calls.append(np.asarray(Pall).shape)
        return np.zeros(Pall.shape[:-1], dtype=bool), 0.0

    monkeypatch.setattr(scan_2d, "analytical_c", fake_analytical_c)

    out = os.path.join(tmp_path, "scan_cm.npz")
    result = run2D(
        N0=4, N1=4,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10, uPA=0.8, dPB=1e-10, uPB=0.1,
        Prest=np.array([0.05, 0.05]),
        fi=out,
        use_ana_k=False,
        use_ana_m=False,
        use_cm=True,
    )

    assert calls == [(4, 4, 4)]
    assert result["ana_k"]["boundaries"] == []
    assert result["ana_m"]["boundaries"] == []
    assert result["counting_matrix"]["enabled"] is True
    assert result["analytical"]["criteria"] == ("counting_matrix",)


def test_run2d_can_skip_counting_matrix(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("analytical_c should have been skipped")

    monkeypatch.setattr(scan_2d, "analytical_c", fail_if_called)

    result = run2D(
        N0=4, N1=4,
        indx=np.array([0, 1, 2, 3], dtype=np.int16),
        dPA=1e-10, uPA=0.8, dPB=1e-10, uPB=0.1,
        Prest=np.array([0.05, 0.05]),
        fi=os.path.join(tmp_path, "scan_no_cm.npz"),
        use_ana_m=False,
        use_cm=False,
    )

    assert result["counting_matrix"]["boundaries"] == []
    assert result["analytical"]["criteria"] == ("ana_k",)


def test_run2d_allows_sweep_order_different_from_analytical_order(tmp_path):
    """The first two indx entries define the scan axes, not sort order."""
    out = os.path.join(tmp_path, "scan_unsorted.npz")
    result = run2D(
        N0=4, N1=4,
        indx=np.array([1, 3, 2, 5, 6], dtype=np.int16),
        dPA=1e-10, uPA=0.3, dPB=1e-10, uPB=0.04,
        Prest=np.array([0.05, 0.001, 0.001]),
        fi=out,
        use_ana_m=False,
    )

    assert result["x"].shape == (4,)
    assert result["y"].shape == (4,)
    assert os.path.exists(out)


def test_run2d_rejects_wrong_prest_length(tmp_path):
    out = os.path.join(tmp_path, "scan_bad_prest.npz")
    with pytest.raises(ValueError, match="Prest must have length"):
        run2D(
            N0=2, N1=2,
            indx=np.array([1, 3, 2, 5, 6], dtype=np.int16),
            Prest=np.array([0.05]),
            fi=out,
            use_ana_m=False,
        )


def test_analyze_2d_accepts_precomputed_probability_grid(tmp_path):
    """Physical-model scans can pass Pall directly instead of sweeping P_i."""
    x_values = np.linspace(0.1, 0.4, 4)
    y_values = np.linspace(0.2, 0.5, 3)
    yy, xx = np.meshgrid(y_values, x_values, indexing="ij")

    Pall = np.empty((len(y_values), len(x_values), 4))
    Pall[..., 0] = 0.55 + 0.05 * np.sin(xx)
    Pall[..., 1] = 0.12 + 0.03 * np.cos(yy)
    Pall[..., 2] = 0.18 + 0.02 * xx
    Pall[..., 3] = 0.04 + 0.01 * yy

    out = os.path.join(tmp_path, "model_scan.npz")
    result = analyze_2d(
        Pall,
        x_values,
        y_values,
        indx=np.array([0, 1, 2, 3]),
        fi=out,
        use_ana_m=False,
    )

    assert result["x"].shape == (len(x_values),)
    assert os.path.exists(out)
    loaded = load_scan_result(out)
    x_saved = loaded["x"]
    y_saved = loaded["y"]
    fixed_saved = loaded["fixed"]
    nonclass_num = loaded["numerical"]["nonclass"]
    nonclass_ana = loaded["analytical"]["nonclass"]
    assert np.array_equal(x_saved, x_values)
    assert np.array_equal(y_saved, y_values)
    assert fixed_saved.shape == (0,)
    assert nonclass_num.shape == (len(y_values), len(x_values))
    assert nonclass_ana.shape == (len(y_values), len(x_values))


def test_analyze_2d_marks_nonphysical_points_nan(monkeypatch, tmp_path):
    Pall = np.zeros((2, 2, 3))
    Pall[0, 0, :] = [0.4, 0.2, 0.1]
    Pall[0, 1, :] = [0.8, 0.3, 0.0]   # sum > 1
    Pall[1, 0, :] = [0.4, -0.1, 0.1]  # negative probability
    Pall[1, 1, :] = [0.2, 0.2, 0.2]

    calls = []

    def fake_max_inner_value(*args, **kwargs):
        return 0.0, 0.0

    def fake_solve(probs, *args, **kwargs):
        calls.append(np.asarray(probs, dtype=float).copy())
        return {"lam": np.zeros(3)}

    monkeypatch.setattr(scan_2d, "max_inner_value", fake_max_inner_value)
    monkeypatch.setattr(scan_2d, "solve_semi_infinite_lp", fake_solve)

    out = os.path.join(tmp_path, "nonphysical.npz")
    with pytest.warns(RuntimeWarning, match="non-physical probability vector"):
        analyze_2d(
            Pall,
            x_values=np.arange(2),
            y_values=np.arange(2),
            indx=np.array([0, 1, 2]),
            fi=out,
            use_ana_m=False,
        )

    assert len(calls) == 2
    assert all(np.all(p >= 0.0) and np.sum(p) <= 1.0 for p in calls)

    loaded = load_scan_result(out)
    nonclass_num = loaded["numerical"]["nonclass"]
    nonclass_ana = loaded["analytical"]["nonclass"]

    assert np.isnan(nonclass_num[0, 1])
    assert np.isnan(nonclass_num[1, 0])
    assert np.isnan(nonclass_ana[0, 1])
    assert np.isnan(nonclass_ana[1, 0])
    assert np.isfinite(nonclass_num[0, 0])
    assert np.isfinite(nonclass_num[1, 1])


def test_analyze_2d_allows_unsorted_indx_for_model_grid(tmp_path):
    x_values = np.linspace(0.1, 0.2, 2)
    y_values = np.linspace(0.3, 0.4, 2)
    Pall = np.zeros((2, 2, 4))
    Pall[..., 0] = 0.1  # P_1
    Pall[..., 1] = 0.05  # P_3
    Pall[..., 2] = 0.6  # P_0
    Pall[..., 3] = 0.2  # P_2

    result = analyze_2d(
        Pall,
        x_values,
        y_values,
        indx=np.array([1, 3, 0, 2]),
        fi=os.path.join(tmp_path, "unsorted_model_scan.npz"),
        use_ana_m=False,
    )

    assert result["indx"].tolist() == [1, 3, 0, 2]


def test_analyze_2d_rejects_coordinate_length_mismatch(tmp_path):
    Pall = np.zeros((3, 4, 2))
    with pytest.raises(ValueError, match="len\\(x_values\\)"):
        analyze_2d(
            Pall,
            x_values=np.arange(5),
            y_values=np.arange(3),
            fi=os.path.join(tmp_path, "bad_coords.npz"),
            use_ana_m=False,
        )


def test_analyze_2d_can_scan_y_in_decreasing_order(monkeypatch, tmp_path):
    """The warm-started LP sweep can visit y rows from top to bottom."""
    Pall = np.zeros((3, 2, 2))
    Pall[:, 0, 0] = [0.1, 0.2, 0.3]
    Pall[:, 1, 0] = [0.4, 0.5, 0.6]
    Pall[..., 1] = 0.01
    x_values = np.array([0.0, 1.0])
    y_values = np.array([1.0, 2.0, 3.0])

    visited = []

    def fake_max_inner_value(*args, **kwargs):
        return 0.0, 0.0

    def fake_solve(probs, *args, **kwargs):
        visited.append(float(probs[0]))
        return {"lam": np.zeros(2)}

    monkeypatch.setattr(scan_2d, "max_inner_value", fake_max_inner_value)
    monkeypatch.setattr(scan_2d, "solve_semi_infinite_lp", fake_solve)

    analyze_2d(
        Pall,
        x_values,
        y_values,
        indx=np.array([0, 1]),
        fi=os.path.join(tmp_path, "decreasing_y.npz"),
        use_ana_m=False,
        y_scan_direction="decreasing",
    )

    assert visited == [0.3, 0.2, 0.1, 0.6, 0.5, 0.4]


def test_analyze_2d_rejects_bad_y_scan_direction(tmp_path):
    Pall = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="y_scan_direction"):
        analyze_2d(
            Pall,
            x_values=np.arange(2),
            y_values=np.arange(2),
            fi=os.path.join(tmp_path, "bad_direction.npz"),
            use_ana_m=False,
            y_scan_direction="sideways",
        )
