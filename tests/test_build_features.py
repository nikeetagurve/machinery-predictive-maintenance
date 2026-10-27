"""Tests for the FD001 feature-generation pipeline (src/features/build_features.py)."""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import COLUMN_NAMES, TRAIN_FILE
from src.features import build_features as bf


def make_raw(n_engines=10, seed=0):
    """Small synthetic table in the raw FD001 layout (26 columns, no header)."""
    rng = np.random.default_rng(seed)
    rows = []
    for engine in range(1, n_engines + 1):
        n_cycles = 30 + 3 * engine
        for cycle in range(1, n_cycles + 1):
            row = {"unit_number": engine, "time_cycles": cycle}
            for col in bf.OP_COLS:
                row[col] = 100.0 if col == "op_setting_3" else rng.normal(0, 0.01)
            for col in bf.SENSOR_COLS:
                row[col] = rng.normal(500, 5)
            row["sensor_1"] = 518.67  # constant everywhere
            # constant on the first 7 (train) engines, varies afterwards
            row["sensor_2"] = 5.0 if engine <= 7 else rng.normal(5, 1)
            rows.append(row)
    return pd.DataFrame(rows)[COLUMN_NAMES]


@pytest.fixture(scope="module")
def raw():
    return make_raw()


@pytest.fixture(scope="module")
def matrix(raw):
    return bf.build_feature_matrix(raw)


def test_labels_and_ids_are_never_model_features(matrix):
    features = bf.feature_columns(matrix)
    for forbidden in ["unit_number", "time_cycles", "rul", "target", "split"]:
        assert forbidden not in features
    assert not any("rul" in c.split("_") or "target" in c for c in features)


def test_constant_channels_dropped_using_train_engines_only(matrix):
    assert "sensor_1" not in matrix.columns  # constant everywhere
    assert "op_setting_3" not in matrix.columns
    # sensor_2 is constant on train engines (1-7) but varies later: still dropped,
    # because the decision is made on training engines only.
    assert "sensor_2" not in matrix.columns
    assert "sensor_3" in matrix.columns


def test_lag_and_rolling_values_use_only_past_cycles(raw, matrix):
    engine = raw[raw.unit_number == 3].sort_values("time_cycles").reset_index(drop=True)
    out = matrix[matrix.unit_number == 3].reset_index(drop=True)
    s = engine["sensor_3"]
    for i in range(1, len(s)):
        assert out.loc[i, "sensor_3_lag_1"] == pytest.approx(s[i - 1])
        past = s[max(0, i - 5):i]
        assert out.loc[i, "sensor_3_rolling_mean_5"] == pytest.approx(past.mean())
        assert out.loc[i, "sensor_3_rolling_max_5"] == pytest.approx(past.max())


def test_changing_future_rows_does_not_change_earlier_features(raw):
    changed = raw.copy()
    mask = (changed.unit_number == 2) & (changed.time_cycles >= 20)
    changed.loc[mask, bf.SENSOR_COLS] += 1000.0

    a = bf.build_feature_matrix(raw)
    b = bf.build_feature_matrix(changed)
    feats = bf.feature_columns(a)
    early = (a.unit_number == 2) & (a.time_cycles < 20)
    pd.testing.assert_frame_equal(a.loc[early, feats], b.loc[early, feats])


def test_warmup_rows_are_left_as_nan(matrix):
    for lag in bf.LAGS:
        col = f"sensor_3_lag_{lag}"
        nan_per_engine = matrix.groupby("unit_number")[col].apply(lambda x: x.isna().sum())
        assert (nan_per_engine == lag).all()


def test_pipeline_is_deterministic(raw):
    pd.testing.assert_frame_equal(bf.build_feature_matrix(raw), bf.build_feature_matrix(raw))


def test_engines_belong_to_one_split_and_labels_follow_the_rule(matrix):
    assert (matrix.groupby("unit_number")["split"].nunique() == 1).all()
    assert set(matrix["split"]) == {"train", "val", "test"}
    assert (matrix["target"] == (matrix["rul"] <= bf.HORIZON).astype(int)).all()
    positives = matrix.groupby("unit_number")["target"].sum()
    lengths = matrix.groupby("unit_number").size()
    assert (positives == np.minimum(bf.HORIZON + 1, lengths)).all()


def test_manifest_covers_every_feature_exactly_once(matrix):
    manifest = bf.describe_features(matrix)
    assert list(manifest["feature"]) == bf.feature_columns(matrix)
    assert manifest["feature"].is_unique
    assert set(manifest["family"]) == {
        "raw_setting", "raw_sensor", "lag",
        "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
    }


def test_run_writes_outputs_and_gives_identical_checksum(tmp_path, raw):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw.to_csv(raw_dir / TRAIN_FILE, sep=" ", header=False, index=False)

    out = tmp_path / "processed" / "features.csv"
    manifest = tmp_path / "reports" / "manifest.csv"
    first = bf.run(raw_dir, out, manifest)
    second = bf.run(raw_dir, out, manifest)

    assert out.is_file() and manifest.is_file()
    assert first["sha256"] == second["sha256"]
    assert first["rows"] == len(raw)
