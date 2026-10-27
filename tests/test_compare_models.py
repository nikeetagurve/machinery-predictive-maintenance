"""Tests for the model-comparison utilities (src/modeling/compare_models.py)."""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import COLUMN_NAMES, TRAIN_FILE
from src.features import build_features as bf
from src.modeling import compare_models as cm


def make_raw(n_engines=12, seed=0):
    """Synthetic raw FD001-style table whose sensors drift as each engine nears failure."""
    rng = np.random.default_rng(seed)
    rows = []
    for engine in range(1, n_engines + 1):
        n_cycles = 60 + 2 * engine
        for cycle in range(1, n_cycles + 1):
            wear = (cycle / n_cycles) ** 3
            row = {"unit_number": engine, "time_cycles": cycle}
            for col in bf.OP_COLS:
                row[col] = 100.0 if col == "op_setting_3" else rng.normal(0, 0.01)
            for i, col in enumerate(bf.SENSOR_COLS):
                row[col] = 500 + rng.normal(0, 1) + (10 if i % 2 else -10) * wear
            row["sensor_1"] = 518.67  # constant channel
            rows.append(row)
    return pd.DataFrame(rows)[COLUMN_NAMES]


@pytest.fixture(scope="module")
def prepared():
    raw, matrix = cm.prepare_data(make_raw())
    selected = bf.feature_columns(matrix)[:20]
    probs = cm.run_sklearn_models(raw, matrix, selected)
    return raw, matrix, probs


@pytest.fixture(scope="module")
def predictions(prepared):
    raw, _, probs = prepared
    return cm.build_prediction_table(raw, probs)


def test_raw_table_and_matrix_are_row_aligned(prepared):
    raw, matrix, _ = prepared
    assert raw[cm.ID_COLS].equals(matrix[cm.ID_COLS])
    assert raw["split"].equals(matrix["split"])


def test_models_score_the_same_validation_rows_and_never_the_others(prepared, predictions):
    raw, _, probs = prepared
    n_val = int((raw["split"] == "val").sum())
    assert len(predictions) == n_val
    for name, p in probs.items():
        assert len(p) == n_val, name
        assert np.all((p >= 0) & (p <= 1)), name
    val_units = set(raw.loc[raw["split"] == "val", "unit_number"])
    assert set(predictions["unit_number"]) == val_units


def test_eligible_rows_are_cycles_from_sequence_length(predictions):
    assert (predictions["eligible"] == (predictions["time_cycles"] >= cm.SEQUENCE_LENGTH)).all()
    assert predictions["eligible"].any() and (~predictions["eligible"]).any()


def test_metrics_confusion_counts_are_consistent(predictions):
    table = cm.build_metrics_table(predictions)
    n_rows = int(predictions["eligible"].sum())
    n_pos = int(predictions.loc[predictions["eligible"], "target"].sum())
    assert set(table["model"]) >= {"dummy_always_0", "lr_raw24", "lr_eng242", "hgb_eng242", "hgb_eng166"}
    for _, r in table.iterrows():
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == n_rows
        assert r["tp"] + r["fn"] == n_pos
    dummy = table.set_index("model").loc["dummy_always_0"]
    assert dummy["tp"] == 0 and dummy["fp"] == 0 and dummy["recall"] == 0
    assert dummy["accuracy"] == pytest.approx(1 - n_pos / n_rows)


def test_compute_metrics_on_a_known_example():
    y = [1, 1, 0, 0, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.1, 0.1]
    m = cm.compute_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_run_sklearn_models_is_deterministic(prepared):
    raw, matrix, probs = prepared
    again = cm.run_sklearn_models(raw, matrix, bf.feature_columns(matrix)[:20])
    for name in probs:
        np.testing.assert_allclose(probs[name], again[name])


def test_alert_summary_lead_time_missed_and_early_alerts():
    # engine 1: alerts at RUL 30 (early), 22, 10 -> lead time 22, one early alert
    # engine 2: alert only at RUL 40 -> no alert inside the window = missed
    rows = []
    for unit, ruls, alerts in [(1, [30, 22, 10, 5], {30, 22, 10}), (2, [40, 20, 5], {40})]:
        for r in ruls:
            rows.append({"unit_number": unit, "time_cycles": 100 - r, "rul": r, "target": int(r <= 24),
                         "eligible": True, "p_x": 0.9 if r in alerts else 0.1})
    summary = cm.alert_summary(pd.DataFrame(rows), "p_x").set_index("unit_number")
    assert summary.loc[1, "lead_time_cycles"] == 22
    assert summary.loc[1, "early_alert_rows"] == 1 and not summary.loc[1, "missed"]
    assert summary.loc[1, "first_alert_rul"] == 30
    assert summary.loc[2, "missed"] and np.isnan(summary.loc[2, "lead_time_cycles"])


def test_load_selected_features_validates_against_matrix(tmp_path):
    f = tmp_path / "proposal.csv"
    pd.DataFrame({"family": ["lag", "lag"]}, index=pd.Index(["a", "b"], name="feature")).to_csv(f)
    assert cm.load_selected_features(f, available=["a", "b", "c"]) == ["a", "b"]
    with pytest.raises(ValueError):
        cm.load_selected_features(f, available=["a"])
    assert cm.load_selected_features(tmp_path / "missing.csv") is None


def test_run_writes_all_outputs(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_raw().to_csv(raw_dir / TRAIN_FILE, sep=" ", header=False, index=False)
    out = tmp_path / "reports"
    result = cm.run(raw_dir, out, with_gru=False, selected_features_file=tmp_path / "none.csv")
    for name in ["model_predictions_val.csv", "model_comparison_metrics.csv", "model_comparison.md"]:
        assert (out / name).is_file()
    text = (out / "model_comparison.md").read_text(encoding="utf-8")
    assert "Validation metrics" in text and "Imbalance handling" in text
    assert "hgb_eng166" not in set(result["metrics"]["model"])  # no proposal file -> model skipped
