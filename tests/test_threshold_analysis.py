"""Tests for the alert-threshold analysis (src/modeling/threshold_analysis.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import COLUMN_NAMES, TRAIN_FILE
from src.features import build_features as bf
from src.modeling import compare_models as cm
from src.modeling import threshold_analysis as ta


def make_raw(n_engines=14, seed=0):
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
            row["sensor_1"] = 518.67
            rows.append(row)
    return pd.DataFrame(rows)[COLUMN_NAMES]


def long_table(per_config):
    """per_config: {(k, threshold): [first_alert_rul per engine]} -> long first-alert table."""
    rows = []
    for (k, thr), values in per_config.items():
        for i, v in enumerate(values):
            rows.append((k, thr, i + 1, v))
    return pd.DataFrame(rows, columns=["k", "threshold", "unit_number", "first_alert_rul"])


# ------------------------------------------------------------------ alert logic
def test_first_alert_rul_with_persistence():
    prob = [0.1, 0.6, 0.7, 0.2, 0.9]
    rul = [4, 3, 2, 1, 0]
    assert ta.first_alert_rul(prob, rul, 0.5, 1) == 3
    assert ta.first_alert_rul(prob, rul, 0.5, 2) == 2     # two consecutive cycles >= 0.5
    assert np.isnan(ta.first_alert_rul(prob, rul, 0.5, 3))
    assert np.isnan(ta.first_alert_rul(prob, rul, 0.95, 1))
    assert ta.first_alert_rul(prob, rul, 0.1, 1) == 4     # threshold is inclusive


def test_engine_outcome_classification_and_cost():
    first = pd.Series([np.nan, 5.0, 30.0, 50.0, 20.0])
    flags = ta.classify(first, min_notice=10, horizon=24, grace=15)
    assert flags["missed"].sum() == 1 and flags["too_late"].sum() == 1
    assert flags["premature"].sum() == 2 and flags["far_premature"].sum() == 1
    summary = ta.summarize_engine_policy(long_table({(1, 0.5): list(first)}), 10, 24, 15, ratios=(10,))
    row = summary.iloc[0]
    assert row["bad"] == 2 and row["far_premature"] == 1
    assert row["cost_r10"] == pytest.approx((10 * 2 + 1) / 5)


def test_mark_stable_requires_next_thresholds_to_be_safe():
    table = long_table({
        (1, 0.1): [20] * 10, (1, 0.2): [20] * 10, (1, 0.3): [20] * 9 + [5], (1, 0.4): [20] * 10,
    })
    summary = ta.mark_stable(ta.summarize_engine_policy(table), margin_steps=2)
    stable = dict(zip(summary["threshold"], summary["stable"]))
    assert stable == {0.1: False, 0.2: False, 0.3: False, 0.4: True}


# ------------------------------------------------------------------ selection
def test_choose_provisional_is_safe_stable_and_prefers_the_simplest_tied_rule():
    far = lambda n_far: [50] * n_far + [20] * (20 - n_far)
    table = long_table({
        (1, 0.1): far(6), (1, 0.2): far(4), (1, 0.3): far(2), (1, 0.4): [5] + [20] * 19,
        (2, 0.1): far(2), (2, 0.2): far(2), (2, 0.3): far(2), (2, 0.4): far(2),
    })
    choice = ta.choose_provisional(table, ratio=10)
    assert choice["safety_constraint_met"]
    assert choice["chosen_row"]["bad"] == 0
    assert (choice["k"], choice["threshold"]) == (2, 0.1)
    # the cheap-looking k=1, threshold 0.3 sits next to the unsafe 0.4 setting and must not be chosen
    assert (choice["k"], choice["threshold"]) != (1, 0.3)


def test_choose_provisional_reports_when_no_setting_is_safe():
    table = long_table({(1, 0.1): [5] + [20] * 9, (1, 0.2): [5] + [20] * 9})
    assert not ta.choose_provisional(table, ratio=10)["safety_constraint_met"]


def test_bootstrap_se_is_zero_for_constant_costs_and_deterministic():
    assert ta.bootstrap_se(np.ones(20)) == 0.0
    v = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=float)
    assert ta.bootstrap_se(v) == ta.bootstrap_se(v) > 0


def test_robust_choice_requires_safety_in_every_variant():
    far = lambda n: [50] * n + [20] * (20 - n)
    late = lambda n: [5] + [20] * 19
    a = long_table({(1, 0.1): far(6), (1, 0.2): far(4), (1, 0.3): far(3), (1, 0.4): far(2), (1, 0.5): far(1)})
    b = long_table({(1, 0.1): far(6), (1, 0.2): far(4), (1, 0.3): far(3), (1, 0.4): late(0), (1, 0.5): late(0)})
    assert ta.choose_provisional(a, ratio=10)["threshold"] > 0.1           # variant A alone would go higher
    robust = ta.robust_choice({"A": a, "B": b}, ratio=10)
    assert robust["safety_constraint_met"] and robust["n_variants"] == 2
    assert robust["threshold"] == 0.1                                       # B makes everything above 0.1 unstable


def test_first_unsafe_threshold():
    summary = ta.summarize_engine_policy(long_table({(2, 0.3): [20] * 5, (2, 0.5): [5] + [20] * 4, (2, 0.7): [5] + [20] * 4}))
    assert ta.first_unsafe_threshold(summary, 2) == 0.5
    assert ta.first_unsafe_threshold(summary, 1) is None


def test_row_level_sweep_counts_and_persistence():
    pred = pd.DataFrame({
        "unit_number": [1] * 6, "time_cycles": range(30, 36), "rul": [5, 4, 3, 2, 1, 0],
        "target": [1, 1, 1, 1, 1, 1], "eligible": True, "p": [0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
    })
    one = ta.row_level_sweep(pred, "p", [0.5], k=1).iloc[0]
    two = ta.row_level_sweep(pred, "p", [0.5], k=2).iloc[0]
    assert (one["tp"], one["fn"]) == (4, 2)
    assert (two["tp"], two["fn"]) == (1, 5)   # only the cycle after two consecutive alerts counts
    assert one["precision"] == 1.0 and one["recall"] == pytest.approx(4 / 6)


# ------------------------------------------------------------------ probabilities
def test_stream_probabilities_skips_rows_without_a_prediction():
    class Stub:
        def __init__(self):
            self.seen = {}

        def predict(self, row):
            n = self.seen[row["unit_number"]] = self.seen.get(row["unit_number"], 0) + 1
            return None if n <= 2 else {"failure_probability": 0.25 * n}

    rows = pd.DataFrame({"unit_number": [2, 1, 1, 1, 2, 2], "time_cycles": [1, 3, 1, 2, 3, 2]})
    out = ta.stream_probabilities(rows, Stub(), "p_x")
    assert list(out["unit_number"]) == [1, 2] and list(out["time_cycles"]) == [3, 3]
    assert list(out["p_x"]) == [0.75, 0.75]


def test_oof_probabilities_score_every_train_engine_once_and_are_deterministic():
    raw, matrix = cm.prepare_data(make_raw())
    a = ta.oof_probabilities(raw, matrix, "lr_eng242", n_splits=3)
    b = ta.oof_probabilities(raw, matrix, "lr_eng242", n_splits=3)
    n_train = int((matrix["split"] == "train").sum())
    assert len(a) == n_train
    assert set(a["unit_number"]) == set(matrix.loc[matrix["split"] == "train", "unit_number"])
    assert a["p_lr_eng242"].between(0, 1).all()
    np.testing.assert_allclose(a["p_lr_eng242"], b["p_lr_eng242"])


def test_oof_regularisation_can_be_varied_for_logistic_models_only():
    raw, matrix = cm.prepare_data(make_raw())
    weak = ta.oof_probabilities(raw, matrix, "lr_eng242", n_splits=3, C=0.001)
    strong = ta.oof_probabilities(raw, matrix, "lr_eng242", n_splits=3, C=100.0)
    assert not np.allclose(weak["p_lr_eng242"], strong["p_lr_eng242"])
    with pytest.raises(ValueError):
        ta.oof_probabilities(raw, matrix, "hgb_eng242", n_splits=3, C=1.0)


def test_unsupported_model_is_rejected():
    raw, matrix = cm.prepare_data(make_raw())
    with pytest.raises(ValueError):
        ta.oof_probabilities(raw, matrix, "gru_raw24")


# ------------------------------------------------------------------ end to end
def test_run_writes_report_json_and_tables(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_raw().to_csv(raw_dir / TRAIN_FILE, sep=" ", header=False, index=False)
    out = tmp_path / "reports"
    result = ta.run(raw_dir, out, "lr_eng242", "oof")

    for name in ["threshold_analysis.md", "provisional_threshold.json", "threshold_row_sweep_val.csv",
                 "threshold_engine_sweep_decision.csv", "threshold_engine_sweep_val.csv",
                 "threshold_decision_predictions.csv"]:
        assert (out / name).is_file(), name
    config = json.loads((out / "provisional_threshold.json").read_text(encoding="utf-8"))
    assert config["status"] == "provisional" and config["model"] == "lr_eng242"
    assert config["alert_threshold"] in ta.THRESHOLDS and config["persistence_cycles"] in ta.PERSISTENCE
    assert config["horizon_cycles"] == bf.HORIZON
    text = (out / "threshold_analysis.md").read_text(encoding="utf-8")
    assert "Recommendation" in text and "Cost of the two error types" in text
    assert "Robustness to the model fit" in text and "variant_choices" in config
    assert config["selection"].startswith("robust")
    # decision engines are TRAIN engines only; the held-out table covers validation engines
    train_units = set(result["decision_pred"]["unit_number"])
    val_units = set(result["val_pred"]["unit_number"])
    assert train_units.isdisjoint(val_units)
