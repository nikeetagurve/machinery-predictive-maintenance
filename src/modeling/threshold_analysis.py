"""Precision-recall trade-off and provisional alert-threshold selection (FD001).

Two views of the same probabilities
  Row view     "is this engine-cycle inside the failure window?"  -> precision / recall / F1 per threshold.
  Engine view  "when does the alert system FIRST fire for each engine?"  -> what an operator actually experiences.
               An engine is: missed (never alerted), too late (first alert < MIN_NOTICE cycles before failure),
               a premature alarm (first alert > HORIZON cycles before failure) or a far-premature alarm
               (first alert > HORIZON + GRACE cycles before failure; treated as a real false alarm).

Alert rule: raise an alert when the failure probability is >= threshold for `k` consecutive cycles
(k = 1 is the plain "probability >= threshold" rule used by predict.py today).

Cost model (relative units, to be confirmed with the client)
  cost per engine = ratio * (missed or too-late) + (far-premature)
  `ratio` = cost of an unplanned failure / cost of one unnecessary maintenance halt. The real ratio is NOT known;
  results are shown for several ratios.

Selection method (see choose_provisional)
  1. Safety first: keep only (threshold, k) settings where NO decision engine is missed or alerted too late, AND the
     next MARGIN_STEPS higher thresholds (same k) are also safe, so the choice is not on the edge of a cliff.
  2. Among those, find the lowest expected cost at the central ratio.
  3. Among settings whose cost is within one bootstrap standard error of that minimum (statistically tied), take the
     simplest rule: fewest persistence cycles, then lowest cost, then the longest minimum warning.
  4. Robustness (logistic regression, default on): the whole analysis is repeated for several regularisation strengths
     (ROBUST_C) and only settings that are stably safe in EVERY variant are eligible. Small changes in the model fit
     (or library versions) move the safety cliff, so a threshold picked from a single fit is not trustworthy.
  5. Confirm once on the held-out validation engines. The reserved test split is never used.

Decision engines: for scikit-learn models, out-of-fold predictions on the TRAIN engines (grouped by engine);
validation engines are then a genuine hold-out check. For models without out-of-fold predictions (e.g. the GRU),
use --decision-source val: the choice is then made on 15 engines only and has no hold-out confirmation.

Run from the repo root:
    python -m src.modeling.threshold_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.load_data import RAW_DIR, REPORTS_DIR, TRAIN_FILE, load_cmapss_file
from src.features import build_features as bf
from src.modeling import compare_models as cm

MACHINE_COL, CYCLE_COL = bf.MACHINE_COL, bf.CYCLE_COL
HORIZON = bf.HORIZON          # 24 cycles (provisional, pending team approval)
MIN_NOTICE = 10               # ASSUMPTION: cycles of warning needed to schedule maintenance
GRACE = 15                    # ASSUMPTION: alerts up to HORIZON + GRACE cycles before failure are acceptable
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.951, 0.05)] + [0.99]
PERSISTENCE = (1, 2, 3)
COST_RATIOS = (5, 10, 20, 50)
CENTRAL_RATIO = 10            # ASSUMPTION, shown with other ratios as sensitivity
MARGIN_STEPS = 2              # a setting must stay safe for the next 2 higher thresholds (+0.10)
ROBUST_C = (0.5, 0.7, 1.0, 1.5, 2.0)  # logistic-regression regularisation values used for the robustness check
N_BOOT = 2000
SEED = 42

SKLEARN_MODELS = ("lr_raw24", "lr_eng242", "hgb_eng242")

OUT_REPORT = REPORTS_DIR / "threshold_analysis.md"
OUT_JSON = REPORTS_DIR / "provisional_threshold.json"
OUT_ROW_SWEEP = REPORTS_DIR / "threshold_row_sweep_val.csv"
OUT_ENGINE_DECISION = REPORTS_DIR / "threshold_engine_sweep_decision.csv"
OUT_ENGINE_VAL = REPORTS_DIR / "threshold_engine_sweep_val.csv"
OUT_DECISION_PRED = REPORTS_DIR / "threshold_decision_predictions.csv"


# ----------------------------------------------------------------------------- alert logic
def first_alert_rul(prob, rul, threshold: float, k: int = 1) -> float:
    """RUL at the first cycle where prob >= threshold for k consecutive cycles (NaN if never).

    `prob` and `rul` must be ordered by cycle (RUL decreasing).
    """
    alert = (np.asarray(prob, dtype=float) >= threshold).astype(int)
    if k > 1:
        alert = (np.convolve(alert, np.ones(k, dtype=int), mode="full")[: len(alert)] >= k).astype(int)
    hits = np.flatnonzero(alert)
    return float(np.asarray(rul)[hits[0]]) if len(hits) else np.nan


def engine_first_alerts(pred: pd.DataFrame, prob_col: str, thresholds=THRESHOLDS, ks=PERSISTENCE) -> pd.DataFrame:
    """Long table: for every (k, threshold, engine) the RUL at the first alert (NaN = never alerted)."""
    data = pred[pred["eligible"]].sort_values([MACHINE_COL, CYCLE_COL])
    rows = []
    for unit, g in data.groupby(MACHINE_COL, sort=True):
        p, r = g[prob_col].to_numpy(), g["rul"].to_numpy()
        for k in ks:
            for thr in thresholds:
                rows.append((k, thr, unit, first_alert_rul(p, r, thr, k)))
    return pd.DataFrame(rows, columns=["k", "threshold", MACHINE_COL, "first_alert_rul"])


def classify(first: pd.Series, min_notice=MIN_NOTICE, horizon=HORIZON, grace=GRACE) -> pd.DataFrame:
    """Per-engine outcome flags from first-alert RUL values."""
    return pd.DataFrame({
        "missed": first.isna(),
        "too_late": first < min_notice,
        "premature": first > horizon,
        "far_premature": first > horizon + grace,
    })


def summarize_engine_policy(first_alerts: pd.DataFrame, min_notice=MIN_NOTICE, horizon=HORIZON,
                            grace=GRACE, ratios=COST_RATIOS) -> pd.DataFrame:
    """One row per (k, threshold): outcome counts, lead-time statistics and expected cost per engine."""
    rows = []
    for (k, thr), g in first_alerts.groupby(["k", "threshold"], sort=True):
        flags = classify(g["first_alert_rul"], min_notice, horizon, grace)
        bad = int((flags["missed"] | flags["too_late"]).sum())
        far = int(flags["far_premature"].sum())
        first = g["first_alert_rul"]
        row = {
            "k": k, "threshold": thr, "engines": len(g),
            "missed": int(flags["missed"].sum()), "too_late": int(flags["too_late"].sum()), "bad": bad,
            "premature": int(flags["premature"].sum()), "far_premature": far,
            "median_first_alert_rul": first.median(), "min_first_alert_rul": first.min(),
            "p10_first_alert_rul": first.quantile(0.10),
        }
        for r in ratios:
            row[f"cost_r{r}"] = (r * bad + far) / len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def row_level_sweep(pred: pd.DataFrame, prob_col: str, thresholds=THRESHOLDS, k: int = 1) -> pd.DataFrame:
    """Row-level precision / recall / F1 / F2 on the eligible rows (alert = k consecutive cycles >= threshold)."""
    data = pred[pred["eligible"]].sort_values([MACHINE_COL, CYCLE_COL])
    y = data["target"].to_numpy().astype(bool)
    rows = []
    for thr in thresholds:
        hit = (data[prob_col] >= thr).astype(int)
        if k > 1:
            hit = hit.groupby(data[MACHINE_COL]).transform(lambda s: s.rolling(k).sum().ge(k).astype(int))
        alert = hit.to_numpy().astype(bool)
        tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
        fn, tn = int((~alert & y).sum()), int((~alert & ~y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
        rows.append({"threshold": thr, "k": k, "precision": prec, "recall": rec, "f1": f1, "f2": f2,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- selection
def _engine_costs(g: pd.DataFrame, ratio: float, min_notice, horizon, grace) -> np.ndarray:
    flags = classify(g["first_alert_rul"], min_notice, horizon, grace)
    bad = (flags["missed"] | flags["too_late"]).to_numpy()
    return ratio * bad + flags["far_premature"].to_numpy()


def bootstrap_se(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> float:
    """Standard error of the mean obtained by resampling engines with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return float(values[idx].mean(axis=1).std(ddof=1))


def mark_stable(summary: pd.DataFrame, margin_steps: int = MARGIN_STEPS) -> pd.DataFrame:
    """Add `stable`: no missed/too-late engine at this threshold nor at the next `margin_steps` higher ones (same k)."""
    out = summary.sort_values(["k", "threshold"]).copy()
    out["stable"] = False
    for _, idx in out.groupby("k").groups.items():
        bad = out.loc[idx, "bad"].to_numpy()
        out.loc[idx, "stable"] = [bool((bad[i:i + margin_steps + 1] == 0).all()) for i in range(len(bad))]
    return out


def choose_provisional(first_alerts: pd.DataFrame, ratio: float = CENTRAL_RATIO, min_notice=MIN_NOTICE,
                       horizon=HORIZON, grace=GRACE, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Pick (threshold, k): stably safe, cost-minimal within one standard error, and as conservative as possible."""
    summary = mark_stable(summarize_engine_policy(first_alerts, min_notice, horizon, grace, ratios=(ratio,)))
    cost_col = f"cost_r{ratio}"
    safe = summary[summary["stable"]]
    safety_met = not safe.empty
    pool = safe if safety_met else summary
    best = pool.loc[pool[cost_col].idxmin()]

    g = first_alerts[(first_alerts["k"] == best["k"]) & (first_alerts["threshold"] == best["threshold"])]
    se = bootstrap_se(_engine_costs(g, ratio, min_notice, horizon, grace), n_boot, seed)
    tied = pool[pool[cost_col] <= best[cost_col] + se].copy()
    tied["_neg_min_lead"] = -tied["min_first_alert_rul"].fillna(-1)
    chosen = tied.sort_values(["k", cost_col, "_neg_min_lead", "threshold"]).iloc[0]
    return {
        "threshold": float(chosen["threshold"]), "k": int(chosen["k"]),
        "cost_per_engine": float(chosen[cost_col]), "min_cost_per_engine": float(best[cost_col]),
        "cost_se": se, "cost_ratio": ratio, "safety_constraint_met": bool(safety_met),
        "n_tied_candidates": int(len(tied)), "chosen_row": chosen.drop("_neg_min_lead").to_dict(),
    }


def first_unsafe_threshold(summary: pd.DataFrame, k: int):
    """Smallest threshold at which persistence k first has a missed/too-late engine (None if never)."""
    bad = summary[(summary["k"] == k) & (summary["bad"] > 0)]
    return float(bad["threshold"].min()) if not bad.empty else None


def robust_choice(variants: dict, ratio: float = CENTRAL_RATIO, min_notice=MIN_NOTICE, horizon=HORIZON,
                  grace=GRACE, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Like choose_provisional, but a setting must be stably safe in EVERY model variant.

    `variants` maps a variant name to its first-alert table. Cost is averaged over variants.
    """
    cost_col = f"cost_r{ratio}"
    summaries = {name: mark_stable(summarize_engine_policy(fa, min_notice, horizon, grace, ratios=(ratio,))).set_index(["k", "threshold"])
                 for name, fa in variants.items()}
    table = pd.DataFrame({
        "stable_all": pd.concat([sm["stable"] for sm in summaries.values()], axis=1).all(axis=1),
        "cost_mean": pd.concat([sm[cost_col] for sm in summaries.values()], axis=1).mean(axis=1),
        "min_lead_worst": pd.concat([sm["min_first_alert_rul"] for sm in summaries.values()], axis=1).min(axis=1),
    }).reset_index()
    safe = table[table["stable_all"]]
    safety_met = not safe.empty
    pool = safe if safety_met else table
    best = pool.loc[pool["cost_mean"].idxmin()]

    first_fa = next(iter(variants.values()))
    g = first_fa[(first_fa["k"] == best["k"]) & (first_fa["threshold"] == best["threshold"])]
    se = bootstrap_se(_engine_costs(g, ratio, min_notice, horizon, grace), n_boot, seed)
    tied = pool[pool["cost_mean"] <= best["cost_mean"] + se].copy()
    tied["_neg_lead"] = -tied["min_lead_worst"].fillna(-1)
    chosen = tied.sort_values(["k", "cost_mean", "_neg_lead", "threshold"]).iloc[0]
    return {
        "threshold": float(chosen["threshold"]), "k": int(chosen["k"]),
        "cost_per_engine": float(chosen["cost_mean"]), "min_cost_per_engine": float(best["cost_mean"]),
        "cost_se": se, "cost_ratio": ratio, "safety_constraint_met": bool(safety_met),
        "n_tied_candidates": int(len(tied)), "worst_case_min_warning": float(chosen["min_lead_worst"]),
        "n_variants": len(variants),
    }


# ----------------------------------------------------------------------------- probabilities
def _model_inputs(raw: pd.DataFrame, matrix: pd.DataFrame, model_name: str):
    if model_name == "lr_raw24":
        return raw, cm.RAW_FEATURES, cm.make_logistic_regression
    if model_name == "lr_eng242":
        return matrix, bf.feature_columns(matrix), cm.make_logistic_regression
    if model_name == "hgb_eng242":
        return matrix, bf.feature_columns(matrix), cm.make_tree_model
    raise ValueError(f"Unsupported model for out-of-fold scoring: {model_name}. Supported: {SKLEARN_MODELS}")


def _prediction_frame(frame: pd.DataFrame, prob: np.ndarray, model_name: str) -> pd.DataFrame:
    out = frame[[MACHINE_COL, CYCLE_COL, "rul", "target"]].reset_index(drop=True).copy()
    out["eligible"] = out[CYCLE_COL] >= cm.SEQUENCE_LENGTH
    out[f"p_{model_name}"] = prob
    return out


def oof_probabilities(raw: pd.DataFrame, matrix: pd.DataFrame, model_name: str, n_splits: int = 5,
                      C: float = None) -> pd.DataFrame:
    """Out-of-fold probabilities for the TRAIN engines (folds are grouped by engine, so no engine is split)."""
    source, cols, factory = _model_inputs(raw, matrix, model_name)
    train = source[source["split"] == "train"].reset_index(drop=True)
    prob = np.zeros(len(train))
    for fit_idx, score_idx in GroupKFold(n_splits=n_splits).split(train, groups=train[MACHINE_COL]):
        model = factory()
        if C is not None:
            if not model_name.startswith("lr"):
                raise ValueError("C can only be varied for the logistic-regression models.")
            model.set_params(classifier__C=C)
        model.fit(train.loc[fit_idx, cols], train.loc[fit_idx, "target"])
        prob[score_idx] = model.predict_proba(train.loc[score_idx, cols])[:, 1]
    return _prediction_frame(train, prob, model_name)


def val_probabilities(raw: pd.DataFrame, matrix: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Validation-engine probabilities from a model fitted on all train engines."""
    source, cols, factory = _model_inputs(raw, matrix, model_name)
    train, val = source[source["split"] == "train"], source[source["split"] == "val"].reset_index(drop=True)
    model = factory()
    model.fit(train[cols], train["target"])
    return _prediction_frame(val, model.predict_proba(val[cols])[:, 1], model_name)


def stream_probabilities(rows: pd.DataFrame, predictor, column: str = "p_gru_artifact") -> pd.DataFrame:
    """Feed rows one at a time to `predictor.predict(row_dict)` (e.g. predict.StreamingPredictor).

    Returns (unit_number, time_cycles, column) for every row where the predictor produced a probability.
    """
    out = []
    for row in rows.sort_values([MACHINE_COL, CYCLE_COL]).to_dict("records"):
        result = predictor.predict(row)
        if result is not None:
            out.append((row[MACHINE_COL], row[CYCLE_COL], float(result["failure_probability"])))
    return pd.DataFrame(out, columns=[MACHINE_COL, CYCLE_COL, column])


# ----------------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame, fmt: dict) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---:"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return lines


def format_report(model_name, choice, decision_summary, val_summary, row_sweep, decision_source,
                  n_decision, n_val, confirm, confirm_default, variant_rows=None) -> str:
    thr, k = choice["threshold"], choice["k"]
    pick = lambda df, kk, tt: df[(df["k"] == kk) & (df["threshold"] == tt)].iloc[0]
    d, v = pick(decision_summary, k, thr), (pick(val_summary, k, thr) if val_summary is not None else None)
    base_d = pick(decision_summary, 1, 0.5)

    show_thr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rs = row_sweep[row_sweep["threshold"].isin(show_thr)][["threshold", "precision", "recall", "f1", "f2", "fp", "fn"]]
    eng = decision_summary[(decision_summary["threshold"].isin([0.3, 0.4, 0.5, 0.7, 0.9]))][
        ["k", "threshold", "missed", "too_late", "premature", "far_premature",
         "median_first_alert_rul", "min_first_alert_rul"]]

    sens = []
    for r in COST_RATIOS:
        cc = f"cost_r{r}"
        pool = mark_stable(decision_summary)
        pool = pool[pool["stable"]] if pool["stable"].any() else pool
        best = pool.loc[pool[cc].idxmin()]
        sens.append({"cost ratio (failure : false alarm)": f"{r}:1", "best safe k": int(best["k"]),
                     "best safe threshold": best["threshold"], "cost / engine": round(best[cc], 3)})
    sens = pd.DataFrame(sens)

    lines = [
        f"# Alert Threshold Analysis — {model_name} (provisional)",
        "",
        "Generated by `python -m src.modeling.threshold_analysis`. **Provisional**: the threshold belongs to this model's "
        "probability scale and must be re-derived for any other model (for example the GRU used by `predict.py`).",
        "",
        "## Recommendation",
        "",
        f"- **Alert rule:** failure probability >= **{thr}** for **{k}** consecutive cycle(s).",
        f"- Decision data: {decision_source} ({n_decision} engines). Cost ratio used: {choice['cost_ratio']}:1 "
        f"(assumption). Safety constraint (no engine missed or warned < {MIN_NOTICE} cycles ahead, and still true for the "
        f"next {MARGIN_STEPS} higher thresholds) met: {choice['safety_constraint_met']}.",
        *( [f"- **Robust choice:** the rule is stably safe in all {choice['n_variants']} model variants tested "
            f"(logistic regression with C in {list(ROBUST_C)}); worst-case shortest warning across variants: "
            f"{choice['worst_case_min_warning']:.0f} cycles."] if "n_variants" in choice else [] ),
        f"- On the decision engines (base fit) this rule gives {int(d['missed'])} missed, {int(d['too_late'])} too late, "
        f"{int(d['far_premature'])} far-premature alarms (alert more than {HORIZON + GRACE} cycles before failure), "
        f"minimum warning {d['min_first_alert_rul']:.0f} cycles; the current default (0.5, k=1) gives "
        f"{int(base_d['too_late'] + base_d['missed'])} missed/late and {int(base_d['far_premature'])} far-premature.",
    ]
    if v is not None:
        vb = pick(val_summary, 1, 0.5)
        lines.append(
            f"- Held-out confirmation on {n_val} validation engines: {int(v['missed'])} missed, {int(v['too_late'])} too late, "
            f"{int(v['far_premature'])} far-premature; minimum warning {v['min_first_alert_rul']:.0f} cycles; "
            f"row-level precision {confirm['precision']:.3f}, recall {confirm['recall']:.3f}, F1 {confirm['f1']:.3f} "
            f"(default 0.5, k=1: {int(vb['missed'] + vb['too_late'])} missed/late, {int(vb['far_premature'])} far-premature; "
            f"precision {confirm_default['precision']:.3f}, recall {confirm_default['recall']:.3f}, F1 {confirm_default['f1']:.3f}).")
    else:
        lines.append("- No hold-out confirmation: the decision was made on the validation engines themselves.")
    lines += [
        "",
        "## Assumptions (to be confirmed with the client / team)",
        "",
        f"- Horizon {HORIZON} cycles (cycles, not hours; provisional).",
        f"- Minimum useful notice {MIN_NOTICE} cycles; alerts up to {HORIZON + GRACE} cycles before failure are acceptable early warnings.",
        "- Cost of an unplanned failure vs one unnecessary maintenance halt is unknown; the brief only says downtime is far "
        "more expensive and false alarms are costly. Ratios of 5:1 to 50:1 are shown.",
        "",
        "## Row-level precision-recall trade-off (validation, same rows as the model comparison; threshold only, k = 1)",
        "",
    ] + _md_table(rs, {"threshold": "{:.2f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
                       "f2": "{:.3f}", "fp": "{:.0f}", "fn": "{:.0f}"}) + [
        "",
        "Raising the threshold buys precision (fewer false alarms) at the price of recall (more rows inside the failure window "
        "flagged late). Row-level misses are mostly rows at the far edge of the window, not missed failures.",
        "",
        f"## Engine-level behaviour ({decision_source}, {n_decision} engines)",
        "",
    ] + _md_table(eng, {"threshold": "{:.2f}", "k": "{:.0f}", "missed": "{:.0f}", "too_late": "{:.0f}",
                        "premature": "{:.0f}", "far_premature": "{:.0f}",
                        "median_first_alert_rul": "{:.1f}", "min_first_alert_rul": "{:.0f}"}) + [
        "",
        "## Robustness to the model fit",
        "",
    ] + ( _md_table(variant_rows, {"chosen_k": "{:.0f}", "chosen_threshold": "{:.2f}",
                                    "k2_first_unsafe_threshold": "{}"}) + [
        "",
        "Each row repeats the selection for a slightly different logistic regression. The preferred persistence is stable, but the "
        "best single threshold and the point where k = 2 stops being safe move with the fit. That is why the final rule is chosen "
        "from settings that are safe in every variant and not from one fit's optimum.",
    ] if variant_rows is not None else ["Not run (robustness check applies to logistic-regression models with out-of-fold predictions)."] ) + [
        "",
        "## Cost sensitivity (stably safe settings only, base fit)",
        "",
    ] + _md_table(sens, {"best safe k": "{:.0f}", "best safe threshold": "{:.2f}", "cost / engine": "{:.3f}"}) + [
        "",
        "## Cost of the two error types",
        "",
        "| Error | What happens | Cost | What the data show |",
        "|---|---|---|---|",
        "| Missed failure (never alerted, or alerted too late to act) | Unplanned stop, no time to schedule maintenance | "
        "Very high (the brief: downtime costs millions) | Rare here; appears only at high thresholds / long persistence rules |",
        "| False alarm (alert far ahead of failure) | Unnecessary maintenance halt, wasted remaining life | "
        "Moderate each, but accumulates across a fleet | Frequent at low thresholds with k = 1 (flickering probabilities) |",
        "",
        "## Limitations",
        "",
        "- Simulated turbofan data; cycles are not hours; the 24-cycle horizon is provisional.",
        "- A handful of engines drive the result (one hard engine decides where the safety cliff sits); treat the threshold as provisional.",
        "- Row-level F1 and engine-level cost point in somewhat different directions; the engine view is closer to operations.",
        "- Thresholds are model-specific. The deployed GRU needs its own analysis (use `--model gru_artifact`).",
        "- The reserved test split was not used.",
        "",
    ]
    return "\n".join(lines)


def run(raw_dir=RAW_DIR, reports_dir=REPORTS_DIR, model_name="lr_eng242", decision_source="oof",
        predictions_file=cm.PREDICTIONS_FILE, predictor=None, ratio=CENTRAL_RATIO, robust=True) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prob_col = f"p_{model_name}"

    if predictor is not None:  # score the deployed GRU artifact on the validation engines
        raw, _ = cm.prepare_data(load_cmapss_file(Path(raw_dir) / TRAIN_FILE))
        val_rows = raw[raw["split"] == "val"]
        probs = stream_probabilities(val_rows, predictor, prob_col)
        val_pred = val_rows[[MACHINE_COL, CYCLE_COL, "rul", "target"]].merge(probs, on=[MACHINE_COL, CYCLE_COL])
        val_pred["eligible"] = True
        decision_source = "val"
    elif model_name in SKLEARN_MODELS:
        raw, matrix = cm.prepare_data(load_cmapss_file(Path(raw_dir) / TRAIN_FILE))
        val_pred = val_probabilities(raw, matrix, model_name)
    else:
        val_pred = pd.read_csv(predictions_file)
        if prob_col not in val_pred.columns:
            raise ValueError(f"{prob_col} not in {predictions_file}. Run compare_models --with-gru first.")
        decision_source = "val"

    if decision_source == "oof":
        decision_pred = oof_probabilities(raw, matrix, model_name)
        decision_label = "train engines, out-of-fold"
    else:
        decision_pred, decision_label = val_pred, "validation engines (no hold-out)"

    fa_decision = engine_first_alerts(decision_pred, prob_col)
    fa_val = engine_first_alerts(val_pred, prob_col)
    variant_rows = None
    if robust and decision_source == "oof" and model_name.startswith("lr"):
        variants = {}
        for C in ROBUST_C:
            pred_c = decision_pred if C == 1.0 else oof_probabilities(raw, matrix, model_name, C=C)
            variants[f"C={C}"] = engine_first_alerts(pred_c, prob_col)
        choice = robust_choice(variants, ratio=ratio)
        rows = []
        for name, fa in variants.items():
            c = choose_provisional(fa, ratio=ratio)
            rows.append({"variant": name, "chosen_k": c["k"], "chosen_threshold": c["threshold"],
                         "k2_first_unsafe_threshold": first_unsafe_threshold(
                             summarize_engine_policy(fa, ratios=(ratio,)), 2)})
        variant_rows = pd.DataFrame(rows)
    else:
        choice = choose_provisional(fa_decision, ratio=ratio)

    decision_summary = summarize_engine_policy(fa_decision)
    val_summary = summarize_engine_policy(fa_val)
    row_sweep = row_level_sweep(val_pred, prob_col)
    confirm = row_level_sweep(val_pred, prob_col, [choice["threshold"]], choice["k"]).iloc[0].to_dict()
    confirm_default = row_level_sweep(val_pred, prob_col, [0.5], 1).iloc[0].to_dict()

    n_dec, n_val = decision_pred[MACHINE_COL].nunique(), val_pred[MACHINE_COL].nunique()
    held_out = None if decision_source != "oof" else val_summary
    (reports_dir / OUT_REPORT.name).write_text(
        format_report(model_name, choice, decision_summary, held_out, row_sweep, decision_label, n_dec, n_val, confirm, confirm_default, variant_rows),
        encoding="utf-8")
    row_sweep.round(5).to_csv(reports_dir / OUT_ROW_SWEEP.name, index=False)
    decision_summary.round(5).to_csv(reports_dir / OUT_ENGINE_DECISION.name, index=False)
    val_summary.round(5).to_csv(reports_dir / OUT_ENGINE_VAL.name, index=False)
    decision_pred.to_csv(reports_dir / OUT_DECISION_PRED.name, index=False)

    config = {
        "status": "provisional", "model": model_name,
        "alert_threshold": choice["threshold"], "persistence_cycles": choice["k"],
        "horizon_cycles": HORIZON, "min_notice_cycles": MIN_NOTICE, "grace_cycles": GRACE,
        "cost_ratio_assumed": choice["cost_ratio"], "decision_data": decision_label,
        "safety_constraint_met": choice["safety_constraint_met"],
        "selection": ("robust: stably safe across logistic-regression variants C=" + ",".join(str(c) for c in ROBUST_C))
                     if variant_rows is not None else "single fit",
        "note": "Threshold is specific to this model's probability scale. Cycles are not hours.",
    }
    if variant_rows is not None:
        config["variant_choices"] = variant_rows.to_dict("records")
    (reports_dir / OUT_JSON.name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"choice": choice, "decision_summary": decision_summary, "val_summary": val_summary,
            "row_sweep": row_sweep, "config": config, "val_pred": val_pred, "decision_pred": decision_pred}


def main() -> None:
    parser = argparse.ArgumentParser(description="Provisional alert-threshold analysis (FD001).")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--model", default="lr_eng242",
                        help="lr_raw24, lr_eng242, hgb_eng242, a p_<name> column in the predictions CSV, or gru_artifact")
    parser.add_argument("--decision-source", choices=["oof", "val"], default="oof")
    parser.add_argument("--no-robust", action="store_true", help="skip the multi-fit robustness check (faster)")
    parser.add_argument("--ratio", type=float, default=CENTRAL_RATIO, help="assumed cost of failure / cost of false alarm")
    args = parser.parse_args()

    predictor = None
    model_name = args.model
    if model_name == "gru_artifact":
        from src.modeling.predict import StreamingPredictor  # needs PyTorch and artifacts/
        predictor = StreamingPredictor()
    result = run(args.raw_dir, args.reports_dir, model_name, args.decision_source, predictor=predictor, ratio=args.ratio, robust=not args.no_robust)
    c = result["choice"]
    print(f"Provisional rule for {model_name}: probability >= {c['threshold']} for {c['k']} consecutive cycle(s)")
    print(f"Safety constraint met: {c['safety_constraint_met']} | cost/engine: {c['cost_per_engine']:.3f} "
          f"(min {c['min_cost_per_engine']:.3f}, SE {c['cost_se']:.3f}, ratio {c['cost_ratio']}:1)")
    print(f"Saved: {Path(args.reports_dir) / OUT_REPORT.name}")


if __name__ == "__main__":
    main()
