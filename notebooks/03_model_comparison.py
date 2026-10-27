# %% [markdown]
# # Model comparison, class imbalance and error analysis — FD001 validation split
# **Axlero Project 2 — Machinery Predictive Maintenance** · Owner: Sreetama Santra · Branch: `feat/sreetama-analysis-docs`
#
# Run first, from the repo root: `python -m src.modeling.compare_models`
# (add `--with-gru` if PyTorch is installed and you want the GRU re-trained).
#
# Setup facts that make the comparison fair:
# - every model is trained on the 70 train engines and scored on the **same 15 validation engines / same rows**;
# - scored rows = validation cycles >= 30, the rows a 30-cycle GRU window exists for;
# - threshold = 0.5 for all models (not tuned); the test split is **not** used;
# - "cycle" = operational cycle, not an hour; the 24-cycle horizon is provisional.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

_cwd = Path.cwd().resolve()
ROOT = next((p for p in [_cwd, *_cwd.parents] if (p / ".git").exists()), _cwd)
sys.path.insert(0, str(ROOT))

from src.modeling import compare_models as cm
from src.features import build_features as bf

REPORTS = ROOT / "reports"
PLOTS = REPORTS / "model_comparison_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

for f in (cm.PREDICTIONS_FILE, cm.METRICS_FILE):
    if not f.is_file():
        raise FileNotFoundError(f"{f} not found. Run from the repo root:\n    python -m src.modeling.compare_models")

pred = pd.read_csv(cm.PREDICTIONS_FILE)
metrics = pd.read_csv(cm.METRICS_FILE)
eligible = pred[pred["eligible"]].copy()
MODELS = [c[2:] for c in pred.columns if c.startswith("p_") and c != "p_dummy_always_0"]
H = bf.HORIZON
print("scored rows:", len(eligible), "| validation engines:", eligible["unit_number"].nunique())
print("models compared:", MODELS)

# %% [markdown]
# ## 1. Class imbalance
# The label is positive for exactly the last 25 cycles (RUL 0..24) of every engine, so the imbalance is structural.
# On the scored validation rows an "always predict no failure" model is already very accurate, which is why accuracy
# cannot be used to rank models.

# %%
feat_file = bf.OUTPUT_FILE
if feat_file.is_file():
    lab = pd.read_csv(feat_file, usecols=["unit_number", "split", "target"])
    prev = lab[lab["split"].isin(["train", "val"])].groupby("split").agg(
        rows=("target", "size"), engines=("unit_number", "nunique"), positives=("target", "sum"))
    prev["positive_rate_%"] = (prev["positives"] / prev["rows"] * 100).round(2)
    print("All rows by split (test split deliberately not shown):"); print(prev)

n_pos = int(eligible["target"].sum()); n = len(eligible)
print(f"\nScored validation rows: {n} | positives: {n_pos} ({n_pos / n:.1%}) | per engine: "
      f"{eligible.groupby('unit_number')['target'].sum().unique()}")
print(f"Accuracy of 'always predict 0': {1 - n_pos / n:.3f}")
m = metrics.set_index("model")
m[["accuracy", "balanced_accuracy", "f1", "pr_auc"]].round(3).sort_values("f1", ascending=False)

# %% [markdown]
# ## 2. Comparison table (same split, same rows)

# %%
table = (metrics.sort_values("f1", ascending=False)
         [["label", "precision", "recall", "f1", "accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn"]]
         .set_index("label").round(4))
table

# %%
real = metrics[metrics["model"] != "dummy_always_0"].set_index("model")
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(real)); w = 0.26
for i, col in enumerate(["precision", "recall", "f1"]):
    ax.bar(x + (i - 1) * w, real[col], w, label=col)
ax.set_xticks(x); ax.set_xticklabels(real.index, rotation=20, ha="right"); ax.set_ylim(0.5, 1.02)
ax.set(title="Validation precision / recall / F1 (threshold 0.5)"); ax.legend()
plt.tight_layout(); plt.savefig(PLOTS / "01_metrics_bars.png", dpi=150); plt.show()

# %% [markdown]
# ## 3. Precision–recall curves (threshold-free view)
# PR curves are the honest comparison under imbalance. Dots mark where each model operates at threshold 0.5.

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
for name in MODELS:
    p = eligible[f"p_{name}"]
    prec, rec, _ = precision_recall_curve(eligible["target"], p)
    ax.plot(rec, prec, label=f"{name} (PR-AUC {m.loc[name, 'pr_auc']:.3f})")
    ax.scatter([m.loc[name, "recall"]], [m.loc[name, "precision"]], s=30)
ax.axhline(n_pos / n, color="grey", ls=":", label="no-skill (prevalence)")
ax.set(xlabel="recall", ylabel="precision", title="Precision–recall, validation", ylim=(0, 1.02)); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PLOTS / "02_pr_curves.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Confusion matrices

# %%
fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.4))
for a, name in zip(np.atleast_1d(axes), MODELS):
    cmx = np.array([[m.loc[name, "tn"], m.loc[name, "fp"]], [m.loc[name, "fn"], m.loc[name, "tp"]]])
    a.imshow(cmx, cmap="Blues")
    for (i, j), v in np.ndenumerate(cmx):
        a.text(j, i, int(v), ha="center", va="center", color="white" if v > cmx.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticks([0, 1]); a.set_yticklabels(["actual 0", "actual 1"])
    a.set_title(name, fontsize=9)
plt.tight_layout(); plt.savefig(PLOTS / "03_confusion_matrices.png", dpi=150); plt.show()

# %% [markdown]
# ## 5. Inspecting false positives and false negatives
# `rul` is the true remaining life. The label window is RUL 0..24, so:
# - a **false negative** is a missed alarm inside the window; its RUL tells how close to the end it happened,
# - a **false positive** is an alarm outside the window; its RUL tells how far outside it happened.

# %%
FN_BINS = [-1, 4, 9, 14, 19, 24]; FN_LAB = ["0-4", "5-9", "10-14", "15-19", "20-24"]
FP_BINS = [24, 29, 39, 59, 10**6]; FP_LAB = ["25-29", "30-39", "40-59", "60+"]

def errors(name):
    p = eligible[f"p_{name}"] >= cm.THRESHOLD
    return eligible[p & (eligible.target == 0)], eligible[~p & (eligible.target == 1)]

rows_fn, rows_fp = {}, {}
for name in MODELS:
    fp, fn = errors(name)
    rows_fn[name] = pd.cut(fn["rul"], FN_BINS, labels=FN_LAB).value_counts().reindex(FN_LAB, fill_value=0)
    rows_fp[name] = pd.cut(fp["rul"], FP_BINS, labels=FP_LAB).value_counts().reindex(FP_LAB, fill_value=0)
fn_tab = pd.DataFrame(rows_fn).T; fp_tab = pd.DataFrame(rows_fp).T
print("FALSE NEGATIVES by true RUL (inside the failure window):"); print(fn_tab.assign(total=fn_tab.sum(axis=1)))
print("\nFALSE POSITIVES by true RUL (outside the failure window):"); print(fp_tab.assign(total=fp_tab.sum(axis=1)))

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
fn_tab.plot.bar(ax=ax[0], stacked=True, colormap="Reds_r", rot=20); ax[0].set(title="False negatives by RUL bucket", ylabel="rows")
fp_tab.plot.bar(ax=ax[1], stacked=True, colormap="Blues_r", rot=20); ax[1].set(title="False positives by RUL bucket", ylabel="rows")
plt.tight_layout(); plt.savefig(PLOTS / "04_errors_by_rul.png", dpi=150); plt.show()

# %%
share = pd.DataFrame({
    "FN at RUL 20-24 (%)": (fn_tab["20-24"] / fn_tab.sum(axis=1) * 100).round(1),
    "FP within 15 cycles of window (RUL 25-39) (%)": ((fp_tab["25-29"] + fp_tab["30-39"]) / fp_tab.sum(axis=1) * 100).round(1),
    "FP at RUL >= 60": fp_tab["60+"],
})
share

# %% [markdown]
# ### 5b. Do the errors concentrate in a few engines? (best non-reference model by F1)

# %%
BEST = real.sort_values("f1", ascending=False).index[0]
fp, fn = errors(BEST)
by_engine = pd.DataFrame({"false_pos": fp.groupby("unit_number").size(), "false_neg": fn.groupby("unit_number").size()}).fillna(0).astype(int)
by_engine = by_engine.reindex(sorted(eligible["unit_number"].unique()), fill_value=0)
print("best model by F1:", BEST)
by_engine.T

# %% [markdown]
# ### 5c. Warning behaviour per engine
# Metrics on rows do not say how many cycles of warning an engine really got. Per engine and model:
# - **lead time** = RUL at the earliest alert inside the window (a perfect alert at the window start is 24),
# - **missed engine** = no alert at all inside the window,
# - **early alerts** = alerts raised while RUL > 24 (these are the false positives).

# %%
rows = []
for name in MODELS:
    s = cm.alert_summary(pred, f"p_{name}")
    rows.append({"model": name, "engines": len(s), "missed_engines": int(s["missed"].sum()),
                 "median_lead_cycles": s["lead_time_cycles"].median(), "min_lead_cycles": s["lead_time_cycles"].min(),
                 "engines_with_early_alerts": int((s["early_alert_rows"] > 0).sum()),
                 "median_first_alert_rul": s["first_alert_rul"].median()})
lead = pd.DataFrame(rows).set_index("model")
lead

# %% [markdown]
# ### 5d. What do the model's probabilities do over an engine's life?
# One panel per validation engine (best model). Red band = true failure window (RUL <= 24).
# x axis runs towards failure on the right.

# %%
fig, axes = plt.subplots(3, 5, figsize=(17, 8), sharey=True)
for a, (unit, g) in zip(axes.ravel(), eligible.groupby("unit_number")):
    a.plot(-g["rul"], g[f"p_{BEST}"], lw=1)
    a.axvspan(-H, 0, color="red", alpha=0.12); a.axhline(cm.THRESHOLD, color="k", ls="--", lw=0.8)
    a.set_title(f"engine {unit}", fontsize=9)
fig.suptitle(f"{BEST}: probability of failure within {H} cycles vs time to failure (-RUL)")
fig.supxlabel("-RUL (cycles before failure)"); fig.supylabel("probability")
plt.tight_layout(); plt.savefig(PLOTS / "05_probability_trajectories.png", dpi=150); plt.show()

# %% [markdown]
# ### 5e. What separates false alarms and misses from correct rows?
# Mean value of the strongest engineered features for each outcome group (best model). If false positives sit
# between true negatives and true positives, they look like "early degradation" rather than random noise.

# %%
if feat_file.is_file():
    top = pd.read_csv(cm.SELECTED_FEATURES_FILE, index_col=0).index[:6].tolist()
    feats = pd.read_csv(feat_file, usecols=["unit_number", "time_cycles"] + top)
    e = eligible.merge(feats, on=["unit_number", "time_cycles"], how="left")
    predicted = e[f"p_{BEST}"] >= cm.THRESHOLD
    e["outcome"] = np.select(
        [~predicted & (e.target == 0), predicted & (e.target == 0), ~predicted & (e.target == 1), predicted & (e.target == 1)],
        ["TN", "FP", "FN", "TP"], default="?")
    sig = e.groupby("outcome")[top].mean().reindex(["TN", "FP", "FN", "TP"])
    print("rows per outcome:", e["outcome"].value_counts().reindex(["TN", "FP", "FN", "TP"]).to_dict())
    display_sig = sig.T.round(3)
    display_sig
else:
    print("features_fd001.csv not found; run python -m src.features.build_features to enable this section.")

# %% [markdown]
# ### 5f. Sensitivity: how much of the false-positive cost is just the label boundary?
# Row RUL=25 is labelled 0 but is almost identical to row RUL=24, which is labelled 1. **Post-hoc, relaxed view only**
# (it does not replace the standard metrics): precision if alarms up to *m* cycles outside the window were not counted
# as false alarms.

# %%
rows = []
for name in MODELS:
    fp, fn = errors(name)
    tp = int(m.loc[name, "tp"])
    row = {"model": name}
    for margin in [0, 5, 10, 15]:
        fp_far = int((fp["rul"] > H + margin).sum())
        row[f"precision_tolerating_{margin}"] = round(tp / (tp + fp_far), 3) if tp + fp_far else np.nan
    rows.append(row)
pd.DataFrame(rows).set_index("model")

# %% [markdown]
# ## 6. Save the error tables

# %%
fn_tab.to_csv(REPORTS / "model_comparison_fn_by_rul.csv"); fp_tab.to_csv(REPORTS / "model_comparison_fp_by_rul.csv")
lead.round(3).to_csv(REPORTS / "model_comparison_alert_summary.csv")
print("saved error tables and plots in reports/ and reports/model_comparison_plots/")

# %% [markdown]
# ## 7. Findings (write after running — your own words, with numbers from the cells above)
# 1. **Same-split check:** does the raw logistic regression reproduce the team's published numbers?
# 2. **Class imbalance:** share of positive rows; what "always predict 0" scores; why accuracy is not used to rank.
# 3. **Ranking:** which models lead on F1 and PR-AUC; how large are the gaps; are they meaningful with only 15 validation engines?
# 4. **What drives the gain:** raw -> engineered features vs. model type (linear vs trees vs GRU).
# 5. **False negatives:** where in the window they fall (RUL buckets); did any engine get no warning at all?
# 6. **False positives:** where they fall relative to the window boundary; how many are far (RUL >= 40) from failure; which engines.
# 7. **Warning behaviour:** typical lead time, minimum lead time, engines with early alerts.
# 8. **Caveats:** one split, simulated data, cycles != hours, threshold 0.5 not tuned, interim tree model, GRU run-to-run variation.
# 9. **Recommendation / next step:** e.g. threshold calibration (Week 4), compare with Shankar's tree model, check the test split only at the end.

# ## 7. Findings
# 1. **Same-split check:** The raw logistic regression reproduces the team's published baseline on the same
#    2,775 validation rows (366 TP, 190 FP, 9 FN, 2210 TN; precision 0.658, recall 0.976, F1 0.786), so every
#    model here is scored on the same engines and rows as the baseline/GRU comparison.
# 2. **Class imbalance:** 12.4% of training rows (1,750 / 14,130) and 13.5% of the scored validation rows
#    (375 / 2,775) are positive; every engine contributes exactly 25 positive rows, so the imbalance is structural.
#    An "always predict 0" model scores 86.5% accuracy, while the raw logistic regression's 92.8% accuracy comes with
#    only 0.786 F1. Accuracy therefore cannot rank models; I use precision, recall, F1 and PR-AUC. All models use
#    class weighting (class_weight=balanced; pos_weight for the GRU).
# 3. **Ranking:** On F1 the logistic regression on engineered features leads (0.892), then trees on the 166
#    filtered features (0.882), trees on all 242 features (0.879) and the raw logistic regression (0.786). The team's
#    GRU (0.889, from the earlier CUDA run) falls inside the same band. The three engineered-feature models are within
#    0.013 F1 and 0.002 PR-AUC of each other; with 15 validation engines and slightly different results on a
#    teammate's machine, I treat them as tied. Trees on 242 features have the lowest recall (0.931, 26 FN).
# 4. **What drives the gain:** Same model (logistic regression), different inputs: moving from raw to engineered
#    features lifts precision from 0.658 to 0.828 and F1 from 0.786 to 0.892, cuts false positives from 190 to 75
#    and raises PR-AUC from 0.950 to 0.976, with recall almost unchanged (0.976 vs 0.965). The gain comes from
#    temporal context (lags and rolling statistics), not from a more complex model.
# 5. **False negatives:** They sit at the far edge of the window. For the logistic regression on engineered
#    features 12 of 13 misses are at RUL 20-24 and none are within the last 15 cycles; for trees 22 of 26 (242
#    features) and 17 of 19 (166 features) are at RUL 20-24. No model misses anything in RUL 0-9.
#    [CHECK missed_engines in the warning-behaviour table: expected 0 for every model.]
# 6. **False positives:** For the engineered models 89-91% of false alarms fall within 15 cycles outside the
#    window (RUL 25-39; e.g. 67 of 75 for logistic regression), 6-9 fall at RUL 40-59 and none at RUL 60 or more.
#    The raw model had 55 at RUL 40-59 and 8 at 60+. Row RUL=25 is labelled 0 but is nearly identical to RUL=24, so many
#    "false" alarms are early warnings next to the label boundary. [CHECK errors-per-engine table: name the
#    engines with the most false positives.]
# 7. **Warning behaviour:** [CHECK warning-behaviour table: median lead time (expected 24 cycles), minimum lead
#    time per model, and how many of the 15 engines raise an early alert before RUL 25.] The probability
#    trajectories (plot 05) rise sharply 25-50 cycles before failure, but on some engines the probability crosses
#    0.5 back and forth near the window edge, so a single-cycle threshold produces flickering alerts.
# 8. **Sensitivity (post-hoc, not a replacement metric):** if alarms up to 15 cycles outside the window were
#    not counted as false alarms, precision for the engineered models would rise from about 0.83 to about 0.98
#    [CHECK against the tolerance table], while raw logistic regression would reach only about 0.85.
#    False alarms and misses both have feature values close to the true-positive group (e.g. sensor_11_rolling_mean_5),
#    i.e. they look like borderline-degraded engines, not random noise [CHECK the feature table].
# 9. **Caveats and next steps:** One split of 15 validation engines, simulated data, cycles are not hours, the
#    threshold 0.5 is not tuned, the tree model is an interim scikit-learn stand-in for Shankar's XGBoost/LightGBM,
#    and the GRU numbers are from the teammate's run, not mine. Results of tree and large logistic models shifted slightly
#    between two machines (library versions are recorded in reports/model_comparison.md). Next: calibrate the
#    threshold and add a persistence rule (e.g. alert after k consecutive cycles) in Week 4, compare against
#    Shankar's model, and only then evaluate once on the reserved test split.git add src/modeling/compare_models.py tests/test_compare_models.py notebooks/03_model_comparison.py reports/
