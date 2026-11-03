# %% [markdown]
# # Alert threshold analysis — precision/recall trade-off, cost of false alarms vs missed failures
# **Axlero Project 2 — Machinery Predictive Maintenance** · Owner: Sreetama Santra · Branch: `feat/sreetama-analysis-docs`
#
# Run first, from the repo root: `python -m src.modeling.threshold_analysis`
#
# What this notebook shows
# 1. How the threshold moves precision and recall (row view).
# 2. What the two error types cost, and why the row view alone is misleading (engine view).
# 3. How the provisional alert rule was chosen, how sensitive it is to the model fit, and what it does on held-out engines.
#
# Ground rules: thresholds are chosen on out-of-fold predictions for the **train engines**; the validation engines are only
# a hold-out check; the test split is **not** used. Thresholds belong to one model's probability scale. Cycles are not hours,
# and the 24-cycle horizon, the minimum notice (10 cycles), the 15-cycle grace and the cost ratios are **assumptions**.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

_cwd = Path.cwd().resolve()
ROOT = next((p for p in [_cwd, *_cwd.parents] if (p / ".git").exists()), _cwd)
sys.path.insert(0, str(ROOT))

from src.modeling import threshold_analysis as ta
from src.modeling import compare_models as cm

REPORTS = ROOT / "reports"
PLOTS = REPORTS / "threshold_plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def show(df, title=None):
    if title:
        print(title)
    print(df.to_string())
    print()


def save_show(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


needed = [ta.OUT_JSON, ta.OUT_DECISION_PRED, ta.OUT_ROW_SWEEP, ta.OUT_ENGINE_DECISION, ta.OUT_ENGINE_VAL, cm.PREDICTIONS_FILE]
missing = [f.name for f in needed if not f.is_file()]
if missing:
    raise FileNotFoundError(f"Missing {missing}. Run from the repo root:\n    python -m src.modeling.threshold_analysis")

config = json.loads(ta.OUT_JSON.read_text(encoding="utf-8"))
MODEL, PROB = config["model"], f"p_{config['model']}"
THR, K = config["alert_threshold"], config["persistence_cycles"]
H, MIN_NOTICE, GRACE = config["horizon_cycles"], config["min_notice_cycles"], config["grace_cycles"]

dec = pd.read_csv(ta.OUT_DECISION_PRED)                     # out-of-fold, train engines
val = pd.read_csv(cm.PREDICTIONS_FILE)                      # validation engines (model comparison output)
row_sweep = pd.read_csv(ta.OUT_ROW_SWEEP)
eng_dec = pd.read_csv(ta.OUT_ENGINE_DECISION)
eng_val = pd.read_csv(ta.OUT_ENGINE_VAL)
print(f"model: {MODEL} | provisional rule: probability >= {THR} for {K} consecutive cycle(s)")
print(f"decision engines: {dec['unit_number'].nunique()} (train, out-of-fold) | validation engines: {val['unit_number'].nunique()}")

# %% [markdown]
# ## 1. Row view: the precision–recall trade-off
# Each threshold turns probabilities into alerts. Lower thresholds catch more of the failure window (higher recall) but
# raise more alarms outside it (lower precision). Validation rows, same rows as the model comparison.

# %%
shown = row_sweep[row_sweep["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])]
show(shown[["threshold", "precision", "recall", "f1", "f2", "tp", "fp", "fn"]].round(3).set_index("threshold"),
     "Row-level metrics by threshold (validation, k = 1):")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for col in ["precision", "recall", "f1", "f2"]:
    ax[0].plot(row_sweep["threshold"], row_sweep[col], label=col)
ax[0].axvline(0.5, color="grey", ls=":", label="current default 0.5"); ax[0].axvline(THR, color="k", ls="--", label=f"provisional {THR}")
ax[0].set(xlabel="threshold", ylabel="score", title="Row-level metrics vs threshold"); ax[0].legend(fontsize=8)

v = val[val["eligible"]]
prec, rec, _ = precision_recall_curve(v["target"], v[PROB])
ax[1].plot(rec, prec)
for thr, style in [(0.5, "s"), (THR, "o")]:
    r = row_sweep[row_sweep["threshold"] == thr].iloc[0]
    ax[1].scatter([r["recall"]], [r["precision"]], marker=style, s=60, label=f"threshold {thr}")
ax[1].set(xlabel="recall", ylabel="precision", title="Precision–recall curve (validation)", ylim=(0.5, 1.02)); ax[1].legend()
save_show(PLOTS / "01_row_tradeoff.png")

# %% [markdown]
# ## 2. What do the two error types cost?
# | Error | Operational meaning | Cost |
# |---|---|---|
# | **Missed failure** (never alerted, or alerted with less than 10 cycles of notice) | Unplanned stop; no time to schedule maintenance | Very high — the brief says unplanned downtime costs millions |
# | **False alarm** (first alert more than 39 cycles before failure) | Unnecessary maintenance halt and wasted remaining life | Moderate each, but it repeats across a fleet; the brief asks for a low false-positive rate |
#
# The real cost ratio is **not known**. Below, cost per engine = ratio × (missed or too-late engines) + (far-premature engines),
# divided by the number of engines, for several ratios. A "row" error is not the same as an "engine" error: a row-level miss at
# RUL 22 is harmless if the same engine is alerted two cycles later.

# %% [markdown]
# ## 3. Engine view: what actually happens to each engine?
# For every engine we look at the **first** alert. `k` is the number of consecutive cycles above the threshold needed to alert.

# %%
cols = ["k", "threshold", "missed", "too_late", "premature", "far_premature", "median_first_alert_rul", "min_first_alert_rul"]
show(eng_dec[eng_dec["threshold"].isin([0.3, 0.5, 0.7, 0.9])][cols].round(1).set_index(["k", "threshold"]),
     f"Engine-level outcomes, {dec['unit_number'].nunique()} train engines (out-of-fold):")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for k, g in eng_dec.groupby("k"):
    ax[0].plot(g["threshold"], g["far_premature"], marker="o", ms=3, label=f"k={k}")
    ax[1].plot(g["threshold"], g["too_late"] + g["missed"], marker="o", ms=3, label=f"k={k}")
    ax[2].plot(g["threshold"], g["min_first_alert_rul"], marker="o", ms=3, label=f"k={k}")
ax[0].set(title=f"False alarms: engines alerted > {H + GRACE} cycles early", xlabel="threshold", ylabel="engines")
ax[1].set(title=f"Missed or warned < {MIN_NOTICE} cycles ahead", xlabel="threshold", ylabel="engines")
ax[2].set(title="Shortest warning on any engine (cycles)", xlabel="threshold", ylabel="cycles"); ax[2].axhline(MIN_NOTICE, color="r", ls=":")
for a in ax:
    a.axvline(THR, color="k", ls="--", lw=0.8); a.legend(fontsize=8)
save_show(PLOTS / "02_engine_view.png")

# %% [markdown]
# **The safety cliff.** Raising the threshold (or the persistence `k`) removes false alarms, but at some point one engine gets
# almost no warning. The hard engine is shown below: its probability rises late, so any rule that waits for strong evidence
# alerts too late.

# %%
dec_el = dec[dec["eligible"]].sort_values(["unit_number", "time_cycles"])
CLIFF_THR, CLIFF_K = 0.5, 2
firsts = {u: ta.first_alert_rul(g[PROB].to_numpy(), g["rul"].to_numpy(), CLIFF_THR, CLIFF_K) for u, g in dec_el.groupby("unit_number")}
hard = min(firsts, key=lambda u: (np.inf if np.isnan(firsts[u]) else firsts[u]))
print(f"hardest engine at threshold {CLIFF_THR}, k={CLIFF_K}: engine {hard}, first alert only {firsts[hard]:.0f} cycles before failure")

fig, ax = plt.subplots(figsize=(8, 3.8))
g = dec_el[dec_el["unit_number"] == hard]
ax.plot(-g["rul"], g[PROB], label="probability (out-of-fold)")
ax.axvspan(-H, 0, color="red", alpha=0.12, label="failure window")
ax.axhline(0.5, color="grey", ls="--", label="0.5"); ax.axhline(THR, color="k", ls=":", label=f"provisional {THR}")
ax.set(xlim=(-80, 0), xlabel="-RUL (cycles before failure)", ylabel="probability", title=f"Engine {hard}: late probability rise")
ax.legend(fontsize=8); save_show(PLOTS / "03_hard_engine.png")

# %% [markdown]
# ## 4. Sensitivity to the (unknown) cost ratio
# `bad` engines (missed or too late) cost `ratio`; far-premature alarms cost 1. Only settings that are safe *and* stay safe
# for the next two higher thresholds are eligible.

# %%
stable = ta.mark_stable(eng_dec)
rows = []
for r in ta.COST_RATIOS:
    pool = stable[stable["stable"]]
    best = pool.loc[pool[f"cost_r{r}"].idxmin()]
    rows.append({"ratio": f"{r}:1", "best_k": int(best["k"]), "best_threshold": best["threshold"], "cost_per_engine": round(best[f"cost_r{r}"], 3)})
show(pd.DataFrame(rows).set_index("ratio"), "Best stably-safe setting per cost ratio:")

fig, ax = plt.subplots(figsize=(8, 4))
for r in [5, 20, 50]:
    g = eng_dec[eng_dec["k"] == K]
    ax.plot(g["threshold"], g[f"cost_r{r}"], marker="o", ms=3, label=f"ratio {r}:1 (k={K})")
ax.axvline(THR, color="k", ls="--", lw=0.8); ax.set(xlabel="threshold", ylabel="expected cost per engine", title="Expected cost vs threshold")
ax.legend(); save_show(PLOTS / "04_cost_sensitivity.png")

# %% [markdown]
# ## 4b. How much does the answer depend on the exact model fit?
# The analysis was repeated for logistic regressions with different regularisation strength (C). The *best* threshold and the
# point where persistence k = 2 stops being safe move a lot between fits, so the final rule is chosen only from settings that
# are safe in **every** variant (`selection` in `provisional_threshold.json`).

# %%
if "variant_choices" in config:
    show(pd.DataFrame(config["variant_choices"]).set_index("variant"),
         "Best rule per model variant (single-fit view) and where k = 2 first becomes unsafe:")
    print("final (robust) rule:", f"probability >= {THR} for {K} consecutive cycle(s)")
else:
    print("robustness check was not run (re-run threshold_analysis without --no-robust)")

# %% [markdown]
# ## 5. First-alert timing: default vs provisional rule
# Distribution over engines of "cycles before failure at the first alert". The failure window starts at 24; alerts further
# left than 39 count as false alarms.

# %%
def firsts_for(df, thr, k):
    e = df[df["eligible"]].sort_values(["unit_number", "time_cycles"])
    return pd.Series({u: ta.first_alert_rul(g[PROB].to_numpy(), g["rul"].to_numpy(), thr, k) for u, g in e.groupby("unit_number")})

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, (name, df) in zip(ax, [("train engines (out-of-fold)", dec), ("validation engines (hold-out)", val)]):
    data = [firsts_for(df, 0.5, 1).dropna(), firsts_for(df, THR, K).dropna()]
    a.boxplot(data, tick_labels=["default\n0.5, k=1", f"provisional\n{THR}, k={K}"])
    a.axhline(H, color="red", ls=":"); a.axhline(H + GRACE, color="red", ls="--"); a.axhline(MIN_NOTICE, color="grey", ls=":")
    a.set(title=name, ylabel="cycles before failure at first alert")
save_show(PLOTS / "05_first_alert_timing.png")

# %% [markdown]
# ## 6. Held-out confirmation (validation engines)

# %%
def pick(df, k, thr):
    return df[(df["k"] == k) & (df["threshold"] == thr)].iloc[0]

compare = pd.DataFrame({
    "default (0.5, k=1)": pick(eng_val, 1, 0.5), f"provisional ({THR}, k={K})": pick(eng_val, K, THR),
}).loc[["missed", "too_late", "premature", "far_premature", "median_first_alert_rul", "min_first_alert_rul"]]
show(compare, f"Engine-level outcomes on {val['unit_number'].nunique()} validation engines:")
rl = pd.concat([ta.row_level_sweep(val, PROB, [0.5], 1), ta.row_level_sweep(val, PROB, [THR], K)])
show(rl[["threshold", "k", "precision", "recall", "f1", "f2", "fp", "fn"]].round(3).set_index(["threshold", "k"]),
     "Row-level metrics on validation, with persistence applied:")

# %% [markdown]
# ## 7. Findings (write after running — your own words, with numbers from the cells above)
# 1. **Row-level trade-off:** how precision and recall move between thresholds 0.3, 0.5 and 0.9; where F1 peaks; why F2 differs.
# 2. **Why the row view is not enough:** row-level misses sit at the far edge of the window; engine-level, how many engines were really missed?
# 3. **Cost of the two errors:** your explanation of missed failure vs false alarm; why the ratio is unknown and which ratios you tested.
# 4. **The provisional rule:** threshold and persistence chosen, how (safety first, cost, simplest rule), and the numbers on the 70 decision engines.
# 5. **The cliff:** which setting breaks safety and why (the hard engine); how far the chosen setting is from it.
# 6. **Hold-out check:** how the rule behaves on the 15 validation engines compared with the 0.5 default; is the difference meaningful?
# 7. **Sensitivity:** does the best setting change with the cost ratio? And with small changes of the model fit (section 4b)?
#    Which part of the answer is stable (e.g. no engine missed, minimum warning) and which is not (the exact threshold)?
# 8. **Caveats:** one model (LR on engineered features), thresholds are not transferable to the GRU, small engine counts, assumptions
#    (24 cycles, 10 minimum notice, 15 grace), simulated data, cycles are not hours, test split untouched.
# 9. **Next steps:** re-run for the model that ships (`--model gru_artifact`), agree the cost ratio with the client, add a persistence rule to `predict.py`, evaluate once on the test split at the end.

# ## 7. Findings
# 1. **Row-level trade-off (validation):** Raising the threshold from 0.3 to 0.9 raises precision from 0.791 to 0.898 and
#    lowers recall from 0.981 to 0.893. F1 peaks around 0.8 (0.906); F2, which weights recall more, is highest near 0.3
#    (0.936). At the current default 0.5: precision 0.828, recall 0.965, F1 0.892 (75 FP, 13 FN).
# 2. **Why the row view is not enough:** Row-level misses sit at the edge of the failure window and are not missed failures.
#    At the engine level no engine is missed or warned fewer than 10 cycles ahead for thresholds up to 0.5 (single cycle),
#    and the shortest warning on any of the 70 out-of-fold training engines is 17 cycles.
# 3. **Cost of the two errors:** A missed failure (never alerted, or alerted with under 10 cycles of notice) means an unplanned
#    stop; the brief says unplanned downtime costs millions. A false alarm (first alert more than 39 cycles before failure)
#    means an unnecessary maintenance halt and wasted remaining life; the brief asks for a low false-positive rate. The real cost
#    ratio is unknown, so I tested 5:1, 10:1, 20:1 and 50:1; the best safe setting did not change with the ratio.
# 4. **Provisional rule:** Safety first (no engine missed or warned under 10 cycles ahead, and still safe for the next two higher
#    thresholds), then lowest expected cost, then the simplest rule. In my run this gave probability >= 0.5 for 2 consecutive
#    cycles: 5 far-premature alarms on 70 engines versus 10 for the default (0.5, single cycle), shortest warning 14 cycles.
# 5. **The safety cliff:** Thresholds above about 0.7 (k=1) leave one engine with only 8 cycles of notice, and k=2 breaks at
#    0.7. The cause is engine 28 in the training folds, whose probability rises late (first alert only 14 cycles before failure
#    at 0.5 with k=2). The chosen setting sits two thresholds below the break.
# 6. **Hold-out check (15 validation engines):** No engine missed or late either way. Far-premature alarms 2 (default) vs 1
#    (provisional rule); median first alert 30 vs 28 cycles before failure; row-level precision 0.828 vs 0.868, recall 0.965 vs
#    0.947, F1 0.892 vs 0.906. The difference rests on one or two engines and is not strong evidence.
# 7. **Sensitivity:** The best setting does not change with the cost ratio (k=3, 0.5, cost 0.043 per engine, base fit).
#    It does change with the model fit: across five regularisation strengths the best rule moved between k=2 and k=3 and
#    between thresholds 0.3 and 0.55, and k=2 first became unsafe between 0.65 and 0.75. In a second environment
#    (different library versions) the cliff for k=2 was at 0.35-0.65 and the robust rule came out as 0.35 with one cycle.
#    What is stable: no missed engines and a shortest warning of at least 17 cycles for single-cycle thresholds of 0.3-0.5.
#    What is not stable: whether 2-cycle persistence stays safe.
# 8. **Caveats:** One model (logistic regression on engineered features); the threshold is specific to its probability scale
#    and does not transfer to the GRU in predict.py. Small number of engines (70 + 15); one hard engine decides the cliff.
#    Assumptions: 24-cycle horizon, 10 cycles minimum notice, 15 cycles grace, cost ratio unknown. Simulated data, cycles are
#    not hours. The test split was not used.
# 9. **Recommendation and next steps:** Keep 0.5 as the provisional alert threshold (it is safe in both environments).
#    Treat 2-cycle persistence as an optional refinement and re-check it on the final model in the team's reference
#    environment. Agree the cost ratio with the client, re-run the analysis for the GRU (--model gru_artifact), and use
#    the test split only once at the end.