# %% [markdown]
# # Feature correlation & quality — FD001 processed feature matrix
# **Axlero Project 2 — Machinery Predictive Maintenance** · Owner: Sreetama Santra · Branch: `feat/sreetama-analysis-docs`
#
# Input: `data/processed/features_fd001.csv` (built by `python -m src.features.build_features`).
#
# Rules used throughout:
# - Every statistic that could influence feature selection is computed on the **train split only** (70 engines).
#   The validation split is used only to check whether findings generalise to unseen engines.
# - `rul`, `target`, `split`, `unit_number`, `time_cycles` are analysis fields, never features.
# - "Cycle" = operational cycle, not an hour. The 24-cycle horizon is provisional.
#
# Outputs: `reports/feature_quality_summary.csv`, `reports/selected_features_proposal.csv`, `reports/feature_plots/`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run from anywhere inside the repo: find the repo root (folder containing .git)
_cwd = Path.cwd().resolve()
ROOT = next((p for p in [_cwd, *_cwd.parents] if (p / ".git").exists()), _cwd)
sys.path.insert(0, str(ROOT))

from src.features import build_features as bf

FEATURES_CSV = ROOT / "data" / "processed" / "features_fd001.csv"
MANIFEST_CSV = ROOT / "reports" / "feature_manifest.csv"
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "feature_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

if not FEATURES_CSV.is_file():
    raise FileNotFoundError(
        f"{FEATURES_CSV} not found. Build it first from the repo root:\n"
        "    python -m src.features.build_features"
    )

df = pd.read_csv(FEATURES_CSV)
manifest = pd.read_csv(MANIFEST_CSV).fillna("")
FEATURES = bf.feature_columns(df)
assert FEATURES == list(manifest["feature"]), "manifest and matrix disagree"

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
print("matrix:", df.shape, "| model features:", len(FEATURES))
print(df.groupby("split").agg(rows=("target", "size"), engines=("unit_number", "nunique"),
                              positive_rate=("target", "mean")).round(3))

# %% [markdown]
# ## 1. What is in the matrix?

# %%
fam = manifest["family"].value_counts().rename("n_features").to_frame()
fam["example"] = manifest.groupby("family")["feature"].first()
fam

# %% [markdown]
# ## 2. Leakage audit
# 1. Label/analysis columns are not in the feature list.
# 2. Empirical causality check: rebuild the features for a few engines from a **truncated** history (cut at cycle 100)
#    and confirm every feature at cycles ≤ 100 is identical. If any feature used future rows, these would differ.

# %%
forbidden = {"unit_number", "time_cycles", "rul", "target", "split"}
print("Label/id columns inside feature list:", sorted(forbidden & set(FEATURES)) or "none")

raw_train = bf.load_cmapss_file(bf.RAW_DIR / bf.TRAIN_FILE)
CUT = 100
checked, mismatches = 0, []
full = bf.build_feature_matrix(raw_train)
for engine in [1, 2, 3, 50, 99]:
    full_e = full[full.unit_number == engine]
    if full_e.time_cycles.max() <= CUT:
        continue
    cut_raw = raw_train[~((raw_train.unit_number == engine) & (raw_train.time_cycles > CUT))]
    cut_e = bf.build_feature_matrix(cut_raw)
    cut_e = cut_e[cut_e.unit_number == engine]
    a = full_e[full_e.time_cycles <= CUT][FEATURES].reset_index(drop=True)
    b = cut_e[cut_e.time_cycles <= CUT][FEATURES].reset_index(drop=True)
    checked += 1
    if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
        mismatches.append(engine)
print(f"Engines checked: {checked} | engines with differing features: {mismatches or 'none'}")

# %% [markdown]
# ## 3. Missingness (expected warm-up NaNs only)
# Lag-k features are NaN for the first k cycles of every engine. Rolling features are NaN on cycle 1 (nothing earlier
# exists) and rolling std is also NaN on cycle 2 (std of one value). The pipeline deliberately does **not** fill these;
# a model pipeline should impute with statistics fitted on the train split only (tree models can handle NaN natively).

# %%
miss = df[FEATURES].isna().mean().mul(100).rename("missing_%").to_frame().join(manifest.set_index("feature")[["family", "parameter"]])
print(miss.groupby("family")["missing_%"].agg(["min", "max"]).round(2))
unexpected = miss[(miss["family"].isin(["raw_sensor", "raw_setting"])) & (miss["missing_%"] > 0)]
print("Raw channels with missing values:", list(unexpected.index) or "none")

lag10 = df.groupby("unit_number")["sensor_11_lag_10"].apply(lambda s: s.isna().sum())
print("sensor_11_lag_10 NaN rows per engine (expect 10):", lag10.unique())

# %% [markdown]
# ## 4. Per-feature quality metrics (train split only)
# - `spearman_rul`: rank correlation with remaining useful life (negative = rises as failure nears).
# - `pearson_target`: correlation with the binary failure-within-24-cycles label.
# - `std`, `skew`, `missing_%`: basic health checks.
# Rows with NaN for a given feature are dropped pairwise for that feature.

# %%
def corr_with(frame, col_y, method, cols=None):
    cols = FEATURES if cols is None else cols
    return frame[cols].apply(lambda s: s.corr(frame[col_y], method=method))

q = pd.DataFrame({
    "spearman_rul": corr_with(train, "rul", "spearman"),
    "pearson_target": corr_with(train, "target", "pearson"),
    "std": train[FEATURES].std(),
    "skew": train[FEATURES].skew(),
    "missing_%": train[FEATURES].isna().mean() * 100,
})
q["abs_spearman_rul"] = q["spearman_rul"].abs()
q = manifest.set_index("feature").join(q)
q.sort_values("abs_spearman_rul", ascending=False).head(15).round(3)

# %%
by_family = (q.groupby("family")["abs_spearman_rul"].agg(["count", "mean", "max"])
             .sort_values("max", ascending=False).round(3))
by_family

# %% [markdown]
# **Do the engineered features add signal beyond the raw sensors?** For each sensor, compare the strongest
# |Spearman with RUL| among its raw value, lags and rolling statistics.

# %%
rows = []
for sensor in sorted(q[q.family == "raw_sensor"].index):
    sub = q[q.source == sensor]
    raw_v = sub.loc[sensor, "abs_spearman_rul"]
    best = sub.drop(index=sensor)["abs_spearman_rul"].sort_values(ascending=False)
    rows.append({"sensor": sensor, "raw": raw_v, "best_engineered": best.iloc[0],
                 "best_feature": best.index[0], "gain": best.iloc[0] - raw_v})
gain = pd.DataFrame(rows).set_index("sensor").sort_values("gain", ascending=False)
print(gain.round(3).to_string())
print("\nsensors where an engineered feature beats the raw value:", int((gain.gain > 0).sum()), "of", len(gain))

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
order = by_family.sort_values("mean", ascending=False).index
data = [q.loc[q.family == f, "abs_spearman_rul"].dropna() for f in order]
ax[0].boxplot(data, showfliers=False)
ax[0].set_xticks(range(1, len(order) + 1)); ax[0].set_xticklabels(order, rotation=45, ha="right")
ax[0].set(title="|Spearman with RUL| by feature family (train)", ylabel="|rho|")
w = q[q.parameter.str.startswith("window=")].copy()
w["window"] = w.parameter.str.replace("window=", "").astype(int)
w.groupby(["window", "family"])["abs_spearman_rul"].mean().unstack().plot(marker="o", ax=ax[1])
ax[1].set(title="Mean |rho| vs rolling window length", xlabel="window (cycles)", ylabel="mean |rho|")
plt.tight_layout(); plt.savefig(PLOTS / "01_family_signal.png", dpi=150); plt.show()

# %% [markdown]
# ## 5. Redundancy (train split only)
# Many engineered features are near-copies of each other (e.g. a 5-cycle rolling mean and lag-1). Pairs with
# |Spearman| > 0.95 are treated as redundant. Within each redundant group the feature with the strongest
# |Spearman with RUL| is kept. This is a **proposal** to be confirmed on validation results, not a final selection.

# %%
THRESH = 0.95
ranked = train[FEATURES].dropna().rank()  # rows with any NaN removed -> consistent sample for all pairs
corr = ranked.corr()
print("rows used for the correlation matrix:", len(ranked))
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pairs = upper[upper.abs() > THRESH]
print(f"feature pairs with |rho| > {THRESH}: {len(pairs):,} of {len(upper):,}")

kept = []
for feat in q.sort_values("abs_spearman_rul", ascending=False).index:
    if all(abs(corr.loc[feat, k]) <= THRESH for k in kept):
        kept.append(feat)
proposal = q.loc[kept].sort_values("abs_spearman_rul", ascending=False)
print(f"features kept after redundancy filter: {len(kept)} of {len(FEATURES)}")
proposal[["family", "source", "parameter", "spearman_rul", "pearson_target"]].head(20).round(3)

# %%
print(proposal.groupby("family").size().rename("kept").to_frame().join(q.groupby("family").size().rename("total")))
fig, ax = plt.subplots(figsize=(10, 8))
top = proposal.index[:30]
im = ax.imshow(corr.loc[top, top], cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=6)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=6)
plt.colorbar(im, ax=ax, shrink=.7); ax.set_title("Spearman correlation among the 30 strongest kept features (train)")
plt.tight_layout(); plt.savefig(PLOTS / "02_kept_feature_correlation.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Does the signal generalise to unseen engines?
# Rank-correlate each feature's |Spearman with RUL| on **train** with the same quantity on the **validation** engines.
# A value close to 1 means the ranking learned from train engines holds on engines it never saw.

# %%
# Features that never change on the validation engines have no defined correlation there (e.g. sensor_6 only takes
# two distinct values). They are reported and left out of the agreement calculation instead of being silently dropped.
const_in_val = [c for c in FEATURES if val[c].nunique() <= 1]
print("constant on validation engines (correlation undefined):", const_in_val or "none")
val_rho = corr_with(val, "rul", "spearman", [c for c in FEATURES if c not in const_in_val]).abs().reindex(FEATURES)
cmp = pd.DataFrame({"train": q["abs_spearman_rul"], "val": val_rho})
agreement = cmp.corr(method="spearman").loc["train", "val"]
print(f"rank agreement of |rho| between train and validation engines: {agreement:.3f}")
top_k = proposal.index[:20]
print("max |train - val| gap among the 20 strongest kept features:",
      round((cmp.loc[top_k, "train"] - cmp.loc[top_k, "val"]).abs().max(), 3))

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(cmp["train"], cmp["val"], s=10, alpha=.6); ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlabel="|rho| train engines", ylabel="|rho| validation engines", title="Signal strength: train vs validation")
plt.tight_layout(); plt.savefig(PLOTS / "03_train_vs_val.png", dpi=150); plt.show()

# %% [markdown]
# ## 7. Save outputs

# %%
summary = q.drop(columns="abs_spearman_rul").join(cmp["val"].rename("abs_spearman_rul_val"))
summary["kept_by_redundancy_filter"] = summary.index.isin(kept)
summary.round(5).to_csv(REPORTS / "feature_quality_summary.csv")
proposal[["family", "source", "parameter", "spearman_rul", "pearson_target"]].round(5).to_csv(
    REPORTS / "selected_features_proposal.csv")
print("saved:", "reports/feature_quality_summary.csv, reports/selected_features_proposal.csv, reports/feature_plots/")

# %% [markdown]
# ## 8. Findings (fill in after running — write in your own words, with numbers from the cells above)
# 1. **Leakage audit:** result of the truncation check and the label-column check.
# 2. **Missingness:** only warm-up NaNs? Which families, how many rows per engine?
# 3. **Strongest features:** top 5 by |Spearman with RUL|; which families dominate.
# 4. **Value of engineering:** how many sensors improve over their raw value; which window lengths help most.
# 5. **Redundancy:** features kept out of the total; what that implies for model input size.
# 6. **Generalisation:** train-vs-validation rank agreement.
# 7. **Caveats:** simulated data; cycles ≠ hours; selection here is a proposal and must be validated by model results;
#    FD001 has no raw vibration waveform, so no FFT features.

# ## 8. Findings
# 1. **Leakage audit:** No label or id columns (unit_number, time_cycles, rul, target, split) are among the 242 model
#    features. I also rebuilt the features for engines 1, 2, 3, 50 and 99 from histories cut at cycle 100; every
#    feature at cycles <= 100 was identical, so no feature uses future rows. The unit tests check the same property.
# 2. **Missingness:** The raw sensors and settings have no missing values. The only NaNs are warm-up rows: a lag-k
#    feature is NaN for the first k cycles of each engine (e.g. sensor_11_lag_10 has exactly 10 NaN rows per engine),
#    rolling features are NaN on cycle 1 (0.48% of rows) and rolling std also on cycle 2 (0.97%). The largest share
#    is 4.85% (lag 10). These are left unfilled on purpose; imputation must be fitted on the train split only.
# 3. **Strongest features (train engines):** The top features are all rolling means: sensor_11_rolling_mean_5
#    (Spearman with RUL -0.757), sensor_17_rolling_mean_10 (-0.757), sensor_3_rolling_mean_10 (-0.755),
#    sensor_21_rolling_mean_5 (+0.750) and sensor_15_rolling_mean_5 (-0.750). By family, mean |rho| is highest for
#    rolling mean (0.631), then rolling min (0.592), rolling max (0.589), raw sensors (0.568) and lags (0.546).
#    Rolling std is weak (0.108), and the two operating settings have almost no correlation with RUL (about 0.001).
# 4. **Value of engineering:** For all 15 sensors, at least one engineered feature correlates more strongly with RUL
#    than the raw value. The largest gains are sensor_6 (0.134 -> 0.369), sensor_14 (0.214 -> 0.379) and
#    sensor_3 (0.613 -> 0.755). Mean |rho| of rolling mean/min/max is nearly flat across windows 5, 10 and 20, with
#    window 10 slightly best for the mean. Rolling std gets stronger with longer windows (0.04 at 5, 0.19 at 20).
#    Part of the gain is just noise reduction from smoothing, so this is a hint, not proof of better model
#    performance. Rolling std may still help a non-linear model; that needs to be tested on validation results.
# 5. **Redundancy:** 480 of 58,564 feature pairs have |Spearman| > 0.95. Keeping the feature with the stronger RUL
#    correlation in each correlated group leaves 166 of 242 features (rolling mean 13/45, rolling min 23/45,
#    rolling max 25/45, lags 43/45, rolling std 45/45, all raw features kept). The list is in
#    reports/selected_features_proposal.csv and is a proposal only, to be confirmed on validation results.
# 6. **Generalisation:** The ranking of features by |rho| on the train engines agrees with the ranking on the
#    validation engines (rank agreement 0.989; largest gap among the top 20 kept features 0.043). Three sensor_6
#    rolling-max features are constant on the validation engines, so their correlation is undefined there.
# 7. **Caveats:** FD001 is simulated turbofan data, not real factory telemetry. A cycle is not an hour, and the
#    24-cycle horizon is provisional. FD001 has no raw vibration waveform, so no FFT features were built.
#    sensor_6 takes only two distinct values and is a weak feature. Correlation with RUL is a screening tool only;
#    the final feature set must be judged by model results on the validation split.git add src/features/build_features.py tests/test_build_features.py notebooks/01b_eda_class_balance_trends_noise.py notebooks/02_feature_quality.py reports/ requirements.txt
