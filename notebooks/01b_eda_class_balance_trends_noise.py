# Week 1 supplementary EDA - NASA C-MAPSS FD001 (class balance, trends vs RUL, noise, trend scores)
# Open in VS Code and click "Run Cell" above each "# %%" block (needs Python + Jupyter extensions).

# %% [markdown]
# # Week 1 EDA — NASA C-MAPSS FD001 (training set)
# **Axlero Project 2 — Machinery Predictive Maintenance** · Owner: Sreetama Santra · Branch: `feat/sreetama-analysis-docs`
#
# Scope (from the daily plan): sensor distributions, time trends, missingness, class imbalance, sensor variance.
# Extra: sensor noise level (informs smoothing choice) and feature-quality checks (informs Week 2 feature work).
#
# **Conventions (from Veda's dataset note):** the label `failure_within_24_cycles = 1 if RUL <= 24` and `RUL` are used here **only for analysis**.
# They are never model inputs. Cycles are *operational cycles, not hours*.
#
# **Outputs:** figures → `reports/figures/`, tables → `reports/`.

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 140)

# --- Paths: resolve from the repo root (works whether the notebook runs from root or notebooks/) ---
CWD = Path.cwd().resolve()
ROOT = next((p for p in [CWD, *CWD.parents] if (p / ".git").exists()), CWD)

# Raw data is read-only: never write into data/raw. Team layout: data/raw/CMAPSSData/train_FD001.txt (do not commit it).
CANDIDATES = [os.environ.get("CMAPSS_PATH", ""), ROOT / "data/raw/CMAPSSData/train_FD001.txt", ROOT / "data/raw/train_FD001.txt", CWD / "train_FD001.txt"]
DATA_PATH = next((Path(p) for p in CANDIDATES if p and Path(p).exists()), None)
assert DATA_PATH is not None, "train_FD001.txt not found. Put it in data/raw/ or set CMAPSS_PATH."

FIG_DIR, REP_DIR = ROOT / "reports/figures", ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 24   # provisional, pending team approval
OP_COLS = [f"op{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP_COLS + SENSORS

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLS)
print("Loaded:", DATA_PATH, df.shape)
df.head()

# %% [markdown]
# ## 1. Integrity checks and missingness

# %%
checks = {
    "rows (expected 20,631)": len(df),
    "columns (expected 26)": df.shape[1],
    "unique engines (expected 100)": df["unit"].nunique(),
    "missing cells": int(df.isna().sum().sum()),
    "duplicate rows": int(df.duplicated().sum()),
    "duplicate (unit, cycle) keys": int(df.duplicated(["unit", "cycle"]).sum()),
}
# cycles must run 1..max with no gaps for every engine, and rows must be sorted
gaps = df.groupby("unit")["cycle"].agg(lambda c: (c.max() != len(c)) or (c.min() != 1))
checks["engines with cycle gaps"] = int(gaps.sum())
checks["sorted by (unit, cycle)"] = bool(df.equals(df.sort_values(["unit", "cycle"]).reset_index(drop=True)))
pd.Series(checks, name="value").to_frame()

# %%
print("Non-numeric columns:", list(df.select_dtypes(exclude="number").columns))
miss = df.isna().mean().mul(100).round(3)
print("Columns with any missing values:", miss[miss > 0].to_dict() or "none")

# %% [markdown]
# ## 2. Engine lifetimes and class imbalance
# The training engines are run-to-failure, so `final_cycle = max(cycle)` per engine is legitimate **for labelling the training set**.

# %%
df["final_cycle"] = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = df["final_cycle"] - df["cycle"]
df["label"] = (df["RUL"] <= HORIZON).astype(int)

life = df.groupby("unit")["cycle"].max()
print(life.describe().round(1).to_string())

counts = df["label"].value_counts().sort_index()
share = (counts / len(df) * 100).round(1)
print("\nClass balance (row level):")
print(pd.DataFrame({"rows": counts, "share_%": share}))

pos_per_engine = df.groupby("unit")["label"].sum()
print("\nPositive rows per engine: min", pos_per_engine.min(), "max", pos_per_engine.max())

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
life.plot.hist(bins=20, ax=ax[0], color="#4c72b0"); ax[0].set(title="Engine lifetime (cycles)", xlabel="final cycle")
ax[1].bar(["0: >24 left", "1: <=24 left"], counts.values, color=["#4c72b0", "#c44e52"])
for i, (v, s) in enumerate(zip(counts.values, share.values)): ax[1].text(i, v, f"{v:,}\n{s}%", ha="center", va="bottom")
ax[1].set(title="Class balance (rows)"); ax[1].set_ylim(0, counts.max() * 1.2)
life.sort_values().reset_index(drop=True).plot(ax=ax[2], color="#55a868"); ax[2].set(title="Sorted lifetimes", xlabel="engine rank", ylabel="cycles")
plt.tight_layout(); plt.savefig(FIG_DIR / "01_lifetimes_class_balance.png", dpi=150); plt.show()

# %% [markdown]
# **Reading this:** the imbalance is *structural*. Every engine contributes (at most) the same final 25 rows as positives, regardless of how long it lived.
# Two consequences to carry into modelling: (1) a row-level random split would leak, because neighbouring cycles of one engine are nearly identical, so splits must be by engine;
# (2) accuracy is meaningless here (predicting all-0 gives ~88%), so use precision/recall/F1 and per-engine warning lead time.

# %% [markdown]
# ## 3. Sensor variance — which channels carry information?

# %%
desc = df[OP_COLS + SENSORS].agg(["mean", "std", "min", "max", "nunique"]).T
desc["cv"] = desc["std"] / desc["mean"].abs().replace(0, np.nan)
desc["range"] = desc["max"] - desc["min"]
# flag: constant, or effectively constant relative to its own scale
desc["constant"] = (desc["nunique"] <= 1) | (desc["std"] < 1e-8)
# NOTE: CV is scale-dependent (a ~640 K temperature with std 0.5 has tiny CV but is real signal),
# so "near-constant" is judged by number of distinct values instead.
desc["near_constant"] = (~desc["constant"]) & (desc["nunique"] <= 3)
desc.round(5).to_csv(REP_DIR / "eda_sensor_variance.csv")
desc.round(5)

# %%
print("Constant channels      :", list(desc.index[desc["constant"]]))
print("Near-constant channels :", list(desc.index[desc["near_constant"]]))
sens = desc.loc[SENSORS].sort_values("cv")
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#c44e52" if (c or n) else "#4c72b0" for c, n in zip(sens["constant"], sens["near_constant"])]
ax.bar(sens.index, sens["cv"].fillna(0), color=colors); ax.set_yscale("symlog", linthresh=1e-4)
ax.set(title="Coefficient of variation per sensor (red = constant / near-constant by distinct-value count)", ylabel="std / |mean|")
plt.tight_layout(); plt.savefig(FIG_DIR / "02_sensor_variance.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Distributions (informative channels only)

# %%
informative = [s for s in SENSORS if not (desc.loc[s, "constant"] or desc.loc[s, "near_constant"])]
print(len(informative), "informative sensors:", informative)
n = len(informative); cols = 4; rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 2.8 * rows))
for a, s in zip(axes.ravel(), informative):
    sns.histplot(df.loc[df.label == 0, s], ax=a, color="#4c72b0", stat="density", bins=40, alpha=.5, label="far (>24)")
    sns.histplot(df.loc[df.label == 1, s], ax=a, color="#c44e52", stat="density", bins=40, alpha=.5, label="near (<=24)")
    a.set(title=s, xlabel="", ylabel="")
for a in axes.ravel()[n:]: a.axis("off")
axes.ravel()[0].legend(fontsize=8)
plt.suptitle("Sensor distributions by class (for EDA only)", y=1.0); plt.tight_layout()
plt.savefig(FIG_DIR / "03_distributions_by_class.png", dpi=150); plt.show()

# skewness / outliers
skew = df[informative].skew().round(2); print("Skewness:", skew.to_dict())
z = (df[informative] - df[informative].mean()) / df[informative].std()
print("Rows with any |z|>4:", int((z.abs() > 4).any(axis=1).sum()))

# %% [markdown]
# ## 5. Time trends — what does degradation look like?
# Left: raw trajectories of a few engines against cycle. Right: all engines aligned on **RUL** (time-to-failure); mean ± 1 std band.
# Alignment on RUL is for *understanding* only, since at inference time the final cycle is unknown.

# %%
rng = np.random.default_rng(0)
sample_units = sorted(rng.choice(df["unit"].unique(), 5, replace=False))
show = informative[:8]
fig, axes = plt.subplots(len(show), 2, figsize=(14, 2.4 * len(show)), sharex="col")
for i, s in enumerate(show):
    for u in sample_units:
        g = df[df.unit == u]; axes[i, 0].plot(g["cycle"], g[s], lw=.8, label=f"engine {u}")
    g = df[df.RUL <= 200].groupby("RUL")[s].agg(["mean", "std"])
    axes[i, 1].plot(g.index, g["mean"], color="#4c72b0"); axes[i, 1].fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=.25)
    axes[i, 1].axvline(HORIZON, color="#c44e52", ls="--", lw=1); axes[i, 1].invert_xaxis()
    axes[i, 0].set_ylabel(s)
axes[0, 0].legend(fontsize=7, ncol=3); axes[0, 0].set_title("Raw vs cycle (5 random engines)"); axes[0, 1].set_title("Mean±std vs RUL (all engines; red = 24-cycle horizon)")
axes[-1, 0].set_xlabel("cycle"); axes[-1, 1].set_xlabel("RUL (cycles to failure)")
plt.tight_layout(); plt.savefig(FIG_DIR / "04_time_trends.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Noise level — does smoothing help, and how much?

# %%
W = 5
def noise_ratio(s):
    # high-frequency residual vs. a causal trailing mean, per engine; ratio to the sensor's within-engine std
    res, tot = [], []
    for _, g in df.groupby("unit"):
        x = g[s]; r = x - x.rolling(W, min_periods=1).mean()
        res.append(r.std()); tot.append(x.std())
    return np.nanmean(res) / np.nanmean(tot)

noise = pd.Series({s: noise_ratio(s) for s in informative}, name="noise_ratio").sort_values()
print("Noise ratio = residual std / within-engine std (higher = noisier, smoothing matters more)")
noise.round(3).to_frame()

# %%
fig, ax = plt.subplots(figsize=(9, 3.5)); noise.plot.bar(ax=ax, color="#8172b2")
ax.set(title=f"Relative high-frequency noise per informative sensor (trailing window={W})", ylabel="ratio"); plt.tight_layout()
plt.savefig(FIG_DIR / "05_noise_ratio.png", dpi=150); plt.show()

# %% [markdown]
# ## 7. Feature quality — signal vs RUL, redundancy
# - **Trend score:** mean per-engine Spearman between sensor and cycle. Near ±1 means a monotonic drift (good for rolling/lag features); near 0 means little degradation signal.
# - **RUL correlation:** Spearman between sensor and RUL, computed on **training engines only**. If this is later used for *selecting* features, redo it inside the training fold, not on the whole set.
# - **Redundancy:** pairs with |r| > 0.90 carry nearly the same information.

# %%
trend = {}
for s in informative:
    rs = [spearmanr(g["cycle"], g[s])[0] for _, g in df.groupby("unit") if g[s].nunique() > 1]
    trend[s] = np.nanmean(rs)
rul_corr = {s: spearmanr(df[s], df["RUL"])[0] for s in informative}
op_corr = {o: spearmanr(df[o], df["RUL"])[0] for o in OP_COLS if not desc.loc[o, "constant"]}
print("Operational settings vs RUL (Spearman):", {k: round(v, 3) for k, v in op_corr.items()})
fq = pd.DataFrame({"trend_score": trend, "spearman_vs_RUL": rul_corr, "noise_ratio": noise}).round(3)
fq["abs_trend"] = fq["trend_score"].abs()
fq.sort_values("abs_trend", ascending=False).drop(columns="abs_trend")

# %%
corr = df[informative].corr(method="spearman")
print("Strongest sensor-sensor pairs:")
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Sensor–sensor Spearman correlation (training set)"); plt.tight_layout()
plt.savefig(FIG_DIR / "06_sensor_correlation.png", dpi=150); plt.show()

pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename("r").reset_index()
         .rename(columns={"level_0": "a", "level_1": "b"}))
red = pairs[pairs["r"].abs() > 0.90].sort_values("r", key=abs, ascending=False)
print("Highly redundant pairs (|r|>0.90):"); red.round(3)

# %% [markdown]
# ## 8. Summary table → `reports/eda_sensor_summary.csv`

# %%
rows = []
for s in OP_COLS + SENSORS:
    if desc.loc[s, "constant"]:       rec, why = "drop", "constant"
    elif s in OP_COLS:                 rec, why = "review", f"op setting; FD001 is one regime, values are tiny jitter (rho vs RUL {op_corr.get(s, float('nan')):+.2f})"
    elif desc.loc[s, "near_constant"]: rec, why = "drop / review", f"only {int(desc.loc[s,'nunique'])} distinct values"
    elif s in fq.index and abs(fq.loc[s, "trend_score"]) < 0.3: rec, why = "review", f"weak trend ({fq.loc[s,'trend_score']:+.2f})"
    else:                              rec, why = "keep", "informative"
    rows.append({"channel": s, "recommendation": rec, "reason": why})
summary = pd.DataFrame(rows).set_index("channel")
summary["spearman_vs_RUL"] = np.nan
summary = summary.drop(columns="spearman_vs_RUL").join(fq[["trend_score", "spearman_vs_RUL", "noise_ratio"]])
for o, v in op_corr.items(): summary.loc[o, "spearman_vs_RUL"] = v
summary.to_csv(REP_DIR / "eda_sensor_summary.csv")
print(summary["recommendation"].value_counts().to_string()); summary

# %% [markdown]
# ## 9. Findings (fill in after running on the real file)
# Write each in your own words, with the number from the cells above:
#
# 1. **Data integrity:** rows / engines / missing / duplicates / gaps → matches Veda's note? 
# 2. **Class imbalance:** rows 0 vs 1 and why it is structural; consequence for the split and metrics.
# 3. **Constant / near-constant channels:** which, and the decision (drop from features; record in the data dictionary).
# 4. **Strongest degradation signals:** top sensors by |trend score| and RUL correlation.
# 5. **Noise:** which sensors are noisiest; suggested smoothing window or filter for the Week 1 preprocessing step.
# 6. **Redundancy:** which pairs to thin out before modelling.
# 7. **Caveats to document:** simulated data; cycles ≠ hours; FD001 has no raw vibration waveform, so FFT is not directly applicable.
