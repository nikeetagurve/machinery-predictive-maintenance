"""NASA C-MAPSS FD001 loading and validation.

Raw files are never modified. Download instructions (do not commit raw data):
  1. Get the C-MAPSS archive from NASA's Prognostics Data Repository
     (see dataset_selection.md for the link and licence note).
  2. Extract train_FD001.txt, test_FD001.txt, RUL_FD001.txt to data/raw/.

Columns: unit, cycle, 3 operational settings, 21 sensors (26 total).
Time axis is the operational CYCLE, not a wall-clock timestamp.

Leakage rule: RUL and failure labels are supervision-only. They must never
appear in the model's input features. `make_labels` keeps them in separate
columns, and `feature_columns` returns the safe input list.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS          # 26 columns
LABEL_COLS = ["RUL", "failure_within_h"]                  # never model inputs

HORIZON_CYCLES = 24                                       # provisional, per team doc

# Observed in the team's inspection (commit 2e390a3)
EXPECTED_TRAIN = {"rows": 20_631, "cols": 26, "units": 100,
                  "positives_h24": 2_500}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_split(path: str | Path) -> pd.DataFrame:
    """Load train_FD001.txt / test_FD001.txt (whitespace separated, no header)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. See module docstring for download steps.")
    df = pd.read_csv(p, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")                    # trailing-space artefacts
    if df.shape[1] != len(ALL_COLS):
        raise ValueError(f"Expected {len(ALL_COLS)} columns, got {df.shape[1]}")
    df.columns = ALL_COLS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def load_rul(path: str | Path) -> pd.Series:
    """RUL_FD001.txt: one true RUL per TEST engine, in engine order 1..N."""
    s = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    s.index = range(1, len(s) + 1)
    s.index.name = "unit"
    return s.rename("RUL_at_last_cycle")


def make_labels(train: pd.DataFrame, horizon: int = HORIZON_CYCLES) -> pd.DataFrame:
    """Add RUL and failure_within_h to a run-to-failure TRAIN frame (copy)."""
    out = train.copy()
    final = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = final - out["cycle"]
    out["failure_within_h"] = (out["RUL"] <= horizon).astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Safe model inputs: settings + sensors only. No ids, no labels."""
    return [c for c in SETTING_COLS + SENSOR_COLS if c in df.columns]


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Validation failed:\n- " + "\n- ".join(self.errors))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def validate_train(df: pd.DataFrame, horizon: int = HORIZON_CYCLES) -> Report:
    r = Report()
    r.stats.update(rows=len(df), cols=df.shape[1], units=int(df["unit"].nunique()))

    # Structure vs the team's documented inspection
    if list(df.columns[:26]) != ALL_COLS:
        r.errors.append("Column names/order differ from the C-MAPSS schema.")
        return r
    for key, got in (("rows", len(df)), ("cols", df.shape[1]), ("units", df["unit"].nunique())):
        if got != EXPECTED_TRAIN[key]:
            r.warnings.append(f"{key}={got}, team inspection recorded {EXPECTED_TRAIN[key]}.")

    # Missing / infinite
    n_na = int(df[ALL_COLS].isna().sum().sum())
    r.stats["missing_cells"] = n_na
    if n_na:
        r.errors.append(f"{n_na} missing cells (team recorded 0).")
    num = df[SETTING_COLS + SENSOR_COLS]
    if not num.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all().all():
        r.errors.append("Non-finite values in settings/sensors.")

    # Per-engine integrity: cycles start at 1, contiguous, strictly increasing
    bad_start, bad_gap, bad_dup = [], [], []
    for u, g in df.groupby("unit"):
        c = g["cycle"].to_numpy()
        if c[0] != 1:
            bad_start.append(int(u))
        if len(set(c)) != len(c):
            bad_dup.append(int(u))
        if (c[1:] - c[:-1] != 1).any():
            bad_gap.append(int(u))
    if bad_start:
        r.warnings.append(f"Engines not starting at cycle 1: {bad_start[:10]}")
    if bad_dup:
        r.errors.append(f"Duplicate cycles in engines: {bad_dup[:10]}")
    if bad_gap:
        r.errors.append(f"Non-contiguous/unsorted cycles in engines: {bad_gap[:10]}")
    r.stats["engine_lengths"] = {
        "min": int(df.groupby("unit")["cycle"].max().min()),
        "max": int(df.groupby("unit")["cycle"].max().max()),
    }

    # Constant columns (known in FD001: settings 3 and several sensors)
    const = [c for c in SETTING_COLS + SENSOR_COLS if df[c].nunique() <= 1]
    r.stats["constant_columns"] = const
    if const:
        r.warnings.append(f"Zero-variance columns (candidates to drop): {const}")

    # Label sanity on a labelled copy
    lab = make_labels(df, horizon)
    pos = int(lab["failure_within_h"].sum())
    r.stats["positives"] = pos
    r.stats["positive_frac"] = round(pos / len(lab), 4)
    if horizon == HORIZON_CYCLES and pos != EXPECTED_TRAIN["positives_h24"]:
        r.warnings.append(f"Positives={pos}, team recorded {EXPECTED_TRAIN['positives_h24']}.")
    leak = set(LABEL_COLS) & set(feature_columns(lab))
    if leak:
        r.errors.append(f"Label columns leaked into features: {leak}")
    return r


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("train_path", nargs="?", default="data/raw/train_FD001.txt")
    ap.add_argument("--out", default="reports/data_validation.json")
    a = ap.parse_args()
    rep = validate_train(load_split(a.train_path))
    rep.to_json(a.out)
    print(f"OK={rep.ok} errors={len(rep.errors)} warnings={len(rep.warnings)}")
    for e in rep.errors:
        print("ERROR:", e)
    for w in rep.warnings:
        print("WARN:", w)
