"""Data loading and validation for the predictive-maintenance pipeline.

Dataset-agnostic: configure column names via DataConfig once the team's
dataset is fixed. Raw data is never modified; all work happens on copies.

Usage:
    from src.data.loader import DataConfig, load_raw, validate

    cfg = DataConfig(path="data/raw/sensors.csv",
                     timestamp_col="timestamp",
                     machine_col="machine_id",
                     target_col="failure",
                     sensor_cols=["temp", "vibration", "pressure"])
    df = load_raw(cfg)
    report = validate(df, cfg)
    report.raise_if_failed()
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    path: str
    timestamp_col: str = "timestamp"
    machine_col: Optional[str] = "machine_id"
    target_col: Optional[str] = None          # set once target is defined
    sensor_cols: list[str] = field(default_factory=list)
    expected_checksum: Optional[str] = None   # sha256 of the raw file
    max_missing_frac: float = 0.30            # per-column warning threshold
    read_kwargs: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.errors))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_raw(cfg: DataConfig) -> pd.DataFrame:
    """Read the raw file (csv/parquet) without altering its contents.

    Timestamps are parsed; no sorting, filling or dropping happens here.
    That is the cleaning step (Oct 14).
    """
    p = Path(cfg.path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data not found: {p}. See README for download steps.")

    if cfg.expected_checksum:
        actual = file_sha256(p)
        if actual != cfg.expected_checksum:
            raise ValueError(f"Checksum mismatch for {p}: {actual} != {cfg.expected_checksum}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, **cfg.read_kwargs)
    else:
        df = pd.read_csv(p, **cfg.read_kwargs)

    if cfg.timestamp_col in df.columns:
        df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")

    log.info("Loaded %s rows x %s cols from %s", *df.shape, p)
    return df


def validate(df: pd.DataFrame, cfg: DataConfig) -> ValidationReport:
    """Run structural and quality checks. Does not modify df."""
    r = ValidationReport()
    r.stats["n_rows"], r.stats["n_cols"] = df.shape

    if df.empty:
        r.errors.append("Dataset is empty.")
        return r

    # 1. Required columns
    required = [cfg.timestamp_col] + cfg.sensor_cols
    for c in (cfg.machine_col, cfg.target_col):
        if c:
            required.append(c)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        r.errors.append(f"Missing required columns: {missing_cols}")
        return r

    # 2. Timestamps
    ts = df[cfg.timestamp_col]
    n_bad_ts = int(ts.isna().sum())
    r.stats["unparseable_timestamps"] = n_bad_ts
    if n_bad_ts:
        r.errors.append(f"{n_bad_ts} rows have missing/unparseable timestamps.")
    r.stats["time_range"] = [str(ts.min()), str(ts.max())]

    # 3. Ordering and duplicates (per machine when available)
    keys = [cfg.machine_col, cfg.timestamp_col] if cfg.machine_col else [cfg.timestamp_col]
    n_dup_keys = int(df.duplicated(subset=keys).sum())
    n_dup_rows = int(df.duplicated().sum())
    r.stats["duplicate_key_rows"] = n_dup_keys
    r.stats["duplicate_full_rows"] = n_dup_rows
    if n_dup_keys:
        r.warnings.append(f"{n_dup_keys} duplicate ({', '.join(keys)}) rows.")

    if cfg.machine_col:
        r.stats["n_machines"] = int(df[cfg.machine_col].nunique())
        unsorted = 0
        for _, g in df.groupby(cfg.machine_col, sort=False):
            if not g[cfg.timestamp_col].is_monotonic_increasing:
                unsorted += 1
        r.stats["machines_not_time_sorted"] = unsorted
        if unsorted:
            r.warnings.append(f"{unsorted} machine series are not time-sorted (fix in cleaning).")
    elif not ts.is_monotonic_increasing:
        r.warnings.append("Timestamps are not sorted (fix in cleaning).")

    # 4. Sensor columns: dtype, missingness, constants, infinities
    sensor_stats = {}
    for c in cfg.sensor_cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            r.errors.append(f"Sensor '{c}' is not numeric (dtype={s.dtype}).")
            continue
        miss = float(s.isna().mean())
        n_inf = int(((s == float("inf")) | (s == float("-inf"))).sum())
        const = s.nunique(dropna=True) <= 1
        sensor_stats[c] = {"missing_frac": round(miss, 4), "n_inf": n_inf,
                           "constant": bool(const),
                           "min": float(s.min()), "max": float(s.max())}
        if miss > cfg.max_missing_frac:
            r.warnings.append(f"Sensor '{c}' is {miss:.0%} missing.")
        if n_inf:
            r.errors.append(f"Sensor '{c}' contains {n_inf} infinite values.")
        if const:
            r.warnings.append(f"Sensor '{c}' is constant (zero variance).")
    r.stats["sensors"] = sensor_stats

    # 5. Target (if defined)
    if cfg.target_col:
        t = df[cfg.target_col]
        n_na = int(t.isna().sum())
        if n_na:
            r.errors.append(f"Target '{cfg.target_col}' has {n_na} missing values.")
        vc = t.value_counts(dropna=True)
        r.stats["target_distribution"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) < 2:
            r.errors.append("Target has fewer than 2 classes.")
        else:
            minority = float(vc.min() / vc.sum())
            r.stats["minority_class_frac"] = round(minority, 5)
            if minority < 0.05:
                r.warnings.append(f"Severe class imbalance (minority {minority:.2%}).")

    return r


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Validate a raw sensor dataset.")
    ap.add_argument("path")
    ap.add_argument("--timestamp", default="timestamp")
    ap.add_argument("--machine", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--sensors", nargs="*", default=[])
    ap.add_argument("--out", default="reports/data_validation.json")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = DataConfig(path=a.path, timestamp_col=a.timestamp, machine_col=a.machine,
                     target_col=a.target, sensor_cols=a.sensors)
    rep = validate(load_raw(cfg), cfg)
    rep.to_json(a.out)
    print(f"OK={rep.ok}  errors={len(rep.errors)}  warnings={len(rep.warnings)}")
    for e in rep.errors:
        print("ERROR:", e)
    for w in rep.warnings:
        print("WARN:", w)
