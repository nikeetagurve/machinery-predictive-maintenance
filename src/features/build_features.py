"""Reproducible feature-generation pipeline for NASA C-MAPSS FD001 (training set).

Pipeline (every step reuses the team's existing, tested modules):

    raw train_FD001.txt
      -> load + validate            src.data.load_data
      -> RUL / failure labels       src.data.target_labels
      -> engine-disjoint split      src.data.temporal_split
      -> drop constant channels     (decided on the TRAIN engines only)
      -> lag features               src.data.lag_features
      -> rolling mean/std/min/max   src.features.rolling_features
      -> data/processed/features_fd001.csv  +  reports/feature_manifest.csv

Leakage rules followed here:
  * Lag and rolling features at cycle t use cycles < t of the SAME engine only.
  * Constant channels are detected on training engines only.
  * `rul`, `target`, `split` and the id columns are never model inputs.
  * Missing values created by the warm-up period (first cycles of an engine) are
    left as NaN. Nothing is imputed here; fit imputers/scalers on train data later.

Run from the repo root:

    python -m src.features.build_features
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.data.lag_features import create_lag_features
from src.data.load_data import (
    COLUMN_NAMES,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    TRAIN_FILE,
    load_cmapss_file,
)
from src.data.target_labels import create_failure_labels
from src.data.temporal_split import temporal_train_val_test_split
from src.features.rolling_features import add_rolling_features

MACHINE_COL = "unit_number"
CYCLE_COL = "time_cycles"
ID_COLS = [MACHINE_COL, CYCLE_COL]
LABEL_COLS = ["rul", "target", "split"]  # analysis/supervision fields, never inputs

OP_COLS = [c for c in COLUMN_NAMES if c.startswith("op_setting_")]
SENSOR_COLS = [c for c in COLUMN_NAMES if c.startswith("sensor_")]

# Pipeline configuration. The horizon is in operational CYCLES, not hours, and is
# still pending team approval (see dataset_selection.md).
HORIZON = 24
LAGS = (1, 5, 10)
WINDOWS = (5, 10, 20)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "features_fd001.csv"
MANIFEST_FILE = REPORTS_DIR / "feature_manifest.csv"


def find_constant_columns(df: pd.DataFrame, columns) -> list[str]:
    """Return the columns that take a single value in `df`."""
    return [c for c in columns if df[c].nunique(dropna=True) <= 1]


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model-input columns: everything except ids and label/analysis fields."""
    excluded = set(ID_COLS) | set(LABEL_COLS)
    return [c for c in matrix.columns if c not in excluded]


def _assign_split(labeled: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Add a `split` column (train/val/test) using the team's engine-disjoint split."""
    parts = temporal_train_val_test_split(
        labeled,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_of_engine = {}
    for name, part in zip(("train", "val", "test"), parts):
        for engine in part[MACHINE_COL].unique():
            split_of_engine[engine] = name
    result = labeled.copy()
    result["split"] = result[MACHINE_COL].map(split_of_engine)
    return result, parts[0]


def build_feature_matrix(
    raw_train: pd.DataFrame,
    horizon: int = HORIZON,
    lags=LAGS,
    windows=WINDOWS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> pd.DataFrame:
    """Turn the raw FD001 training table into the processed feature matrix.

    Returns one row per (engine, cycle) with: id columns, model features,
    then `rul`, `target`, `split`.
    """
    labeled = create_failure_labels(
        raw_train,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        prediction_window=horizon,
    )
    labeled, train_part = _assign_split(labeled, train_ratio, val_ratio)

    # Decide which channels to drop using TRAIN engines only (no val/test peeking).
    constant = set(find_constant_columns(train_part, OP_COLS + SENSOR_COLS))
    kept_ops = [c for c in OP_COLS if c not in constant]
    kept_sensors = [c for c in SENSOR_COLS if c not in constant]

    base = labeled[ID_COLS + kept_ops + kept_sensors + LABEL_COLS]

    # Lag and rolling features (both are strictly past-only, per engine).
    with_lags = create_lag_features(
        base,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        sensor_cols=kept_sensors,
        lags=list(lags),
    )
    with_rolling = add_rolling_features(
        with_lags, windows=tuple(windows), sensor_columns=kept_sensors
    )

    ordered = (
        ID_COLS
        + [c for c in with_rolling.columns if c not in ID_COLS + LABEL_COLS]
        + LABEL_COLS
    )
    matrix = with_rolling[ordered]
    return matrix.sort_values(ID_COLS, kind="stable").reset_index(drop=True)


def describe_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per model feature: which family it belongs to and where it came from."""
    rows = []
    for col in feature_columns(matrix):
        if "_lag_" in col:
            source, lag = col.split("_lag_")
            rows.append((col, "lag", source, f"lag={lag}"))
        elif "_rolling_" in col:
            source, rest = col.split("_rolling_")
            stat, window = rest.rsplit("_", 1)
            rows.append((col, f"rolling_{stat}", source, f"window={window}"))
        else:
            family = "raw_setting" if col.startswith("op_setting_") else "raw_sensor"
            rows.append((col, family, col, ""))
    return pd.DataFrame(rows, columns=["feature", "family", "source", "parameter"])


def file_sha256(path: Path) -> str:
    """Checksum used to prove two runs produce an identical processed file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(raw_dir=RAW_DIR, output_file=OUTPUT_FILE, manifest_file=MANIFEST_FILE) -> dict:
    """Load raw data, build features, write the matrix and manifest, return a summary."""
    raw_train = load_cmapss_file(Path(raw_dir) / TRAIN_FILE)
    matrix = build_feature_matrix(raw_train)
    manifest = describe_features(matrix)

    output_file = Path(output_file)
    manifest_file = Path(manifest_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_file, index=False)
    manifest.to_csv(manifest_file, index=False)

    split_rows = matrix["split"].value_counts().to_dict()
    split_engines = matrix.groupby("split")[MACHINE_COL].nunique().to_dict()
    return {
        "rows": len(matrix),
        "engines": int(matrix[MACHINE_COL].nunique()),
        "model_features": len(manifest),
        "feature_families": manifest["family"].value_counts().to_dict(),
        "split_rows": split_rows,
        "split_engines": split_engines,
        "positive_rate_%": round(100 * matrix["target"].mean(), 2),
        "output_file": str(output_file),
        "manifest_file": str(manifest_file),
        "sha256": file_sha256(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FD001 processed feature matrix.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Folder containing train_FD001.txt")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Processed feature matrix (CSV)")
    parser.add_argument("--manifest", default=str(MANIFEST_FILE), help="Feature manifest (CSV)")
    args = parser.parse_args()

    summary = run(args.raw_dir, args.output, args.manifest)
    print("Feature pipeline completed")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
