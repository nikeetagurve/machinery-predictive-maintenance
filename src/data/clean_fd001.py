"""Clean and validate the NASA C-MAPSS FD001 training dataset."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.preprocessing import validate_cycle_sequences


COLUMN_NAMES = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def clean_fd001(input_path: str, output_path: str) -> dict:
    """Validate, sort, and save FD001 without inventing observations."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    df = pd.read_csv(input_file, sep=r"\s+", header=None)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"Expected 26 columns, found {df.shape[1]}.")

    df.columns = COLUMN_NAMES

    original_rows = len(df)

    if df.isna().any().any():
        missing_counts = df.isna().sum()
        raise ValueError(
            "Missing values found; review them before cleaning:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    duplicate_pairs = int(df.duplicated(["engine_id", "cycle"]).sum())
    if duplicate_pairs:
        raise ValueError(
            f"Found {duplicate_pairs} duplicate engine-cycle pairs."
        )

    df = validate_cycle_sequences(df, machine_col="engine_id", cycle_col="cycle")
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return {
        "input_rows": original_rows,
        "output_rows": len(df),
        "columns": len(df.columns),
        "engines": int(df["engine_id"].nunique()),
        "duplicate_engine_cycle_pairs": duplicate_pairs,
        "missing_cells": int(df.isna().sum().sum()),
        "output_file": str(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and clean NASA C-MAPSS FD001 data."
    )
    parser.add_argument(
        "--input",
        default="data/train_FD001.txt",
        help="Path to the original FD001 training text file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/train_FD001_clean.csv",
        help="Path for the cleaned CSV output.",
    )
    args = parser.parse_args()

    summary = clean_fd001(args.input, args.output)

    print("FD001 cleaning completed successfully")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
