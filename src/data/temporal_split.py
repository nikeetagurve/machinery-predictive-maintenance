
"""Engine-disjoint train/validation/test splitting for NASA FD001."""

import pandas as pd

from src.data.preprocessing import validate_cycle_sequences


def temporal_train_val_test_split(
    df: pd.DataFrame,
    machine_col: str = "unit_number",
    cycle_col: str = "time_cycles",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Split complete engine histories without sharing engines between sets."""

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and leave data for testing.")

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    result = validate_cycle_sequences(df, machine_col, cycle_col)

    engine_ids = sorted(result[machine_col].unique())
    n_engines = len(engine_ids)

    n_train = int(n_engines * train_ratio)
    n_val = int(n_engines * val_ratio)

    if n_train == 0 or n_val == 0 or n_train + n_val >= n_engines:
        raise ValueError("Not enough engines for three non-empty splits.")

    train_ids = set(engine_ids[:n_train])
    val_ids = set(engine_ids[n_train:n_train + n_val])
    test_ids = set(engine_ids[n_train + n_val:])

    train_df = result[result[machine_col].isin(train_ids)].copy()
    val_df = result[result[machine_col].isin(val_ids)].copy()
    test_df = result[result[machine_col].isin(test_ids)].copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
