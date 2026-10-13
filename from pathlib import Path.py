from pathlib import Path
import pandas as pd
import pytest
from src.data_loader import (
    EXPECTED_TOTAL_COLS,
    get_immutable_data,
    load_raw_split,
    validate_raw_data,
)


def create_dummy_data(rows=10, cols=26):
    data = {f"c{i}": [float(j) + 1.0 for j in range(rows)] for i in range(cols)}
    df = pd.DataFrame(data)
    df.columns = [
        "unit_nr",
        "time_cycles",
        "setting_1",
        "setting_2",
        "setting_3",
    ] + [f"s{i}" for i in range(1, 22)]
    return df


def test_validate_raw_data_success():
    df = create_dummy_data()
    assert validate_raw_data(df, "test_valid") is True


def test_validate_raw_data_missing_column():
    df = create_dummy_data()
    df = df.drop(columns=["s21"])
    with pytest.raises(ValueError):
        validate_raw_data(df, "test_missing_col")


def test_validate_raw_data_null_values():
    df = create_dummy_data()
    df.loc[0, "s1"] = None
    with pytest.raises(ValueError):
        validate_raw_data(df, "test_nulls")


def test_validate_raw_data_invalid_cycles():
    df = create_dummy_data()
    df.loc[0, "time_cycles"] = 0
    with pytest.raises(ValueError):
        validate_raw_data(df, "test_zero_cycle")