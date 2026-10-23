
import pandas as pd
import pytest

from src.data.temporal_split import temporal_train_val_test_split


def make_sample_data():
    """Create 20 engines with 5 operating cycles each."""
    rows = []
    for engine in range(1, 21):
        for cycle in range(1, 6):
            rows.append({
                "unit_number": engine,
                "time_cycles": cycle,
                "sensor_1": engine * 10 + cycle,
            })
    return pd.DataFrame(rows)


def test_split_has_no_engine_overlap():
    df = make_sample_data()

    train, val, test = temporal_train_val_test_split(df)

    train_ids = set(train["unit_number"])
    val_ids = set(val["unit_number"])
    test_ids = set(test["unit_number"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_all_rows_are_preserved():
    df = make_sample_data()

    train, val, test = temporal_train_val_test_split(df)

    assert len(train) + len(val) + len(test) == len(df)


def test_cycles_are_sorted():
    df = make_sample_data().sample(frac=1, random_state=42)

    train, val, test = temporal_train_val_test_split(df)

    for split in (train, val, test):
        for _, group in split.groupby("unit_number"):
            assert group["time_cycles"].is_monotonic_increasing


def test_split_is_deterministic():
    df = make_sample_data()

    first = temporal_train_val_test_split(df)
    second = temporal_train_val_test_split(df)

    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_supports_cleaned_column_names():
    df = make_sample_data().rename(columns={
        "unit_number": "engine_id",
        "time_cycles": "cycle",
    })

    train, val, test = temporal_train_val_test_split(
        df,
        machine_col="engine_id",
        cycle_col="cycle",
    )

    assert all(not split.empty for split in (train, val, test))


def test_invalid_ratios():
    df = make_sample_data()

    with pytest.raises(ValueError):
        temporal_train_val_test_split(
            df, train_ratio=0.9, val_ratio=0.2
        )


def test_duplicate_cycles_rejected():
    df = make_sample_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError):
        temporal_train_val_test_split(df)


def test_input_is_not_modified():
    df = make_sample_data()
    original = df.copy(deep=True)

    temporal_train_val_test_split(df)

    pd.testing.assert_frame_equal(df, original)
