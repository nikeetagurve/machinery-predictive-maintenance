import pandas as pd
import pytest

from src.models.evaluate_models import (
    prepare_dataset,
    split_by_engine,
)


def sample_data():
    rows = []
    for engine in range(1, 11):
        for cycle in range(1, 31):
            rows.append({
                "engine_id": engine,
                "cycle": cycle,
                "setting_1": float(engine),
                "sensor_1": float(cycle + engine),
            })
    return pd.DataFrame(rows)


def test_prepare_dataset_creates_binary_target():
    X, y, groups = prepare_dataset(sample_data())

    assert set(y.unique()) == {0, 1}
    assert "engine_id" not in X.columns
    assert "rul" not in X.columns
    assert "target" not in X.columns
    assert len(X) == len(y) == len(groups)


def test_prepare_dataset_does_not_modify_input():
    df = sample_data()
    original = df.copy(deep=True)

    prepare_dataset(df)

    pd.testing.assert_frame_equal(df, original)


def test_engine_groups_do_not_overlap():
    X, y, groups = prepare_dataset(sample_data())

    X_train, X_test, y_train, y_test, train_groups, test_groups = (
        split_by_engine(X, y, groups)
    )

    assert train_groups.isdisjoint(test_groups)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_duplicate_engine_cycle_is_rejected():
    df = sample_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(df)


def test_missing_engine_column_is_rejected():
    df = sample_data().drop(columns=["engine_id"])

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_dataset(df)


def test_rul_is_not_used_as_a_model_feature():
    df = sample_data()
    df["rul"] = 999

    X, _, _ = prepare_dataset(df)

    assert "rul" not in X.columns
