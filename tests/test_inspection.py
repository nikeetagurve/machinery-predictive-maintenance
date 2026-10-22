"""Basic tests. All data below is SYNTHETIC (tiny fake tables), not the real NASA dataset."""
import pandas as pd
import pytest

from src.data.inspect_dataset import (
    check_rul_count,
    count_duplicates,
    engine_cycle_summary,
    find_constant_columns,
    make_data_dictionary,
    missing_value_table,
)
from src.data.load_data import COLUMN_NAMES, check_files, load_cmapss_file


def fake_table():
    """Two fake engines (3 and 2 cycles) with the 26 FD001 columns."""
    rows = [[unit, cycle] + [float(unit + cycle + i) for i in range(24)]
            for unit, n in [(1, 3), (2, 2)] for cycle in range(1, n + 1)]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df["sensor_1"] = 518.67  # constant on purpose
    return df


def write_text_file(df, path):
    path.write_text("\n".join(" ".join(str(v) for v in row) + "  " for row in df.to_numpy()) + "\n")


def test_schema_has_26_columns():
    assert len(COLUMN_NAMES) == 26 and len(set(COLUMN_NAMES)) == 26


def test_loads_valid_file(tmp_path):
    write_text_file(fake_table(), tmp_path / "train_FD001.txt")
    df = load_cmapss_file(tmp_path / "train_FD001.txt")
    assert list(df.columns) == COLUMN_NAMES and len(df) == 5


def test_missing_and_empty_file(tmp_path):
    assert set(check_files(tmp_path).values()) == {"missing"}
    with pytest.raises(FileNotFoundError):
        load_cmapss_file(tmp_path / "train_FD001.txt")
    (tmp_path / "train_FD001.txt").write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_cmapss_file(tmp_path / "train_FD001.txt")


def test_wrong_column_count(tmp_path):
    (tmp_path / "train_FD001.txt").write_text(" ".join(["1"] * 25) + "\n")
    with pytest.raises(ValueError, match="expected 26"):
        load_cmapss_file(tmp_path / "train_FD001.txt")


def test_missing_values_and_duplicates():
    df = fake_table()
    df.loc[0, "sensor_2"] = float("nan")
    assert missing_value_table(df).loc["sensor_2", "missing_count"] == 1
    assert count_duplicates(fake_table()) == {"full_row_duplicates": 0, "unit_time_duplicates": 0}
    doubled = pd.concat([fake_table(), fake_table().iloc[[0]]])
    assert count_duplicates(doubled) == {"full_row_duplicates": 1, "unit_time_duplicates": 1}


def test_engine_summary_and_constants_and_rul():
    df = fake_table()
    assert engine_cycle_summary(df)["n_rows"].tolist() == [3, 2]
    assert find_constant_columns(df) == ["sensor_1"]
    assert check_rul_count(df, pd.DataFrame({"rul": [5, 9]}))["match"]
    assert not check_rul_count(df, pd.DataFrame({"rul": [5]}))["match"]


def test_data_dictionary_has_26_rows_and_no_invented_units():
    d = make_data_dictionary(fake_table())
    assert len(d) == 26 and d.loc[d.column_name == "sensor_1", "unit"].iloc[0] == "Not specified/verified"
    assert (make_data_dictionary(None)["observed_dtype"] == "pending").all()
