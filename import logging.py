import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
EXPECTED_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
EXPECTED_TOTAL_COLS = 26


def load_raw_split(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"data file not found at: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=EXPECTED_COLUMNS,
        index_col=False,
    )
    return df


def validate_raw_data(df, dataset_name="unspecified"):
    errors = []

    if df.shape[1] != EXPECTED_TOTAL_COLS:
        errors.append(
            f"expected {EXPECTED_TOTAL_COLS} columns, found {df.shape[1]}"
        )

    if df.empty:
        errors.append("loaded dataframe is empty")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        errors.append(f"detected {null_count} missing values")

    invalid_types = []
    for col in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            invalid_types.append(col)
    if invalid_types:
        errors.append(
            f"non-numeric columns present: {', '.join(invalid_types)}"
        )

    if (df["time_cycles"] < 1).any():
        errors.append("non-positive values detected in time_cycles")

    if (df["unit_nr"] < 1).any():
        errors.append("non-positive values detected in unit_nr")

    if errors:
        error_msg = f"validation failure for {dataset_name}: " + "; ".join(
            errors
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(
        f"validation passed for {dataset_name}. shape: {df.shape}, units: {df['unit_nr'].nunique()}"
    )
    return True


def get_immutable_data(file_path, dataset_name="dataset"):
    df = load_raw_split(file_path)
    validate_raw_data(df, dataset_name=dataset_name)
    return df.copy(deep=True)


if __name__ == "__main__":
    sample_file = Path("data/raw/train_FD001.txt")
    if sample_file.exists():
        raw_df = get_immutable_data(sample_file, dataset_name="train_FD001")
        print(raw_df.head())
    else:
        print(f"place dataset text files in {sample_file.parent} to run")