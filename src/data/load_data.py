"""Load the NASA C-MAPSS FD001 files (read-only: raw files are never changed)."""
from pathlib import Path

import pandas as pd

# Paths are relative to the project root (two folders above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"
REQUIRED_FILES = [TRAIN_FILE, TEST_FILE, RUL_FILE]

# The 26 columns of the train/test files, in order (the files have no header row).
COLUMN_NAMES = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DOWNLOAD_HELP = (
    "Download CMAPSSData.zip from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data "
    "and copy train_FD001.txt, test_FD001.txt and RUL_FD001.txt into data/raw/CMAPSSData/ "
    "(see README.md)."
)


def check_files(raw_dir=RAW_DIR):
    """Return {file name: 'ok' / 'missing' / 'empty'} for the three required files."""
    status = {}
    for name in REQUIRED_FILES:
        path = Path(raw_dir) / name
        if not path.is_file():
            status[name] = "missing"
        elif path.stat().st_size == 0:
            status[name] = "empty"
        else:
            status[name] = "ok"
    return status


def _read_numbers(path):
    """Read a whitespace-separated numeric text file that has no header row."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}\n{DOWNLOAD_HELP}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}\n{DOWNLOAD_HELP}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        raise ValueError(f"{path.name} has non-numeric values. Was a header row added to the file?")
    return df


def load_cmapss_file(path):
    """Load train_FD001.txt or test_FD001.txt. Checks for 26 columns BEFORE naming them."""
    df = _read_numbers(path)
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"{Path(path).name} has {df.shape[1]} columns, expected {len(COLUMN_NAMES)}.")
    df.columns = COLUMN_NAMES
    return df


def load_rul_file(path):
    """Load RUL_FD001.txt: one number per line, returned as a column called 'rul'."""
    df = _read_numbers(path)
    if df.shape[1] != 1:
        raise ValueError(f"{Path(path).name} has {df.shape[1]} columns, expected 1.")
    df.columns = ["rul"]
    return df


def load_fd001(raw_dir=RAW_DIR):
    """Load (train, test, rul) from the raw data folder."""
    raw_dir = Path(raw_dir)
    train = load_cmapss_file(raw_dir / TRAIN_FILE)
    test = load_cmapss_file(raw_dir / TEST_FILE)
    rul = load_rul_file(raw_dir / RUL_FILE)
    return train, test, rul
