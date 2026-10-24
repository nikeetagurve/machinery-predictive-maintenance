"""Train a Logistic Regression baseline for NASA C-MAPSS FD001."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_fd001, REPORTS_DIR
from src.data.target_labels import create_failure_labels
from src.data.temporal_split import temporal_train_val_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = REPORTS_DIR / "baseline_metrics.md"

MACHINE_COL = "unit_number"
CYCLE_COL = "time_cycles"
TARGET_COL = "target"

# Use operating settings and sensor readings, not engine IDs, cycle,
# RUL, or target. These are the 24 raw measurements.
FEATURE_COLS = (
    [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main() -> None:
    """Train the baseline and save validation metrics."""

    train_df, _, _ = load_fd001()

    labeled_df = create_failure_labels(
        train_df,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        prediction_window=24,
    )

    train, val, _test = temporal_train_val_test_split(
        labeled_df,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET_COL]

    # Imputation and scaling are fitted on training data only.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )
    accuracy = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    report = f"""# Logistic Regression Baseline â€” NASA C-MAPSS FD001

## Setup

- Target: failure within the next 24 operational cycles
- Positive class: `1` (failure within 24 cycles)
- Model: Logistic Regression with `class_weight="balanced"`
- Input features: 3 operating settings and 21 sensor readings
- Split: 70% train, 15% validation, 15% test, separated by engine
- Evaluation: validation set only
- Preprocessing: median imputation and standard scaling fitted on training data only
- Random state: 42

## Dataset sizes

| Split | Rows | Engines |
|---|---:|---:|
| Train | {len(train)} | {train[MACHINE_COL].nunique()} |
| Validation | {len(val)} | {val[MACHINE_COL].nunique()} |
| Test (reserved) | {_test.shape[0]} | {_test[MACHINE_COL].nunique()} |

## Validation metrics

| Metric | Score |
|---|---:|
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |
| Accuracy | {accuracy:.4f} |

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | {tn} (TN) | {fp} (FP) |
| Actual 1 | {fn} (FN) | {tp} (TP) |

- True negatives (TN): {tn}
- False positives (FP): {fp}
- False negatives (FN): {fn}
- True positives (TP): {tp}

## Notes and limitations

- `unit_number`, `time_cycles`, `rul`, and `target` are not model input features.
- The test split was reserved and not used for model fitting or metric reporting.
- The engine-disjoint split is deterministic but is not a global chronological split.
- These results are a baseline for comparison with later models, not a final production evaluation.
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(report, encoding="utf-8")

    print("Baseline training completed.")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(f"Test rows reserved: {len(_test)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(cm)
    print(f"\nReport saved to: {METRICS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
