"""Compare baseline classifiers for FD001 failure-within-24-cycle prediction."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_FD001_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTION_WINDOW = 24
TEST_SIZE = 0.20
RANDOM_STATE = 42


def prepare_dataset(df):
    """Create binary labels and model inputs without using future RUL as a feature."""
    required = {"engine_id", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["engine_id", "cycle"]].isna().any().any():
        raise ValueError("Engine IDs and cycles cannot contain missing values.")

    if df.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle combinations found.")

    if not pd.api.types.is_numeric_dtype(df["cycle"]):
        raise ValueError("Cycle values must be numeric.")

    final_cycles = df.groupby("engine_id")["cycle"].transform("max")
    rul = final_cycles - df["cycle"]
    y = (rul <= PREDICTION_WINDOW).astype(int)

    # Engine ID is used only for grouping, never as a model feature.
    # RUL is used only to create labels, never as a model feature.
    excluded = {"engine_id", "rul", "target"}
    feature_columns = [col for col in df.columns if col not in excluded]
    X = df[feature_columns].copy()

    if X.empty:
        raise ValueError("No model input features are available.")

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("All model input features must be numeric.")

    if X.isna().any().any():
        raise ValueError("Model input features contain missing values.")

    groups = df["engine_id"].copy()

    return X, y, groups


def split_by_engine(X, y, groups):
    """Create a deterministic train/validation split with no shared engines."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    if train_groups & test_groups:
        raise ValueError("Engine leakage detected between train and validation.")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both classes must appear in training and validation. "
            "Review the engine-grouped split before evaluating."
        )

    return X_train, X_test, y_train, y_test, train_groups, test_groups


def build_models():
    """Return baseline and candidate classifiers."""
    return {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(X_train, X_test, y_train, y_test):
    """Fit and evaluate classifiers on the same held-out engines."""
    results = []

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
        })

    return pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    ).reset_index(drop=True)


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, groups = prepare_dataset(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
    ) = split_by_engine(X, y, groups)

    results = evaluate_models(X_train, X_test, y_train, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "model_evaluation_results.csv"
    summary_path = REPORTS_DIR / "model_evaluation_summary.md"

    results.to_csv(results_path, index=False)

    summary = f"""# FD001 Model Evaluation — Week 3

## Objective

Compare baseline classifiers for predicting whether an engine will fail
within the next {PREDICTION_WINDOW} operational cycles.

## Dataset and labels

- Dataset: NASA C-MAPSS FD001 cleaned training data
- Total rows: {len(df)}
- Total engines: {groups.nunique()}
- Input features: {X.shape[1]}
- Prediction target: 1 when RUL <= {PREDICTION_WINDOW}; otherwise 0
- Normal rows: {int((y == 0).sum())}
- Failure-within-window rows: {int((y == 1).sum())}

## Validation method

- Split: engine-grouped 80/20 train/validation
- Random state: {RANDOM_STATE}
- Training engines: {len(train_groups)}
- Validation engines: {len(test_groups)}
- Training rows: {len(X_train)}
- Validation rows: {len(X_test)}
- Engine overlap: none

## Model comparison

{results.to_markdown(index=False, floatfmt=".4f")}

## Interpretation

PR-AUC is included because the failure class is a minority class.
Recall measures how many actual impending failures are detected.
Precision measures how many predicted impending failures are correct.
F1 balances precision and recall.

The Dummy Classifier is the baseline reference. The candidate models
must be compared using the same held-out engines.

## Model decision

No final model is selected automatically by this script. The results
must be reviewed against the project's priorities, especially failure
detection recall, precision, PR-AUC, and operational false alarms.

## Limitations

- Evaluation uses one fixed engine-grouped holdout split.
- Results are not yet cross-validated across multiple engine splits.
- The target assumes complete run-to-failure training histories.
- The model predicts a binary 24-cycle warning, not exact remaining life.
- The threshold is the classifier's default decision threshold.
- Validation results do not establish real-world deployment performance.
"""

    summary_path.write_text(summary, encoding="utf-8")

    print("\n===== DATASET =====")
    print(f"Rows: {len(df)}")
    print(f"Engines: {groups.nunique()}")
    print(f"Features: {X.shape[1]}")
    print(f"Training engines: {len(train_groups)}")
    print(f"Validation engines: {len(test_groups)}")

    print("\n===== TARGET DISTRIBUTION =====")
    print(y.value_counts().sort_index())

    print("\n===== MODEL COMPARISON =====")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
