"""Compare Logistic Regression and GRU on the same validation rows."""

import random

import numpy as np
import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_fd001, REPORTS_DIR
from src.data.target_labels import create_failure_labels
from src.data.temporal_split import temporal_train_val_test_split
from src.modeling.train_gru import (
    FEATURE_COLS,
    MACHINE_COL,
    CYCLE_COL,
    TARGET_COL,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_STATE,
    GRUClassifier,
    make_sequences,
)


REPORT_FILE = REPORTS_DIR / "baseline_gru_comparison.md"


def get_metrics(y_true, y_pred):
    """Return standard binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print(f"Comparison device: {device}")

    raw_train, _, _ = load_fd001()
    labeled = create_failure_labels(
        raw_train,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        prediction_window=24,
    )

    train, val, _test = temporal_train_val_test_split(
        labeled,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # Fit preprocessing on training rows only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train[FEATURE_COLS]))
    X_val = scaler.transform(imputer.transform(val[FEATURE_COLS]))

    # Use the same eligible validation rows for both models:
    # only cycles that have a full 30-cycle history in their engine.
    val_sorted = val.sort_values([MACHINE_COL, CYCLE_COL]).reset_index(drop=True)
    eligible_mask = val_sorted.groupby(MACHINE_COL).cumcount() >= SEQUENCE_LENGTH - 1
    eligible_val = val_sorted.loc[eligible_mask].copy().reset_index(drop=True)

    # Logistic Regression trained with the baseline configuration.
    baseline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    baseline.fit(train[FEATURE_COLS], train[TARGET_COL])

    # Reproduce GRU preprocessing and sequence creation.
    X_train_rows = imputer.transform(train[FEATURE_COLS])
    X_train_rows = scaler.transform(X_train_rows)
    X_val_rows = scaler.transform(imputer.transform(val[FEATURE_COLS]))

    X_gru_train, y_gru_train = make_sequences(
        train, X_train_rows, SEQUENCE_LENGTH
    )
    X_gru_val, y_gru_val = make_sequences(
        val, X_val_rows, SEQUENCE_LENGTH
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_gru_train),
            torch.from_numpy(y_gru_train),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = GRUClassifier(len(FEATURE_COLS), HIDDEN_SIZE).to(device)

    positives = float(y_gru_train.sum())
    negatives = float(len(y_gru_train) - positives)
    if positives == 0:
        raise ValueError("No positive labels in GRU training sequences.")

    pos_weight = torch.tensor(
        [negatives / positives], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            seen += len(batch_y)

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {total_loss / seen:.4f}")

    # Keep the GRU predictions aligned with validation engine-cycle rows.
    gru_index = []
    for _, group in val_sorted.groupby(MACHINE_COL, sort=True):
        indices = group.index.to_numpy()
        for end in range(SEQUENCE_LENGTH - 1, len(indices)):
            gru_index.append(indices[end])

    if len(gru_index) != len(X_gru_val):
        raise ValueError("GRU sequence index alignment check failed.")

    aligned_val = val_sorted.loc[gru_index].reset_index(drop=True)
    eligible_val = eligible_val.sort_values(
        [MACHINE_COL, CYCLE_COL]
    ).reset_index(drop=True)
    aligned_val = aligned_val.sort_values(
        [MACHINE_COL, CYCLE_COL]
    ).reset_index(drop=True)

    if not aligned_val[[MACHINE_COL, CYCLE_COL]].equals(
        eligible_val[[MACHINE_COL, CYCLE_COL]]
    ):
        raise ValueError("Baseline and GRU validation rows do not align.")

    baseline_pred = baseline.predict(eligible_val[FEATURE_COLS])

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_gru_val).to(device))
        gru_prob = torch.sigmoid(val_logits).cpu().numpy()
    gru_pred = (gru_prob >= 0.5).astype(int)

    y_true = eligible_val[TARGET_COL].to_numpy().astype(int)
    if not np.array_equal(y_true, y_gru_val.astype(int)):
        raise ValueError("Validation labels do not align.")

    baseline_metrics = get_metrics(y_true, baseline_pred)
    gru_metrics = get_metrics(y_true, gru_pred)

    def metric_table_row(name, metrics):
        return (
            f"| {name} | {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} "
            f"| {metrics['accuracy']:.4f} |"
        )

    report = f"""# Logistic Regression vs GRU — Comparable Validation

## Evaluation setup

- Dataset: NASA C-MAPSS FD001 training data
- Target: failure within the next 24 operational cycles
- Split: engine-disjoint 70% train, 15% validation, 15% test
- Validation rows: same engines and same eligible cycle rows for both models
- Sequence length: {SEQUENCE_LENGTH}
- GRU hidden size: {HIDDEN_SIZE}
- GRU epochs: {EPOCHS}
- GRU device: {device}
- Classification threshold: 0.5
- Test split: reserved and unused

## Eligible validation size

- Validation engines: {eligible_val[MACHINE_COL].nunique()}
- Eligible validation rows/sequences: {len(eligible_val)}

## Metrics

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
{metric_table_row("Logistic Regression", baseline_metrics)}
{metric_table_row("GRU", gru_metrics)}

## Confusion matrices

Rows are actual labels; columns are predicted labels.

### Logistic Regression

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | {baseline_metrics['tn']} | {baseline_metrics['fp']} |
| Actual 1 | {baseline_metrics['fn']} | {baseline_metrics['tp']} |

### GRU

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | {gru_metrics['tn']} | {gru_metrics['fp']} |
| Actual 1 | {gru_metrics['fn']} | {gru_metrics['tp']} |

## Notes

- Both models were evaluated on identical validation engine-cycle rows.
- Logistic Regression was retrained using the existing baseline configuration.
- GRU was retrained using the prototype configuration and random seed.
- Preprocessing was fitted on training rows only.
- These are validation results; the test split remains reserved.
- GPU training can be nondeterministic across hardware and library versions.
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\nComparable validation results")
    print(f"Eligible rows: {len(eligible_val)}")
    for name, metrics in [
        ("Logistic Regression", baseline_metrics),
        ("GRU", gru_metrics),
    ]:
        print(
            f"{name}: Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}"
        )
        print(
            f"  Confusion matrix: "
            f"[[{metrics['tn']}, {metrics['fp']}], "
            f"[{metrics['fn']}, {metrics['tp']}]]"
        )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
