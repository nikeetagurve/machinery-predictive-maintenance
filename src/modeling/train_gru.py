"""Train a GPU-enabled GRU prototype for NASA C-MAPSS FD001."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_fd001, REPORTS_DIR
from src.data.target_labels import create_failure_labels
from src.data.temporal_split import temporal_train_val_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = REPORTS_DIR / "gru_metrics.md"

MACHINE_COL = "unit_number"
CYCLE_COL = "time_cycles"
TARGET_COL = "target"

FEATURE_COLS = (
    [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


class GRUClassifier(nn.Module):
    """Small GRU classifier for cycle-level failure prediction."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, number_of_features)
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.classifier(last_hidden).squeeze(1)


def make_sequences(
    df: pd.DataFrame,
    features: np.ndarray,
    sequence_length: int,
):
    """Create windows within each engine; never cross engine boundaries."""

    sequences = []
    labels = []

    work = df[[MACHINE_COL, CYCLE_COL, TARGET_COL]].reset_index(drop=True)

    for _, group in work.groupby(MACHINE_COL, sort=True):
        row_indices = group.index.to_numpy()

        # Only consecutive cycles of the same engine form a sequence.
        cycles = group[CYCLE_COL].to_numpy()
        if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
            raise ValueError("Non-consecutive cycles found within an engine.")

        if len(row_indices) < sequence_length:
            continue

        for end in range(sequence_length - 1, len(row_indices)):
            start = end - sequence_length + 1
            indices = row_indices[start:end + 1]

            sequences.append(features[indices])
            labels.append(int(work.loc[row_indices[end], TARGET_COL]))

    if not sequences:
        raise ValueError(
            f"No sequences created. Need at least {sequence_length} cycles per engine."
        )

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y


def main() -> None:
    """Train the GRU on CUDA when available and report validation metrics."""

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, _, _ = load_fd001()

    labeled_df = create_failure_labels(
        train_df,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        prediction_window=24,
    )

    train, val, test = temporal_train_val_test_split(
        labeled_df,
        machine_col=MACHINE_COL,
        cycle_col=CYCLE_COL,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # Fit preprocessing only on training rows to avoid validation/test leakage.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_rows = imputer.fit_transform(train[FEATURE_COLS])
    X_train_rows = scaler.fit_transform(X_train_rows)

    X_val_rows = scaler.transform(imputer.transform(val[FEATURE_COLS]))
    X_test_rows = scaler.transform(imputer.transform(test[FEATURE_COLS]))

    X_train, y_train = make_sequences(
        train, X_train_rows, SEQUENCE_LENGTH
    )
    X_val, y_val = make_sequences(
        val, X_val_rows, SEQUENCE_LENGTH
    )
    X_test, y_test = make_sequences(
        test, X_test_rows, SEQUENCE_LENGTH
    )

    print(f"Train sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    print(f"Test sequences reserved: {len(X_test)}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = GRUClassifier(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)

    if positives == 0:
        raise ValueError("Training sequences contain no positive labels.")

    # Compensate for class imbalance using training labels only.
    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(
                device, non_blocking=(device.type == "cuda")
            )
            batch_y = batch_y.to(
                device, non_blocking=(device.type == "cuda")
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            seen += len(batch_y)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- loss: {total_loss / seen:.4f}"
        )

    # Evaluate only on validation data; test remains reserved.
    model.eval()
    val_tensor = torch.from_numpy(X_val).to(device)

    with torch.no_grad():
        val_logits = model(val_tensor)
        val_probabilities = torch.sigmoid(val_logits).cpu().numpy()

    y_pred = (val_probabilities >= 0.5).astype(np.int64)
    y_true = y_val.astype(np.int64)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = f"""# GRU Prototype — NASA C-MAPSS FD001

## Configuration

- Task: predict failure within the next 24 operational cycles
- Model: one-layer GRU followed by a binary classification head
- Sequence length: {SEQUENCE_LENGTH} consecutive cycles
- Input features: 3 operating settings and 21 sensor readings
- Hidden size: {HIDDEN_SIZE}
- Batch size: {BATCH_SIZE}
- Epochs: {EPOCHS}
- Optimizer: Adam
- Learning rate: {LEARNING_RATE}
- Loss: BCEWithLogitsLoss with positive-class weighting
- Device: {device}
- Random state: {RANDOM_STATE}
- Prediction threshold: 0.5

## Split and sequence sizes

| Split | Rows | Engines | Sequences |
|---|---:|---:|---:|
| Train | {len(train)} | {train[MACHINE_COL].nunique()} | {len(X_train)} |
| Validation | {len(val)} | {val[MACHINE_COL].nunique()} | {len(X_val)} |
| Test (reserved) | {len(test)} | {test[MACHINE_COL].nunique()} | {len(X_test)} |

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

## Notes and limitations

- Windows contain consecutive cycles from a single engine only.
- The label for each window is the target of its final cycle.
- Imputation and scaling were fitted on training rows only.
- The engine-disjoint validation split was used for evaluation.
- The test split was not used for fitting or model selection.
- The 0.5 threshold is a starting point and has not been tuned.
- Results depend on the random seed and this prototype's training configuration.
- Compare these metrics with the Logistic Regression baseline; do not assume the GRU is better.
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(report, encoding="utf-8")

    print("\nGRU training completed.")
    print(f"Device: {device}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(cm)
    print(f"\nReport saved to: {METRICS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()