# GRU Prototype — NASA C-MAPSS FD001

## Configuration

- Task: predict failure within the next 24 operational cycles
- Model: one-layer GRU followed by a binary classification head
- Sequence length: 30 consecutive cycles
- Input features: 3 operating settings and 21 sensor readings
- Hidden size: 32
- Batch size: 64
- Epochs: 20
- Optimizer: Adam
- Learning rate: 0.001
- Loss: BCEWithLogitsLoss with positive-class weighting
- Device: cuda
- Random state: 42
- Prediction threshold: 0.5

## Split and sequence sizes

| Split | Rows | Engines | Sequences |
|---|---:|---:|---:|
| Train | 14130 | 70 | 12100 |
| Validation | 3210 | 15 | 2775 |
| Test (reserved) | 3291 | 15 | 2856 |

## Validation metrics

| Metric | Score |
|---|---:|
| Precision | 0.8276 |
| Recall | 0.9600 |
| F1-score | 0.8889 |
| Accuracy | 0.9676 |

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2325 (TN) | 75 (FP) |
| Actual 1 | 15 (FN) | 360 (TP) |

## Notes and limitations

- Windows contain consecutive cycles from a single engine only.
- The label for each window is the target of its final cycle.
- Imputation and scaling were fitted on training rows only.
- The engine-disjoint validation split was used for evaluation.
- The test split was not used for fitting or model selection.
- The 0.5 threshold is a starting point and has not been tuned.
- Results depend on the random seed and this prototype's training configuration.
- Compare these metrics with the Logistic Regression baseline; do not assume the GRU is better.
