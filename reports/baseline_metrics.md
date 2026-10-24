# Logistic Regression Baseline â€” NASA C-MAPSS FD001

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
| Train | 14130 | 70 |
| Validation | 3210 | 15 |
| Test (reserved) | 3291 | 15 |

## Validation metrics

| Metric | Score |
|---|---:|
| Precision | 0.6583 |
| Recall | 0.9760 |
| F1-score | 0.7863 |
| Accuracy | 0.9380 |

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2645 (TN) | 190 (FP) |
| Actual 1 | 9 (FN) | 366 (TP) |

- True negatives (TN): 2645
- False positives (FP): 190
- False negatives (FN): 9
- True positives (TP): 366

## Notes and limitations

- `unit_number`, `time_cycles`, `rul`, and `target` are not model input features.
- The test split was reserved and not used for model fitting or metric reporting.
- The engine-disjoint split is deterministic but is not a global chronological split.
- These results are a baseline for comparison with later models, not a final production evaluation.
