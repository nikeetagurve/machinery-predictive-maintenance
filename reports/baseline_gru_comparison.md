# Logistic Regression vs GRU — Comparable Validation

## Evaluation setup

- Dataset: NASA C-MAPSS FD001 training data
- Target: failure within the next 24 operational cycles
- Split: engine-disjoint 70% train, 15% validation, 15% test
- Validation rows: same engines and same eligible cycle rows for both models
- Sequence length: 30
- GRU hidden size: 32
- GRU epochs: 20
- GRU device: cuda
- Classification threshold: 0.5
- Test split: reserved and unused

## Eligible validation size

- Validation engines: 15
- Eligible validation rows/sequences: 2775

## Metrics

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6583 | 0.9760 | 0.7863 | 0.9283 |
| GRU | 0.8276 | 0.9600 | 0.8889 | 0.9676 |

## Confusion matrices

Rows are actual labels; columns are predicted labels.

### Logistic Regression

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2210 | 190 |
| Actual 1 | 9 | 366 |

### GRU

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2325 | 75 |
| Actual 1 | 15 | 360 |

## Notes

- Both models were evaluated on identical validation engine-cycle rows.
- Logistic Regression was retrained using the existing baseline configuration.
- GRU was retrained using the prototype configuration and random seed.
- Preprocessing was fitted on training rows only.
- These are validation results; the test split remains reserved.
- GPU training can be nondeterministic across hardware and library versions.
