# Model Decision Record — FD001

## Decision

Select Logistic Regression as the provisional preferred model for
binary prediction of failure within the next 24 operational cycles.

## Context

The dataset contains 20,631 rows from 100 engines. The target is
positive when remaining useful life (RUL) is less than or equal to
24 cycles.

The target is imbalanced:
- Normal: 18,131 rows (87.88%)
- Failure within 24 cycles: 2,500 rows (12.12%)

## Evaluation method

- Engine-grouped 80/20 train/validation split.
- 80 engines for training and 20 for validation.
- Random state: 42.
- No engine overlap between training and validation.
- Models compared: Dummy Classifier, Logistic Regression,
  Random Forest.

## Validation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 94.15% | 96.83% |
| Balanced accuracy | 95.72% | 92.86% |
| Precision | 68.30% | 86.73% |
| Recall | 97.80% | 87.60% |
| F1-score | 80.43% | 87.16% |
| ROC-AUC | 99.26% | 99.26% |
| PR-AUC | 95.42% | 95.46% |

## Confusion matrix findings

Validation set: 4,070 rows, including 500 positive cases.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 3,343 | 227 | 11 | 489 |
| Random Forest | 3,503 | 67 | 62 | 438 |
| Dummy Classifier | 3,570 | 0 | 500 | 0 |

## Rationale

Logistic Regression is provisionally preferred because it detects
489 of the 500 actual impending-failure cases, missing only 11.
Its recall is 97.80%.

Random Forest detects 438 cases and misses 62, but produces fewer
false alarms (67 compared with 227).

For an early-warning predictive-maintenance system, missing an
impending failure may be more costly than generating an unnecessary
inspection alert. This makes Logistic Regression the preferred
candidate under the current assumed priority.

## Trade-offs

- Logistic Regression: higher recall, lower precision, more false alarms.
- Random Forest: higher precision, fewer false alarms, lower recall.
- Dummy Classifier: baseline only; it detects no impending failures.

## Limitations and follow-up

1. Results use one fixed engine-grouped holdout split.
2. Multi-split or grouped cross-validation is still required.
3. The operational costs of missed failures and false alarms
   have not yet been quantified.
4. The default classification threshold has not been optimized.
5. Deployment performance has not been established.

The selection is provisional and should be reviewed after
further validation and threshold analysis.
