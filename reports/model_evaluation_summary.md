# FD001 Model Evaluation — Week 3

## Objective

Compare baseline classifiers for predicting whether an engine will fail
within the next 24 operational cycles.

## Dataset and labels

- Dataset: NASA C-MAPSS FD001 cleaned training data
- Total rows: 20631
- Total engines: 100
- Input features: 25
- Prediction target: 1 when RUL <= 24; otherwise 0
- Normal rows: 18131
- Failure-within-window rows: 2500

## Validation method

- Split: engine-grouped 80/20 train/validation
- Random state: 42
- Training engines: 80
- Validation engines: 20
- Training rows: 16561
- Validation rows: 4070
- Engine overlap: none

## Model comparison

| model               |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |
|:--------------------|-----------:|--------------------:|------------:|---------:|-------:|----------:|---------:|
| Random Forest       |     0.9683 |              0.9286 |      0.8673 |   0.8760 | 0.8716 |    0.9926 |   0.9546 |
| Logistic Regression |     0.9415 |              0.9572 |      0.6830 |   0.9780 | 0.8043 |    0.9926 |   0.9542 |
| Dummy Classifier    |     0.8771 |              0.5000 |      0.0000 |   0.0000 | 0.0000 |    0.5000 |   0.1229 |

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


## Confusion matrix

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 3343 | 227 | 11 | 489 |
| Random Forest | 3503 | 67 | 62 | 438 |
| Dummy Classifier | 3570 | 0 | 500 | 0 |
