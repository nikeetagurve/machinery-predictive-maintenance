# Week 3 Checkpoint — Model Evaluation

## Project

NASA C-MAPSS FD001 Predictive Maintenance

## Work completed

- Verified the cleaned FD001 dataset.
- Confirmed 20,631 rows and 100 engines.
- Generated a binary target for failure within 24 cycles.
- Measured class distribution and identified class imbalance.
- Implemented an engine-grouped 80/20 validation split.
- Ensured training and validation engines do not overlap.
- Implemented Dummy Classifier, Logistic Regression,
  and Random Forest.
- Evaluated accuracy, balanced accuracy, precision, recall,
  F1-score, ROC-AUC, and PR-AUC.
- Examined confusion matrices and false alarm / missed-failure counts.
- Added model evaluation tests.
- Added scikit-learn to project requirements.

## Main findings

Logistic Regression:
- Recall: 97.80%
- Precision: 68.30%
- F1-score: 80.43%
- Missed failures: 11
- False alarms: 227

Random Forest:
- Recall: 87.60%
- Precision: 86.73%
- F1-score: 87.16%
- Missed failures: 62
- False alarms: 67

## Current decision

Logistic Regression is the provisional preferred model because
failure detection recall is the current priority.

Random Forest remains an alternative when fewer false alarms
are more important.

## Verification

- Full test suite: 42 passed.
- Git diff whitespace check: passed.
- Work performed on branch: feat/nikeeta-team-lead.

## Next steps

1. Perform grouped cross-validation across multiple engine splits.
2. Evaluate alternative classification thresholds.
3. Compare false-negative and false-positive costs.
4. Review final model choice after further validation.
5. Prepare the next project checkpoint.
