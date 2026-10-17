# Machinery Predictive Maintenance

A team project focused on using time-series machinery sensor data and machine learning to identify potential equipment failures and support predictive maintenance.

## Project Objectives

- Inspect and clean machinery sensor data.
- Perform exploratory data analysis (EDA).
- Engineer time-series features such as lag and rolling statistics where appropriate.
- Train and compare baseline and machine-learning models.
- Evaluate models using a chronological data split to avoid future-data leakage.
- Investigate alert thresholds and simulate predictions on incoming data.

## Planned Workflow

1. Dataset selection and documentation
2. Data validation, cleaning, and EDA
3. Time-series feature engineering and target definition
4. Chronological train, validation, and test split
5. Baseline model and tree-based model training
6. Model comparison and alert-threshold analysis
7. Prediction workflow, testing, and documentation

## Team Responsibilities

- **Nikeeta Gurve — Team Lead:** Repository coordination, integration, pull-request reviews, and quality assurance.
- **Veda M V:** Dataset research, target definition, temporal validation, and threshold analysis.
- **Aditya Bhosale:** Data loading, cleaning, EDA, baseline model, and streaming prediction.
- **Shankar Savalgi:** Preprocessing, feature engineering, tree-based modelling, and tests.
- **Akif:** Target labelling, model comparison, and model artifacts.
- **Sreetama Santra:** Feature-quality analysis, experiment analysis, documentation, and final report.

## Repository Structure

- `data/` — Dataset documentation and data-related files
- `notebooks/` — Exploration and experiments
- `src/` — Data processing, features, models, and monitoring
- `tests/` — Automated tests
- `models/` — Model artifacts when appropriate
- `reports/` — Results and final report
- `docs/` — Work distribution and project documentation

## Development Workflow

1. Work on your assigned feature branch.
2. Make meaningful commits for completed work.
3. Push your branch to GitHub.
4. Open a pull request to `main`.
5. Request review before merging.

## Project Status

Initial repository setup and team branch creation are complete. Dataset selection, implementation, and model evaluation are pending.

## Dataset and Evaluation

The dataset source, license, prediction target, evaluation metrics, and final model results will be documented after the team validates its selected dataset.

## Limitations

Model performance and practical usefulness depend on the selected dataset, target definition, data quality, and evaluation results. No performance claims are made before experiments are completed.
