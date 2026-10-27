"""Compare models on the same engine-disjoint validation split of NASA C-MAPSS FD001.

What this script guarantees
  * Every model is trained on the TRAIN engines only and scored on the SAME validation rows.
  * The shared scored rows are validation cycles >= SEQUENCE_LENGTH (30). That is exactly the set of rows a
    30-cycle GRU window can be built for, so row models and the GRU are compared like for like.
  * The reserved test split is never scored or looked at.
  * Imputation/scaling are fitted on training rows only (inside each model pipeline).
  * The 0.5 threshold is the team default and is not tuned here.

Models
  dummy_always_0  reference: always predicts "no failure soon"
  lr_raw24        logistic regression on the 24 raw settings/sensors (same config as the team baseline)
  lr_eng242       logistic regression on the engineered feature matrix
  hgb_eng242      gradient-boosted trees (scikit-learn HistGradientBoosting) on the engineered features.
                  INTERIM tree model: replace/add Shankar's XGBoost/LightGBM when it lands on main.
  hgb_eng166      same trees on the redundancy-filtered proposal from the feature-quality notebook
  gru_raw24       the team's GRU prototype (optional: needs PyTorch, run with --with-gru)

Run from the repo root:

    python -m src.modeling.compare_models
    python -m src.modeling.compare_models --with-gru
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import COLUMN_NAMES, RAW_DIR, REPORTS_DIR, TRAIN_FILE, load_cmapss_file
from src.features import build_features as bf

MACHINE_COL = bf.MACHINE_COL
CYCLE_COL = bf.CYCLE_COL
ID_COLS = bf.ID_COLS
RAW_FEATURES = list(COLUMN_NAMES[2:])  # 3 operating settings + 21 sensors

SEQUENCE_LENGTH = 30  # same as the GRU prototype; defines the shared scored rows
THRESHOLD = 0.5
RANDOM_STATE = 42

PREDICTIONS_FILE = REPORTS_DIR / "model_predictions_val.csv"
METRICS_FILE = REPORTS_DIR / "model_comparison_metrics.csv"
REPORT_FILE = REPORTS_DIR / "model_comparison.md"
SELECTED_FEATURES_FILE = REPORTS_DIR / "selected_features_proposal.csv"

MODEL_LABELS = {
    "dummy_always_0": "Always predict 0 (reference)",
    "lr_raw24": "Logistic Regression, raw features",
    "lr_eng242": "Logistic Regression, engineered features",
    "hgb_eng242": "Gradient-boosted trees, engineered features (interim)",
    "hgb_eng166": "Gradient-boosted trees, filtered engineered features (interim)",
    "gru_raw24": "GRU (30-cycle windows), raw features",
}
IMBALANCE_HANDLING = {
    "dummy_always_0": "none",
    "lr_raw24": "class_weight=balanced",
    "lr_eng242": "class_weight=balanced",
    "hgb_eng242": "class_weight=balanced",
    "hgb_eng166": "class_weight=balanced",
    "gru_raw24": "pos_weight = negatives/positives (train)",
}


# ----------------------------------------------------------------------------- data
def prepare_data(raw_train: pd.DataFrame):
    """Return (raw table with labels+split, engineered matrix), both sorted by (engine, cycle)."""
    matrix = bf.build_feature_matrix(raw_train)
    labels = matrix[ID_COLS + bf.LABEL_COLS]
    raw = raw_train.merge(labels, on=ID_COLS, how="inner", validate="one_to_one")
    raw = raw.sort_values(ID_COLS, kind="stable").reset_index(drop=True)
    matrix = matrix.sort_values(ID_COLS, kind="stable").reset_index(drop=True)
    if not raw[ID_COLS].equals(matrix[ID_COLS]):
        raise ValueError("Raw table and feature matrix are not row-aligned.")
    return raw, matrix


def load_selected_features(path=SELECTED_FEATURES_FILE, available=None):
    """Features from the redundancy-filtered proposal (None if the file is missing)."""
    path = Path(path)
    if not path.is_file():
        return None
    names = pd.read_csv(path, index_col=0).index.tolist()
    if available is not None:
        missing = [n for n in names if n not in set(available)]
        if missing:
            raise ValueError(f"Proposal lists features that are not in the matrix: {missing[:5]}")
    return names


# ----------------------------------------------------------------------------- models
def make_logistic_regression():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def make_tree_model():
    # early_stopping=False keeps the run deterministic and uses all training rows for fitting.
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        learning_rate=0.1,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def _fit_predict(model, X_train, y_train, X_val):
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def run_sklearn_models(raw: pd.DataFrame, matrix: pd.DataFrame, selected_features=None) -> dict:
    """Train on train engines, return {model name: probability of failure for EVERY validation row}."""
    train_mask = (raw["split"] == "train").to_numpy()
    val_mask = (raw["split"] == "val").to_numpy()
    y_train = raw.loc[train_mask, "target"].to_numpy()
    eng_cols = bf.feature_columns(matrix)

    probs = {"dummy_always_0": np.zeros(int(val_mask.sum()))}
    probs["lr_raw24"] = _fit_predict(
        make_logistic_regression(), raw.loc[train_mask, RAW_FEATURES], y_train, raw.loc[val_mask, RAW_FEATURES]
    )
    probs["lr_eng242"] = _fit_predict(
        make_logistic_regression(), matrix.loc[train_mask, eng_cols], y_train, matrix.loc[val_mask, eng_cols]
    )
    probs["hgb_eng242"] = _fit_predict(
        make_tree_model(), matrix.loc[train_mask, eng_cols], y_train, matrix.loc[val_mask, eng_cols]
    )
    if selected_features:
        probs["hgb_eng166"] = _fit_predict(
            make_tree_model(), matrix.loc[train_mask, selected_features], y_train,
            matrix.loc[val_mask, selected_features],
        )
    return probs


def gru_probabilities(raw: pd.DataFrame) -> np.ndarray:
    """Retrain the team's GRU prototype and return probabilities for the eligible validation rows.

    PyTorch is imported here (not at module import) so the rest of the script works without it.
    Reuses the model class, sequence builder and settings from src/modeling/train_gru.py.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.modeling import train_gru as tg

    random.seed(tg.RANDOM_STATE)
    np.random.seed(tg.RANDOM_STATE)
    torch.manual_seed(tg.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tg.RANDOM_STATE)
    print(f"GRU training device: {device}")

    train = raw[raw["split"] == "train"]
    val = raw[raw["split"] == "val"]

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_train_rows = scaler.fit_transform(imputer.fit_transform(train[tg.FEATURE_COLS]))
    X_val_rows = scaler.transform(imputer.transform(val[tg.FEATURE_COLS]))

    X_train, y_train = tg.make_sequences(train, X_train_rows, tg.SEQUENCE_LENGTH)
    X_val, y_val = tg.make_sequences(val, X_val_rows, tg.SEQUENCE_LENGTH)

    # Sequence i corresponds to the i-th validation row (in engine, cycle order) with cycle >= SEQUENCE_LENGTH.
    eligible = val[val[CYCLE_COL] >= tg.SEQUENCE_LENGTH]
    if len(eligible) != len(X_val) or not np.array_equal(eligible["target"].to_numpy(), y_val.astype(int)):
        raise ValueError("GRU sequences do not line up with the eligible validation rows.")

    model = tg.GRUClassifier(input_size=len(tg.FEATURE_COLS), hidden_size=tg.HIDDEN_SIZE).to(device)
    pos = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - pos) / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=tg.LEARNING_RATE)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=tg.BATCH_SIZE, shuffle=True, num_workers=0,
    )
    for epoch in range(tg.EPOCHS):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            seen += len(yb)
        print(f"GRU epoch {epoch + 1:02d}/{tg.EPOCHS} - loss: {total / seen:.4f}")

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y_true, prob, threshold: float = THRESHOLD) -> dict:
    """Threshold metrics plus threshold-free ranking metrics (PR-AUC is the honest one under imbalance)."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def build_prediction_table(raw: pd.DataFrame, probs: dict, gru_prob=None) -> pd.DataFrame:
    """One row per validation engine-cycle with each model's probability (NaN where a model has no score)."""
    val = raw[raw["split"] == "val"][ID_COLS + ["rul", "target"]].reset_index(drop=True)
    val["eligible"] = val[CYCLE_COL] >= SEQUENCE_LENGTH
    for name, p in probs.items():
        val[f"p_{name}"] = np.asarray(p, dtype=float)
    if gru_prob is not None:
        col = np.full(len(val), np.nan)
        col[val["eligible"].to_numpy()] = np.asarray(gru_prob, dtype=float)
        val["p_gru_raw24"] = col
    return val


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every model on the shared eligible rows."""
    eligible = predictions[predictions["eligible"]]
    rows = []
    for col in [c for c in predictions.columns if c.startswith("p_")]:
        name = col[2:]
        if eligible[col].isna().any():
            raise ValueError(f"{name} has no score for some eligible rows.")
        row = {"model": name, "label": MODEL_LABELS.get(name, name), "rows_scored": len(eligible)}
        row.update(compute_metrics(eligible["target"], eligible[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def alert_summary(predictions: pd.DataFrame, prob_col: str, threshold: float = THRESHOLD,
                  horizon: int = bf.HORIZON) -> pd.DataFrame:
    """Per validation engine: when did the model first alert, relative to the failure window?

    lead_time_cycles = RUL at the EARLIEST alert inside the label window (how many cycles of warning the
    engine actually got). early_alert_rows = alerts raised while RUL > horizon (false alarms).
    """
    rows = []
    eligible = predictions[predictions["eligible"]].sort_values(ID_COLS)
    for unit, g in eligible.groupby(MACHINE_COL, sort=True):
        alerts = g[g[prob_col] >= threshold]
        in_window = alerts[alerts["rul"] <= horizon]
        early = alerts[alerts["rul"] > horizon]
        rows.append({
            MACHINE_COL: unit,
            "first_alert_rul": alerts["rul"].max() if len(alerts) else np.nan,
            "lead_time_cycles": in_window["rul"].max() if len(in_window) else np.nan,
            "missed": len(in_window) == 0,
            "early_alert_rows": len(early),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def format_report(metrics: pd.DataFrame, predictions: pd.DataFrame, n_engineered: int, n_selected, with_gru: bool) -> str:
    eligible = predictions[predictions["eligible"]]
    n_pos = int(eligible["target"].sum())
    prevalence = n_pos / len(eligible)
    ordered = metrics.sort_values("f1", ascending=False)

    lines = [
        "# Model Comparison — NASA C-MAPSS FD001 (validation)",
        "",
        "Generated by `python -m src.modeling.compare_models`. Tables are produced by code; read them with the notes below.",
        "",
        "## Evaluation setup",
        "",
        f"- Target: failure within the next {bf.HORIZON} operational cycles (cycles, not hours; horizon pending team approval)",
        "- Split: engine-disjoint 70% train / 15% validation / 15% test (team `temporal_train_val_test_split`)",
        f"- Scored rows: validation cycles >= {SEQUENCE_LENGTH} ({len(eligible)} rows, "
        f"{eligible[MACHINE_COL].nunique()} engines), so row models and the GRU use identical rows",
        f"- Class balance on scored rows: {n_pos} positive / {len(eligible) - n_pos} negative "
        f"({prevalence:.1%} positive); an all-negative predictor already scores {1 - prevalence:.1%} accuracy",
        f"- Threshold: {THRESHOLD} for every model, not tuned. Test split not used.",
        "- Imputation and scaling fitted on training rows only; trees use scikit-learn HistGradientBoosting "
        "(interim stand-in for the XGBoost/LightGBM model).",
        f"- Engineered features: {n_engineered} columns"
        + (f"; filtered proposal: {n_selected}" if n_selected else ""),
    ]
    if not with_gru:
        lines.append("- GRU: not re-run in this report (use `--with-gru`); see `reports/baseline_gru_comparison.md` for the team's run.")
    lines += [
        "",
        "## Validation metrics (sorted by F1)",
        "",
        "| Model | Precision | Recall | F1 | Accuracy | Balanced acc. | PR-AUC | ROC-AUC | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ordered.iterrows():
        lines.append(
            f"| {r['label']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        )
    lines += [
        "",
        "## Imbalance handling",
        "",
        "| Model | Handling |",
        "|---|---|",
    ] + [f"| {MODEL_LABELS.get(m, m)} | {IMBALANCE_HANDLING.get(m, '-')} |" for m in ordered["model"]]
    lines += [
        "",
        "## Notes",
        "",
        "- Accuracy is not a useful ranking metric here; use precision, recall, F1 and PR-AUC.",
        "- All metrics are on one validation split of 15 engines; differences of a few percent are within noise.",
        "- Per-row probabilities are in `reports/model_predictions_val.csv` for false-positive / false-negative inspection.",
        "- GPU/CPU training of the GRU is not bit-for-bit reproducible.",
        "",
    ]
    return "\n".join(lines)


def run(raw_dir=RAW_DIR, reports_dir=REPORTS_DIR, with_gru: bool = False,
        selected_features_file=SELECTED_FEATURES_FILE) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_train = load_cmapss_file(Path(raw_dir) / TRAIN_FILE)
    raw, matrix = prepare_data(raw_train)
    selected = load_selected_features(selected_features_file, available=matrix.columns)

    probs = run_sklearn_models(raw, matrix, selected)
    gru_prob = gru_probabilities(raw) if with_gru else None

    predictions = build_prediction_table(raw, probs, gru_prob)
    metrics = build_metrics_table(predictions)

    predictions.to_csv(reports_dir / PREDICTIONS_FILE.name, index=False)
    metrics.to_csv(reports_dir / METRICS_FILE.name, index=False)
    (reports_dir / REPORT_FILE.name).write_text(
        format_report(metrics, predictions, len(bf.feature_columns(matrix)),
                      len(selected) if selected else None, with_gru),
        encoding="utf-8",
    )
    return {"metrics": metrics, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on the FD001 validation split.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--with-gru", action="store_true", help="Also retrain the GRU (needs PyTorch).")
    args = parser.parse_args()

    result = run(args.raw_dir, args.reports_dir, with_gru=args.with_gru)
    shown = result["metrics"][["model", "precision", "recall", "f1", "pr_auc", "tp", "fp", "fn", "tn"]]
    print(shown.round(4).to_string(index=False))
    print(f"\nSaved: {Path(args.reports_dir) / REPORT_FILE.name}")


if __name__ == "__main__":
    main()
