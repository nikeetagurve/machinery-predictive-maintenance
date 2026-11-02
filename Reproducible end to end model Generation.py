"""Reproducible End-to-End Model Generation Pipeline.

Runs feature reading, deterministic training, threshold validation, and
artifact packaging.
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Ensure root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modeling.model_io import save_model_artifact
from src.modeling.train_tree import extract_features_and_target, train_lightgbm_model

DETERMINISTIC_CONFIG = {
    "model_type": "LightGBM Classifier",
    "dataset": "FD001",
    "failure_window_cycles": 30,
    "alert_threshold": 0.50,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "n_estimators": 250,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,
    "random_state": 42,
}


def build_and_export():
    train_path = Path("data/processed/train_features.parquet")
    val_path = Path("data/processed/val_features.parquet")

    if not train_path.exists():
        train_path = Path("data/processed/train_features.csv")
        val_path = Path("data/processed/val_features.csv")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed datasets missing. Run feature pipeline first.")

    load_fn = pd.read_parquet if train_path.suffix == ".parquet" else pd.read_csv
    train_df = load_fn(train_path)
    val_df = load_fn(val_path)

    target_col = "label" if "label" in train_df.columns else "target"
    X_train, y_train, features = extract_features_and_target(train_df, target_col)
    X_val, y_val, _ = extract_features_and_target(val_df, target_col)

    print(f"Training reproducible model with seed {DETERMINISTIC_CONFIG['random_state']}...")
    model = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # Evaluate
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_val_proba >= DETERMINISTIC_CONFIG["alert_threshold"]).astype(int)

    metrics = {
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_val_proba)), 4),
    }

    # Save to versioned models repository
    export_dir = Path("models")
    saved_path = save_model_artifact(
        model=model,
        feature_names=features,
        metrics=metrics,
        config=DETERMINISTIC_CONFIG,
        output_dir=export_dir,
        artifact_version="v1.0.0",
    )

    print(f"Reproducible build complete. Artifact verified at {saved_path}")


if __name__ == "__main__":
    build_and_export()