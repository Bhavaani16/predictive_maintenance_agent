"""
tools/train_model.py
─────────────────────
FabGuardian – Model Training Pipeline

Trains two complementary models on the UCI Machine Predictive Maintenance
dataset (or any CSV with the same schema) and saves them to models/.

Usage:
    python tools/train_model.py                           # use default dataset
    python tools/train_model.py --data path/to/data.csv  # custom dataset
    python tools/train_model.py --evaluate                # print full eval report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent.parent
_DATA_PATH  = _ROOT / "data" / "predictive_maintenance.csv"
_MODELS_DIR = _ROOT / "models"

FEATURES = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Power_kW",
    "Vibration_mms",
    "Type_enc",
]
TARGET = "Machine_failure"


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(data_path: Path, evaluate: bool = True) -> None:
    logger.info("Loading dataset: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset shape: %s | Failure rate: %.2f%%",
                df.shape, df[TARGET].mean() * 100)

    # --- Feature engineering ---
    le = LabelEncoder()
    df["Type_enc"] = le.fit_transform(df["Type"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model 1: IsolationForest (unsupervised) ---
    logger.info("Training IsolationForest on healthy samples only ...")
    healthy_train = X_train[y_train == 0]
    iso = Pipeline([
        ("scaler", StandardScaler()),
        ("iso",    IsolationForest(n_estimators=200, contamination=0.05, random_state=42)),
    ])
    iso.fit(healthy_train)

    # Compute score-based thresholds from known-failure test samples
    iso_scores_failure = iso.score_samples(X_test[y_test == 1])
    high_threshold   = float(np.percentile(iso_scores_failure, 25))
    medium_threshold = float(np.percentile(iso_scores_failure, 60))
    logger.info("IsolationForest thresholds → HIGH: %.4f | MEDIUM: %.4f",
                high_threshold, medium_threshold)

    # --- Model 2: RandomForestClassifier (supervised) ---
    logger.info("Training RandomForestClassifier ...")
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    rf.fit(X_train, y_train)

    if evaluate:
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        logger.info("\n%s", classification_report(y_test, y_pred))
        logger.info("ROC-AUC: %.4f", auc)
    else:
        y_prob = rf.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)

    # --- Save artefacts ---
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(iso, _MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(rf,  _MODELS_DIR / "random_forest.joblib")
    joblib.dump(le,  _MODELS_DIR / "label_encoder.joblib")

    meta = {
        "features":               FEATURES,
        "dataset":                str(data_path.name),
        "total_samples":          len(df),
        "failure_rate_pct":       round(float(y.mean() * 100), 2),
        "train_samples":          len(X_train),
        "test_samples":           len(X_test),
        "iso_score_thresholds":   {
            "HIGH":   round(high_threshold, 4),
            "MEDIUM": round(medium_threshold, 4),
        },
        "rf_roc_auc":             round(auc, 4),
    }
    with open(_MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Models and metadata saved to %s", _MODELS_DIR)
    logger.info("Training complete. RF ROC-AUC = %.4f", auc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FabGuardian failure prediction models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=_DATA_PATH,
        help="Path to the training CSV (default: data/predictive_maintenance.csv)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Print full classification report after training (default: True)",
    )
    args = parser.parse_args()
    train(data_path=args.data, evaluate=args.evaluate)
