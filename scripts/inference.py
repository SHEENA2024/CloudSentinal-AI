
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Locate the project and saved model artifacts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "cloudsentinel_isolation_forest_v3.joblib"
BASELINE_PATH = MODEL_DIR / "cloudsentinel_resource_baselines_v3.joblib"
CONFIG_PATH = MODEL_DIR / "cloudsentinel_v3_inference_config.joblib"


def load_artifacts():
    """Load the trained model, resource baselines, and configuration."""
    for path in [MODEL_PATH, BASELINE_PATH, CONFIG_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    model = joblib.load(MODEL_PATH)
    resource_stats = joblib.load(BASELINE_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, resource_stats, config


def predict_anomalies(metrics):
    """
    Predict anomalies in new cloud-resource metric records.

    Accepts a pandas DataFrame or a list of dictionaries.
    Returns the original records with prediction columns.
    """
    if isinstance(metrics, dict):
        data = pd.DataFrame([metrics])
    elif isinstance(metrics, pd.DataFrame):
        data = metrics.copy()
    else:
        data = pd.DataFrame(metrics)

    if data.empty:
        raise ValueError("Input data contains no records.")

    model, resource_stats, config = load_artifacts()

    original_features = config["original_features"]
    baseline_features = config["baseline_features"]
    feature_columns = config["feature_columns"]

    required_columns = ["resource_id"] + original_features
    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["resource_id"] = data["resource_id"].astype(str)

    # Each resource must have a baseline learned during training.
    known_resources = set(resource_stats.index.astype(str))
    unknown_resources = sorted(
        set(data["resource_id"]) - known_resources
    )

    if unknown_resources:
        raise ValueError(
            "No training baseline exists for resource(s): "
            f"{unknown_resources}"
        )

    # Validate and convert model input metrics to numeric values.
    for col in original_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[original_features].isna().any().any():
        raise ValueError(
            "Input metrics contain missing or non-numeric values."
        )

    if not np.isfinite(data[original_features].to_numpy()).all():
        raise ValueError("Input metrics contain infinite values.")

    # Recreate the same deviation features used during training.
    for col in baseline_features:
        means = data["resource_id"].map(
            resource_stats[(col, "mean")]
        )
        stds = data["resource_id"].map(
            resource_stats[(col, "std")]
        )

        data[f"{col}_deviation"] = (
            data[col] - means
        ) / (stds + 1e-6)

    X = data[feature_columns]

    # Isolation Forest: -1 means anomaly; 1 means normal.
    raw_predictions = model.predict(X)
    data["is_anomaly_prediction"] = (
        raw_predictions == -1
    ).astype(int)

    data["prediction_label"] = np.where(
        data["is_anomaly_prediction"] == 1,
        "anomaly",
        "normal"
    )

    # Higher values indicate a more anomalous observation.
    data["anomaly_score"] = -model.decision_function(X)

    return data
