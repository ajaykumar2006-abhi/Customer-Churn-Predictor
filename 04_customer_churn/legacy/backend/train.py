"""
Customer Churn Prediction - Training Pipeline
---------------------------------------------
Predicts whether a telecom customer will churn from account and service features.

Data is synthetic (no public dataset is downloaded), shaped after the IBM Telco
churn layout so the feature set transfers to the real dataset unchanged.
Categorical columns are encoded with explicit, ordered category maps that are
saved alongside the model; inference reuses those maps instead of refitting.
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORY_MAPS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = NUMERIC_FEATURES + list(CATEGORY_MAPS)


def create_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customers whose churn label depends on known risk factors."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, n_rows),
            "MonthlyCharges": rng.uniform(20, 120, n_rows),
            "TotalCharges": rng.uniform(100, 8000, n_rows),
        }
    )
    for column, options in CATEGORY_MAPS.items():
        df[column] = rng.choice(options, n_rows)

    risk = (
        0.35 * (df["Contract"] == "Month-to-month")
        + 0.20 * (df["PaymentMethod"] == "Electronic check")
        + 0.15 * (df["tenure"] < 12)
        + 0.10 * (df["InternetService"] == "Fiber optic")
        - 0.10 * (df["TechSupport"] == "Yes")
        + rng.normal(0, 0.08, n_rows)
    )
    df["Churn"] = (risk > 0.45).astype(int)

    logger.info("Generated %d customers, churn rate %.1f%%", n_rows, df["Churn"].mean() * 100)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Map each categorical column to integer codes using the fixed category order."""
    encoded = df.copy()
    for column, options in CATEGORY_MAPS.items():
        unknown = set(encoded[column].astype(str)) - set(options)
        if unknown:
            raise ValueError(f"Unknown values in {column}: {sorted(unknown)}")
        encoded[column] = pd.Categorical(encoded[column], categories=options).codes
    return encoded


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def train() -> dict:
    df = encode_categoricals(create_dataset())

    X = df[FEATURE_COLUMNS]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    best_name, best_model, best_auc = None, None, -1.0

    for name, candidate in build_models().items():
        candidate.fit(X_train, y_train)
        probabilities = candidate.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, candidate.predict(X_test))
        results[name] = {"roc_auc": round(float(auc), 4), "accuracy": round(float(accuracy), 4)}
        logger.info("%-20s accuracy=%.4f roc_auc=%.4f", name, accuracy, auc)

        if auc > best_auc:
            best_name, best_model, best_auc = name, candidate, auc

    logger.info("Selected %s (roc_auc=%.4f)", best_name, best_auc)

    with open(os.path.join(MODEL_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "feature_columns": FEATURE_COLUMNS,
                "numeric_features": NUMERIC_FEATURES,
                "category_maps": CATEGORY_MAPS,
            },
            f,
        )

    metrics = {
        "model_type": best_name,
        "roc_auc": round(float(best_auc), 4),
        "accuracy": results[best_name]["accuracy"],
        "candidates": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "feature_columns": FEATURE_COLUMNS,
        "note": "Trained on synthetic data; metrics do not transfer to production traffic.",
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()
