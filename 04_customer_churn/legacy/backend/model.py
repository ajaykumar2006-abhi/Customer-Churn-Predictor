"""Model loading and churn inference for the customer churn API."""

import json
import logging
import os
import pickle
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


class ChurnModel:
    """Wraps the trained classifier plus the encodings it was trained with."""

    def __init__(self) -> None:
        self.model = None
        self.feature_columns: List[str] = []
        self.category_maps: Dict[str, List[str]] = {}
        self.metrics: Dict = {}
        self.model_name = "unknown"
        self.loaded = False

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            logger.error("Model file not found: %s", MODEL_PATH)
            return False

        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
        except Exception as exc:  # corrupt or incompatible pickle
            logger.error("Could not load model: %s", exc)
            return False

        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.category_maps = bundle["category_maps"]
        self.model_name = bundle.get("model_name", "unknown")
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                self.metrics = json.load(f)
        self.loaded = True
        logger.info("Loaded %s from %s", self.model_name, MODEL_PATH)
        return True

    def _encode(self, customer: Dict) -> pd.DataFrame:
        row = {}
        for column in self.feature_columns:
            if column in self.category_maps:
                options = self.category_maps[column]
                value = str(customer[column])
                if value not in options:
                    raise ValueError(f"{column} must be one of {options}")
                row[column] = options.index(value)
            else:
                row[column] = float(customer[column])
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, customer: Dict) -> Dict:
        if not self.loaded and not self.load():
            return {"error": "Model not available"}

        frame = self._encode(customer)
        probability = float(self.model.predict_proba(frame)[0][1])
        label = "Yes" if probability >= 0.5 else "No"
        if probability >= 0.7:
            risk = "high"
        elif probability >= 0.4:
            risk = "medium"
        else:
            risk = "low"

        return {
            "churn": label,
            "churn_probability": round(probability, 4),
            "retention_probability": round(1 - probability, 4),
            "risk_level": risk,
            "model": self.model_name,
        }

    def get_metrics(self) -> Dict:
        return self.metrics


model = ChurnModel()


def get_model() -> ChurnModel:
    if not model.loaded:
        model.load()
    return model
