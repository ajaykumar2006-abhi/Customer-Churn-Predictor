"""Data loading, training, validation and inference for the customer churn model.

Dataset: IBM "Telco Customer Churn" (7,043 customers, one row per customer, label column
``Churn``). Place the CSV at ``data/Telco-Customer-Churn.csv`` (see README).

Leakage controls
----------------
* ``customerID`` is dropped and duplicate IDs are rejected.
* A 20% stratified hold-out (the *test set*) is split off first and is only scored once,
  after the model and threshold are fixed.
* Every learned step (median imputer, scaler, model weights) lives inside one sklearn
  ``Pipeline`` and is fitted on the development split only. Model choice and the decision
  threshold come from 5-fold out-of-fold predictions on the development split.
* The inference path loads the same fitted pipeline and feeds it raw, validated values in the
  training column order, so no encoding is re-implemented at serving time.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCHEMA_VERSION = 4

TARGET = "Churn"
ID_COLUMN = "customerID"

# Category lists are the exact values in the training CSV (Yes/No columns, plus the
# service-dependent "No internet service" value). They are fixed, documented schema.
CATEGORY_MAPS: dict[str, list[str]] = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
}
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = NUMERIC_FEATURES + list(CATEGORY_MAPS)

# Units: tenure in whole months; charges in US dollars. Limits are the ranges the
# dataset covers (tenure 0-72 months); anything outside is rejected, not guessed.
NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "tenure": (0, 72),
    "MonthlyCharges": (0.0, 150.0),
    "TotalCharges": (0.0, 10_000.0),
}
FEATURE_LABELS = {
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Contract": "Contract type",
    "PaymentMethod": "Payment method",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "TechSupport": "Tech support",
}

# Probability bands used for the Low / Moderate / High risk label (documented in the UI).
RISK_HIGH = 0.7
RISK_MEDIUM = 0.4


def risk_band(probability: float) -> str:
    if probability >= RISK_HIGH:
        return "high"
    if probability >= RISK_MEDIUM:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- data


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Read and check the raw CSV. Returns FEATURE_COLUMNS + TARGET, one row per customer."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Customer Churn CSV and save it there."
        )
    raw = pd.read_csv(path)
    required = [ID_COLUMN, TARGET, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if raw[ID_COLUMN].duplicated().any():
        raise ValueError("Dataset contains duplicate customerID values; refusing to train.")

    frame = raw[required].copy()
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    bad_labels = set(frame[TARGET]) - {"Yes", "No"}
    if bad_labels:
        raise ValueError(f"Churn must be Yes/No; found {sorted(bad_labels)}")
    frame[TARGET] = (frame[TARGET] == "Yes").astype(int)

    # TotalCharges is text in the source file; blanks are customers with no billed months yet.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].astype(str).str.strip(), errors="coerce")
    for column in CATEGORY_MAPS:
        frame[column] = frame[column].astype(str).str.strip()
    for column, options in CATEGORY_MAPS.items():
        unknown = set(frame[column]) - set(options)
        if unknown:
            raise ValueError(f"Unknown values in {column}: {sorted(unknown)}")
    return frame[FEATURE_COLUMNS + [TARGET]].reset_index(drop=True)


def audit_dataset(raw_path: str = DATA_PATH) -> dict[str, Any]:
    """Data-quality facts saved with the metrics so the report can state them."""
    raw = pd.read_csv(raw_path)
    frame = load_dataset(raw_path)
    churn = frame[TARGET]
    return {
        "source_file": os.path.basename(raw_path),
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "duplicate_customer_ids": int(raw[ID_COLUMN].duplicated().sum()),
        "duplicate_feature_rows": int(frame.duplicated().sum()),
        "missing_values_total": int(raw.isna().sum().sum()),
        "blank_total_charges": int(
            raw["TotalCharges"].astype(str).str.strip().eq("").sum()
        ),
        "churn_yes": int(churn.sum()),
        "churn_no": int((1 - churn).sum()),
        "churn_rate": round(float(churn.mean()), 4),
        "target_column": TARGET,
        "id_column_excluded": ID_COLUMN,
    }


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. The 20% test part is returned untouched for one final score."""
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_dev, X_test, y_dev, y_test


# ------------------------------------------------------------------- pipeline


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        categories=[CATEGORY_MAPS[column] for column in CATEGORY_MAPS],
        handle_unknown="error",
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, list(CATEGORY_MAPS)),
        ]
    )


def candidate_models() -> dict[str, Any]:
    """Fresh, unfitted estimators. Class weights address the 26% churn share."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED,
        ),
    }


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Pick the probability cut-off that maximises F1 on out-of-fold predictions."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    observed, predicted_mean = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    prevalence = float(np.mean(y_true))
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": round(prevalence, 4),
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "pr_auc_baseline": round(prevalence, 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "brier_baseline": round(float(brier_score_loss(y_true, np.full(len(y_true), prevalence))), 4),
        "log_loss": round(float(log_loss(y_true, clipped, labels=[0, 1])), 4),
        "log_loss_baseline": round(float(log_loss(y_true, np.full(len(y_true), prevalence), labels=[0, 1])), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": [
            {"predicted": round(float(p), 3), "observed": round(float(o), 3)}
            for p, o in zip(predicted_mean, observed)
        ],
    }


def reliability_bins(y_true: Any, probabilities: Any, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width reliability bins on [0, 1]. Empty bins are kept with count 0 and no means."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        bins.append(
            {
                "lower": round(b / n_bins, 2),
                "upper": round((b + 1) / n_bins, 2),
                "count": count,
                "predicted": round(float(p[mask].mean()), 4) if count else None,
                "observed": round(float(y[mask].mean()), 4) if count else None,
            }
        )
    return bins


def reliability_report(y_true: Any, raw: Any, calibrated: Any, n_bins: int = 10) -> dict[str, Any]:
    """Raw and calibrated reliability on the same labelled rows, with their Brier scores."""
    y = np.asarray(y_true, dtype=float)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "brier_raw": round(float(brier_score_loss(y, raw)), 4),
        "brier_calibrated": round(float(brier_score_loss(y, calibrated)), 4),
        "raw": reliability_bins(y, raw, n_bins),
        "calibrated": reliability_bins(y, calibrated, n_bins),
    }


SMALL_BIN_SIZE = 30


def reliability_points(report: dict[str, Any]) -> pd.DataFrame:
    """Chart rows for the raw and calibrated curves. Empty bins are dropped; small bins are flagged."""
    rows = []
    for series, key in (("Raw model", "raw"), ("Calibrated", "calibrated")):
        for index, item in enumerate(report[key], start=1):
            if item["count"] == 0:
                continue
            rows.append(
                {
                    "Series": series,
                    "Bin": index,
                    "Predicted": item["predicted"],
                    "Observed": item["observed"],
                    "Customers": item["count"],
                    "Small bin": item["count"] < SMALL_BIN_SIZE,
                }
            )
    return pd.DataFrame(rows, columns=["Series", "Bin", "Predicted", "Observed", "Customers", "Small bin"])


# ------------------------------------------------------------------ calibration

CALIBRATION_METHODS = ("sigmoid", "isotonic")
CALIBRATION_LABELS = {
    "none": "Not calibrated (the raw model scores were already the best of the options tested)",
    "sigmoid": "Calibrated with Platt scaling (sigmoid curve)",
    "isotonic": "Calibrated with isotonic regression",
}
_EPS = 1e-6


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1.0 - probabilities))


def _clip(scores: Any) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), _EPS, 1.0 - _EPS)


class Calibrator:
    """Maps raw model scores to calibrated churn probabilities.

    ``sigmoid`` is Platt scaling: a logistic curve fitted to the logit of the score.
    ``isotonic`` is a monotone step function fitted without a shape assumption. Both are fitted
    only on out-of-fold development scores, so no test label is ever used.
    """

    def __init__(self, method: str):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._fitted: Any = None

    def fit(self, scores: Any, labels: Any) -> "Calibrator":
        scores = _clip(scores)
        labels = np.asarray(labels).astype(int)
        if self.method == "sigmoid":
            self._fitted = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), labels)
        else:
            self._fitted = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, labels)
        return self

    def transform(self, scores: Any) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("The calibrator has not been fitted.")
        scores = _clip(scores)
        if self.method == "sigmoid":
            return self._fitted.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return np.asarray(self._fitted.predict(scores), dtype=float)


def cross_calibrated(method: str, scores: np.ndarray, labels: pd.Series, folds: Any) -> np.ndarray:
    """Out-of-fold calibrated probabilities: each row is calibrated by a calibrator that never saw its label."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if method == "none":
        return scores.copy()
    out = np.zeros(len(scores))
    for train_idx, test_idx in folds.split(scores, labels):
        calibrator = Calibrator(method).fit(scores[train_idx], labels[train_idx])
        out[test_idx] = calibrator.transform(scores[test_idx])
    return out


def select_calibration(scores: np.ndarray, labels: pd.Series, folds: Any) -> tuple[str, dict[str, float]]:
    """Choose the calibration with the lowest out-of-fold Brier score on the development split.

    "none" is listed first, so a calibrator is only used when it strictly beats the raw scores.
    """
    briers = {}
    for method in ("none", *CALIBRATION_METHODS):
        calibrated = cross_calibrated(method, scores, labels, folds)
        briers[method] = float(brier_score_loss(labels, calibrated))
    best = min(briers, key=briers.get)
    return best, {method: round(value, 4) for method, value in briers.items()}


def calibration_label(bundle: dict[str, Any]) -> str:
    return CALIBRATION_LABELS[bundle.get("calibration_method", "none")]


def train_model(data_path: str = DATA_PATH) -> dict[str, Any]:
    """Compare candidates with CV on the development split, fit the winner once, score the test split once."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    audit = audit_dataset(data_path)
    frame = load_dataset(data_path)
    X_dev, X_test, y_dev, y_test = split_data(frame)

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    estimators = candidate_models()
    candidates: dict[str, dict[str, float]] = {}
    oof_by_name: dict[str, np.ndarray] = {}
    for name, estimator in estimators.items():
        oof = cross_val_predict(
            build_pipeline(clone(estimator)), X_dev, y_dev, cv=folds, method="predict_proba"
        )[:, 1]
        oof_by_name[name] = oof
        candidates[name] = {
            "cv_roc_auc": round(float(roc_auc_score(y_dev, oof)), 4),
            "cv_pr_auc": round(float(average_precision_score(y_dev, oof)), 4),
        }

    best_name = max(candidates, key=lambda name: candidates[name]["cv_roc_auc"])
    best_oof = oof_by_name[best_name]

    # Calibration is chosen and fitted on out-of-fold development scores only. The test set is untouched.
    cal_folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED + 1)
    calibration_method, calibration_briers = select_calibration(best_oof, y_dev, cal_folds)
    raw_threshold = select_threshold(y_dev, best_oof)
    dev_calibrated = cross_calibrated(calibration_method, best_oof, y_dev, cal_folds)
    threshold = select_threshold(y_dev, dev_calibrated)

    final = build_pipeline(estimators[best_name])
    final.fit(X_dev, y_dev)
    background_mean = final.named_steps["preprocess"].transform(X_dev).mean(axis=0)
    calibrator = Calibrator(calibration_method).fit(best_oof, y_dev) if calibration_method != "none" else None

    # The test set is scored exactly once per variant, after every choice above is fixed.
    raw_test = final.predict_proba(X_test)[:, 1]
    test_uncalibrated = evaluate(y_test, raw_test, raw_threshold)
    calibrated_test = calibrator.transform(raw_test) if calibrator is not None else raw_test
    test_metrics = evaluate(y_test, calibrated_test, threshold)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": final,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "category_maps": CATEGORY_MAPS,
        "threshold": threshold,
        "background_mean": background_mean,
        "calibration_method": calibration_method,
        "calibrator": calibrator,
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(bundle, handle)

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": best_name,
        "decision_threshold": round(threshold, 2),
        "threshold_rule": "Maximises F1 on out-of-fold, calibrated probabilities of the development split.",
        "calibration": {
            "method": calibration_method,
            "label": CALIBRATION_LABELS[calibration_method],
            "rule": (
                "Chosen by the lowest out-of-fold Brier score on the development split. "
                "A calibrator is used only if it beats the raw scores."
            ),
            "dev_cv_brier": calibration_briers,
            "fitted_on": "Out-of-fold development scores; the test set was never used to fit or choose it.",
        },
        "selection_rule": f"Highest {CV_FOLDS}-fold CV ROC-AUC on the development split.",
        "split": {
            "development_rows": int(len(y_dev)),
            "test_rows": int(len(y_test)),
            "test_size": TEST_SIZE,
            "stratified": True,
            "seed": SEED,
        },
        "data": audit,
        "candidates": candidates,
        "dev_reliability": reliability_report(y_dev, best_oof, dev_calibrated),
        "test": test_metrics,
        "test_uncalibrated": test_uncalibrated,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_limits": NUMERIC_LIMITS,
        "category_maps": CATEGORY_MAPS,
        "note": (
            "Evaluated on a 20% stratified hold-out. Estimates describe this public telco dataset; "
            "they are not guaranteed for other businesses or future customers."
        ),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    load_model.cache_clear()
    return metrics


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any] | None:
    """Load the saved bundle once per process; None when the model has not been trained."""
    if not model_exists():
        return None
    with open(MODEL_PATH, "rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Saved model is from an older version. Press Train model to rebuild it.")
    return bundle


def read_metrics() -> dict[str, Any] | None:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------ validation


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_customer(customer: dict[str, Any]) -> str | None:
    """Return the first friendly problem with the submitted values, or None when usable."""
    labels = {**FEATURE_LABELS}
    for field in NUMERIC_FEATURES:
        number = _as_number(customer.get(field))
        if number is None:
            return f"{labels[field]} is required and must be a number."
        low, high = NUMERIC_LIMITS[field]
        if not low <= number <= high:
            unit = "months" if field == "tenure" else "dollars"
            return (
                f"{labels[field]} must be between {low:g} and {high:g} {unit} "
                f"(you entered {number:g})."
            )
    tenure = _as_number(customer["tenure"])
    if tenure is not None and tenure != int(tenure):
        return "Tenure must be a whole number of months."
    for field, options in CATEGORY_MAPS.items():
        value = customer.get(field)
        if value is None or str(value) not in options:
            return f"{labels[field]} must be one of: {', '.join(options)}."
    return None


def encode_customer(customer: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    """Build the one-row raw frame the saved pipeline expects, in training column order."""
    problem = validate_customer(customer)
    if problem:
        raise ValueError(problem)
    row: dict[str, Any] = {}
    for field in NUMERIC_FEATURES:
        row[field] = float(customer[field])
    for field in bundle["category_maps"]:
        row[field] = str(customer[field])
    return pd.DataFrame([row], columns=bundle["feature_columns"])


# ----------------------------------------------------------------- explanation


def _feature_of(column_name: str) -> str | None:
    """Map a transformed column such as 'cat__Contract_One year' back to 'Contract'."""
    _, _, rest = column_name.partition("__")
    for feature in FEATURE_COLUMNS:
        if rest == feature or rest.startswith(feature + "_"):
            return feature
    return None


def make_explainer(bundle: dict[str, Any]) -> Any:
    """Build the SHAP TreeExplainer once. Returns None for logistic regression, which needs no explainer."""
    model = bundle["pipeline"].named_steps["model"]
    if isinstance(model, LogisticRegression):
        return None
    return shap.TreeExplainer(model)


def _factor_matrix(names: Any) -> np.ndarray:
    """0/1 matrix mapping each transformed column to the input feature it came from."""
    membership = np.zeros((len(names), len(FEATURE_COLUMNS)))
    for row, name in enumerate(names):
        feature = _feature_of(name)
        if feature is not None:
            membership[row, FEATURE_COLUMNS.index(feature)] = 1.0
    return membership


def explain_frame(
    bundle: dict[str, Any], frame: pd.DataFrame, top: int = 5, explainer: Any = None
) -> list[list[dict[str, Any]]]:
    """Per-row contribution of each input towards churn, for a whole frame in one pass.

    Logistic regression uses its exact linear decomposition (coef * (x - training mean));
    tree models use SHAP TreeExplainer. Contributions are in model units, so only their sign
    and rank are shown to the user. Pass a shared ``explainer`` from make_explainer to avoid
    rebuilding it for every chunk.
    """
    pipeline = bundle["pipeline"]
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = np.asarray(preprocess.transform(frame), dtype=float)
    names = preprocess.get_feature_names_out()

    if isinstance(model, LogisticRegression):
        values = model.coef_[0] * (transformed - np.asarray(bundle["background_mean"]))
    else:
        if explainer is None:
            explainer = make_explainer(bundle)
        raw = explainer.shap_values(transformed, check_additivity=False)
        if isinstance(raw, list):
            raw = raw[1]
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw[:, :, 1]
        values = raw

    grouped = values @ _factor_matrix(names)
    results = []
    for row_index in range(len(frame)):
        row = frame.iloc[row_index]
        factors = []
        for feature, contribution in zip(FEATURE_COLUMNS, grouped[row_index]):
            contribution = float(contribution)
            if abs(contribution) < 1e-9:
                continue
            value = row[feature]
            factors.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "value": format_value(feature, value),
                    "raw": float(value) if feature == "tenure" else None,
                    "contribution": round(contribution, 4),
                    "direction": "raises" if contribution > 0 else "lowers",
                }
            )
        factors.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        results.append(factors[:top])
    return results


def explain(bundle: dict[str, Any], frame: pd.DataFrame, top: int = 5) -> list[dict[str, Any]]:
    """Explanation for the first row of ``frame`` (single-customer prediction)."""
    return explain_frame(bundle, frame.iloc[:1], top)[0]


def format_value(feature: str, value: Any) -> str:
    if feature == "tenure":
        return f"{int(float(value))} months"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"${float(value):,.2f}"
    return str(value)


# ------------------------------------------------------------------ inference


def predict_probabilities(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Calibrated churn probabilities for a frame of raw feature rows (shared by single and batch scoring)."""
    raw = bundle["pipeline"].predict_proba(frame)[:, 1]
    calibrator = bundle.get("calibrator")
    return calibrator.transform(raw) if calibrator is not None else raw


def predict(customer: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one customer with the saved pipeline. Raises ValueError for unusable input."""
    bundle = bundle if bundle is not None else load_model()
    if bundle is None:
        raise RuntimeError("No trained model found. Train the model first.")

    frame = encode_customer(customer, bundle)
    uncalibrated = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    probability = float(predict_probabilities(bundle, frame)[0])
    threshold = float(bundle["threshold"])
    return {
        "churn": "Yes" if probability >= threshold else "No",
        "churn_probability": probability,
        "uncalibrated_probability": uncalibrated,
        "calibration": calibration_label(bundle),
        "retention_probability": 1.0 - probability,
        "risk_level": risk_band(probability),
        "threshold": threshold,
        "model": bundle["model_name"],
        "factors": explain(bundle, frame),
    }
