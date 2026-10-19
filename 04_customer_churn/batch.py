"""Batch scoring: read an uploaded customer file, validate every row, and score the valid rows.

Every row is checked on its own. A row that fails any check is listed in the validation errors
with its row number and is never scored, so each prediction stays aligned with the input row
that produced it. Uploaded data is processed in memory in this process only and is not stored.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

import model as churn_model
from model import CATEGORY_MAPS, FEATURE_COLUMNS, FEATURE_LABELS, ID_COLUMN, NUMERIC_FEATURES, NUMERIC_LIMITS
from utils import PRESETS, RISK_LABELS

MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS = 20_000
CHUNK_ROWS = 2_000
SUPPORTED_SUFFIXES = (".csv", ".xlsx")
RISK_ORDER = ["High", "Moderate", "Low"]

SCORED_COLUMNS = [
    "row_number",
    "customer_id",
    "churn_prediction",
    "churn_probability",
    "risk_level",
    "validation_status",
    "main_factors",
]
ERROR_COLUMNS = ["row_number", "customer_id", "problem"]

# Documented meaning of each output column (shown in the app and the README).
OUTPUT_COLUMN_GUIDE = {
    "row_number": "Position of the customer in the uploaded file, counting the first data row as 1.",
    "customer_id": "The customerID from the upload, or blank when the file has no customerID column.",
    "churn_prediction": "Yes when the calibrated probability reaches the decision threshold, otherwise No.",
    "churn_probability": "Calibrated estimate of churn probability between 0 and 1. A model estimate, not a guarantee.",
    "risk_level": "Low, Moderate or High, using the same probability bands as single-customer predictions.",
    "validation_status": "Scored for every row in this file. Rows that fail validation are listed separately.",
    "main_factors": (
        "Up to three inputs with the largest influence on this customer's model score, e.g. "
        "'Contract = Month-to-month (raises risk score)'. These describe what the model relies on, "
        "not what caused churn."
    ),
}

FACTORS_PER_CUSTOMER = 3


class BatchError(ValueError):
    """A file-level problem the user must fix before any row can be scored."""


@dataclass
class BatchResult:
    file_name: str
    total_rows: int
    scored: pd.DataFrame
    errors: pd.DataFrame
    threshold: float
    model_name: str
    calibration: str
    ignored_columns: list[str] = field(default_factory=list)

    @property
    def scored_count(self) -> int:
        return len(self.scored)

    @property
    def invalid_count(self) -> int:
        return len(self.errors)

    def risk_counts(self) -> dict[str, int]:
        counts = self.scored["risk_level"].value_counts()
        return {label: int(counts.get(label, 0)) for label in RISK_ORDER}

    def predicted_churn_count(self) -> int:
        return int((self.scored["churn_prediction"] == "Yes").sum())


def read_upload(data: bytes, file_name: str) -> pd.DataFrame:
    """Read a CSV or Excel upload as text so that no value is silently converted."""
    if not data:
        raise BatchError("The file is empty. Upload a file with a header row and at least one customer.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise BatchError(f"The file is larger than {MAX_UPLOAD_MB} MB. Split it into smaller files.")
    suffix = os.path.splitext(file_name or "")[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise BatchError(f"'{file_name}' is not supported. Upload a .csv or .xlsx file.")
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        else:
            frame = pd.read_excel(io.BytesIO(data), dtype=str, engine="openpyxl")
    except pd.errors.EmptyDataError as error:
        raise BatchError("The file has no header row or customer rows.") from error
    except Exception as error:
        raise BatchError(
            "The file could not be read. Check that it is a valid, unprotected CSV or Excel file."
        ) from error
    return frame


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise column names and cell text. Checks the columns before any cell is touched."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    ignored = check_columns(list(frame.columns))
    frame = frame.fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame, ignored


def check_columns(columns: list[str]) -> list[str]:
    """Raise for duplicate or missing required columns; return the extra columns that will be ignored."""
    duplicated = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated:
        raise BatchError(f"These columns appear more than once: {', '.join(duplicated)}. Keep one column per field.")
    missing = [column for column in FEATURE_COLUMNS if column not in columns]
    if missing:
        names = ", ".join(f"{FEATURE_LABELS[column]} ({column})" for column in missing)
        raise BatchError(f"Required columns are missing: {names}. Download the sample template to see the layout.")
    return [column for column in columns if column not in FEATURE_COLUMNS and column != ID_COLUMN]


def _check_row(record: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one row without filling in any missing value. Returns (clean values, None) or (None, problem)."""
    clean: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        label = FEATURE_LABELS[column]
        raw = record[column]
        if raw == "":
            return None, f"{label} is missing."
        number = churn_model._as_number(raw)
        if number is None:
            return None, f"{label} must be a number (found '{raw}')."
        low, high = NUMERIC_LIMITS[column]
        if not low <= number <= high:
            return None, f"{label} must be between {low:g} and {high:g} (found {raw})."
        clean[column] = number
    for column, options in CATEGORY_MAPS.items():
        label = FEATURE_LABELS[column]
        raw = record[column]
        if raw == "":
            return None, f"{label} is missing."
        match = next((option for option in options if option.lower() == raw.lower()), None)
        if match is None:
            return None, f"{label} must be one of: {', '.join(options)} (found '{raw}')."
        clean[column] = match
    problem = churn_model.validate_customer(clean)
    return (clean, None) if problem is None else (None, problem)


def format_factors(factors: list[dict[str, Any]]) -> str:
    """'Label = value (raises risk score); ...' for one customer, or blank when nothing stands out."""
    return "; ".join(
        f"{item['label']} = {item['value']} ({'raises' if item['direction'] == 'raises' else 'lowers'} risk score)"
        for item in factors
    )


def score_frame(
    frame: pd.DataFrame,
    file_name: str,
    bundle: dict[str, Any],
    progress: Callable[[float], None] | None = None,
) -> BatchResult:
    """Validate every row, then score the valid rows in chunks with the saved model and calibrator."""
    frame, ignored = _prepare(frame)
    total = len(frame)
    if total == 0:
        raise BatchError("The file has a header row but no customer rows.")
    if total > MAX_ROWS:
        raise BatchError(f"The file has {total:,} rows; the limit is {MAX_ROWS:,} per upload. Split it into smaller files.")

    ids = frame[ID_COLUMN].tolist() if ID_COLUMN in frame.columns else [""] * total
    id_counts = pd.Series(ids, dtype=object).value_counts()
    records = frame.to_dict("records")

    valid_rows: list[dict[str, Any]] = []
    valid_numbers: list[int] = []
    valid_ids: list[str] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row_number = index + 1
        customer_id = str(ids[index])
        clean, problem = _check_row(record)
        if problem is None and customer_id and int(id_counts.get(customer_id, 0)) > 1:
            count = int(id_counts[customer_id])
            problem = (
                f"Customer ID '{customer_id}' appears {count} times. Each customer must appear once, "
                "so none of these rows were scored."
            )
        if problem is not None:
            errors.append({"row_number": row_number, "customer_id": customer_id, "problem": problem})
            continue
        valid_rows.append(clean)
        valid_numbers.append(row_number)
        valid_ids.append(customer_id)

    probabilities = np.zeros(len(valid_rows))
    factor_text: list[str] = [""] * len(valid_rows)
    if valid_rows:
        features = pd.DataFrame(valid_rows)[bundle["feature_columns"]]
        explainer = churn_model.make_explainer(bundle)
        for start in range(0, len(features), CHUNK_ROWS):
            chunk = features.iloc[start : start + CHUNK_ROWS]
            probabilities[start : start + len(chunk)] = churn_model.predict_probabilities(bundle, chunk)
            chunk_factors = churn_model.explain_frame(bundle, chunk, FACTORS_PER_CUSTOMER, explainer)
            factor_text[start : start + len(chunk)] = [format_factors(factors) for factors in chunk_factors]
            if progress is not None:
                progress(min(1.0, (start + len(chunk)) / len(features)))
    elif progress is not None:
        progress(1.0)

    threshold = float(bundle["threshold"])
    scored = pd.DataFrame(
        {
            "row_number": valid_numbers,
            "customer_id": valid_ids,
            "churn_prediction": np.where(probabilities >= threshold, "Yes", "No"),
            "churn_probability": np.round(probabilities, 4),
            "risk_level": [RISK_LABELS[churn_model.risk_band(float(p))] for p in probabilities],
            "validation_status": "Scored",
            "main_factors": factor_text,
        },
        columns=SCORED_COLUMNS,
    )
    error_frame = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    return BatchResult(
        file_name=file_name,
        total_rows=total,
        scored=scored,
        errors=error_frame,
        threshold=threshold,
        model_name=bundle["model_name"],
        calibration=churn_model.calibration_label(bundle),
        ignored_columns=ignored,
    )


def score_upload(data: bytes, file_name: str, bundle: dict[str, Any], progress: Callable[[float], None] | None = None) -> BatchResult:
    return score_frame(read_upload(data, file_name), file_name, bundle, progress)


def template_csv() -> bytes:
    """A sample input file with three example customers, using the exact expected column names."""
    rows = []
    for index, preset in enumerate(list(PRESETS.values())[:3], start=1):
        rows.append({ID_COLUMN: f"SAMPLE-{index:03d}", **{column: preset[column] for column in FEATURE_COLUMNS}})
    return pd.DataFrame(rows, columns=[ID_COLUMN, *FEATURE_COLUMNS]).to_csv(index=False).encode("utf-8")


def scored_csv(result: BatchResult) -> bytes:
    return result.scored.to_csv(index=False).encode("utf-8")


def errors_csv(result: BatchResult) -> bytes:
    return result.errors.to_csv(index=False).encode("utf-8")
