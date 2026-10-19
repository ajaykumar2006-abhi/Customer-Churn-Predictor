"""Tests for the customer churn predictor.

Run with:  python -m pytest tests.py -q
"""

from __future__ import annotations

import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

import batch
import model as churn
from report import build_report, report_to_json, report_to_pdf
from utils import DEFAULT_CUSTOMER, PRESETS, consistency_note, format_percent, retention_actions

SAFE = PRESETS["Long-tenure two-year customer"]
RISKY = PRESETS["New month-to-month customer"]


@pytest.fixture(scope="module")
def trained():
    """The saved model trained on the real dataset (trained once if the artefact is missing)."""
    if not churn.model_exists() or churn.read_metrics() is None:
        churn.train_model()
    churn.load_model.cache_clear()
    return churn.load_model()


@pytest.fixture(scope="module")
def metrics(trained):
    return churn.read_metrics()


# ----------------------------------------------------------------- risk bands


def test_risk_bands_follow_the_documented_thresholds():
    assert churn.risk_band(0.0) == "low"
    assert churn.risk_band(0.399) == "low"
    assert churn.risk_band(0.4) == "medium"
    assert churn.risk_band(0.699) == "medium"
    assert churn.risk_band(0.7) == "high"
    assert churn.risk_band(1.0) == "high"


# ----------------------------------------------------------------- validation


def test_default_and_preset_customers_are_valid():
    assert churn.validate_customer(DEFAULT_CUSTOMER) is None
    for preset in PRESETS.values():
        assert churn.validate_customer(preset) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"tenure": 73},
        {"tenure": -1},
        {"MonthlyCharges": 150.01},
        {"MonthlyCharges": -0.01},
        {"TotalCharges": 10_000.01},
    ],
)
def test_values_outside_the_documented_ranges_are_rejected(changes):
    assert churn.validate_customer({**DEFAULT_CUSTOMER, **changes}) is not None


def test_boundary_values_are_accepted():
    assert churn.validate_customer({**DEFAULT_CUSTOMER, "tenure": 0, "TotalCharges": 0}) is None
    assert churn.validate_customer({**DEFAULT_CUSTOMER, "tenure": 72, "MonthlyCharges": 150}) is None


def test_fractional_tenure_is_rejected():
    assert "whole number" in churn.validate_customer({**DEFAULT_CUSTOMER, "tenure": 12.5})


@pytest.mark.parametrize("value", [None, "", "lots", float("nan"), float("inf"), True])
def test_missing_or_non_numeric_values_are_rejected(value):
    assert churn.validate_customer({**DEFAULT_CUSTOMER, "TotalCharges": value}) is not None


def test_unknown_or_missing_category_is_rejected():
    assert "Contract" in churn.validate_customer({**DEFAULT_CUSTOMER, "Contract": "Lifetime"})
    customer = {k: v for k, v in DEFAULT_CUSTOMER.items() if k != "PaymentMethod"}
    assert "Payment method" in churn.validate_customer(customer)


def test_consistency_note_flags_impossible_totals():
    assert consistency_note({**DEFAULT_CUSTOMER, "tenure": 24, "MonthlyCharges": 100, "TotalCharges": 100})
    assert consistency_note(DEFAULT_CUSTOMER) is None


# ----------------------------------------------------------------- data


def test_dataset_loads_with_the_churn_target_and_no_duplicate_customers():
    frame = churn.load_dataset()
    assert len(frame) == 7043
    assert set(frame[churn.TARGET].unique()) == {0, 1}
    assert list(frame.columns) == churn.FEATURE_COLUMNS + [churn.TARGET]
    audit = churn.audit_dataset()
    assert audit["duplicate_customer_ids"] == 0
    assert audit["blank_total_charges"] == 11


def test_split_is_stratified_and_disjoint():
    frame = churn.load_dataset()
    X_dev, X_test, y_dev, y_test = churn.split_data(frame)
    assert len(X_dev) + len(X_test) == len(frame)
    assert set(X_dev.index).isdisjoint(X_test.index)
    assert abs(y_dev.mean() - y_test.mean()) < 0.01


def test_preprocessing_learns_only_from_the_training_split():
    frame = churn.load_dataset()
    X_dev, X_test, y_dev, _ = churn.split_data(frame)
    pipeline = churn.build_pipeline(churn.candidate_models()["Logistic Regression"])
    pipeline.fit(X_dev, y_dev)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    expected_median = X_dev["TotalCharges"].median()
    # The imputer's median must come from the development rows only, never the test rows.
    assert imputer.statistics_[2] == pytest.approx(expected_median)
    assert len(X_test) > 0


def test_saved_metrics_describe_the_real_dataset(metrics):
    assert metrics["data"]["rows"] == 7043
    assert metrics["data"]["target_column"] == "Churn"
    assert metrics["split"]["test_rows"] == 1409
    assert metrics["model_type"] in metrics["candidates"]
    assert 0.05 <= metrics["decision_threshold"] <= 0.95
    test = metrics["test"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
        assert 0.0 <= test[key] <= 1.0
    cm = test["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == test["n"]


def test_training_writes_consistent_artifacts(tmp_path, monkeypatch):
    """Train on a 1,000-row slice into a temp folder; artefacts must agree with each other."""
    path = tmp_path / "subset.csv"
    pd.read_csv(churn.DATA_PATH).iloc[:1000].to_csv(path, index=False)
    monkeypatch.setattr(churn, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(churn, "MODEL_PATH", str(tmp_path / "m.pkl"))
    monkeypatch.setattr(churn, "METRICS_PATH", str(tmp_path / "metrics.json"))
    result = churn.train_model(str(path))
    assert (tmp_path / "m.pkl").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["model_type"] == result["model_type"]
    assert saved["data"]["rows"] == 1000
    churn.load_model.cache_clear()
    bundle = churn.load_model()
    assert bundle["model_name"] == saved["model_type"]
    assert bundle["threshold"] == saved["decision_threshold"]
    churn.load_model.cache_clear()


# ----------------------------------------------------------------- prediction


def test_prediction_returns_probability_outcome_and_factors(trained):
    result = churn.predict(DEFAULT_CUSTOMER, trained)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["retention_probability"] == pytest.approx(1 - result["churn_probability"])
    assert result["churn"] in {"Yes", "No"}
    assert (result["churn"] == "Yes") == (result["churn_probability"] >= result["threshold"])
    assert result["risk_level"] in {"low", "medium", "high"}
    assert 1 <= len(result["factors"]) <= 5


def test_repeated_predictions_are_identical(trained):
    first = churn.predict(RISKY, trained)
    second = churn.predict(RISKY, trained)
    assert first["churn_probability"] == second["churn_probability"]
    assert first["factors"] == second["factors"]


def test_representative_customers_score_without_error(trained):
    customers = [DEFAULT_CUSTOMER, RISKY, SAFE,
                 {**DEFAULT_CUSTOMER, "tenure": 0, "TotalCharges": 0.0},
                 {**DEFAULT_CUSTOMER, "tenure": 72, "Contract": "Two year", "InternetService": "No",
                  "OnlineSecurity": "No internet service", "TechSupport": "No internet service"}]
    for customer in customers:
        result = churn.predict(customer, trained)
        assert 0.0 <= result["churn_probability"] <= 1.0


def test_risky_profile_scores_above_safe_profile(trained):
    assert churn.predict(RISKY, trained)["churn_probability"] > churn.predict(SAFE, trained)["churn_probability"]


def test_invalid_input_is_rejected_by_predict(trained):
    with pytest.raises(ValueError):
        churn.predict({**DEFAULT_CUSTOMER, "Contract": "Lifetime"}, trained)
    with pytest.raises(ValueError):
        churn.predict({**DEFAULT_CUSTOMER, "tenure": None}, trained)


def test_unknown_internet_option_is_not_silently_encoded(trained):
    with pytest.raises(ValueError):
        churn.predict({**DEFAULT_CUSTOMER, "InternetService": "Satellite"}, trained)


def test_factor_list_is_sorted_by_size_and_uses_input_labels(trained):
    factors = churn.predict(RISKY, trained)["factors"]
    sizes = [abs(f["contribution"]) for f in factors]
    assert sizes == sorted(sizes, reverse=True)
    labels = {f["label"] for f in factors}
    assert labels <= set(churn.FEATURE_LABELS.values())


def test_load_model_rejects_an_outdated_artefact(tmp_path, monkeypatch):
    import pickle

    path = tmp_path / "old.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
    monkeypatch.setattr(churn, "MODEL_PATH", str(path))
    churn.load_model.cache_clear()
    with pytest.raises(RuntimeError, match="older version"):
        churn.load_model()
    churn.load_model.cache_clear()


# ----------------------------------------------------------------- guidance


def test_retention_actions_only_use_factors_that_raised_the_estimate():
    factors = [
        {"feature": "Contract", "direction": "raises"},
        {"feature": "tenure", "direction": "lowers"},
        {"feature": "TechSupport", "direction": "raises"},
    ]
    actions = retention_actions(factors, "high")
    assert any("contract" in a.lower() for a in actions)
    assert not any("early-life" in a for a in actions)
    assert len(retention_actions(factors * 3, "high", limit=2)) <= 2


def test_early_life_action_is_skipped_for_long_tenure_customers():
    long_tenure = [{"feature": "tenure", "direction": "raises", "raw": 58.0}]
    new_customer = [{"feature": "tenure", "direction": "raises", "raw": 3.0}]
    assert not any("early-life" in a for a in retention_actions(long_tenure, "medium"))
    assert any("early-life" in a for a in retention_actions(new_customer, "medium"))





def test_low_risk_without_raising_factors_gets_a_no_action_message():
    actions = retention_actions([], "low")
    assert actions and "No specific action" in actions[0]


def test_formatters():
    assert format_percent(0.4321) == "43.2%"


# ----------------------------------------------------------------- report


def _report_for(trained, metrics, customer=RISKY):
    result = churn.predict(customer, trained)
    return build_report(customer, result, metrics, "CHR-TEST000001"), result


def test_json_report_matches_the_prediction(trained, metrics):
    report, result = _report_for(trained, metrics)
    data = json.loads(report_to_json(report))
    assert data["report_id"] == "CHR-TEST000001"
    assert data["prediction"]["churn_probability"] == pytest.approx(result["churn_probability"], abs=1e-4)
    assert data["prediction"]["predicted_outcome"] == (
        "Likely to churn" if result["churn"] == "Yes" else "Likely to stay"
    )
    assert data["model"]["name"] == result["model"]
    assert data["disclaimer"] and data["limitations"]


def test_pdf_report_contains_the_same_prediction(trained, metrics):
    report, result = _report_for(trained, metrics)
    pdf = report_to_pdf(report)
    assert pdf.startswith(b"%PDF")
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert "CHR-TEST000001" in text
    assert format_percent(result["churn_probability"]) in text
    assert "Recommended retention actions" in text


# ----------------------------------------------------------------- workflow


def _click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button {label!r} not found; have {[b.label for b in at.button]}"
    matches[0].click()
    return at.run(timeout=120)


def test_five_step_workflow_end_to_end(trained):
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    assert any("Customer Churn Predictor" in t.value for t in at.title)

    at = _click(at, "Start")
    assert any("Customer information" in h.value for h in at.header)

    at = _click(at, "Load example: new customer")
    at = _click(at, "Predict churn risk")
    assert not at.exception
    assert any("Estimated probability of churn" in m.value for m in at.markdown)

    at = _click(at, "See the explanation")
    assert any(h.value == "Key factors" for h in at.subheader)
    at = _click(at, "Continue to the report")
    assert any(h.value == "Download report" for h in at.header)
    assert not at.exception


# ----------------------------------------------------------- calibration (1, 2, 3)


def test_calibrator_is_fitted_only_on_development_rows(tmp_path, monkeypatch):
    """Every calibrator fit must use development rows only; the 200 test rows are never seen."""
    path = tmp_path / "subset.csv"
    pd.read_csv(churn.DATA_PATH).iloc[:1000].to_csv(path, index=False)
    monkeypatch.setattr(churn, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(churn, "MODEL_PATH", str(tmp_path / "m.pkl"))
    monkeypatch.setattr(churn, "METRICS_PATH", str(tmp_path / "metrics.json"))
    fit_sizes: list[int] = []
    original_fit = churn.Calibrator.fit

    def spy(self, scores, labels):
        fit_sizes.append(len(labels))
        return original_fit(self, scores, labels)

    monkeypatch.setattr(churn.Calibrator, "fit", spy)
    churn.train_model(str(path))
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    dev_rows = saved["split"]["development_rows"]
    assert dev_rows == 800 and saved["split"]["test_rows"] == 200
    assert fit_sizes, "calibration candidates should have been fitted"
    assert max(fit_sizes) <= dev_rows
    assert saved["calibration"]["fitted_on"].startswith("Out-of-fold development scores")
    churn.load_model.cache_clear()


def test_calibration_selection_uses_only_out_of_fold_scores():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 600)
    labels = pd.Series((rng.uniform(size=600) < raw**2).astype(int))
    folds = churn.StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    method, briers = churn.select_calibration(raw, labels, folds)
    assert set(briers) == {"none", "sigmoid", "isotonic"}
    assert method == min(briers, key=briers.get)
    # Out-of-fold predictions: no row is calibrated by a model that saw its own label.
    oof = churn.cross_calibrated("sigmoid", raw, labels, folds)
    assert oof.shape == raw.shape and np.all((oof > 0) & (oof < 1))


def test_evaluate_reports_brier_log_loss_and_threshold_metrics():
    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    out = churn.evaluate(y, p, 0.5)
    assert out["accuracy"] == 1.0 and out["precision"] == 1.0 and out["recall"] == 1.0
    assert out["brier"] == pytest.approx(float(np.mean((p - y.to_numpy()) ** 2)), abs=1e-4)
    assert out["log_loss"] == pytest.approx(
        float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), abs=1e-4
    )


def test_saved_before_and_after_calibration_metrics_match_the_method(metrics):
    before = metrics["test_uncalibrated"]
    after = metrics["test"]
    assert before["n"] == after["n"] == 1409
    assert before["positives"] == after["positives"] == 374
    assert before["prevalence"] == after["prevalence"]
    # A monotone calibrator cannot change the ranking.
    assert after["roc_auc"] == pytest.approx(before["roc_auc"], abs=1e-3)
    if metrics["calibration"]["method"] == "none":
        assert after["brier"] == before["brier"]
    else:
        assert after["brier"] < before["brier"]
        assert after["log_loss"] < before["log_loss"]
    assert len(after["calibration"]) == len(before["calibration"]) == 10


def test_calibrator_survives_pickle_and_artifact_reload(trained):
    assert trained["calibration_method"] in ("none", "sigmoid", "isotonic")
    if trained["calibration_method"] != "none":
        calibrator = trained["calibrator"]
        restored = pickle.loads(pickle.dumps(calibrator))
        scores = np.linspace(0.05, 0.95, 19)
        np.testing.assert_allclose(calibrator.transform(scores), restored.transform(scores))
    churn.load_model.cache_clear()
    reloaded = churn.load_model()
    assert reloaded["calibration_method"] == trained["calibration_method"]
    assert churn.calibration_label(reloaded) == churn.CALIBRATION_LABELS[trained["calibration_method"]]


def test_single_customer_prediction_uses_the_calibrated_pipeline(trained):
    result = churn.predict(SAFE, trained)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["calibration"] == churn.calibration_label(trained)
    frame = churn.encode_customer(SAFE, trained)
    direct = churn.predict_probabilities(trained, frame)[0]
    assert result["churn_probability"] == pytest.approx(float(direct), abs=1e-4)


# ------------------------------------------------------ batch scoring (4-10)


def _batch_frame(n: int = 3) -> pd.DataFrame:
    base = pd.read_csv(io.BytesIO(batch.template_csv()), dtype=str, keep_default_na=False)
    frame = base.iloc[[i % len(base) for i in range(n)]].reset_index(drop=True)
    frame[churn.ID_COLUMN] = [f"CUST-{i + 1:04d}" for i in range(n)]
    return frame


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def test_template_is_a_valid_batch_file(trained):
    template = batch.template_csv()
    result = batch.score_upload(template, "customer_template.csv", trained)
    assert result.total_rows == len(PRESETS)
    assert result.scored_count == len(PRESETS) and result.invalid_count == 0
    assert list(pd.read_csv(io.BytesIO(template)).columns) == [churn.ID_COLUMN, *churn.FEATURE_COLUMNS]


def test_valid_batch_csv_scores_every_row(trained):
    result = batch.score_upload(_csv_bytes(_batch_frame(7)), "c.csv", trained)
    assert result.total_rows == 7 and result.scored_count == 7
    scored = result.scored
    assert set(scored["risk_level"]) <= set(batch.RISK_ORDER)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["churn_prediction"]) <= {"Yes", "No"}
    assert result.calibration == churn.calibration_label(trained)


def test_missing_and_extra_columns(trained):
    frame = _batch_frame()
    missing = frame.drop(columns=["MonthlyCharges"])
    with pytest.raises(batch.BatchError, match="MonthlyCharges"):
        batch.score_frame(missing, "m.csv", trained)
    extra = frame.assign(Notes="call back")
    result = batch.score_frame(extra, "e.csv", trained)
    assert result.ignored_columns == ["Notes"]
    assert result.scored_count == 3
    duplicated = pd.concat([frame, frame[["Contract"]]], axis=1)
    with pytest.raises(batch.BatchError, match="more than once"):
        batch.score_frame(duplicated, "d.csv", trained)


def test_invalid_values_and_missing_data_are_reported_per_row(trained):
    frame = _batch_frame(5)
    frame.loc[0, "tenure"] = "abc"
    frame.loc[1, "MonthlyCharges"] = ""
    frame.loc[2, "Contract"] = "Weekly"
    frame.loc[3, "tenure"] = "200"
    result = batch.score_frame(frame, "v.csv", trained)
    assert result.invalid_count == 4 and result.scored_count == 1
    problems = dict(zip(result.errors["row_number"], result.errors["problem"]))
    assert "must be a number" in problems[1]
    assert "is missing" in problems[2]
    assert "must be one of" in problems[3]
    assert "must be between" in problems[4]
    assert result.scored["row_number"].tolist() == [5]


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"", "empty.csv"),
        (b"customerID,tenure\n", "header_only.csv"),
        (b"not an excel file", "broken.xlsx"),
        (b"a,b\n1,2\n", "notes.txt"),
    ],
)
def test_empty_malformed_and_unsupported_files_fail_safely(trained, data, name):
    with pytest.raises(batch.BatchError):
        batch.score_upload(data, name, trained)


def test_oversized_upload_is_rejected():
    too_big = b"x" * (batch.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(batch.BatchError, match="larger than"):
        batch.read_upload(too_big, "big.csv")


def test_row_limit_is_enforced(trained, monkeypatch):
    monkeypatch.setattr(batch, "MAX_ROWS", 2)
    with pytest.raises(batch.BatchError, match="limit"):
        batch.score_frame(_batch_frame(3), "rows.csv", trained)


def test_duplicate_customer_ids_are_not_scored(trained):
    frame = _batch_frame(4)
    frame.loc[1, churn.ID_COLUMN] = frame.loc[0, churn.ID_COLUMN]
    result = batch.score_frame(frame, "dup.csv", trained)
    assert result.scored["row_number"].tolist() == [3, 4]
    assert result.errors["row_number"].tolist() == [1, 2]
    assert all("appears 2 times" in text for text in result.errors["problem"])


def test_rows_without_customer_id_are_scored_with_blank_id(trained):
    result = batch.score_frame(_batch_frame(2).drop(columns=[churn.ID_COLUMN]), "n.csv", trained)
    assert result.scored_count == 2
    assert result.scored["customer_id"].tolist() == ["", ""]


def test_row_order_and_counts_stay_aligned(trained):
    frame = _batch_frame(5)
    frame.loc[2, "tenure"] = "-4"
    result = batch.score_frame(frame, "order.csv", trained)
    assert result.total_rows == 5
    assert result.scored_count + result.invalid_count == 5
    assert result.scored["row_number"].tolist() == [1, 2, 4, 5]
    assert result.scored["customer_id"].tolist() == ["CUST-0001", "CUST-0002", "CUST-0004", "CUST-0005"]
    single = batch.score_frame(frame.iloc[[3]].reset_index(drop=True), "one.csv", trained)
    assert single.scored["churn_probability"].iloc[0] == result.scored["churn_probability"].iloc[2]


def test_chunked_scoring_gives_the_same_predictions(trained, monkeypatch):
    frame = _batch_frame(7)
    whole = batch.score_frame(frame, "w.csv", trained).scored
    monkeypatch.setattr(batch, "CHUNK_ROWS", 2)
    chunked = batch.score_frame(frame, "c.csv", trained).scored
    pd.testing.assert_frame_equal(whole, chunked)


def test_excel_upload_is_scored(trained):
    buffer = io.BytesIO()
    _batch_frame(3).to_excel(buffer, index=False)
    result = batch.score_upload(buffer.getvalue(), "customers.xlsx", trained)
    assert result.scored_count == 3


# ----------------------------------------------------- exports and reports (11, 12)


def test_scored_csv_download_contents(trained):
    result = batch.score_frame(_batch_frame(3), "d.csv", trained)
    exported = pd.read_csv(io.BytesIO(batch.scored_csv(result)), dtype=str)
    assert list(exported.columns) == batch.SCORED_COLUMNS
    assert len(exported) == result.scored_count == 3
    assert exported["customer_id"].tolist() == ["CUST-0001", "CUST-0002", "CUST-0003"]
    assert not {"password", "api_key", "token"} & set(exported.columns)


def test_validation_error_download_contents(trained):
    frame = _batch_frame(2)
    frame.loc[1, "tenure"] = "x"
    result = batch.score_frame(frame, "e.csv", trained)
    exported = pd.read_csv(io.BytesIO(batch.errors_csv(result)), dtype=str)
    assert list(exported.columns) == batch.ERROR_COLUMNS
    assert exported["row_number"].tolist() == ["2"]
    assert exported["customer_id"].tolist() == ["CUST-0002"]


def test_batch_pdf_report_is_generated(trained, metrics):
    frame = _batch_frame(3)
    frame.loc[0, "tenure"] = "bad"
    result = batch.score_frame(frame, "report.csv", trained)
    report = build_batch_report_for(result, metrics)
    pdf = batch_report_bytes(report)
    assert pdf.startswith(b"%PDF")
    text = "".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert "Customer Churn Batch Scoring Report" in text
    assert "report.csv" in text
    assert report["summary"]["invalid_rows"] == 1 and report["summary"]["scored_rows"] == 2


def build_batch_report_for(result, metrics):
    from report import build_batch_report

    return build_batch_report(result, metrics, "TEST-BATCH")


def batch_report_bytes(report):
    from report import batch_report_to_pdf

    return batch_report_to_pdf(report)


# ------------------------------------------- reliability curve (development only)


def test_reliability_bins_count_every_row_and_keep_empty_bins():
    bins = churn.reliability_bins([0, 1, 1, 0, 1], [0.05, 0.08, 0.55, 0.57, 0.97], n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 5
    assert bins[0]["count"] == 2 and bins[0]["observed"] == 0.5
    assert bins[0]["predicted"] == pytest.approx(0.065, abs=1e-4)
    assert bins[3]["count"] == 0 and bins[3]["predicted"] is None and bins[3]["observed"] is None
    assert bins[5]["count"] == 2 and bins[5]["observed"] == 0.5
    assert bins[9]["count"] == 1 and bins[9]["observed"] == 1.0


def test_probability_of_exactly_one_falls_in_the_top_bin():
    bins = churn.reliability_bins([1], [1.0], n_bins=10)
    assert bins[9]["count"] == 1 and bins[9]["observed"] == 1.0


def test_reliability_report_brier_scores_match_the_formula():
    y = np.array([0, 1, 1, 0])
    raw = np.array([0.2, 0.6, 0.9, 0.4])
    calibrated = np.array([0.1, 0.8, 0.8, 0.2])
    report = churn.reliability_report(y, raw, calibrated)
    assert report["n"] == 4 and report["positives"] == 2
    assert report["brier_raw"] == pytest.approx(float(np.mean((raw - y) ** 2)), abs=1e-4)
    assert report["brier_calibrated"] == pytest.approx(float(np.mean((calibrated - y) ** 2)), abs=1e-4)


def test_reliability_points_drop_empty_bins_and_flag_small_ones():
    empty = {"count": 0, "predicted": None, "observed": None}
    report = {
        "raw": [dict(empty), {"count": 40, "predicted": 0.15, "observed": 0.2}],
        "calibrated": [dict(empty), {"count": 12, "predicted": 0.16, "observed": 0.18}],
    }
    points = churn.reliability_points(report)
    assert points["Series"].tolist() == ["Raw model", "Calibrated"]
    assert points["Bin"].tolist() == [2, 2]
    assert points["Small bin"].tolist() == [False, True]
    assert points.loc[0, "Observed"] == 0.2


def test_saved_reliability_curve_uses_development_rows_only(metrics):
    dev = metrics["dev_reliability"]
    assert dev["n"] == metrics["split"]["development_rows"]
    assert dev["n"] != metrics["test"]["n"]
    assert sum(b["count"] for b in dev["raw"]) == dev["n"]
    assert sum(b["count"] for b in dev["calibrated"]) == dev["n"]
    assert 0 <= dev["brier_raw"] <= 1 and 0 <= dev["brier_calibrated"] <= 1
    if metrics["calibration"]["method"] != "none":
        assert dev["brier_calibrated"] < dev["brier_raw"]


# ------------------------------------------- per-customer explanations in batch


def test_batch_main_factors_match_single_customer_explanations(trained):
    presets = list(PRESETS.values())
    result = batch.score_frame(_batch_frame(3), "f.csv", trained)
    assert result.scored["row_number"].tolist() == [1, 2, 3]
    for position, row in result.scored.iterrows():
        customer = presets[position % len(presets)]
        single = churn.predict(customer, trained)
        expected = batch.format_factors(
            churn.explain(trained, churn.encode_customer(customer, trained), batch.FACTORS_PER_CUSTOMER)
        )
        assert row["main_factors"] == expected
        assert row["churn_probability"] == pytest.approx(single["churn_probability"], abs=1e-4)


def test_invalid_rows_get_no_explanation_and_row_numbers_stay_aligned(trained):
    presets = list(PRESETS.values())
    frame = _batch_frame(3)
    frame.loc[0, "tenure"] = "abc"
    result = batch.score_frame(frame, "v.csv", trained)
    assert result.scored["row_number"].tolist() == [2, 3]
    assert result.errors["row_number"].tolist() == [1]
    expected = batch.format_factors(
        churn.explain(trained, churn.encode_customer(presets[1 % len(presets)], trained), batch.FACTORS_PER_CUSTOMER)
    )
    assert result.scored.loc[result.scored["row_number"] == 2, "main_factors"].item() == expected


def test_factor_text_names_only_input_features_and_direction(trained):
    result = batch.score_frame(_batch_frame(3), "f.csv", trained)
    for text in result.scored["main_factors"]:
        assert text != ""
        for part in text.split("; "):
            assert any(part.startswith(label + " = ") for label in churn.FEATURE_LABELS.values())
            assert part.endswith("(raises risk score)") or part.endswith("(lowers risk score)")
    assert batch.format_factors([]) == ""


def test_shared_explainer_gives_the_same_factors_as_a_fresh_one(trained):
    frame = pd.concat([churn.encode_customer(p, trained) for p in PRESETS.values()], ignore_index=True)
    shared = churn.make_explainer(trained)
    assert churn.explain_frame(trained, frame, 3, shared) == churn.explain_frame(trained, frame, 3)


def test_logistic_explanations_are_the_exact_linear_decomposition():
    frame = churn.load_dataset()
    X_dev, _, y_dev, _ = churn.split_data(frame)
    pipeline = churn.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_dev, y_dev)
    background = pipeline.named_steps["preprocess"].transform(X_dev).mean(axis=0)
    bundle = {"pipeline": pipeline, "background_mean": background}
    assert churn.make_explainer(bundle) is None

    rows = X_dev.iloc[:5]
    explained = churn.explain_frame(bundle, rows, top=50)
    transformed = pipeline.named_steps["preprocess"].transform(rows)
    expected_total = transformed @ pipeline.named_steps["model"].coef_[0] - background @ pipeline.named_steps["model"].coef_[0]
    for index, factors in enumerate(explained):
        # Each contribution is rounded to 4 decimals, so the tolerance grows with the number of factors.
        tolerance = len(factors) * 5e-5 + 1e-9
        assert sum(f["contribution"] for f in factors) == pytest.approx(expected_total[index], abs=tolerance)


# ------------------------------------------------------ app regression (13)


def test_batch_mode_renders_template_and_guides(trained):
    at = AppTest.from_file("app.py", default_timeout=120).run()
    at.sidebar.radio[0].set_value("Batch file").run()
    assert not at.exception
    assert any("Batch customer scoring" in t.value for t in at.title)
    assert any("Download sample template" in str(b.label) for b in at.get("download_button"))

    at.sidebar.radio[0].set_value("One customer").run()
    assert not at.exception
    assert any("Customer Churn Predictor" in t.value for t in at.title)
