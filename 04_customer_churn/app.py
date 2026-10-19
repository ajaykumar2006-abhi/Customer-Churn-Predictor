"""Customer Churn Predictor: a five-step guided workflow.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

import batch
import model as churn_model
from batch import BatchError
from report import (
    batch_report_to_pdf,
    build_batch_report,
    build_report,
    new_report_id,
    report_to_json,
    report_to_pdf,
)
from utils import (
    DEFAULT_CUSTOMER,
    PRESETS,
    RISK_EXPLANATIONS,
    RISK_LABELS,
    consistency_note,
    field_help,
    format_percent,
    retention_actions,
)

REPORT_DIR = os.path.join(churn_model.BASE_DIR, "reports")
STEPS = ["Welcome", "Customer information", "Prediction", "Explanation", "Download report"]
FORM_FIELDS = list(DEFAULT_CUSTOMER)

RISK_COLOURS = {"Low": "#15803d", "Moderate": "#b45309", "High": "#b91c1c"}

CSS = """
<style>
.block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.01em; }
.step-label { color: #5b6472; font-size: 0.9rem; margin-bottom: 0.25rem; }
.card { background: #ffffff; border: 1px solid #dfe4ec; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.result-title { font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0; }
.result-prob { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.muted { color: #5b6472; font-size: 0.95rem; }
.risk-pill { display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; color: #ffffff; font-weight: 600; font-size: 0.9rem; }
</style>
"""


# ----------------------------------------------------------------- state helpers


def _key(field: str) -> str:
    return f"field_{field}"


def init_state() -> None:
    st.session_state.setdefault("step", 0)
    for field, value in DEFAULT_CUSTOMER.items():
        st.session_state.setdefault(_key(field), value)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("report", None)


def current_customer() -> dict:
    return {field: st.session_state[_key(field)] for field in FORM_FIELDS}


def go(step: int) -> None:
    st.session_state.step = step


def load_example(name: str) -> None:
    for field, value in PRESETS[name].items():
        st.session_state[_key(field)] = value


def start_over() -> None:
    for field, value in DEFAULT_CUSTOMER.items():
        st.session_state[_key(field)] = value
    st.session_state.result = None
    st.session_state.report = None
    st.session_state.step = 1


def model_status() -> tuple[dict | None, str | None]:
    """Return (bundle, error). bundle is None when the model has not been trained yet."""
    try:
        return churn_model.load_model(), None
    except Exception as exc:  # corrupt or outdated artefact: explain, do not crash
        return None, str(exc)


def train_model_now() -> None:
    try:
        with st.spinner("Training the model on the telco dataset. This takes under a minute..."):
            churn_model.train_model()
        st.session_state.training_error = None
    except FileNotFoundError as exc:
        st.session_state.training_error = str(exc)
    except Exception as exc:
        st.session_state.training_error = f"Training failed: {exc}"


def create_prediction(bundle: dict) -> None:
    """Validate the form, score the customer, build the report and go to the result page."""
    customer = current_customer()
    problem = churn_model.validate_customer(customer)
    if problem:
        st.error(f"Please check the customer information: {problem}")
        return

    result = churn_model.predict(customer, bundle)
    metrics = churn_model.read_metrics()
    report_id = new_report_id()
    report = build_report(customer, result, metrics, report_id)

    pdf_bytes: bytes | None = None
    pdf_error: str | None = None
    try:
        pdf_bytes = report_to_pdf(report)
    except Exception as exc:
        pdf_error = f"The PDF could not be created ({exc}). You can still download the JSON version."

    json_bytes = report_to_json(report)
    saved_to: str | None = None
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        base = os.path.join(REPORT_DIR, report_id)
        if pdf_bytes is not None:
            with open(base + ".pdf", "wb") as handle:
                handle.write(pdf_bytes)
        with open(base + ".json", "wb") as handle:
            handle.write(json_bytes)
        saved_to = REPORT_DIR
    except OSError:
        saved_to = None

    st.session_state.result = result
    st.session_state.report = {
        "id": report_id,
        "pdf": pdf_bytes,
        "pdf_error": pdf_error,
        "json": json_bytes,
        "saved_to": saved_to,
    }
    go(2)


# ------------------------------------------------------------------ components


def step_header(index: int) -> None:
    st.markdown(
        f'<div class="step-label">Step {index + 1} of {len(STEPS)} &middot; {STEPS[index]}</div>',
        unsafe_allow_html=True,
    )
    st.progress((index + 1) / len(STEPS))


def nav_buttons(back: int | None, forward: int | None, forward_label: str = "Next", forward_disabled: bool = False) -> None:
    left, _, right = st.columns([1, 2, 1])
    if back is not None:
        left.button("Back", on_click=go, args=(back,), key=f"back_{back}_{st.session_state.step}")
    if forward is not None:
        right.button(
            forward_label,
            type="primary",
            on_click=go,
            args=(forward,),
            disabled=forward_disabled,
            key=f"fwd_{forward}_{st.session_state.step}",
        )


def risk_pill(label: str) -> str:
    colour = RISK_COLOURS.get(label, "#374151")
    return f'<span class="risk-pill" style="background:{colour}">{label} risk</span>'


# ------------------------------------------------------------------------ steps


def render_welcome(bundle: dict | None, error: str | None) -> None:
    st.title("Customer Churn Predictor")
    st.markdown(
        """
        **Churn** means a customer stops using a service. Keeping an existing customer is usually
        cheaper than winning a new one, so many businesses want to spot at-risk customers early enough
        to act.

        This tool estimates how likely a customer is to leave, based on their contract, bill and
        services. It then explains the main factors behind that estimate and suggests practical
        retention steps.
        """
    )
    st.markdown(
        """
        **What you will do**

        1. Enter a customer's details.
        2. See the predicted outcome and its probability.
        3. Review the factors and suggested actions.
        4. Download a report to share.

        The estimate is a model's best guess from past customers. It is not a guarantee of what this
        customer will do.
        """
    )

    if error:
        st.error(f"The saved model could not be loaded. {error}")
    if st.session_state.get("training_error"):
        st.error(st.session_state.training_error)

    metrics = churn_model.read_metrics()
    if bundle is None:
        st.warning(
            "The model has not been trained on this computer yet. Press **Train model** once; "
            "it takes under a minute."
        )
        if st.button("Train model", type="primary"):
            train_model_now()
            st.rerun()
    else:
        test = metrics["test"] if metrics else None
        if test:
            st.markdown(
                f"""
                <div class="card">
                <b>Model ready:</b> {bundle['model_name']}, trained on {metrics['data']['rows']:,} telco customers.<br>
                <span class="muted">On {test['n']} customers the model had not seen, it reached ROC-AUC {test['roc_auc']:.2f} and
                accuracy {format_percent(test['accuracy'])}. Customers are flagged as likely to churn when their
                estimated probability is {format_percent(bundle['threshold'])} or more.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if bundle is not None and metrics and metrics.get("test_uncalibrated"):
        render_calibration_evidence(metrics)
    nav_buttons(back=None, forward=1, forward_label="Start", forward_disabled=bundle is None)


def render_customer_form(bundle: dict | None) -> None:
    st.header("Customer information")
    st.markdown(
        '<p class="muted">Fill in the details below. Each field explains what it means. '
        "You can also load an example to see how the tool works.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    cols[0].button("Load example: new customer", on_click=load_example, args=("New month-to-month customer",))
    cols[1].button("Load example: long-term customer", on_click=load_example, args=("Long-tenure two-year customer",))

    help_text = field_help()
    with st.form("customer_form", border=True):
        st.subheader("Account")
        a, b = st.columns(2)
        a.number_input(
            "Tenure (months)", min_value=0, max_value=72, step=1, key=_key("tenure"), help=help_text["tenure"]
        )
        b.selectbox("Contract", churn_model.CATEGORY_MAPS["Contract"], key=_key("Contract"), help=help_text["Contract"])
        c, d = st.columns(2)
        c.selectbox(
            "Payment method", churn_model.CATEGORY_MAPS["PaymentMethod"], key=_key("PaymentMethod"),
            help=help_text["PaymentMethod"],
        )
        d.selectbox(
            "Internet service", churn_model.CATEGORY_MAPS["InternetService"], key=_key("InternetService"),
            help=help_text["InternetService"],
        )

        st.subheader("Charges")
        e, f = st.columns(2)
        e.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=150.0, step=0.5, format="%.2f",
            key=_key("MonthlyCharges"), help=help_text["MonthlyCharges"],
        )
        f.number_input(
            "Total charges to date ($)", min_value=0.0, max_value=10000.0, step=1.0, format="%.2f",
            key=_key("TotalCharges"), help=help_text["TotalCharges"],
        )

        st.subheader("Add-on services")
        g, h = st.columns(2)
        g.selectbox(
            "Online security", churn_model.CATEGORY_MAPS["OnlineSecurity"], key=_key("OnlineSecurity"),
            help=help_text["OnlineSecurity"],
        )
        h.selectbox(
            "Tech support", churn_model.CATEGORY_MAPS["TechSupport"], key=_key("TechSupport"),
            help=help_text["TechSupport"],
        )

        submitted = st.form_submit_button("Predict churn risk", type="primary", disabled=bundle is None)

    note = consistency_note(current_customer())
    if note:
        st.warning(note)
    if bundle is None:
        st.info("Train the model on the Welcome step before predicting.")

    if submitted and bundle is not None:
        create_prediction(bundle)
        st.rerun()

    nav_buttons(back=0, forward=None)


def render_prediction() -> None:
    result = st.session_state.result
    report = st.session_state.report
    risk = RISK_LABELS[result["risk_level"]]
    outcome = "Likely to churn" if result["churn"] == "Yes" else "Likely to stay"
    probability = result["churn_probability"]

    st.header("Prediction")
    st.markdown(
        f"""
        <div class="card">
          <div class="muted">Report ID {report['id']}</div>
          <div class="result-title">{outcome}</div>
          <div class="muted">Estimated probability of churn</div>
          <div class="result-prob">{format_percent(probability)}</div>
          <p style="margin:0.5rem 0">{risk_pill(risk)}</p>
          <p class="muted" style="margin-bottom:0">{RISK_EXPLANATIONS[result['risk_level']]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability)
    st.caption(
        f"Decision threshold used: {format_percent(result['threshold'])}. "
        "Customers at or above this probability are predicted to churn. Risk bands: Low below "
        f"{churn_model.RISK_MEDIUM:.0%}, Moderate {churn_model.RISK_MEDIUM:.0%} to under "
        f"{churn_model.RISK_HIGH:.0%}, High {churn_model.RISK_HIGH:.0%} and above."
    )
    st.caption(
        f"Probability calibration: {result.get('calibration', 'raw model scores')}. "
        "Calibration aligns the estimate with the churn rates observed in past customers."
    )
    st.info(
        "This is a model estimate based on past customers, not a guarantee of what this customer will do. "
        "Use it alongside your own knowledge of the account."
    )
    nav_buttons(back=1, forward=3, forward_label="See the explanation")
    st.button("Start a new prediction", on_click=start_over)


def render_explanation() -> None:
    result = st.session_state.result
    risk_level = result["risk_level"]
    metrics = churn_model.read_metrics()

    st.header("Explanation")
    st.subheader("Key factors")
    if result["factors"]:
        st.markdown(
            "These inputs had the largest effect on this customer's estimate. They show patterns the model "
            "found in past customers. They are associations, not proven causes."
        )
        rows = ["| Input | This customer | Effect on the estimate |", "|---|---|---|"]
        for factor in result["factors"]:
            effect = "Pushes the estimate **up**" if factor["direction"] == "raises" else "Pushes the estimate **down**"
            rows.append(f"| {factor['label']} | {factor['value']} | {effect} |")
        st.markdown("\n".join(rows))
    else:
        st.write("No single input moved this customer's estimate noticeably.")

    st.subheader("Recommended actions")
    st.markdown("Suggested only where an input above pushed the estimate up. Use your judgement before acting.")
    for action in retention_actions(result["factors"], risk_level):
        st.markdown(f"- {action}")

    st.subheader("About the model")
    if metrics:
        test = metrics["test"]
        st.markdown(
            f"""
            - **Model:** {result['model']}. Selection rule: {metrics['selection_rule']}
            - **Trained on:** {metrics['data']['rows']:,} customers from the IBM Telco Customer Churn dataset
            - **Tested on:** {test['n']} customers the model did not see during training
            - **Accuracy:** {format_percent(test['accuracy'])} · **Precision:** {format_percent(test['precision'])} ·
              **Recall:** {format_percent(test['recall'])} · **F1-score:** {format_percent(test['f1'])}
            - **ROC-AUC:** {test['roc_auc']:.2f} · **PR-AUC:** {test['pr_auc']:.2f} (a random guess scores {test['pr_auc_baseline']:.2f})
            """
        )
        st.caption(
            "What the probability means: it is the model's estimate for customers like this one. It is not a "
            "frequency for this exact person, and the model cannot see support calls, usage or competitor offers."
        )
    nav_buttons(back=2, forward=4, forward_label="Continue to the report")


def render_download() -> None:
    report = st.session_state.report
    st.header("Download report")
    st.markdown(
        '<p class="muted">Your report contains the customer details you entered, the prediction, the key '
        "factors, suggested actions, and the model's test results with its limitations.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(f"Report ID: **{report['id']}**")

    if report["pdf"] is not None:
        st.download_button(
            "Download Report",
            data=report["pdf"],
            file_name=f"{report['id']}.pdf",
            mime="application/pdf",
            type="primary",
            key="download_pdf",
        )
    else:
        st.error(report["pdf_error"] or "The PDF report is not available.")

    st.download_button(
        "Download JSON",
        data=report["json"],
        file_name=f"{report['id']}.json",
        mime="application/json",
        key="download_json",
        help="Machine-readable copy of the same prediction, factors and model details.",
    )
    if report["saved_to"]:
        st.caption("A copy was also saved to this folder:")
        st.code(report["saved_to"], language=None)
    nav_buttons(back=3, forward=None)
    st.button("Start a new prediction", on_click=start_over)


def render_calibration_evidence(metrics: dict) -> None:
    """Before and after calibration on the untouched test set. Every number comes from metrics.json."""
    before = metrics["test_uncalibrated"]
    after = metrics["test"]
    calibration = metrics.get("calibration", {})
    with st.expander("How reliable are the probabilities? Before and after calibration"):
        st.markdown(f"**Calibration used:** {calibration.get('label', 'none')}")
        if after["brier"] < before["brier"] and after["log_loss"] < before["log_loss"]:
            verdict = (
                f"Calibration lowered the Brier score from {before['brier']:.3f} to {after['brier']:.3f} and "
                f"log loss from {before['log_loss']:.3f} to {after['log_loss']:.3f}, so the probabilities "
                "match observed churn rates more closely. ROC-AUC measures ranking, not probability accuracy, "
                f"and is {before['roc_auc']:.3f} before and {after['roc_auc']:.3f} after."
            )
        else:
            verdict = (
                "Calibration did not improve the Brier score or log loss on the test set, "
                "so the raw model scores are used."
            )
        st.markdown(verdict)
        table = pd.DataFrame(
            [
                ("Brier score (lower is better)", f"{before['brier']:.4f}", f"{after['brier']:.4f}"),
                ("Log loss (lower is better)", f"{before['log_loss']:.4f}", f"{after['log_loss']:.4f}"),
                ("ROC-AUC", f"{before['roc_auc']:.4f}", f"{after['roc_auc']:.4f}"),
                ("Decision threshold", format_percent(before["threshold"]), format_percent(after["threshold"])),
                ("Accuracy", format_percent(before["accuracy"]), format_percent(after["accuracy"])),
                ("Precision", format_percent(before["precision"]), format_percent(after["precision"])),
                ("Recall", format_percent(before["recall"]), format_percent(after["recall"])),
                ("F1-score", format_percent(before["f1"]), format_percent(after["f1"])),
            ],
            columns=["Measure (held-out test set)", "Before calibration", "After calibration"],
        )
        st.dataframe(table, hide_index=True)
        st.caption(
            f"Test set: {after['n']:,} customers, {after['positives']:,} churned "
            f"({format_percent(after['prevalence'])} churn rate). The threshold is chosen on training data only; "
            "the test set was scored once."
        )
        rows = []
        for index, (raw, cal) in enumerate(zip(before["calibration"], after["calibration"]), start=1):
            rows.append(
                {
                    "Bin": index,
                    "Before: predicted": format_percent(raw["predicted"]),
                    "Before: observed": format_percent(raw["observed"]),
                    "After: predicted": format_percent(cal["predicted"]),
                    "After: observed": format_percent(cal["observed"]),
                }
            )
        st.markdown("**Reliability table** (customers grouped into 10 bands by predicted risk)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        render_reliability_chart(metrics)


def render_reliability_chart(metrics: dict) -> None:
    """Reliability diagram from development cross-validation only; the held-out test set is not used here."""
    st.markdown("**Reliability diagram: development cross-validation**")
    report = metrics.get("dev_reliability")
    if not report:
        st.info("This model was trained before the reliability diagram was added. Press Train model to create it.")
        return
    st.caption(
        f"Out-of-fold predictions on {report['n']:,} development customers ({report['positives']:,} churned). "
        f"Brier score (lower is better): raw model {report['brier_raw']:.4f}, calibrated {report['brier_calibrated']:.4f}. "
        "A point on the dashed diagonal means the predicted risk matched the observed churn rate. "
        "Bins with fewer than 30 customers are less reliable."
    )
    points = churn_model.reliability_points(report)
    if points.empty:
        st.info("No customers fell into any probability band, so the diagram cannot be drawn.")
        return
    diagonal = alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})).mark_line(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q", y="y:Q")
    curves = (
        alt.Chart(points)
        .mark_line(point=True)
        .encode(
            x=alt.X("Predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted churn probability (bin average)"),
            y=alt.Y("Observed:Q", scale=alt.Scale(domain=[0, 1]), title="Observed churn rate"),
            color=alt.Color("Series:N", title="Probabilities"),
            tooltip=["Series", "Predicted", "Observed", "Customers", "Small bin"],
        )
    )
    st.altair_chart((diagonal + curves).properties(height=340), use_container_width=True)


# ------------------------------------------------------------------- batch scoring


def batch_column_guide() -> pd.DataFrame:
    rows = [{"Column": churn_model.ID_COLUMN, "Meaning": "Customer ID (optional)", "Allowed values": "Text, unique in the file"}]
    for column in churn_model.FEATURE_COLUMNS:
        if column in churn_model.NUMERIC_FEATURES:
            low, high = churn_model.NUMERIC_LIMITS[column]
            allowed = f"Number from {low:g} to {high:g}"
        else:
            allowed = ", ".join(churn_model.CATEGORY_MAPS[column])
        rows.append({"Column": column, "Meaning": churn_model.FEATURE_LABELS[column], "Allowed values": allowed})
    return pd.DataFrame(rows)


def render_batch(bundle: dict | None, error: str | None) -> None:
    st.title("Batch customer scoring")
    st.markdown(
        f"""
        Score many customers at once from a CSV or Excel file, one customer per row. Each valid row gets
        a churn probability, a predicted outcome and a risk level. Rows with problems are listed with the
        reason and are not scored. Files up to {batch.MAX_UPLOAD_MB} MB and {batch.MAX_ROWS:,} rows are accepted.
        Uploaded data is processed in this session only and is not saved to disk.
        """
    )
    if error:
        st.error(f"The saved model could not be loaded. {error}")
    if bundle is None:
        st.warning("Train the model first: choose **One customer** in the sidebar and press **Train model**.")
        return

    st.download_button(
        "Download sample template (CSV)",
        data=batch.template_csv(),
        file_name="customer_template.csv",
        mime="text/csv",
    )
    with st.expander("Expected input columns"):
        st.markdown(
            "Column names must match exactly. Every column below except `customerID` is required. "
            "Extra columns are ignored."
        )
        st.dataframe(batch_column_guide(), hide_index=True)
    with st.expander("Output columns"):
        st.dataframe(
            pd.DataFrame(
                [{"Column": name, "Meaning": text} for name, text in batch.OUTPUT_COLUMN_GUIDE.items()]
            ),
            hide_index=True,
        )

    upload = st.file_uploader("Customer file (.csv or .xlsx)", type=["csv", "xlsx"], key="batch_upload")
    if upload is None:
        st.session_state.batch = None
        return

    if st.button("Score customers", type="primary"):
        bar = st.progress(0.0, text="Checking rows...")
        try:
            result = batch.score_upload(
                upload.getvalue(),
                upload.name,
                bundle,
                progress=lambda fraction: bar.progress(fraction, text="Scoring customers..."),
            )
        except BatchError as exc:
            bar.empty()
            st.session_state.batch = None
            st.error(str(exc))
        else:
            bar.empty()
            report_id = new_report_id()
            pdf: bytes | None = None
            pdf_error: str | None = None
            try:
                pdf = batch_report_to_pdf(build_batch_report(result, churn_model.read_metrics(), report_id))
            except Exception as exc:
                pdf_error = f"The batch PDF could not be created ({exc}). The CSV downloads are still available."
            st.session_state.batch = {
                "file_name": upload.name,
                "report_id": report_id,
                "result": result,
                "pdf": pdf,
                "pdf_error": pdf_error,
            }

    state = st.session_state.get("batch")
    if state and state["file_name"] == upload.name:
        render_batch_results(state)


def render_batch_results(state: dict) -> None:
    result = state["result"]
    counts = result.risk_counts()

    st.subheader("Results")
    if result.scored_count == 0:
        st.error("No rows could be scored. Check the problems listed below.")
    elif result.invalid_count == 0:
        st.success(f"All {result.scored_count:,} customers were scored.")
    else:
        st.warning(
            f"{result.scored_count:,} of {result.total_rows:,} rows were scored. "
            f"{result.invalid_count:,} rows need fixing and were not scored."
        )

    cols = st.columns(4)
    cols[0].metric("Customers uploaded", f"{result.total_rows:,}")
    cols[1].metric("Successfully scored", f"{result.scored_count:,}")
    cols[2].metric("Invalid rows", f"{result.invalid_count:,}")
    cols[3].metric("Predicted to churn", f"{result.predicted_churn_count():,}")
    cols = st.columns(3)
    for col, label in zip(cols, batch.RISK_ORDER):
        col.metric(f"{label} risk", f"{counts[label]:,}")

    if result.scored_count:
        st.bar_chart(pd.DataFrame({"Customers": [counts[label] for label in batch.RISK_ORDER]}, index=batch.RISK_ORDER))
        st.caption(
            f"Probabilities are calibrated estimates ({result.calibration}). Risk bands: Low below "
            f"{churn_model.RISK_MEDIUM:.0%}, Moderate to under {churn_model.RISK_HIGH:.0%}, High from "
            f"{churn_model.RISK_HIGH:.0%}."
        )
    if result.ignored_columns:
        st.info(f"Extra columns were ignored: {', '.join(result.ignored_columns)}.")

    stem = f"churn_scores_{state['report_id']}"
    cols = st.columns(3)
    cols[0].download_button(
        "Download Scored CSV",
        data=batch.scored_csv(result),
        file_name=f"{stem}.csv",
        mime="text/csv",
        disabled=result.scored_count == 0,
    )
    if result.invalid_count:
        cols[1].download_button(
            "Download Validation Errors",
            data=batch.errors_csv(result),
            file_name=f"{stem}_validation_errors.csv",
            mime="text/csv",
        )
    if state["pdf"] is not None:
        cols[2].download_button(
            "Download Batch Report",
            data=state["pdf"],
            file_name=f"{stem}_report.pdf",
            mime="application/pdf",
        )
    elif state["pdf_error"]:
        st.error(state["pdf_error"])

    if result.scored_count:
        st.subheader("Predictions")
        st.caption("Click a column heading to sort.")
        st.dataframe(result.scored, hide_index=True)
    if result.invalid_count:
        st.subheader("Rows that could not be scored")
        st.dataframe(result.errors, hide_index=True)


# ------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    st.session_state.setdefault("training_error", None)

    mode = st.sidebar.radio(
        "Mode",
        ["One customer", "Batch file"],
        key="mode",
        help="Score one customer step by step, or upload a file with many customers.",
    )
    bundle, error = model_status()
    if mode == "Batch file":
        render_batch(bundle, error)
        return

    if st.session_state.step not in range(len(STEPS)):
        go(0)
    if st.session_state.step >= 2 and st.session_state.result is None:
        go(1)
    step = st.session_state.step

    if step == 0:
        step_header(0)
        render_welcome(bundle, error)
    elif step == 1:
        step_header(1)
        render_customer_form(bundle)
    elif step == 2:
        step_header(2)
        render_prediction()
    elif step == 3:
        step_header(3)
        render_explanation()
    else:
        step_header(4)
        render_download()


main()
