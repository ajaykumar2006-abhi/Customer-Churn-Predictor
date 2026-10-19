"""Form defaults, plain-English copy, and retention actions linked to model factors.

Validation itself lives in model.validate_customer so the app and the model share one schema.
Nothing here imports Streamlit, so the rules can be tested without a browser.
"""

from __future__ import annotations

from typing import Any

from model import NUMERIC_LIMITS

DEFAULT_CUSTOMER: dict[str, Any] = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
}

PRESETS: dict[str, dict[str, Any]] = {
    "New month-to-month customer": {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    },
    "Long-tenure two-year customer": {
        "tenure": 58,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2610.0,
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
    },
}

RISK_LABELS = {"high": "High", "medium": "Moderate", "low": "Low"}

RISK_EXPLANATIONS = {
    "high": "The model estimates this customer is at high risk of leaving. Contact them before renewal.",
    "medium": "The model estimates a moderate risk of leaving. A friendly check-in is worthwhile.",
    "low": "The model estimates a low risk of leaving. No urgent action is indicated.",
}

# Action text for each input that the model can associate with higher churn. Each action is
# only offered when that input actually pushed this customer's score up, and the wording stays
# "associated with", because the model shows patterns in the data, not proven causes.
_ACTIONS: dict[str, str] = {
    "Contract": "Offer a discount for moving to a one- or two-year contract.",
    "tenure": "Schedule an early-life check-in; newer customers leave more often in this data.",
    "PaymentMethod": "Invite the customer to an automatic payment method (bank transfer or card).",
    "MonthlyCharges": "Review the monthly price against the customer's usage and offer a better-fitting plan.",
    "TechSupport": "Offer a free trial of tech support.",
    "OnlineSecurity": "Offer a trial of the online-security add-on.",
    "InternetService": "Ask about speed and reliability; fiber accounts churn more often in this data.",
}


def retention_actions(factors: list[dict[str, Any]], risk_level: str, limit: int = 4) -> list[str]:
    """Turn the factors that raised this customer's score into practical retention actions."""
    actions: list[str] = []
    for factor in factors:
        if factor["direction"] != "raises":
            continue
        # The early-life check-in only makes sense for genuinely new customers.
        raw_tenure = factor.get("raw")
        if factor["feature"] == "tenure" and raw_tenure is not None and raw_tenure >= 12:
            continue
        text = _ACTIONS.get(factor["feature"])
        if text and text not in actions:
            actions.append(text)
        if len(actions) >= limit:
            break
    if actions:
        return actions
    if risk_level == "low":
        return ["No specific action is needed. Keep normal service and billing check-ins."]
    return ["No single input stands out. Review the account's overall service experience."]


def consistency_note(customer: dict[str, Any]) -> str | None:
    """Flag total charges that look too low for the tenure and monthly charge entered.

    The model still scores the account; this only asks the user to double-check the input.
    """
    try:
        tenure = float(customer["tenure"])
        monthly = float(customer["MonthlyCharges"])
        total = float(customer["TotalCharges"])
    except (KeyError, TypeError, ValueError):
        return None
    if tenure >= 3 and total < monthly * (tenure - 1) * 0.5:
        return (
            f"Total charges (${total:,.0f}) look low for {tenure:g} months at "
            f"${monthly:,.0f} per month. Please double-check the input."
        )
    return None


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def field_help() -> dict[str, str]:
    """Plain-English help shown under each form field."""
    low, high = NUMERIC_LIMITS["tenure"]
    return {
        "tenure": f"How many whole months the customer has been with the company ({low:g}–{high:g}).",
        "MonthlyCharges": "The amount the customer is billed each month, in US dollars.",
        "TotalCharges": "Everything the customer has been billed so far, in US dollars. "
        "Use 0 for a brand-new customer who has not been billed yet.",
        "Contract": "How long the customer has committed to stay.",
        "PaymentMethod": "How the customer pays each bill.",
        "InternetService": "The type of home internet connection, if any.",
        "OnlineSecurity": "Whether the customer pays for the online-security add-on.",
        "TechSupport": "Whether the customer pays for the tech-support add-on.",
    }

