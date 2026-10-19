"""FastAPI service for customer churn prediction (default port 8004)."""

import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from model import get_model  # noqa: E402  (path insert must run first)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(
    title="Customer Churn Prediction",
    description="Estimates churn probability for a telecom customer profile.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    tenure: int = Field(12, ge=0, le=120, description="Months as a customer")
    MonthlyCharges: float = Field(50.0, ge=0, description="Current monthly bill")
    TotalCharges: float = Field(600.0, ge=0, description="Lifetime charges")
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    TechSupport: str = "No"


class ChurnPrediction(BaseModel):
    churn: str
    churn_probability: float
    retention_probability: float
    risk_level: str
    model: str
    error: Optional[str] = None


@app.get("/health")
def health():
    loaded = get_model().loaded
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}


@app.get("/metrics")
def metrics():
    values = get_model().get_metrics()
    if not values:
        raise HTTPException(status_code=503, detail="Metrics unavailable - run train.py first")
    return values


@app.post("/predict", response_model=ChurnPrediction)
def predict(features: CustomerFeatures):
    try:
        return get_model().predict(features.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/")
def root():
    return {"name": "Customer Churn Prediction", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8004)))
