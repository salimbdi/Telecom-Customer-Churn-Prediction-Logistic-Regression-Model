"""
app.py — FastAPI backend for Churn Prediction.

Endpoints:
  GET  /          → serves the frontend UI
  GET  /health    → health check
  POST /predict   → returns churn probability and risk label

Run: uvicorn app:app --reload --port 8000
"""

import os
import pickle
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal


# ─────────────────────────────────────────────
# Load model  (auto-train if not present)
# ─────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists("model.pkl") or not os.path.exists("dv.pkl"):
        print("⚠  model.pkl not found — running train.py first …")
        import train
        train.train()

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("dv.pkl", "rb") as f:
        dv = pickle.load(f)
    print("✓ Model and DictVectorizer loaded")
    return model, dv


MODEL, DV = load_artifacts()

# ─────────────────────────────────────────────
# Feature lists (must match train.py exactly)
# ─────────────────────────────────────────────
CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]


# ─────────────────────────────────────────────
# Pydantic schema
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    gender:           Literal["male", "female"]               = Field(..., example="female")
    seniorcitizen:    Literal["0", "1"]                        = Field(..., example="0")
    partner:          Literal["yes", "no"]                     = Field(..., example="yes")
    dependents:       Literal["yes", "no"]                     = Field(..., example="no")
    tenure:           float                                    = Field(..., ge=0, le=72, example=12)
    phoneservice:     Literal["yes", "no"]                     = Field(..., example="yes")
    multiplelines:    Literal["yes", "no", "no_phone_service"] = Field(..., example="no")
    internetservice:  Literal["dsl", "fiber_optic", "no"]      = Field(..., example="fiber_optic")
    onlinesecurity:   Literal["yes", "no", "no_internet_service"] = Field(..., example="no")
    onlinebackup:     Literal["yes", "no", "no_internet_service"] = Field(..., example="yes")
    deviceprotection: Literal["yes", "no", "no_internet_service"] = Field(..., example="no")
    techsupport:      Literal["yes", "no", "no_internet_service"] = Field(..., example="no")
    streamingtv:      Literal["yes", "no", "no_internet_service"] = Field(..., example="no")
    streamingmovies:  Literal["yes", "no", "no_internet_service"] = Field(..., example="no")
    contract:         Literal["month-to-month", "one_year", "two_year"] = Field(..., example="month-to-month")
    paperlessbilling: Literal["yes", "no"]                     = Field(..., example="yes")
    paymentmethod:    Literal[
                          "electronic_check",
                          "mailed_check",
                          "bank_transfer_(automatic)",
                          "credit_card_(automatic)"
                      ]                                        = Field(..., example="electronic_check")
    monthlycharges:   float                                    = Field(..., ge=0, example=70.35)
    totalcharges:     float                                    = Field(..., ge=0, example=844.0)


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predict whether a telecom customer will churn using Logistic Regression.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helper: top risk factors
# ─────────────────────────────────────────────
def get_risk_factors(customer_dict: dict, top_n: int = 5) -> list[dict]:
    """Returns the top-N features driving the prediction."""
    feature_names = DV.get_feature_names_out()
    coefs         = MODEL.coef_[0]

    # Get the feature vector for this customer
    x = DV.transform([customer_dict])[0]

    # Contribution = coefficient × feature value
    contributions = coefs * x

    # Sort by absolute contribution
    indices = np.argsort(np.abs(contributions))[::-1]

    factors = []
    for idx in indices[:top_n]:
        name = feature_names[idx]
        impact = float(contributions[idx])
        if abs(impact) < 0.001:
            continue
        factors.append({
            "feature": name.replace("_", " ").replace("=", ": "),
            "impact": round(impact, 4),
            "direction": "increases risk" if impact > 0 else "reduces risk",
        })

    return factors


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "LogisticRegression"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        customer_dict = customer.model_dump()

        # Build feature dict using only the trained columns
        features = {k: customer_dict[k] for k in CATEGORICAL + NUMERICAL}

        X = DV.transform([features])
        prob = float(MODEL.predict_proba(X)[0, 1])

        # Risk label
        if prob < 0.30:
            risk = "Low"
            color = "#22c55e"
        elif prob < 0.60:
            risk = "Medium"
            color = "#f59e0b"
        else:
            risk = "High"
            color = "#ef4444"

        risk_factors = get_risk_factors(features)

        return {
            "churn_probability": round(prob, 4),
            "churn_probability_pct": round(prob * 100, 1),
            "risk_level": risk,
            "risk_color": color,
            "will_churn": prob >= 0.5,
            "risk_factors": risk_factors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in ./frontend/</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
