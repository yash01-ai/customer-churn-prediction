"""Score a single customer record with the persisted pipeline.

Because the whole preprocessing chain is baked into the saved Pipeline, scoring is just
"build a one-row frame -> predict_proba". The caller passes a plain dict of *raw* feature
values (exactly the columns in the source CSV, minus the ID and target); the pipeline does
its own cleaning-compatible feature engineering and encoding.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
DECISION_PATH = MODELS_DIR / "decision.json"

# The raw input schema the pipeline expects. Column ORDER does not matter (the
# ColumnTransformer selects by name), but every one of these names must be present — so
# predict_one validates an incoming record against this list and fails loudly on a missing
# field instead of letting it surface as a cryptic KeyError deep inside the transformer.
REQUIRED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

# Fallback only if evaluate.py hasn't written the business threshold yet. 0.5 is flagged as
# a fallback precisely because it is NOT the business-optimal choice — see evaluate.py.
DEFAULT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _load_pipeline():
    # WHY cache the load: joblib.load reads and de-pickles a fitted model from disk. In a
    # long-lived process (the Streamlit app, a batch loop) reloading per call is wasted I/O;
    # the model is immutable after training, so one cached instance is safe to share.
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"No model at {PIPELINE_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(PIPELINE_PATH)


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    if DECISION_PATH.exists():
        return float(json.loads(DECISION_PATH.read_text())["threshold"])
    return DEFAULT_THRESHOLD


def predict_one(record: dict, threshold: float | None = None) -> dict:
    """Score one customer.

    Returns the churn probability, the label at the business threshold, and which threshold
    was applied — so a caller can never misread a probability as a decision without knowing
    the cutoff behind it.
    """
    threshold = _load_threshold() if threshold is None else threshold
    pipeline = _load_pipeline()

    # Validate up front so a caller who forgets a field gets one clear message naming exactly
    # what is missing, rather than a KeyError raised from inside the ColumnTransformer that
    # points at sklearn internals instead of their input.
    missing = [f for f in REQUIRED_FEATURES if f not in record]
    if missing:
        raise ValueError(f"record is missing required feature(s): {missing}")

    # WHY a one-row DataFrame and not a bare array: the ColumnTransformer selects columns by
    # name, so the input must be a labeled frame. The engineered num_addon_services column is
    # intentionally NOT required from the caller — the pipeline's FunctionTransformer derives it.
    frame = pd.DataFrame([record])
    proba = float(pipeline.predict_proba(frame)[0, 1])

    return {
        "churn_probability": round(proba, 4),
        "churn_label": int(proba >= threshold),
        "threshold": round(float(threshold), 4),
    }


# A representative month-to-month, fiber, electronic-check customer — the classic high-risk
# profile — handy for smoke-testing predict/serving without hand-typing 19 fields.
EXAMPLE_RECORD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 140.70,
}


if __name__ == "__main__":
    print(predict_one(EXAMPLE_RECORD))
