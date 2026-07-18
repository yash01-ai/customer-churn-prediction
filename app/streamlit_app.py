"""Streamlit scoring app for the churn pipeline.

Run with:  streamlit run app/streamlit_app.py

Loads the persisted Pipeline, takes a single customer's raw features from a form, and
returns the churn probability, the label at the business-cost threshold, and the SHAP
contributions that drove *this* prediction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

# WHY force the non-interactive Agg backend: the app renders on a headless server and hands
# finished Figure objects to st.pyplot — there is no desktop display for an interactive backend
# to draw into, and letting matplotlib pick a GUI backend here can hang or error on the server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# WHY mutate sys.path here: `streamlit run app/streamlit_app.py` executes the file as a
# top-level script, so the project root is not on the import path and `import src...`
# would fail. Adding the repo root makes the app runnable from anywhere without packaging.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the loaders from src.predict on purpose: model loading and the business threshold have
# exactly one definition there, so the app and the CLI/API can never score against a different
# model or cutoff. (Underscore-prefixed, but this is the same codebase, not an external import.)
from src.features import get_feature_names  # noqa: E402
from src.predict import _load_pipeline, _load_threshold  # noqa: E402

DECISION_PATH = ROOT / "models" / "decision.json"

st.set_page_config(page_title="Customer Churn Scoring", page_icon="📉", layout="centered")


@st.cache_resource
def get_pipeline():
    # cache_resource so the model de-pickles once per server process, not once per rerun
    # (Streamlit re-executes the whole script on every widget change).
    return _load_pipeline()


@st.cache_data
def get_threshold() -> float:
    return _load_threshold()


def input_form() -> dict:
    """Render the raw-feature form and return a record dict matching the CSV schema."""
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        # The optional-service selects share the same three-way domain as the raw data,
        # including the "No internet service" sentinel, so the encoder sees familiar values.
        opts = ["No", "Yes", "No internet service"]
        online_sec = st.selectbox("Online security", opts)
        online_bak = st.selectbox("Online backup", opts)
        device = st.selectbox("Device protection", opts)
        tech = st.selectbox("Tech support", opts)
        stream_tv = st.selectbox("Streaming TV", opts)
        stream_mov = st.selectbox("Streaming movies", opts)
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    monthly = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
    # WHY default TotalCharges to a tenure-consistent value: a brand-new customer's total
    # should track tenure*monthly, not an arbitrary number, so the demo stays realistic.
    total = st.number_input("Total charges ($)", min_value=0.0,
                            value=float(round(monthly * max(tenure, 1), 2)), step=10.0)

    return {
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak, "DeviceProtection": device, "TechSupport": tech,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mov, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total),
    }


def local_shap(pipeline, record: dict):
    """Return this one customer's top SHAP contributions (mirrors evaluate.py's global version).

    WHY explain the TRANSFORMED matrix and not the raw record: SHAP attributes the prediction to
    the columns the classifier actually sees — the scaled numerics and one-hot dummies — so we
    push the raw record through the already-fitted preprocessing (pipeline[:-1], never refit) and
    explain the resulting named row. Explaining the raw fields would attribute nothing, since the
    classifier never sees them directly.
    """
    pre = pipeline[:-1]
    clf = pipeline.named_steps["clf"]
    frame = pd.DataFrame([record])
    x_trans = pre.transform(frame)
    names = get_feature_names(pipeline.named_steps["preprocess"])
    x_df = pd.DataFrame(x_trans, columns=names)

    # TreeExplainer is exact and fast for the RF/XGB winners; fall back to the model-agnostic
    # explainer for the linear baseline.
    try:
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(x_df)
    except Exception:
        explainer = shap.Explainer(clf, x_df)
        values = explainer(x_df).values

    # Collapse SHAP's version-dependent output to the positive-class 2-D matrix: newer shap
    # returns (n, features, classes) for classifiers, older versions a [class0, class1] list;
    # either way we want the churn (index 1) contributions.
    sv = np.asarray(values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif isinstance(values, list):
        sv = np.asarray(values[1])
    contrib = pd.DataFrame({"feature": names, "shap_value": sv[0]})
    contrib["abs"] = contrib["shap_value"].abs()
    # Rank by absolute impact so the strongest churn-pushers and churn-dampeners both surface.
    return contrib.sort_values("abs", ascending=False).head(8)


def main():
    st.title("Customer Churn Scoring")
    st.caption(
        "Scores a single customer with the trained pipeline. The label uses the "
        "business-cost threshold, not a naive 0.5 cutoff."
    )

    if not (ROOT / "models" / "churn_pipeline.joblib").exists():
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    pipeline = get_pipeline()
    threshold = get_threshold()
    record = input_form()

    if st.button("Score customer", type="primary"):
        proba = float(pipeline.predict_proba(pd.DataFrame([record]))[0, 1])
        label = int(proba >= threshold)

        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{proba:.1%}")
        c2.metric(f"Decision @ {threshold:.2f}", "WILL CHURN" if label else "will stay")

        if label:
            st.warning("Flagged for a retention offer.")
        else:
            st.success("Not flagged — no offer needed at the current threshold.")

        st.subheader("Why — top SHAP contributors for this customer")
        contrib = local_shap(pipeline, record)
        st.caption(
            "Positive values push toward churn, negative toward staying "
            "(effect on the model's log-odds)."
        )
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in contrib["shap_value"]]
        ax.barh(contrib["feature"][::-1], contrib["shap_value"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on churn log-odds)")
        fig.tight_layout()
        st.pyplot(fig)

        if DECISION_PATH.exists():
            with st.expander("Threshold & cost assumptions"):
                st.json(json.loads(DECISION_PATH.read_text()))


if __name__ == "__main__":
    main()
