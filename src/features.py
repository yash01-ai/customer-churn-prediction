"""Feature engineering and the leakage-safe preprocessing transformer.

The single most important design choice in this file: the preprocessor is a
`ColumnTransformer` that is meant to be dropped *inside* a model Pipeline and fit only
on the training split. Nothing here touches raw statistics at import time. See train.py
for how it is wrapped so cross-validation stays honest.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# The six optional add-on services. Each is "Yes" / "No" / "No internet service".
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Continuous columns that get standardized. num_addon_services is engineered below.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "num_addon_services"]

# WHY SeniorCitizen is passed through untouched instead of scaled or one-hot encoded: it
# already ships as a clean 0/1 indicator. One-hot encoding it would just produce two
# perfectly collinear columns, and scaling a binary flag distorts its interpretation for
# the linear baseline without helping the trees. Passthrough keeps it exactly as-is.
BINARY_PASSTHROUGH = ["SeniorCitizen"]

# Every remaining string column. Enumerated explicitly (not inferred by dtype) so the
# transformer's contract is visible and stable even if column order in the CSV changes.
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add num_addon_services: how many optional services the customer subscribes to.

    WHY this feature earns its place: retention is driven by switching cost. A customer
    paying for five add-ons (security, backup, tech support, streaming…) has far more to
    unwind before leaving than a bare phone-only line, and the data bears this out —
    add-on count is strongly, monotonically inversely related to churn. The model *can*
    in principle recover this by summing interactions across six one-hot columns, but a
    linear baseline cannot express that sum, and even the trees have to spend depth to
    approximate it. Handing them the count directly gives a single, highly interpretable
    "stickiness" signal. It is a pure row-wise count with no fitted statistic, so it is
    leakage-safe wherever it runs; it lives in the Pipeline so inference stays one call.
    """
    X = X.copy()
    # "== 'Yes'" deliberately excludes the "No internet service" sentinel, which is a
    # not-applicable marker, not an active subscription.
    X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
    return X


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that standardizes numerics and one-hot encodes categoricals.

    Returned unfitted on purpose: the caller wraps it in a Pipeline so `.fit` runs on
    training folds only.
    """
    return ColumnTransformer(
        transformers=[
            # WHY StandardScaler and not raw values: LogisticRegression with L2 penalizes
            # coefficients on the same scale, so an unscaled TotalCharges (0–8000) would be
            # regularized far more harshly than tenure (0–72), silently dropping its signal.
            # Scaling is harmless to the tree models, so one shared preprocessor serves all.
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # WHY handle_unknown="ignore": at scoring time a category never seen in training
            # (a new PaymentMethod, say) would otherwise raise and take the service down.
            # "ignore" encodes it as an all-zeros row instead, degrading gracefully. Fitting
            # happens on train folds only, so this leaks nothing.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_PASSTHROUGH),
        ],
        remainder="drop",
        # Keep column count stable and predictable for downstream SHAP naming.
        verbose_feature_names_out=True,
    )


def _engineered_feature_names(_transformer, input_features) -> list[str]:
    # This step APPENDS one column, so it is not one-to-one; sklearn's set_output name
    # check needs the exact output schema (inputs + the new column) or it raises.
    return list(input_features) + ["num_addon_services"]


def build_feature_engineering_step() -> FunctionTransformer:
    """Wrap the engineered-feature function as a stateless Pipeline step.

    WHY a FunctionTransformer rather than computing the column in data.py: keeping it in
    the Pipeline means the persisted model accepts raw customer records and does its own
    feature engineering, so training and inference can never disagree about how the
    feature is built. `validate=False` preserves the DataFrame (and its column names) so
    the ColumnTransformer downstream can still select columns by name.
    """
    return FunctionTransformer(
        add_engineered_features, validate=False, feature_names_out=_engineered_feature_names
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Return the output column names of a fitted preprocessor (for SHAP / coefficients)."""
    return list(fitted_preprocessor.get_feature_names_out())
