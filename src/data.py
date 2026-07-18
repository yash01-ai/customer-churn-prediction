"""Load and clean the raw Telco churn dataset.

This module does the *deterministic* cleaning that must happen before any modeling
decision — the kind of fixes that are objectively correct regardless of train/test
split (type coercion, dropping an identifier, encoding the target). Anything that
*learns* from the data (scaling, encoding categories, imputing a learned statistic)
deliberately lives in the preprocessing Pipeline instead, so it can be fit on the
training folds only. See features.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# WHY a module-level constant and not a hard-coded string at every call site: the raw
# path is a single fact about the project layout. Centralizing it means a relocation is
# one edit, and tests/notebooks all resolve the same file.
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"

# The three genuinely-continuous columns. Everything else in this dataset is categorical
# (including SeniorCitizen, which is a 0/1 flag, not a magnitude — see features.py).
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped, with no cleaning applied."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the split-independent cleaning steps and return a modeling-ready frame."""
    df = df.copy()

    # WHY coerce TotalCharges instead of trusting the dtype: the raw column ships as an
    # *object* (string) because 11 rows contain a blank " " instead of a number. pandas
    # therefore refuses to treat the whole column as numeric. errors="coerce" turns those
    # blanks into NaN so we can handle them explicitly rather than have them silently
    # poison arithmetic later.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # WHY fill those NaNs with 0 and not the mean/median: every blank TotalCharges belongs
    # to a customer with tenure == 0 — a brand-new signup who has not been billed a full
    # cycle yet. Their true lifetime charge is genuinely $0, so 0 is the *correct* value,
    # not an imputation guess. Using the column mean here would invent ~$2000 of spend for
    # a customer who has spent nothing, distorting the tenure/charges relationship the
    # model relies on. This fill is deterministic (independent of any train/test split),
    # so doing it here rather than in the Pipeline introduces no leakage.
    zero_tenure_blanks = df["TotalCharges"].isna()
    assert (df.loc[zero_tenure_blanks, "tenure"] == 0).all(), (
        "Unexpected missing TotalCharges for a customer with non-zero tenure — "
        "the $0 fill is only valid for never-billed (tenure==0) customers."
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # WHY drop customerID: it is a unique per-row identifier with zero predictive signal.
    # Left in, a tree model could memorize individual customers (pure overfitting) and a
    # one-hot encoder would explode it into 7043 useless columns.
    df = df.drop(columns=[ID_COLUMN])

    # WHY map the target to 1/0 with churn as the positive class: scikit-learn metrics
    # (precision, recall, average_precision) treat 1 as the class of interest. Churn is the
    # event we want to detect and price, so it must be the positive label — otherwise recall
    # would measure how well we find *retained* customers, which is not the business goal.
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def load_clean(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Convenience: load the raw CSV and return the cleaned modeling frame."""
    return clean(load_raw(path))


def load_xy(path: str | Path = RAW_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with the target split out — the shape every trainer expects."""
    df = load_clean(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    frame = load_clean()
    features, target = frame.drop(columns=[TARGET]), frame[TARGET]
    print(f"rows={len(frame)}  features={features.shape[1]}")
    print(f"churn rate={target.mean():.4f}  (positives={int(target.sum())})")
    print(f"any NaN remaining: {bool(frame.isna().any().any())}")
