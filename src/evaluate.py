"""Evaluate the persisted pipeline on the held-out test set.

Produces the honest generalization numbers (PR-AUC first, never accuracy), the probability
calibration check, the business-cost-driven decision threshold, and SHAP explanations.
Everything here reads the test split exactly once and never refits the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

# WHY force the non-interactive Agg backend before importing pyplot: this script runs
# headless (CI / a plain terminal) and must save PNGs, not try to open a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import get_feature_names  # noqa: E402
from src.train import PIPELINE_PATH, split_data  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DECISION_PATH = MODELS_DIR / "decision.json"

# --- Business cost assumptions -------------------------------------------------------
# WHY these two numbers, stated loudly: the "right" decision threshold is a business
# decision, not a statistical default of 0.5. We assume:
#   * A false negative (we fail to flag a churner, they leave) costs FN_COST — the lost
#     margin / customer lifetime value we forfeit. Illustrative value: $500.
#   * A false positive (we flag a loyal customer and hand them a retention offer they did
#     not need) costs FP_COST — the wasted incentive/outreach. Illustrative value: $100.
# The 5:1 ratio is what matters, not the absolute dollars: missing a churner is far more
# expensive than a wasted offer, so the optimal threshold sits well below 0.5 — we
# deliberately accept more false alarms to catch more real churners. Swap these for real
# finance numbers and the threshold sweep re-optimizes automatically.
FN_COST = 500.0
FP_COST = 100.0


def _positive_probabilities(pipeline, X) -> np.ndarray:
    """Return P(churn) for each row from the pipeline's predict_proba."""
    return pipeline.predict_proba(X)[:, 1]


def sweep_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Find the probability threshold minimizing expected business cost.

    WHY sweep cost rather than maximize F1 or accuracy: F1 implicitly weights precision and
    recall equally, and accuracy weights every error equally — but here a missed churner
    costs 5x a false alarm. Only an explicit cost function encodes that asymmetry, so the
    chosen threshold reflects the actual economics instead of a metric's built-in symmetry.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        # Simplifying assumption (commented so nobody mistakes it for gospel): we price only
        # the two error types. Retention offers to correctly-flagged churners (TP) are a
        # separate campaign-budget question, held out of the threshold math here.
        cost = FN_COST * fn + FP_COST * fp
        rows.append(
            {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "total_cost": cost, "cost_per_customer": cost / len(y_true)}
        )
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["total_cost"].idxmin(), "threshold"]
    return float(best), sweep


def _metrics_at(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
    }


def _save_curves(y_test, proba, sweep, best_threshold):
    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve — held-out test"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=120); plt.close()

    # Precision-Recall curve (the one that matters for imbalanced churn).
    prec, rec, _ = precision_recall_curve(y_test, proba)
    baseline = y_test.mean()
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f"PR (AP={average_precision_score(y_test, proba):.3f})")
    plt.hlines(baseline, 0, 1, colors="grey", linestyles="--",
               label=f"no-skill ({baseline:.3f})")
    plt.xlabel("Recall (churners caught)"); plt.ylabel("Precision (offers well spent)")
    plt.title("Precision-Recall curve — held-out test"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=120); plt.close()

    # Calibration curve.
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    plt.xlabel("Mean predicted P(churn)"); plt.ylabel("Observed churn frequency")
    plt.title("Calibration curve — held-out test"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=120); plt.close()

    # Cost vs threshold.
    plt.figure(figsize=(6, 4))
    plt.plot(sweep["threshold"], sweep["cost_per_customer"])
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"min-cost @ {best_threshold:.3f}")
    plt.axvline(0.5, color="grey", linestyle=":", label="default 0.5")
    plt.xlabel("Decision threshold"); plt.ylabel("Expected cost per customer ($)")
    plt.title(f"Business cost vs threshold (FN=${FN_COST:.0f}, FP=${FP_COST:.0f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cost_vs_threshold.png", dpi=120); plt.close()

    # Confusion matrix at the chosen business threshold.
    pred = (proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["stay", "churn"]); plt.yticks([0, 1], ["stay", "churn"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix @ {best_threshold:.3f}"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120); plt.close()


def _shap_analysis(pipeline, X_test):
    """Global SHAP summary + dependence plots for the top drivers.

    WHY explain the transformed matrix and not the raw frame: SHAP attributes importance to
    the columns the classifier actually sees — the scaled numerics and one-hot dummies —
    so we run the raw rows through the fitted preprocessing (never refitting it) and feed
    the resulting named matrix to the explainer.
    """
    pre = pipeline[:-1]  # engineer + preprocess, already fitted
    clf = pipeline.named_steps["clf"]
    X_trans = pre.transform(X_test)
    names = get_feature_names(pipeline.named_steps["preprocess"])
    X_df = pd.DataFrame(X_trans, columns=names)

    # TreeExplainer covers the RF/XGB winners exactly; fall back to the model-agnostic
    # explainer for the linear baseline.
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_df)
    except Exception:
        explainer = shap.Explainer(clf, X_df)
        shap_values = explainer(X_df).values

    # Normalize to the positive-class 2-D matrix across shap's version-dependent shapes.
    sv = np.asarray(shap_values)
    if sv.ndim == 3:  # (n, features, classes) — take churn class
        sv = sv[:, :, 1]
    elif isinstance(shap_values, list):  # [class0, class1]
        sv = np.asarray(shap_values[1])

    shap.summary_plot(sv, X_df, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Dependence plots for the three globally most important features.
    top = np.argsort(np.abs(sv).mean(axis=0))[::-1][:3]
    for idx in top:
        feat = names[idx]
        shap.dependence_plot(idx, sv, X_df, show=False)
        safe = feat.replace("__", "_").replace(" ", "_").replace("/", "_")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{safe}.png", dpi=120, bbox_inches="tight")
        plt.close()
    return [names[i] for i in top]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _, X_test, _, y_test = split_data()
    pipeline = joblib.load(PIPELINE_PATH)

    proba = _positive_probabilities(pipeline, X_test)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    best_threshold, sweep = sweep_threshold(y_test, proba)
    _save_curves(y_test, proba, sweep, best_threshold)
    top_features = _shap_analysis(pipeline, X_test)

    default_metrics = _metrics_at(y_test, proba, 0.5)
    business_metrics = _metrics_at(y_test, proba, best_threshold)
    min_cost_row = sweep.loc[sweep["threshold"].sub(best_threshold).abs().idxmin()]
    cost_at_half = sweep.loc[sweep["threshold"].sub(0.5).abs().idxmin(), "cost_per_customer"]

    # Ranking metrics are threshold-independent (they summarize the probability ranking
    # over all cutoffs), so they get one column. Accuracy is deliberately absent — see the
    # module and train.py docstrings for why it is the wrong headline here.
    print("\n=== Held-out test results ===")
    print("Ranking quality (threshold-independent):")
    ranking = pd.DataFrame(
        {"metric": ["ROC-AUC", "PR-AUC (average precision)"],
         "value": [round(roc_auc, 4), round(pr_auc, 4)]}
    )
    print(ranking.to_string(index=False))

    # Threshold-dependent metrics: contrast the naive 0.5 cutoff against the cost-optimal one
    # so the tradeoff (business threshold trades precision for the recall that saves churners)
    # is explicit.
    print("\nDecision quality (threshold-dependent):")
    decisions = pd.DataFrame(
        {
            "metric": ["precision", "recall", "f1"],
            "@ default 0.5": [default_metrics["precision"], default_metrics["recall"], default_metrics["f1"]],
            f"@ business {best_threshold:.3f}": [business_metrics["precision"], business_metrics["recall"], business_metrics["f1"]],
        }
    )
    print(decisions.to_string(index=False))

    print(f"\nBusiness-optimal threshold: {best_threshold:.3f} (FN=${FN_COST:.0f}, FP=${FP_COST:.0f})")
    print(f"Expected cost/customer: ${min_cost_row['cost_per_customer']:.2f} at {best_threshold:.3f} "
          f"vs ${cost_at_half:.2f} at 0.5  "
          f"(saving ${cost_at_half - min_cost_row['cost_per_customer']:.2f}/customer)")
    print(f"Top SHAP drivers: {top_features}")

    # Persist the decision so predict.py and the app score at the SAME threshold the
    # business analysis chose — never a silent 0.5.
    DECISION_PATH.write_text(json.dumps(
        {"threshold": best_threshold, "fn_cost": FN_COST, "fp_cost": FP_COST,
         "roc_auc": round(float(roc_auc), 4), "pr_auc": round(float(pr_auc), 4),
         "top_features": top_features}, indent=2))
    print(f"\nsaved decision -> {DECISION_PATH}")


if __name__ == "__main__":
    main()
