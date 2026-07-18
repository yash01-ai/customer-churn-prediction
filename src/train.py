"""Train, tune, and select the churn model, then persist the full Pipeline.

Every candidate is a full Pipeline (feature engineering -> preprocessing -> classifier),
so cross-validation refits the preprocessing on each training fold and inference is a
single `.predict_proba` call that can never drift from how the model was trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import load_xy
from src.features import build_feature_engineering_step, build_preprocessor

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.json"

# WHY the threading backend instead of joblib's default (loky) process pool: loky spawns a
# multiprocessing resource_tracker whose child process throws a harmless but alarming
# ChildProcessError traceback at interpreter exit on CPython 3.12 / macOS — after a fully
# successful run. Because RandomForest and XGBoost build trees in Cython/C++ that release
# the GIL, threads still parallelize the fits, so we get clean output with no meaningful
# speed loss and, critically, no scary shutdown noise for anyone running the script.
PARALLEL_BACKEND = "threading"

# WHY average_precision (PR-AUC) is the selection metric and not ROC-AUC or accuracy:
#   - Accuracy is disqualified: at a 26.5% churn rate, predicting "nobody churns" scores
#     73.5% accuracy while catching zero churners. Optimizing it rewards the useless model.
#   - ROC-AUC counts true-negatives in its favor. With retained customers 3:1 more common,
#     a model looks great on ROC just by ranking the easy majority well. PR-AUC ignores true
#     negatives entirely and summarizes precision/recall on the *positive* (churn) class —
#     exactly the customers we spend money to retain. It is the honest headline for an
#     imbalanced detect-the-minority problem.
SCORING = "average_precision"

# WHY 5-fold stratified and shuffled: stratification keeps the ~26.5% churn ratio in every
# fold (an unstratified split can hand a fold too few churners to score PR-AUC stably);
# shuffling with a fixed seed removes any ordering artifact while staying reproducible.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def split_data(test_size: float = 0.20):
    """Return X_train, X_test, y_train, y_test.

    WHY this split lives in one function imported by both train.py and evaluate.py: both
    must see the *identical* held-out test set. Because train_test_split is deterministic
    under a fixed random_state and stratify, calling this from either module reconstructs
    the exact same partition — the test rows never leak into training, and evaluation
    never accidentally scores on rows the model saw.
    """
    X, y = load_xy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        # WHY stratify=y: preserve the churn rate in both splits so the test PR-AUC is
        # measured against a realistic base rate, not a lucky/unlucky draw of churners.
        stratify=y,
    )


def _make_pipeline(classifier) -> Pipeline:
    """Wrap any classifier with the shared, leakage-safe preprocessing steps."""
    return Pipeline(
        steps=[
            ("engineer", build_feature_engineering_step()),
            ("preprocess", build_preprocessor()),
            ("clf", classifier),
        ]
    )


def _candidate_models(y_train) -> dict:
    """Define the three candidate estimators and their imbalance handling."""
    # WHY class_weight="balanced" / scale_pos_weight instead of SMOTE oversampling:
    #   - Correctness: SMOTE must be fit *inside each CV fold* (only on that fold's training
    #     rows) or it synthesizes minority points using validation neighbors and leaks. Cost
    #     reweighting has no such trap — it only changes the loss, never the data — so it is
    #     leakage-safe by construction with an ordinary Pipeline.
    #   - Simplicity/honesty: reweighting makes each missed churner as expensive to the loss
    #     as the class imbalance is skewed, directly targeting the costly error (a missed
    #     churner is a lost customer; a false alarm is a cheap retention offer) without
    #     inventing synthetic customers that may not resemble anyone real.
    # (imbalanced-learn is still pinned in requirements so a SMOTE variant is a drop-in
    #  experiment, but class weighting is the correct default here.)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    logreg = LogisticRegression(
        class_weight="balanced",
        # WHY max_iter raised well above the default 100: the L-BFGS solver on this scaled,
        # one-hot-expanded feature space needs more iterations to converge; the default warns
        # and stops early, giving a misleadingly weak baseline.
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        # WHY n_jobs=1 on the estimator even though RF is embarrassingly parallel: the outer
        # RandomizedSearchCV/cross_val_score already fan out across cores (see PARALLEL_BACKEND
        # above). If the inner model ALSO grabbed every core, the two layers would multiply into
        # far more workers than CPUs — oversubscription that thrashes the scheduler and runs
        # slower, not faster. Parallelize once, at the search layer, where there are many more
        # independent fits (n_iter x n_folds) to keep the cores busy.
        n_jobs=1,
    )

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        # WHY logloss eval metric + hist tree method: logloss matches the probabilistic
        # objective we ultimately threshold on; "hist" is the fast, well-tested split finder
        # and keeps tuning tractable on repeated CV fits.
        eval_metric="logloss",
        tree_method="hist",
        # Single-threaded for the same reason as RF above — the search layer parallelizes.
        n_jobs=1,
    )

    return {"logreg": logreg, "random_forest": rf, "xgboost": xgb}


# Search spaces kept deliberately compact: this dataset is small, so wide grids mostly buy
# variance. These ranges span the meaningful regularization/capacity trade-offs for each model.
RF_PARAMS = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [None, 6, 10, 16, 24],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5],
}

XGB_PARAMS = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [3, 4, 5, 6, 8],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
    "clf__reg_lambda": [1.0, 2.0, 5.0],
}


def train_and_select(n_iter: int = 25):
    """Train all candidates, tune the tree models, and return (best_name, best_pipeline, report)."""
    X_train, X_test, y_train, y_test = split_data()
    models = _candidate_models(y_train)

    results: dict[str, float] = {}

    # Baseline: plain 5-fold CV (no search space). WHY leave LR untuned: it is the
    # interpretable floor the tree models must beat to justify their opacity. Tuning it too
    # would blur that comparison; the default regularized LR is exactly the "simplest thing
    # that could work" reference.
    logreg_pipe = _make_pipeline(models["logreg"])
    with parallel_backend(PARALLEL_BACKEND):
        logreg_cv = cross_val_score(logreg_pipe, X_train, y_train, scoring=SCORING, cv=CV, n_jobs=-1)
    results["logreg"] = float(logreg_cv.mean())
    print(f"[logreg ] CV PR-AUC = {logreg_cv.mean():.4f} +/- {logreg_cv.std():.4f}")

    fitted: dict[str, Pipeline] = {"logreg": logreg_pipe.fit(X_train, y_train)}

    # WHY RandomizedSearchCV over GridSearchCV: with 4-7 hyperparameters, a full grid is
    # thousands of fits for marginal gain. A fixed-seed random sample of n_iter points
    # explores the same ranges at a fraction of the cost and is reproducible.
    for name, params in (("random_forest", RF_PARAMS), ("xgboost", XGB_PARAMS)):
        search = RandomizedSearchCV(
            _make_pipeline(models[name]),
            param_distributions=params,
            n_iter=n_iter,
            scoring=SCORING,
            cv=CV,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        with parallel_backend(PARALLEL_BACKEND):
            search.fit(X_train, y_train)
        results[name] = float(search.best_score_)
        fitted[name] = search.best_estimator_
        print(f"[{name:8s}] CV PR-AUC = {search.best_score_:.4f}  best={search.best_params_}")

    # WHY select on cross-validated PR-AUC and not on test-set score: the test set is the
    # single honest estimate of generalization and must not influence model choice, or it
    # stops being held-out. CV on the training data is the selection signal; the test set is
    # touched only once, in evaluate.py.
    best_name = max(results, key=results.get)
    best_pipeline = fitted[best_name]
    print(f"\nselected: {best_name}  (CV PR-AUC = {results[best_name]:.4f})")

    report = {
        "selection_metric": SCORING,
        "cv_pr_auc": results,
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_churn_rate": float(y_train.mean()),
    }
    return best_name, best_pipeline, report


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _, best_pipeline, report = train_and_select()

    # Persist the FULL fitted Pipeline: preprocessing + classifier travel together as one
    # artifact, so scoring is load-and-call with zero chance of a preprocessing mismatch.
    joblib.dump(best_pipeline, PIPELINE_PATH)
    METADATA_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nsaved pipeline -> {PIPELINE_PATH}")
    print(f"saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
