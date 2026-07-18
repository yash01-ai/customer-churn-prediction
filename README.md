# Customer Churn Prediction

Predict which telecom customers are about to leave, so retention spend can be aimed at the
customers who are both **likely to churn** and **expensive to lose** — not sprayed evenly or
wasted on people who were never going to leave.

This is an end-to-end, production-shaped project: cleaning, leakage-safe preprocessing,
model selection, business-cost threshold tuning, explainability, a single-record scoring
API, and a Streamlit app — all wired through one persisted scikit-learn `Pipeline`.

## Problem framing

Telecom churn is a **class-imbalanced, cost-asymmetric** problem:

- Only **~26.5%** of customers churn, so a model that predicts "nobody leaves" is ~73.5%
  accurate and completely useless. **Accuracy is the wrong headline metric.**
- The two mistakes cost very different amounts. Missing a churner (false negative) forfeits
  a customer's lifetime value; flagging a loyal customer (false positive) wastes a small
  retention offer. The decision threshold has to reflect that asymmetry, not default to 0.5.

The project optimizes and reports **PR-AUC (average precision)** and **recall on the churn
class**, and chooses its operating threshold by **minimizing expected business cost**.

## Dataset

IBM sample **Telco Customer Churn** — 7,043 customers, 19 predictive features, binary
`Churn` target.

- Source: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Place the CSV at `data/raw/telco_churn.csv` (the `data/` directory is git-ignored).

Notable data quirk handled deliberately: `TotalCharges` ships as text because **11 brand-new
customers (`tenure == 0`) have a blank value**. Their true total spend is `$0`, so they are
filled with 0 — not the column mean, which would fabricate ~$2,000 of spend. See
`src/data.py`.

## Approach

1. **Clean** (`src/data.py`) — split-independent fixes only: coerce `TotalCharges`, fill the
   tenure-0 blanks with 0, drop `customerID`, map `Churn` to 1/0.
2. **Preprocess, leakage-safe** (`src/features.py`) — a `ColumnTransformer`
   (`StandardScaler` for numerics, `OneHotEncoder(handle_unknown='ignore')` for categoricals)
   that lives **inside** the model `Pipeline`, so it is only ever fit on training folds.
   One engineered feature, `num_addon_services` (a stickiness signal), earns its place.
3. **Train & select** (`src/train.py`) — three candidates sharing that preprocessor:
   Logistic Regression (interpretable baseline), Random Forest, and XGBoost. Imbalance is
   handled by **class weighting** (`class_weight='balanced'` / `scale_pos_weight`), not
   SMOTE. Tree models are tuned with `RandomizedSearchCV` scored on **PR-AUC** under
   stratified 5-fold CV. The best pipeline by CV PR-AUC is persisted whole.
4. **Evaluate** (`src/evaluate.py`) — held-out test metrics, ROC / PR / calibration curves,
   a **business-cost threshold sweep**, and SHAP global + dependence explanations.
5. **Serve** (`src/predict.py`, `app/streamlit_app.py`) — score a single customer as a dict
   or through a form, returning probability, the label at the business threshold, and the
   SHAP drivers behind that specific prediction.

## Key decisions (the ones worth defending)

- **No leakage, ever.** Every transformer is fit inside the `Pipeline` on training folds
  only. Nothing — scaler, encoder, engineered feature — sees the test rows or the validation
  slice of a CV fold. The train/test split is defined once and shared by training and
  evaluation so the held-out set is touched exactly once.
- **PR-AUC over ROC-AUC over accuracy.** With a 3:1 majority, accuracy rewards the trivial
  model and ROC-AUC is flattered by easy true-negatives. PR-AUC summarizes precision/recall
  on the churn class we actually spend money on.
- **Class weighting over SMOTE.** SMOTE only avoids leakage if it is re-fit inside every CV
  fold; cost reweighting changes only the loss, never the data, so it is leakage-safe by
  construction and simpler to reason about.
- **Business-cost threshold, not 0.5.** We assume a missed churner costs ~5× a wasted
  retention offer (illustrative $500 vs $100) and pick the threshold that minimizes expected
  cost. Swap in real finance numbers and the threshold re-optimizes. The chosen threshold is
  saved to `models/decision.json` and used by both the API and the app.
- **One `Pipeline` artifact.** Preprocessing and model are persisted together, so inference
  is a single `predict_proba` call that cannot drift from how the model was trained.

## Results

Held-out 20% test set (stratified). Selected model: **XGBoost** (highest CV PR-AUC).

**Model comparison (5-fold CV PR-AUC on the training split):**

| Model | CV PR-AUC |
|---|---|
| Logistic Regression (baseline) | 0.660 |
| Random Forest | 0.663 |
| **XGBoost (selected)** | **0.670** |

**Held-out test performance (XGBoost):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.847 |
| **PR-AUC (average precision)** | **0.663** |

| Decision metric | @ default 0.5 | @ business threshold (0.295) |
|---|---|---|
| Precision | 0.52 | 0.43 |
| Recall | 0.80 | **0.93** |
| F1 | 0.63 | 0.59 |

Lowering the threshold from 0.5 to the cost-optimal **0.295** trades some precision to lift
recall from 80% to **93%** — catching far more real churners — and reduces expected cost from
**$45.92 to $41.87 per customer** (~$4/customer, ~9%) under the stated 5:1 cost assumption.

**Top churn drivers (SHAP):** month-to-month contract, low tenure, and no online-security
add-on — consistent with the EDA, and exactly the levers a retention team can act on.

Figures are written to `reports/figures/` — ROC, precision-recall, calibration, the
cost-vs-threshold curve, the confusion matrix at the business threshold, and SHAP summary /
dependence plots.

## Repository structure

```
customer-churn-prediction/
├── data/
│   ├── raw/telco_churn.csv        # source data (git-ignored)
│   └── processed/
├── notebooks/01_eda.ipynb         # narrative EDA -> modeling decisions
├── src/
│   ├── data.py                    # load + clean
│   ├── features.py                # engineered feature + leakage-safe ColumnTransformer
│   ├── train.py                   # train, tune, select, persist the Pipeline
│   ├── evaluate.py                # metrics, curves, cost threshold, SHAP
│   └── predict.py                 # score a single customer record
├── models/                        # churn_pipeline.joblib + decision.json (generated)
├── reports/figures/               # saved plots
├── app/streamlit_app.py           # interactive scoring UI
├── requirements.txt
└── README.md
```

## Running it

```bash
# 1. Create / activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt
# macOS + XGBoost needs the OpenMP runtime:  brew install libomp

# 3. Put the dataset at data/raw/telco_churn.csv
#    (download from the Kaggle link above)

# 4. Train, select, and persist the winning pipeline
python -m src.train

# 5. Evaluate on the held-out test set (writes figures + models/decision.json)
python -m src.evaluate

# 6. Score a single customer from the command line
python -m src.predict

# 7. Launch the interactive app
streamlit run app/streamlit_app.py
```

All randomness is seeded (`random_state=42`) for reproducibility.
