[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-approval-ml-njvc7tbe3vlww4dja7bjs8.streamlit.app/)
# Credit Card Approval Probability Prediction

A machine learning system that estimates the probability of credit card approval based on applicant financial and demographic attributes. The project simulates real-world underwriting logic and provides an interactive dashboard for evaluating approval likelihood before submitting a formal credit application.

---

## Problem Statement

Submitting a credit card application triggers a **hard credit inquiry**, which can temporarily lower an applicant's credit score. This project builds a predictive system that estimates the **probability of approval before applying** - helping applicants assess their chances without impacting their credit profile.

---

## System Architecture
```
User Input → Streamlit App → AWS S3 (model load) → Gradient Boosting Pipeline → Probability Output
```

---

## Dataset

**Source:** UCI Credit Approval Dataset (`crx.csv`)

Features include age, debt level, years employed, income, credit score, prior default history, gender, marital status, citizenship, and employment status. Column identities are anonymized in the original dataset; labels have been mapped to interpretable names.

**Target:** Binary approval label (`+` → 1, `-` → 0)

---

## Project Workflow

### 1. Preprocessing Pipeline

Built using a Scikit-learn `ColumnTransformer` inside a `Pipeline` to prevent data leakage — all imputation and scaling is fit on training data only.

- **Numeric features:** Median imputation → Standard scaling
- **Categorical features:** Most-frequent imputation → One-hot encoding
- **Train/test split:** 80/20 with stratification on the target label

### 2. Model Benchmarking

Three models were benchmarked at the default 0.50 threshold:

| Model | Recall | ROC-AUC |
|---|---|---|
| Gradient Boosting | — | 0.9607 |
| SVM | — | 0.9593 |
| AdaBoost | — | — |

Primary metric: **Recall** - minimizes false rejections of creditworthy applicants.

### 3. Model Selection: Gradient Boosting

Despite SVM achieving slightly higher recall at the default threshold, Gradient Boosting was selected as the final model for three reasons:

1. **Equivalent discrimination power** - ROC-AUC scores are nearly identical (0.9607 vs 0.9593), confirming both models carry the same underlying signal. The recall gap at 0.50 is a deployment artifact, not a model quality difference.
2. **Threshold tunability** - Gradient Boosting outputs well-calibrated probabilities, making threshold optimization principled and auditable. This is the correct way to encode credit policy into a model.
3. **Interpretability** - Gradient Boosting provides native feature importances. SVM (RBF kernel) is a black box — a practical liability in any deployed credit decision system.

### 4. Hyperparameter Tuning

`RandomizedSearchCV` with `scoring='recall'` over 5-fold stratified cross-validation. Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_split`.

### 5. Threshold Optimization (Credit Policy Alignment)

The default 0.50 threshold assumes symmetric misclassification costs. In credit lending, costs are asymmetric and cycle-dependent.

**Expansionary policy:** Scan the precision-recall curve to find the highest-precision threshold that still achieves **≥ 90% recall**. This directly encodes a bull-market lending objective - capturing creditworthy applicants at a controlled false positive rate.

### 6. Final Model Performance (Tuned Threshold)

| Metric | Score |
|---|---|
| Recall | ≥ 0.90 |
| ROC-AUC | ~0.96 |
| Validation | 5-Fold Stratified Cross-Validation |

---

## Model Interpretability

The Streamlit dashboard integrates **SHAP explanations**, allowing users to see which features contributed most to each prediction — mirroring explainability requirements used in real-world financial models.

---

## Deployment

Trained model and optimal threshold are stored on **AWS S3** and loaded dynamically by the application.
```
User Input → Streamlit App → AWS S3 (model + threshold) → Prediction + Gauge Chart
```

---

## Tech Stack

Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · SHAP · AWS S3

---

## Repository Structure
```
credit-card-approval-ml/
│
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       └── crx.csv
│
├── models/
│   ├── gb_credit_model.pkl
│   └── optimal_threshold.pkl
│
├── assets/
│   └── architecture.png
│
└── notebooks/
    ├── eda.ipynb
    └── modeling.ipynb
```

---

## Running the Application
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements

- Add model monitoring for distribution drift
- Deploy dashboard publicly via Streamlit Community Cloud
- Expand SHAP visualizations within the dashboard
- Incorporate additional financial features

---

## Author

**Disha Patel** — Computer Science Student  
Interests: Machine Learning · Financial Modeling
