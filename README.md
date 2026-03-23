# Credit Card Approval Probability Prediction

A machine learning system that estimates the probability of credit card approval based on applicant financial and demographic attributes.  
The project simulates real-world underwriting logic and provides an interactive dashboard for evaluating approval likelihood before submitting a formal credit application.

---

## Problem Statement

Submitting a credit card application typically triggers a **hard credit inquiry**, which can temporarily lower an applicant's credit score.

This project builds a predictive system that estimates the **probability of approval before applying**, helping applicants assess their chances without impacting their credit profile.

---

## System Architecture
```
User Input → Streamlit App → AWS S3 (model + threshold) → Gradient Boosting Pipeline → Probability Output
```

---

## Dataset

**Source:** UCI Credit Approval Dataset (crx.csv)

The dataset contains anonymized applicant attributes including age, debt level, employment history, income, credit score, prior default history, and demographic features. These variables simulate features commonly used in credit underwriting models.

> **Note:** Column identities are anonymized in the original dataset. This project is a portfolio demonstration — a production system would use live bureau data (CIBIL, Experian, etc.).

**Target:** Binary approval label (+ → 1, - → 0)  
**Size:** 690 applicants, 15 features

---

## Project Workflow

### 1. Exploratory Data Analysis

- Missing value analysis (12 nulls in Gender/Age, handled via pipeline imputation)
- Outlier detection via boxplots
- Statistical hypothesis testing:
  - t-test: Approved applicants are significantly older (mean 33.7 vs 29.8, p < 0.00001)
  - t-test on log-income: Significant income difference between approved/rejected (p < 10⁻¹⁸)
  - Chi-square: PriorDefault is the strongest categorical predictor (χ² = 355, p < 10⁻⁷⁹)
- Correlation heatmap across numeric features
- Class balance: 55.5% rejected, 44.5% approved — minimal imbalance

---

### 2. Preprocessing Pipeline

Built using a Scikit-learn `ColumnTransformer` inside a `Pipeline` to prevent data leakage — all imputation and scaling is fit on training data only.

- **Numeric features:** Median imputation → Standard scaling
- **Categorical features:** Most-frequent imputation → One-hot encoding
- **Train/test split:** 80/20 with stratification on target label

---

### 3. Model Benchmarking (Default Threshold = 0.50)

Three models benchmarked on the held-out test set. Primary metric: **Recall** — minimizes false rejections of creditworthy applicants.

| Model | Recall | ROC-AUC | F1-Score | Precision |
| --- | --- | --- | --- | --- |
| SVM | 0.9180 | 0.9593 | 0.8889 | 0.8615 |
| AdaBoost | 0.9016 | 0.9606 | 0.8871 | 0.8730 |
| **Gradient Boosting** | **0.8852** | **0.9607** | **0.8780** | **0.8710** |

---

### 4. Model Selection: Gradient Boosting

SVM achieved the highest recall at the default threshold, and AdaBoost 
matched Gradient Boosting's ROC-AUC almost exactly (0.9606 vs 0.9607). 
Despite this, Gradient Boosting was selected as the final model for 
three reasons:

1. **Threshold tunability** - Gradient Boosting outputs well-calibrated 
   probabilities, making threshold optimization principled and auditable. 
   After tuning, Gradient Boosting achieves ≥ 90% recall closing the 
   gap with SVM at the default threshold while giving full control over 
   the precision-recall tradeoff.

2. **Interpretability** - Gradient Boosting provides native feature 
   importances and is compatible with SHAP explainability. SVM (RBF kernel) 
   is a black box, a practical liability in any deployed credit decision 
   system where rejections must be justifiable.

3. **Probability calibration** - Unlike SVM, Gradient Boosting outputs 
   reliable approval probabilities, not just binary decisions. This allows 
   the system to communicate confidence levels to end users, which is 
   critical in a lending context.


---

### 5. Hyperparameter Tuning

`RandomizedSearchCV` with `scoring='recall'` over 5-fold stratified cross-validation.

**Best parameters:**
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1
- subsample: 1.0
- min_samples_split: 2

---

### 6. Threshold Optimization (Credit Policy Alignment)

The default 0.50 threshold assumes symmetric misclassification costs. In credit lending, costs are asymmetric and cycle-dependent.

**Expansionary policy:** Find the highest-precision threshold that still achieves ≥ 90% recall — capturing creditworthy applicants at a controlled false positive rate.

**Selected threshold: 0.449**

---

### 7. Final Model Performance (Tuned Threshold = 0.449)


| Metric | Score |
| --- | --- |
| Recall | 0.9016 |
| ROC-AUC | 0.9607 |
| F1-Score | 0.8871 |
| Precision | 0.8730 |

Validated via **5-fold stratified cross-validation:**
- Mean Recall: 0.8534 ± 0.0459
- Mean ROC-AUC: 0.9363 ± 0.0117

---

### 8. Feature Importance (Top 5)

| Feature | Importance |
| --- | --- |
| PriorDefault (No History) | 0.383 |
| PriorDefault (Has History) | 0.194 |
| Debt | 0.057 |
| Employment Status | 0.055 |
| Income | 0.055 |

Prior default history is overwhelmingly the strongest predictor — consistent with real-world credit underwriting.

---

## Model Interpretability

The dashboard integrates **SHAP explanations**, allowing users to see which features contributed most to each individual prediction — mirroring explainability requirements in real-world financial models.

---

## Deployment

Trained model and optimal threshold stored on **AWS S3**, loaded dynamically by the Streamlit application.
```
User Input → Streamlit App → AWS S3 → Gradient Boosting Pipeline → Probability + Gauge Chart
```

---

## Tech Stack

Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · SHAP · AWS S3

---

## Repository Structure
```
credit-card-approval-ml/
│
├── app.py
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
│   ├── architecture.png
│   ├── feature_importance.png
│   └── model_evaluation_curves.png
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

- Add Expected Loss framework (PD × LGD × EAD)
- Deploy dashboard publicly via Streamlit Community Cloud
- Add model monitoring for distribution drift
- Expand SHAP visualizations within the dashboard
- Incorporate additional financial features (debt-to-income ratio, transaction velocity)

---

## Author

Disha Patel — Computer Science Student  
Interests: Machine Learning · Financial Modeling · Credit Risk