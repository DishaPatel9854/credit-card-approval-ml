import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import requests
import io

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Credit Risk Approval Engine",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
DEV_MODE = os.getenv('DEV_MODE', 'False').lower() in ('1', 'true', 'yes')
try:
    dev_secret = st.secrets.get('DEV_MODE') if hasattr(st, 'secrets') else None
    if dev_secret is not None:
        DEV_MODE = str(dev_secret).lower() in ('1', 'true', 'yes')
except Exception:
    DEV_MODE = DEV_MODE

@st.cache_resource
def load_model():
    model_url     = "https://credit-risk-model-bucket.s3.eu-north-1.amazonaws.com/gb_credit_model.pkl"
    threshold_url = "https://credit-risk-model-bucket.s3.eu-north-1.amazonaws.com/optimal_threshold.pkl"

    try:
        r = requests.get(model_url, timeout=10)
        r.raise_for_status()
        model = joblib.load(io.BytesIO(r.content))
    except Exception as e:
        return None, None, f"Failed to load model from {model_url}: {e}"

    try:
        r = requests.get(threshold_url, timeout=10)
        r.raise_for_status()
        threshold = joblib.load(io.BytesIO(r.content))
    except Exception as e:
        return None, None, f"Failed to load threshold from {threshold_url}: {e}"

    return model, threshold, None

model, threshold, load_err = load_model()
if load_err is not None:
    st.error(load_err)
    st.stop()

if DEV_MODE:
    with st.expander('Model debug info (pre-check)'):
        try:
            feat_names = model.named_steps['preprocessing'].get_feature_names_out()
            st.write('Preprocessor feature names:', list(feat_names))
        except Exception:
            st.write('Could not extract preprocessor feature names from the pipeline.')
        st.write('Model object:', type(model))
        st.write(f'Decision threshold (loaded): {threshold}')

# ----------------------------
# Alias Mappings (UI -> Model Code)
# ----------------------------
gender_map = {"Male": "a", "Female": "b"}

marital_map = {"Single": "u", "Married": "y"}

bank_customer_map = {
    "Existing Customer": "g",
    "New Customer": "p"
}

prior_default_map = {
    "Previous Default": "t",
    "No Default History": "f"
}

employment_map = {
    "Employed": "t",
    "Unemployed": "f"
}

drivers_license_map = {
    "Has License": "t",
    "No License": "f"
}

citizen_map = {
    "Citizen": "g",
    "Permanent Resident": "p",
    "Other Status": "s"
}

# NOTE: Education & Ethnicity labels are anonymized groups
education_map = {
    "Education Group 1": "w",
    "Education Group 2": "q",
    "Education Group 3": "m",
    "Education Group 4": "r",
    "Education Group 5": "x"
}

ethnicity_map = {
    "Group A": "v",
    "Group B": "h",
    "Group C": "bb",
    "Group D": "j",
    "Group E": "n"
}

# ----------------------------
# App Title
# ----------------------------
st.title("Credit Card Approval Probability Engine")

st.markdown("""
This dashboard estimates the probability of **credit card approval**  
based on applicant financial and demographic attributes.

**Model Used:** Gradient Boosting Classifier  
**Validation:** 5-Fold Stratified Cross Validation  
**Objective:** Maximize recall under expansionary credit conditions
""")

st.divider()

# ----------------------------
# Input Layout
# ----------------------------
st.subheader("Applicant Financial Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18.0, value=30.0)
    debt = st.number_input("Debt Level", value=2.0)
    years_employed = st.number_input("Years Employed", value=1.0)
    income = st.number_input("Annual Income", value=500.0)
    credit_score = st.number_input("Credit Score Index", value=1.0)

with col2:
    gender = st.selectbox("Gender", list(gender_map.keys()))
    marital = st.selectbox("Marital Status", list(marital_map.keys()))
    bank_customer = st.selectbox("Bank Relationship", list(bank_customer_map.keys()))
    prior_default = st.selectbox("Default History", list(prior_default_map.keys()))
    employed = st.selectbox("Employment Status", list(employment_map.keys()))
    drivers_license = st.selectbox("Driver's License", list(drivers_license_map.keys()))
    citizen = st.selectbox("Citizenship Status", list(citizen_map.keys()))
    zip_code = st.text_input("Zip Code", value="")

# Advanced section
with st.expander("Advanced Demographic Attributes"):
    education = st.selectbox("Education Category", list(education_map.keys()))
    ethnicity = st.selectbox("Ethnicity Category", list(ethnicity_map.keys()))

st.divider()

# ----------------------------
# Prediction
# ----------------------------
if st.button("Evaluate Approval Probability"):

    input_data = pd.DataFrame([{
        "Gender": gender_map[gender],
        "Age": age,
        "Debt": debt,
        "Married": marital_map[marital],
        "BankCustomer": bank_customer_map[bank_customer],
        "EducationLevel": education_map[education],
        "Ethnicity": ethnicity_map[ethnicity],
        "YearsEmployed": years_employed,
        "PriorDefault": prior_default_map[prior_default],
        "Employed": employment_map[employed],
        "CreditScore": credit_score,
        "DriversLicense": drivers_license_map[drivers_license],
        "Citizen": citizen_map[citizen],
        "Income": income,
        "ZipCode": zip_code
    }])

    try:
        probability = model.predict_proba(input_data)[0][1]
        prediction  = int((probability >= threshold).astype(int))
    except Exception as e:
        err_msg = "Prediction failed during scoring. Please verify the submitted fields and retry."
        st.error(err_msg)
        if DEV_MODE:
            # Provide actionable diagnostics only in developer mode
            try:
                expected = model.named_steps['preprocessing'].feature_names_in_
            except Exception:
                expected = None
            with st.expander('Model debug info'):
                st.write('Model object:', type(model))
                st.write('Input Data columns provided:', list(input_data.columns))
                if expected is not None:
                    st.write('Preprocessor feature names:', list(expected))
        st.stop()

    st.subheader("Approval Probability")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Approval Likelihood (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "#ff4d4d"},
                {"range": [50, 75], "color": "#ffd633"},
                {"range": [75, 100], "color": "#66cc66"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Decision:** {('✅ APPROVED' if prediction == 1 else '❌ DENIED')} (Threshold: {threshold:.3f})")

    st.subheader("Applicant Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Income",
            "Debt Level",
            "Years Employed",
            "Prior Default"
        ],
        "Value": [
            age,
            income,
            debt,
            years_employed,
            prior_default
        ]
    })

    st.table(summary)

    with st.expander("Model Information"):
        st.markdown(f"""
### Model Details

• Algorithm: Gradient Boosting Classifier  
• Validation: 5-Fold Stratified Cross-Validation  
• Tuning: RandomizedSearchCV (optimized for recall)  
• Decision Threshold: {threshold:.3f}  

### Objective

The model was optimized for **recall**, prioritizing identification of creditworthy applicants during expansionary lending conditions.

### Pipeline

The prediction pipeline includes:

- Missing value imputation
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Gradient Boosting classification

### Decision Threshold

Predictions are made by comparing the probability against {threshold:.3f}.  
Applicants above this threshold are approved under the expansionary policy.

**Dataset Note:** Built on the UCI Credit Approval dataset as a learning portfolio project.  
A production system would use live bureau data (CIBIL, Experian, etc.).
""")