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
@st.cache_resource
def load_model():

    url = "https://credit-risk-model-bucket.s3.amazonaws.com/svm_credit_model.pkl"

    response = requests.get(url)

    model = joblib.load(io.BytesIO(response.content))

    return model

model = load_model()

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

**Model Used:** Support Vector Machine  
**Validation:** 5-Fold Cross Validation  
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
    zipcode = st.text_input("Zip Code", "000")

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
        "ZipCode": zipcode,
        "Income": income
    }])

    probability = model.predict_proba(input_data)[0][1]

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
        st.markdown("""
### Model Details

• Algorithm: Support Vector Machine  
• Validation: 5-Fold Stratified Cross-Validation  
• Mean Recall: ~0.91  
• Mean ROC-AUC: ~0.93  

### Objective

The model was optimized for **recall**, prioritizing identification of creditworthy applicants during expansionary lending conditions.

### Pipeline

The prediction pipeline includes:

- Missing value imputation
- One-hot encoding for categorical features
- Standard scaling for numeric features
- SVM classification
""")