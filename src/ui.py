import streamlit as st
import requests

st.title("Loan Approval Prediction")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Income", min_value=0)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_lenght = st.number_input("Employment Length (years)", min_value=0.0, value=5.0)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=1000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_status = st.selectbox("Loan Status", [0, 1])
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.2)
cb_person_default_on_file = st.selectbox("Previously Defaulted", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0)

data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_lenght": person_emp_lenght,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_status": loan_status,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length
}

if st.button("Predict"):
    try:
        response = requests.post("http://localhost:8000/pred", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Class: {result['predicted_class']}")
            st.info(f"Probability: {result['probability']} | Threshold: {result['threshold']}")
        else:
            st.error("API Error: Unable to get prediction.")
    except Exception as e:
        st.error(f"Connection error: {e}")