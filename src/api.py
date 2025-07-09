from src import utils, preprocessing

import pandas as pd
import joblib
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

app = FastAPI()

rf_model = joblib.load("models/models_pkl/rf_model.pkl")

ohe_ownership = joblib.load('models/ohe_home_ownership.pkl')

ohe_loan_intent = joblib.load('models/ohe_loan_intent.pkl')

ohe_loan_grade = joblib.load('models/ohe_loan_grade.pkl')

ohe_default =joblib.load('models/ohe_default_on_file.pkl')

class Item(BaseModel):

    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_lenght: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.post("/pred")
def predict(data: Item):
    row = data.dict()
    df = pd.DataFrame([row])

    df.rename(columns=lambda x: x.replace("person_home_ownership", "home_ownership")
                            .replace("cb_person_default_on_file", "default_onfile")
                            .replace("person_emp_lenght", "person_emp_length"), inplace=True)
    

    # One-hot encoding
    ownership_encoded = pd.DataFrame(
        ohe_ownership.transform(df[['home_ownership']]),
        columns=ohe_ownership.get_feature_names_out(['home_ownership'])
    )
    intent_encoded = pd.DataFrame(
        ohe_loan_intent.transform(df[['loan_intent']]),
        columns=ohe_loan_intent.get_feature_names_out(['loan_intent'])
    )
    grade_encoded = pd.DataFrame(
        ohe_loan_grade.transform(df[['loan_grade']]),
        columns=ohe_loan_grade.get_feature_names_out(['loan_grade'])
    )
    default_encoded = pd.DataFrame(
        ohe_default.transform(df[['default_onfile']]),
        columns=ohe_default.get_feature_names_out(['default_onfile'])
    )

    # Rename sesuai nama saat training

    df.drop(columns=[
        'home_ownership', 'loan_intent', 'loan_grade', 'default_onfile'
    ], inplace=True)

    final_df = pd.concat([df, ownership_encoded, intent_encoded, grade_encoded, default_encoded], axis=1)

    # Predict probability
    proba = rf_model.predict_proba(final_df)[0][1]  # Probabilitas kelas 1
    threshold = 0.4  # Ubah sesuai threshold Anda
    predicted_class = 1 if proba >= threshold else 0

    return {
        "probability": round(proba, 4),
        "threshold": threshold,
        "predicted_class": predicted_class
    }

@app.get("/")
def read_root():
    return {"message": "API is not working"}



