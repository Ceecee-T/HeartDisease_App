import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="centered")

st.title("Explainable Stacked Ensemble Model for Heart Disease Risk Prediction")

st.write(
    "This interface predicts heart disease risk using the trained stacked ensemble model."
)

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Framingham Heart Study Model", "UCI Heart Disease Model"]
)

if model_choice == "Framingham Heart Study Model":
    model = joblib.load("framingham_model.joblib")
    threshold = 0.36

    st.subheader("Enter Patient Clinical Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    male = st.selectbox("Sex", ["Female", "Male"])
    current_smoker = st.selectbox("Current Smoker", ["No", "Yes"])
    cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=80, value=0)
    bp_meds = st.selectbox("On Blood Pressure Medication", ["No", "Yes"])
    prevalent_stroke = st.selectbox("History of Stroke", ["No", "Yes"])
    prevalent_hyp = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    tot_chol = st.number_input("Total Cholesterol", min_value=100, max_value=700, value=220)
    sys_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
    dia_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose", min_value=40, max_value=400, value=85)
    education = st.selectbox("Education Level", [1, 2, 3, 4])

    input_data = pd.DataFrame([{
        "male": 1 if male == "Male" else 0,
        "age": age,
        "education": education,
        "currentSmoker": 1 if current_smoker == "Yes" else 0,
        "cigsPerDay": cigs_per_day,
        "BPMeds": 1 if bp_meds == "Yes" else 0,
        "prevalentStroke": 1 if prevalent_stroke == "Yes" else 0,
        "prevalentHyp": 1 if prevalent_hyp == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "totChol": tot_chol,
        "sysBP": sys_bp,
        "diaBP": dia_bp,
        "BMI": bmi,
        "heartRate": heart_rate,
        "glucose": glucose
    }])

else:
    model = joblib.load("uci_model.joblib")
    threshold = 0.37

    st.subheader("Enter Patient Clinical Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=55)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=130)
    chol = st.number_input("Cholesterol", min_value=100, max_value=700, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    dataset = st.selectbox("Clinical Site", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

if st.button("Predict Heart Disease Risk"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    st.write(f"Predicted Probability: **{probability:.3f}**")
    st.write(f"Decision Threshold: **{threshold}**")

    if prediction == 1:
        st.error("Prediction: High Heart Disease Risk")
    else:
        st.success("Prediction: Low Heart Disease Risk")