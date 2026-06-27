import os
import joblib
import pandas as pd
import streamlit as st

SAVE_DIR = "saved_streamlit_frameworks"

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

@st.cache_resource
def load_components(prefix):
    return {
        "preprocessor": joblib.load(os.path.join(SAVE_DIR, f"{prefix}_preprocessor.pkl")),
        "model": joblib.load(os.path.join(SAVE_DIR, f"{prefix}_final_model.pkl")),
        "threshold": joblib.load(os.path.join(SAVE_DIR, f"{prefix}_threshold.pkl")),
        "selected_features": joblib.load(os.path.join(SAVE_DIR, f"{prefix}_selected_features.pkl")),
    }

def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if name == "numeric":
            feature_names.extend(columns)

        elif name == "categorical":
            encoder = transformer.named_steps["encoder"]
            encoded_names = encoder.get_feature_names_out(columns)
            feature_names.extend(encoded_names)

    return list(feature_names)

def predict_from_raw(raw_df, components):
    processed_array = components["preprocessor"].transform(raw_df)
    feature_names = get_feature_names(components["preprocessor"])

    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    selected_df = processed_df[components["selected_features"]]

    probability = components["model"].predict_proba(selected_df)[:, 1][0]
    threshold = components["threshold"]

    prediction = 1 if probability >= threshold else 0
    label = "High Heart Disease Risk" if prediction == 1 else "Low Heart Disease Risk"

    return probability, threshold, prediction, label

framingham = load_components("Framingham")
uci = load_components("UCI_Heart_Disease")

st.title("Explainable Stacked Ensemble Model for Heart Disease Risk Prediction")

st.warning(
    "This application is for academic research demonstration only and is not a substitute for professional medical diagnosis."
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Framingham", "UCI Heart Disease"]
)

if model_choice == "Framingham":
    st.subheader("Framingham Patient Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        male = st.selectbox("Sex", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 50)
        education = st.selectbox("Education Level", [1.0, 2.0, 3.0, 4.0])
        currentSmoker = st.selectbox("Current Smoker", ["No", "Yes"])
        cigsPerDay = st.number_input("Cigarettes Per Day", 0.0, 100.0, 0.0)

    with col2:
        BPMeds = st.selectbox("On BP Medication", ["No", "Yes"])
        prevalentStroke = st.selectbox("History of Stroke", ["No", "Yes"])
        prevalentHyp = st.selectbox("Hypertension", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        totChol = st.number_input("Total Cholesterol", 80.0, 700.0, 220.0)

    with col3:
        sysBP = st.number_input("Systolic BP", 70.0, 260.0, 120.0)
        diaBP = st.number_input("Diastolic BP", 40.0, 160.0, 80.0)
        BMI = st.number_input("BMI", 10.0, 80.0, 25.0)
        heartRate = st.number_input("Heart Rate", 40.0, 220.0, 75.0)
        glucose = st.number_input("Glucose", 40.0, 500.0, 85.0)

    input_df = pd.DataFrame([{
        "male": 1 if male == "Male" else 0,
        "age": age,
        "education": education,
        "currentSmoker": 1 if currentSmoker == "Yes" else 0,
        "cigsPerDay": cigsPerDay,
        "BPMeds": 1 if BPMeds == "Yes" else 0,
        "prevalentStroke": 1 if prevalentStroke == "Yes" else 0,
        "prevalentHyp": 1 if prevalentHyp == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose
    }])

    components = framingham

else:
    st.subheader("UCI Heart Disease Patient Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 55)
        sex = st.selectbox("Sex", ["Female", "Male"])
        dataset = st.selectbox("Clinical Site", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("Resting BP", 70.0, 260.0, 130.0)

    with col2:
        chol = st.number_input("Cholesterol", 80.0, 700.0, 240.0)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"])
        restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Max Heart Rate", 50.0, 250.0, 150.0)
        exang = st.selectbox("Exercise Induced Angina", ["False", "True"])

    with col3:
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
        ca = st.number_input("Number of Major Vessels", 0.0, 4.0, 0.0)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    input_df = pd.DataFrame([{
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

    components = uci

st.markdown("---")
st.subheader("Input Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Heart Disease Risk"):
    probability, threshold, prediction, label = predict_from_raw(input_df, components)

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    c1.metric("Risk Probability", f"{probability:.3f}")
    c2.metric("Decision Threshold", f"{threshold:.3f}")
    c3.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")

    if prediction == 1:
        st.error(label)
    else:
        st.success(label)