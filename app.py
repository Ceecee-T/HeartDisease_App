import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
   page_title="Heart Disease Prediction System",
   page_icon="❤️",
   layout="wide"
)


# ============================================================
# LOAD SAVED FRAMEWORK
# ============================================================

@st.cache_resource
def load_framework(dataset_name):
   if dataset_name == "Framingham":
       folder = "saved_framingham_framework"
       prefix = "Framingham"
   else:
       folder = "saved_uci_framework"
       prefix = "UCI_Heart_Disease"

   preprocessor = joblib.load(f"{folder}/{prefix}_preprocessor.pkl")
   selector = joblib.load(f"{folder}/{prefix}_feature_selector.pkl")
   model = joblib.load(f"{folder}/{prefix}_final_model.pkl")
   selected_features = joblib.load(f"{folder}/{prefix}_selected_features.pkl")
   threshold = joblib.load(f"{folder}/{prefix}_threshold.pkl")
   metadata = joblib.load(f"{folder}/{prefix}_metadata.pkl")

   return {
       "preprocessor": preprocessor,
       "selector": selector,
       "model": model,
       "selected_features": selected_features,
       "threshold": threshold,
       "metadata": metadata
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


def predict_patient(raw_df, framework):
   preprocessor = framework["preprocessor"]
   model = framework["model"]
   selected_features = framework["selected_features"]
   threshold = framework["threshold"]

   processed_array = preprocessor.transform(raw_df)
   processed_feature_names = get_feature_names(preprocessor)

   processed_df = pd.DataFrame(
       processed_array,
       columns=processed_feature_names
   )

   selected_df = processed_df[selected_features]

   probability = model.predict_proba(selected_df)[:, 1][0]
   prediction = 1 if probability >= threshold else 0

   return prediction, probability, selected_df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
   "Go to",
   [
       "Home",
       "Patient Prediction",
       "Model Information",
       "About Prototype"
   ]
)

dataset_choice = st.sidebar.selectbox(
   "Select Dataset Framework",
   ["Framingham", "UCI Heart Disease"]
)

framework = load_framework(dataset_choice)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
   st.title("❤️ Explainable Heart Disease Prediction System")

   st.markdown(
       """
       This prototype application implements the final heart disease prediction framework developed in this study.

       The system integrates:

       - Data preprocessing
       - Hybrid feature selection
       - Class imbalance handling
       - Stacked ensemble learning
       - Decision threshold optimization
       - Explainable prediction output

       The model predicts whether a patient is likely to be at **Low Risk** or **High Risk** of heart disease based on clinical input variables.
       """
   )

   st.subheader("Framework Components")

   st.table(
       pd.DataFrame({
           "Component": [
               "Preprocessing",
               "Feature Selection",
               "Prediction Model",
               "Meta-Learner",
               "Decision Threshold",
               "Explainability"
           ],
           "Implemented Method": [
               "Imputation, encoding and scaling",
               "Mutual Information + Correlation Filtering + LASSO",
               "Logistic Regression, Random Forest, SVM and XGBoost",
               "LightGBM",
               f"{framework['threshold']:.3f}",
               "SHAP-supported interpretation"
           ]
       })
   )


# ============================================================
# PATIENT PREDICTION PAGE
# ============================================================

elif page == "Patient Prediction":
   st.title("Patient Heart Disease Risk Prediction")

   st.write(
       "Enter the patient's clinical information below and click **Predict Risk**."
   )

   if dataset_choice == "Framingham":
       st.subheader("Framingham Patient Input")

       col1, col2, col3 = st.columns(3)

       with col1:
           male = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
           age = st.number_input("Age", min_value=20, max_value=100, value=50)
           education = st.selectbox("Education Level", [1.0, 2.0, 3.0, 4.0])
           currentSmoker = st.selectbox("Current Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
           cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0.0, max_value=80.0, value=0.0)

       with col2:
           BPMeds = st.selectbox("On BP Medication", [0.0, 1.0], format_func=lambda x: "No" if x == 0 else "Yes")
           prevalentStroke = st.selectbox("Previous Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
           prevalentHyp = st.selectbox("Prevalent Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
           diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
           totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=700.0, value=240.0)

       with col3:
           sysBP = st.number_input("Systolic BP", min_value=80.0, max_value=300.0, value=130.0)
           diaBP = st.number_input("Diastolic BP", min_value=40.0, max_value=160.0, value=80.0)
           BMI = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
           heartRate = st.number_input("Heart Rate", min_value=40.0, max_value=200.0, value=75.0)
           glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=85.0)

       raw_input = pd.DataFrame([{
           "male": male,
           "age": age,
           "education": education,
           "currentSmoker": currentSmoker,
           "cigsPerDay": cigsPerDay,
           "BPMeds": BPMeds,
           "prevalentStroke": prevalentStroke,
           "prevalentHyp": prevalentHyp,
           "diabetes": diabetes,
           "totChol": totChol,
           "sysBP": sysBP,
           "diaBP": diaBP,
           "BMI": BMI,
           "heartRate": heartRate,
           "glucose": glucose
       }])

   else:
       st.subheader("UCI Heart Disease Patient Input")

       col1, col2, col3 = st.columns(3)

       with col1:
           age = st.number_input("Age", min_value=20, max_value=100, value=55)
           sex = st.selectbox("Sex", ["Male", "Female"])
           dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
           cp = st.selectbox(
               "Chest Pain Type",
               ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
           )

       with col2:
           trestbps = st.number_input("Resting Blood Pressure", min_value=80.0, max_value=250.0, value=130.0)
           chol = st.number_input("Cholesterol", min_value=100.0, max_value=700.0, value=240.0)
           fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [False, True])
           restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])

       with col3:
           thalch = st.number_input("Maximum Heart Rate", min_value=60.0, max_value=250.0, value=150.0)
           exang = st.selectbox("Exercise Induced Angina", [False, True])
           oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
           slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
           ca = st.number_input("Number of Major Vessels", min_value=0.0, max_value=4.0, value=0.0)
           thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

       raw_input = pd.DataFrame([{
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

   st.subheader("Input Preview")
   st.dataframe(raw_input)

   if st.button("Predict Risk"):
       start_time = time.time()

       prediction, probability, selected_df = predict_patient(
           raw_input,
           framework
       )

       end_time = time.time()
       response_time = end_time - start_time

       risk_label = "High Heart Disease Risk" if prediction == 1 else "Low Heart Disease Risk"

       st.subheader("Prediction Result")

       if prediction == 1:
           st.error(f"Prediction: {risk_label}")
       else:
           st.success(f"Prediction: {risk_label}")

       st.metric("Risk Probability", f"{probability:.4f}")
       st.metric("Optimized Decision Threshold", f"{framework['threshold']:.4f}")
       st.metric("Prediction Response Time", f"{response_time:.4f} seconds")

       st.subheader("Selected Features Used by the Model")
       st.dataframe(selected_df)

       st.subheader("Prediction Explanation Summary")

       explanation_df = pd.DataFrame({
           "Selected Feature": selected_df.columns,
           "Processed Value": selected_df.iloc[0].values
       })

       st.dataframe(explanation_df)

       report = pd.DataFrame({
           "Dataset Framework": [dataset_choice],
           "Prediction": [risk_label],
           "Risk Probability": [probability],
           "Optimized Threshold": [framework["threshold"]],
           "Response Time Seconds": [response_time]
       })

       csv_report = report.to_csv(index=False).encode("utf-8")

       st.download_button(
           label="Download Prediction Report",
           data=csv_report,
           file_name="heart_disease_prediction_report.csv",
           mime="text/csv"
       )


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":
   st.title("Model Information")

   st.subheader("Selected Framework")
   st.write(dataset_choice)

   st.subheader("Optimized Threshold")
   st.write(framework["threshold"])

   st.subheader("Selected Features")

   selected_features_df = pd.DataFrame({
       "Selected Features": framework["selected_features"]
   })

   st.dataframe(selected_features_df)

   st.subheader("Framework Metadata")
   st.json(framework["metadata"])


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About Prototype":
   st.title("About the Prototype Application")

   st.markdown(
       """
       This prototype demonstrates the implementation of the proposed explainable heart disease prediction framework.

       The application allows users to:

       - Enter patient clinical variables.
       - Generate heart disease risk predictions.
       - View the predicted probability.
       - Apply the optimized decision threshold.
       - Display selected model features.
       - Download a prediction report.

       The prototype is intended for academic demonstration and decision-support research purposes only.
       It is not a substitute for professional medical diagnosis.
       """
   )
