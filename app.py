import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

SAVE_DIR = "saved_streamlit_frameworks"

# friendly labels for each raw input column, used when explaining a
# prediction in plain language instead of showing the encoded/scaled names
FRIENDLY_NAMES = {
    # framingham
    "male": "being male",
    "age": "age",
    "education": "education level",
    "currentSmoker": "being a current smoker",
    "cigsPerDay": "cigarettes smoked per day",
    "BPMeds": "being on blood pressure medication",
    "prevalentStroke": "history of stroke",
    "prevalentHyp": "hypertension",
    "diabetes": "diabetes",
    "totChol": "total cholesterol",
    "sysBP": "systolic blood pressure",
    "diaBP": "diastolic blood pressure",
    "BMI": "BMI",
    "heartRate": "heart rate",
    "glucose": "glucose level",
    # uci
    "sex": "sex",
    "dataset": "clinical site",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ECG result",
    "thalch": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "number of major vessels",
    "thal": "thalassemia result",
}

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
        "background": joblib.load(os.path.join(SAVE_DIR, f"{prefix}_background.pkl")),
    }

@st.cache_resource
def build_explainer(prefix, _model, _background):
    # treats the whole stacked ensemble as a black box, since it's base
    # learners + a meta-learner rather than a single tree model
    def predict_fn(data):
        return _model.predict_proba(data)[:, 1]

    return shap.KernelExplainer(predict_fn, _background)

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

def base_column_name(processed_name, raw_columns):
    # processed names from one-hot encoding look like "male_1.0" or "cp_asymptomatic"
    # strip back down to the original raw column name so it can be looked up
    # in input_df and in FRIENDLY_NAMES
    if processed_name in raw_columns:
        return processed_name

    for raw_col in raw_columns:
        if processed_name.startswith(raw_col + "_"):
            return raw_col

    return processed_name

def describe_feature(processed_name, raw_df):
    raw_col = base_column_name(processed_name, raw_df.columns)
    friendly = FRIENDLY_NAMES.get(raw_col, raw_col)
    raw_value = raw_df.iloc[0][raw_col] if raw_col in raw_df.columns else None
    return friendly, raw_value

def predict_from_raw(raw_df, components):
    processed_array = components["preprocessor"].transform(raw_df)
    feature_names = get_feature_names(components["preprocessor"])

    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    selected_df = processed_df[components["selected_features"]]

    probability = components["model"].predict_proba(selected_df)[:, 1][0]
    threshold = components["threshold"]

    prediction = 1 if probability >= threshold else 0
    label = "High Heart Disease Risk" if prediction == 1 else "Low Heart Disease Risk"

    return probability, threshold, prediction, label, selected_df

def explain_prediction(selected_df, raw_df, components, prefix, top_n=3):
    explainer = build_explainer(prefix, components["model"], components["background"])

    # nsamples kept modest so a click stays responsive
    shap_values = explainer.shap_values(selected_df, nsamples=100)

    row_shap = np.array(shap_values)[0]

    # pair each processed column with its shap value, then translate to
    # a friendly label + the patient's actual raw input value
    raw_contributions = list(zip(selected_df.columns, row_shap))
    raw_contributions.sort(key=lambda c: abs(c[1]), reverse=True)

    described = []
    seen_friendly_names = set()

    for processed_name, shap_value in raw_contributions:
        friendly, raw_value = describe_feature(processed_name, raw_df)

        # avoid listing the same underlying field twice (can happen if a
        # categorical field produced more than one encoded column)
        if friendly in seen_friendly_names:
            continue
        seen_friendly_names.add(friendly)

        described.append((friendly, raw_value, shap_value))

        if len(described) == top_n:
            break

    return described

def format_explanation(contributions, prediction):
    direction_word = "increasing" if prediction == 1 else "lowering"

    parts = []
    for friendly_name, value, shap_value in contributions:
        if isinstance(value, (int, float, np.floating, np.integer)):
            value_str = f"{float(value):.2f}".rstrip("0").rstrip(".")
            parts.append(f"**{friendly_name}** ({value_str})")
        elif isinstance(value, str):
            parts.append(f"**{friendly_name}** ({value})")
        else:
            parts.append(f"**{friendly_name}**")

    if len(parts) > 1:
        feature_list = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    else:
        feature_list = parts[0]

    risk_word = "elevated" if prediction == 1 else "low"

    return (
        f"Your predicted risk is {risk_word} mainly because of {feature_list}, "
        f"which had the largest effect on {direction_word} this prediction."
    )

def risk_tier(probability, threshold):
    # a continuous probability is reduced to three plain-language bands,
    # using the model's own decision threshold as the boundary for "high"
    # and the midpoint between 0 and the threshold as the boundary for
    # "borderline", so the three bands scale with whatever threshold a
    # given dataset's model was tuned to
    if probability >= threshold:
        return "High"
    elif probability >= threshold * 0.6:
        return "Borderline"
    else:
        return "Low"

def plot_contributions(contributions):
    # horizontal bar chart of the top contributing factors, colored by
    # whether each one pushed risk up (red) or down (green)
    labels = [c[0] for c in contributions][::-1]
    shap_vals = [c[2] for c in contributions][::-1]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(labels, shap_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect on predicted risk")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

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
        st.markdown("**Demographics**")
        male = st.selectbox("Sex", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 50)
        education = st.selectbox("Education Level", [1.0, 2.0, 3.0, 4.0])
        st.markdown("**Smoking**")
        currentSmoker = st.selectbox("Current Smoker", ["No", "Yes"])
        cigsPerDay = st.number_input("Cigarettes Per Day", 0.0, 100.0, 0.0)

    with col2:
        st.markdown("**Medical History**")
        BPMeds = st.selectbox("On BP Medication", ["No", "Yes"])
        prevalentStroke = st.selectbox("History of Stroke", ["No", "Yes"])
        prevalentHyp = st.selectbox("Hypertension", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        totChol = st.number_input("Total Cholesterol", 80.0, 700.0, 220.0)

    with col3:
        st.markdown("**Vitals**")
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
    prefix = "Framingham"

else:
    st.subheader("UCI Heart Disease Patient Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age", 18, 100, 55)
        sex = st.selectbox("Sex", ["Female", "Male"])
        dataset = st.selectbox("Clinical Site", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
        st.markdown("**Symptoms**")
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("Resting BP", 70.0, 260.0, 130.0)

    with col2:
        st.markdown("**Blood Work**")
        chol = st.number_input("Cholesterol", 80.0, 700.0, 240.0)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"])
        st.markdown("**Heart Activity**")
        restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Max Heart Rate", 50.0, 250.0, 150.0)
        exang = st.selectbox("Exercise Induced Angina", ["False", "True"])

    with col3:
        st.markdown("**Exercise Test Results**")
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
    prefix = "UCI_Heart_Disease"

st.markdown("---")
st.subheader("Input Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Heart Disease Risk"):
    probability, threshold, prediction, label, selected_df = predict_from_raw(input_df, components)
    tier = risk_tier(probability, threshold)

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Probability", f"{probability:.3f}")
    c2.metric("Decision Threshold", f"{threshold:.3f}")
    c3.metric("Risk Tier", tier)

    # visual probability bar with the decision threshold marked, so it's
    # immediately clear how close the result was to the cutoff
    tier_colors = {"Low": "#2ca02c", "Borderline": "#ff7f0e", "High": "#d62728"}
    st.markdown(
        f"""
        <div style="background-color:#eee; border-radius:8px; height:28px; position:relative; width:100%;">
            <div style="background-color:{tier_colors[tier]}; height:28px; border-radius:8px; width:{probability*100:.1f}%;"></div>
            <div style="position:absolute; left:{threshold*100:.1f}%; top:0; height:28px; border-left:2px dashed #333;"></div>
        </div>
        <p style="font-size:12px; color:#666; margin-top:4px;">Dashed line marks the decision threshold ({threshold:.2f})</p>
        """,
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.error(label)
    else:
        st.success(label)

    with st.spinner("Working out why..."):
        try:
            contributions = explain_prediction(selected_df, input_df, components, prefix)
            explanation_sentence = format_explanation(contributions, prediction)

            st.subheader("Why this prediction?")
            st.markdown(explanation_sentence)

            fig = plot_contributions(contributions)
            st.pyplot(fig)

            with st.expander("See the underlying contribution values"):
                contrib_df = pd.DataFrame(
                    contributions,
                    columns=["Factor", "Your Value", "Contribution (SHAP value)"]
                )
                st.dataframe(contrib_df, use_container_width=True)

        except Exception as e:
            st.info(
                "An explanation could not be generated for this prediction. "
                "The risk result above is still valid."
            )