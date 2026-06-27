import streamlit as st

st.set_page_config(
    page_title="About | Heart Disease Risk Prediction",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("About This Application")

st.markdown(
    """
This application demonstrates an explainable stacked ensemble model for
predicting heart disease risk, built as part of an MSc Computer Science
thesis on heart disease prediction using machine learning.
"""
)

st.subheader("How the model works")
st.markdown(
    """
The model combines four base learners, Logistic Regression, Random Forest,
Support Vector Machine, and XGBoost, with a LightGBM meta-learner trained on
their out-of-fold predictions (a stacked ensemble). Before training, a
three-stage hybrid feature-selection pipeline narrows the input features down
to the most relevant and non-redundant subset for each dataset:

1. **Mutual Information filtering** — ranks features by their statistical
   dependency on the outcome.
2. **Pearson correlation filtering** — removes features that are highly
   correlated with one another.
3. **LASSO regularization** — shrinks the coefficients of weak or redundant
   predictors to zero, with a minimum-feature safeguard to prevent
   over-pruning.

Each prediction's operating threshold (the cutoff used to decide "high risk"
versus "low risk") was selected by F1-score during development, rather than
a default of 0.50, since both datasets are imbalanced to differing degrees.
"""
)

st.subheader("The two datasets")
st.markdown(
    """
**Framingham Heart Study** — a long-horizon cohort dataset of modifiable and
non-modifiable cardiovascular risk factors (age, blood pressure, cholesterol,
smoking, and similar), used to estimate longer-term heart disease risk.

**UCI Heart Disease** — a pooled dataset from four clinical sites (Cleveland,
Hungary, Switzerland, and the VA Long Beach), built from diagnostic-test
results obtained from patients already undergoing cardiac evaluation
(chest pain type, ST depression, exercise test results, and similar).

These two datasets describe meaningfully different clinical situations, a
long-horizon risk profile versus a point-of-care diagnostic signature, so the
two models in this app are trained, evaluated, and explained separately
rather than combined into one.
"""
)

st.subheader("Explainability")
st.markdown(
    """
Each prediction is accompanied by a SHAP-based explanation, computed with a
model-agnostic explainer (KernelExplainer) wrapped around the full stacked
ensemble. Rather than presenting a raw SHAP plot, the app translates the
result into a plain-language sentence naming the input factors that
contributed most to that specific prediction.
"""
)

st.subheader("Limitations")
st.markdown(
    """
- This is an academic research demonstration, not a diagnostic tool, and is
  not a substitute for professional medical advice.
- The stacked ensemble did not outperform every individual base learner
  under every evaluation condition; its main advantage in this work is a
  tunable precision/recall balance rather than uniform superiority.
- The two models are trained and evaluated independently. An attempt at
  cross-dataset validation (training on one dataset, testing on the other)
  could not be completed, since the two datasets' independently selected
  features shared only one common predictor (age).
- Both models were trained on public research datasets, which may not
  reflect the demographic or clinical diversity of the general population.
"""
)

st.markdown("---")
st.caption(
    "Built as part of an MSc Computer Science thesis on explainable machine "
    "learning for heart disease prediction."
)