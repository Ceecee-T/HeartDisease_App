# builds the small background samples shap needs to explain a prediction
# run this once locally, then commit the two .joblib files it creates
# alongside framingham_model.joblib and uci_model.joblib

import pandas as pd
import joblib

N_BACKGROUND_ROWS = 100

# raw training data, same column names/order as what the app builds
# in input_data, since the pipeline does its own preprocessing internally
framingham_raw_df = pd.read_csv("framingham.csv")
uci_raw_df = pd.read_csv("uci_heart_disease.csv")

framingham_background = framingham_raw_df.sample(
    n=min(N_BACKGROUND_ROWS, len(framingham_raw_df)),
    random_state=42
).reset_index(drop=True)

uci_background = uci_raw_df.sample(
    n=min(N_BACKGROUND_ROWS, len(uci_raw_df)),
    random_state=42
).reset_index(drop=True)

joblib.dump(framingham_background, "framingham_background.joblib")
joblib.dump(uci_background, "uci_background.joblib")

print("saved framingham_background.joblib", framingham_background.shape)
print("saved uci_background.joblib", uci_background.shape)