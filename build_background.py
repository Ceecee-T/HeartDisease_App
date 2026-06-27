# builds the small background samples shap needs to explain a prediction
# matches SAVE_DIR in app.py, run once locally then commit the two new
# *_background.pkl files into saved_streamlit_frameworks alongside the rest

import os
import joblib
import pandas as pd

SAVE_DIR = "saved_streamlit_frameworks"
N_BACKGROUND_ROWS = 100
framingham_raw_df = pd.read_csv("framingham.csv")
uci_raw_df = pd.read_csv("heart_disease_uci.csv")


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


def build_background(prefix, raw_df):
    preprocessor = joblib.load(os.path.join(SAVE_DIR, f"{prefix}_preprocessor.pkl"))
    selected_features = joblib.load(os.path.join(SAVE_DIR, f"{prefix}_selected_features.pkl"))

    sample_raw = raw_df.sample(
        n=min(N_BACKGROUND_ROWS, len(raw_df)),
        random_state=42
    ).reset_index(drop=True)

    processed_array = preprocessor.transform(sample_raw)
    feature_names = get_feature_names(preprocessor)

    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    selected_df = processed_df[selected_features]

    output_path = os.path.join(SAVE_DIR, f"{prefix}_background.pkl")
    joblib.dump(selected_df, output_path)

    print(f"saved {output_path}", selected_df.shape)


# raw training csvs, same column names the app's input_df builds
framingham_raw_df = pd.read_csv("framingham.csv")
uci_raw_df = pd.read_csv("heart_disease_uci.csv")

build_background("Framingham", framingham_raw_df)
build_background("UCI_Heart_Disease", uci_raw_df)