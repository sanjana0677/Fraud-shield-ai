
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS AND PAGE CONFIGURATION
# ---------------------------------------------------------

# Supports files uploaded directly to GitHub root
# and the original models/, data/, results/ folders.

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" if (BASE_DIR / "models").is_dir() else BASE_DIR
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else BASE_DIR
RESULTS_DIR = BASE_DIR / "results" if (BASE_DIR / "results").is_dir() else BASE_DIR

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_PATH = MODEL_DIR / "final_xgboost_model.json"
FEATURES_PATH = MODEL_DIR / "selected_features.json"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"

SAMPLE_PATH = DATA_DIR / "sample_transaction.csv"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #172554;
    }

    .sub-title {
        color: #64748b;
        font-size: 15px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1e293b;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# ---------------------------------------------------------

@st.cache_resource
def load_model_files():

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURES_PATH, "r") as file:
        feature_data = json.load(file)

    if isinstance(feature_data, dict):
        features = feature_data.get("selected_features", [])
    else:
        features = feature_data

    # Load saved threshold or use 0.85 if missing.
    if THRESHOLD_PATH.exists():

        with open(THRESHOLD_PATH, "r") as file:
            threshold_data = json.load(file)

        if isinstance(threshold_data, dict):
            threshold = threshold_data.get(
                "threshold",
                threshold_data.get("best_threshold", 0.85)
            )
        else:
            threshold = threshold_data

    else:
        threshold = 0.85

    return preprocessor, model, features, float(threshold)


@st.cache_data
def load_sample_transaction():

    if not SAMPLE_PATH.exists():
        return None

    return pd.read_csv(SAMPLE_PATH)


@st.cache_resource
def load_shap_explainer(_model):

    return shap.TreeExplainer(_model)


try:

    preprocessor, model, FEATURES, THRESHOLD = load_model_files()

    if not FEATURES:
        st.error("Selected features could not be loaded.")
        st.stop()

except Exception as error:

    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ---------------------------------------------------------
# GENERAL HELPER FUNCTIONS
# ---------------------------------------------------------

def page_header(title, subtitle):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sub-title">{subtitle}</div>',
        unsafe_allow_html=True
    )

    st.divider()


def result_file(relative_path):

    # First check the original results folder structure.
    nested_path = RESULTS_DIR / relative_path

    if nested_path.exists():
        return nested_path

    # Then check the file directly in the GitHub root.
    root_path = BASE_DIR / Path(relative_path).name

    return root_path


def prepare_input(data):

    """
    Ensures that the input contains exactly the selected
    61 features in the correct order.
    """

    data = data.copy()

    missing = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    data = data[FEATURES]

    return data


def get_prediction(input_data):

    """
    Uses the saved preprocessing pipeline and trained model.
    No retraining takes place.
    """

    input_data = prepare_input(input_data)

    transformed = preprocessor.transform(input_data)

    probabilities = model.predict_proba(transformed)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    return probabilities, predictions


def prediction_label(probability):

    if probability >= THRESHOLD:
        return "Potential Fraud"

    return "Legitimate Risk"


def display_prediction(probability):

    prediction = prediction_label(probability)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{THRESHOLD * 100:.0f}%"
        )

    if probability >= THRESHOLD:

        st.error(
            "🚨 Potential Fraud Detected. "
            "This transaction should be reviewed."
        )

    else:

        st.success(
            "✅ Legitimate Risk. "
            "The model did not classify this transaction as fraud."
        )


def save_current_transaction(data, source):

    """
    Stores the latest transaction in session state.
    The Explainable AI page reads this same transaction.
    """

    data = prepare_input(data)

    st.session_state["current_transaction"] = data.copy()
    st.session_state["current_source"] = source

    st.session_state.pop("current_shap_values", None)


def get_current_transaction():

    if "current_transaction" in st.session_state:
        return st.session_state["current_transaction"].copy()

    sample = load_sample_transaction()

    if sample is not None:

        sample = prepare_input(sample.head(1))

        return sample

    return None


def get_preprocessor_categorical_features():

    """
    Attempts to identify categorical input features from
    the fitted ColumnTransformer.
    """

    categorical = set()

    try:

        for name, transformer, columns in preprocessor.transformers_:

            if name.lower() in ("cat", "categorical"):

                if isinstance(columns, (list, tuple, np.ndarray)):
                    categorical.update(columns)

    except Exception:
        pass

    return categorical


# ---------------------------------------------------------
# MANUAL TRANSACTION INPUT
# ---------------------------------------------------------

def manual_transaction_input():

    st.markdown("### Enter Transaction Details")

    st.info(
        "Enter transaction attributes below. "
        "The existing trained model will predict whether "
        "the transaction is potentially fraudulent."
    )

    sample = load_sample_transaction()

    if sample is None:

        st.error(
            "Sample transaction file is required to provide "
            "default values for the 61 input fields."
        )

        return

    default_row = sample.iloc[0].copy()

    categorical_features = get_preprocessor_categorical_features()

    entered_values = {}

    st.caption(
        "The form contains your model's selected input features. "
        "You can change the default values before prediction."
    )

    with st.form("manual_transaction_form"):

        columns = st.columns(2)

        for index, feature in enumerate(FEATURES):

            default_value = default_row.get(feature, np.nan)

            with columns[index % 2]:

                if feature in categorical_features:

                    if pd.isna(default_value):
                        default_text = ""
                    else:
                        default_text = str(default_value)

                    entered_values[feature] = st.text_input(
                        feature,
                        value=default_text,
                        key=f"manual_{feature}"
                    )

                elif pd.api.types.is_numeric_dtype(
                    pd.Series([default_value])
                ):

                    if pd.isna(default_value):
                        default_number = 0.0
                    else:
                        default_number = float(default_value)

                    entered_values[feature] = st.number_input(
                        feature,
                        value=default_number,
                        format="%.6f",
                        key=f"manual_{feature}"
                    )

                else:

                    default_text = (
                        "" if pd.isna(default_value)
                        else str(default_value)
                    )

                    entered_values[feature] = st.text_input(
                        feature,
                        value=default_text,
                        key=f"manual_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict Transaction",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            new_transaction = pd.DataFrame(
                [entered_values],
                columns=FEATURES
            )

            for feature in FEATURES:

                if feature in categorical_features:

                    value = new_transaction.at[0, feature]

                    if value == "":
                        new_transaction.at[0, feature] = np.nan

                else:

                    new_transaction[feature] = pd.to_numeric(
                        new_transaction[feature],
                        errors="coerce"
                    )

            probabilities, predictions = get_prediction(
                new_transaction
            )

            save_current_transaction(
                new_transaction,
                "Manual Input"
            )

            st.session_state["last_probability"] = float(
                probabilities[0]
            )

            st.session_state["last_prediction"] = int(
                predictions[0]
            )

            st.success("Transaction submitted successfully.")

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# ---------------------------------------------------------
# CSV TRANSACTION UPLOAD
# ---------------------------------------------------------

def csv_transaction_upload():

    st.markdown("### Upload Transaction CSV")

    st.write(
        "Upload a CSV containing new transactions. "
        "The file must contain all 61 selected model features."
    )

    uploaded_file = st.file_uploader(
        "Choose transaction CSV",
        type=["csv"],
        key="transaction_csv_upload"
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.write("Uploaded file preview")

            st.dataframe(
                uploaded_data.head(),
                use_container_width=True
            )

            missing_features = [
                feature for feature in FEATURES
                if feature not in uploaded_data.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded file is missing these required "
                    f"model features: {', '.join(missing_features)}"
                )

                return

            transaction_data = prepare_input(uploaded_data)

            st.write(
                f"Number of transactions: **{len(transaction_data)}**"
            )

            if st.button(
                "Predict Uploaded Transactions",
                type="primary",
                use_container_width=True
            ):

                probabilities, predictions = get_prediction(
                    transaction_data
                )

                results = transaction_data.copy()

                results["Fraud_Probability"] = probabilities

                results["Prediction"] = [
                    prediction_label(value)
                    for value in probabilities
                ]

                results["Predicted_Class"] = predictions

                st.session_state["csv_prediction_results"] = results

                save_current_transaction(
                    transaction_data.iloc[[0]],
                    "CSV Upload - Row 1"
                )

                st.session_state["last_probability"] = float(
                    probabilities[0]
                )

                st.session_state["last_prediction"] = int(
                    predictions[0]
                )

                st.success(
                    f"Successfully predicted {len(results)} transactions."
                )

        except Exception as error:

            st.error(f"Unable to process CSV: {error}")

    if "csv_prediction_results" in st.session_state:

        results = st.session_state["csv_prediction_results"]

        st.divider()

        st.markdown("### Prediction Results")

        total = len(results)

        fraud_count = int(
            (results["Predicted_Class"] == 1).sum()
        )

        legitimate_count = total - fraud_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Potential Fraud", fraud_count)
        col3.metric("Legitimate Risk", legitimate_count)

        st.dataframe(
            results,
            use_container_width=True
        )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Report",
            data=csv_data,
            file_name="fraudshield_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------------------------------------------------
# SHAP INDIVIDUAL EXPLANATION
# ---------------------------------------------------------

def calculate_individual_shap(transaction):

    transaction = prepare_input(transaction)

    transformed = preprocessor.transform(transaction)

    explainer = load_shap_explainer(model)

    shap_values = explainer.shap_values(transformed)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, -1]
    else:
        shap_values = shap_values.reshape(-1)

    feature_names = preprocessor.get_feature_names_out()

    if len(feature_names) != len(shap_values):

        feature_names = [
            f"Feature_{i}"
            for i in range(len(shap_values))
        ]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values
    })

    explanation["Absolute_SHAP"] = explanation[
        "SHAP_Value"
    ].abs()

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    )

    return explanation


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.markdown("## 🛡️ FraudShield AI")

st.sidebar.caption("Intelligent E-Commerce Fraud Detection")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔍 Fraud Prediction",
        "🧠 Explainable AI",
        "🎛️ What-if Simulator",
        "🚨 Burst Detection",
        "📊 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model deployment active")

st.sidebar.caption("Powered by XGBoost")


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    page_header(
        "FraudShield AI Dashboard",
        "Intelligent E-Commerce Fraud Detection System"
    )

    st.markdown(
        "### Welcome to FraudShield AI"
    )

    st.write(
        "FraudShield AI is an explainable machine learning system "
        "that identifies potentially fraudulent e-commerce transactions "
        "using a trained XGBoost classification model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected Input Features",
        len(FEATURES)
    )

    col2.metric(
        "Decision Threshold",
        f"{THRESHOLD * 100:.0f}%"
    )

    col3.metric(
        "Machine Learning Model",
        "XGBoost"
    )

    st.divider()

    st.markdown("### System Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 🔍 Fraud Prediction")

        st.write(
            "Enter transaction details manually or upload "
            "a CSV file to obtain fraud predictions."
        )

        st.markdown("#### 🧠 Explainable AI")

        st.write(
            "Understand the features that influenced "
            "individual and overall model predictions using SHAP."
        )

        st.markdown("#### 🎛️ What-if Simulator")

        st.write(
            "Change transaction amounts and observe "
            "how the model's prediction responds."
        )

    with col2:

        st.markdown("#### 🚨 Burst Detection")

        st.write(
            "Analyze transaction windows to identify "
            "suspicious bursts of activity."
        )

        st.markdown("#### 📊 Model Performance")

        st.write(
            "Review model comparison results and "
            "final XGBoost test performance."
        )

    st.divider()

    st.info(
        "The application loads the saved trained model. "
        "It does not retrain the model when a new transaction is submitted."
    )


# =========================================================
# PAGE 2: FRAUD PREDICTION
# =========================================================

elif page == "🔍 Fraud Prediction":

    page_header(
        "Fraud Prediction",
        "Predict whether an e-commerce transaction is potentially fraudulent."
    )

    st.markdown("### Choose Input Method")

    input_method = st.radio(
        "Select how you want to provide transactions:",
        [
            "Manual Transaction Input",
            "CSV File Upload"
        ],
        horizontal=True
    )

    if input_method == "Manual Transaction Input":

        manual_transaction_input()

    else:

        csv_transaction_upload()

    if "last_probability" in st.session_state:

        st.divider()

        st.markdown("### Latest Transaction Prediction")

        probability = st.session_state["last_probability"]

        display_prediction(probability)

        st.caption(
            "Open the Explainable AI page to view the SHAP explanation "
            "for this exact transaction."
        )
