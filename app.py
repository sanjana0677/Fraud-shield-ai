
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


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename, folder=None):
    """Find a file in the root folder or its original subfolder."""

    if folder:
        nested = BASE_DIR / folder / filename
        if nested.exists():
            return nested

    root_file = BASE_DIR / filename

    if root_file.exists():
        return root_file

    return BASE_DIR / folder / filename if folder else root_file


PREPROCESSOR_PATH = find_file("preprocessor.pkl", "models")
MODEL_PATH = find_file("final_xgboost_model.json", "models")
FEATURES_PATH = find_file("selected_features.json", "models")
THRESHOLD_PATH = find_file("decision_threshold.json", "models")
SAMPLE_PATH = find_file("sample_transaction.csv", "data")


st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background-color: #F5F7FC;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
    }

    [data-testid="stSidebar"] {
        background-color: #EDF2FF;
        border-right: 1px solid #DCE5F5;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #243B64;
        line-height: 1.5 !important;
        padding-top: 10px;
        padding-bottom: 8px;
        overflow: visible;
    }

    .sub-title {
        color: #71809B;
        font-size: 15px;
        line-height: 1.5;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #344B73;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE5F5;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(36, 59, 100, 0.05);
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #647FC4;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        padding: 9px 18px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #4D68AB;
        color: white;
        border: none;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE5F5;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE5F5;
        border-radius: 10px;
    }

    hr {
        border-color: #DCE5F5;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

@st.cache_resource
def load_model_files():

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Selected features file not found: {FEATURES_PATH}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURES_PATH, "r") as file:
        feature_data = json.load(file)

    if isinstance(feature_data, dict):
        features = feature_data.get("selected_features", [])
    else:
        features = feature_data

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


# =========================================================
# GENERAL HELPER FUNCTIONS
# =========================================================

def page_header(title, subtitle):

    st.markdown(
        f'<h1 class="main-title">{title}</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sub-title">{subtitle}</div>',
        unsafe_allow_html=True
    )

    st.divider()


def result_file(relative_path):

    parts = Path(relative_path).parts

    if len(parts) > 1:
        nested_path = BASE_DIR.joinpath(*parts)

        if nested_path.exists():
            return nested_path

    return BASE_DIR / Path(relative_path).name


def prepare_input(data):

    data = data.copy()

    missing = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    return data[FEATURES]


def get_prediction(input_data):

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

    data = prepare_input(data)

    st.session_state["current_transaction"] = data.copy()
    st.session_state["current_source"] = source

    st.session_state.pop("current_shap_values", None)


def get_current_transaction():

    if "current_transaction" in st.session_state:
        return st.session_state["current_transaction"].copy()

    sample = load_sample_transaction()

    if sample is not None:
        return prepare_input(sample.head(1))

    return None


def get_preprocessor_categorical_features():

    categorical = set()

    try:

        for name, transformer, columns in preprocessor.transformers_:

            if name.lower() in ("cat", "categorical"):

                if isinstance(columns, (list, tuple, np.ndarray)):
                    categorical.update(columns)

    except Exception:
        pass

    return categorical


# =========================================================
# MANUAL TRANSACTION INPUT
# =========================================================

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

                    default_text = (
                        "" if pd.isna(default_value)
                        else str(default_value)
                    )

                    entered_values[feature] = st.text_input(
                        feature,
                        value=default_text,
                        key=f"manual_{feature}"
                    )

                elif pd.api.types.is_numeric_dtype(
                    pd.Series([default_value])
                ):

                    default_number = (
                        0.0 if pd.isna(default_value)
                        else float(default_value)
                    )

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

            # Clear previous CSV results after manual submission.
            st.session_state.pop("csv_prediction_results", None)
            st.session_state.pop("shap_csv_row_selector", None)

            st.session_state["last_probability"] = float(
                probabilities[0]
            )

            st.session_state["last_prediction"] = int(
                predictions[0]
            )

            st.success("Transaction submitted successfully.")

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# =========================================================
# CSV TRANSACTION UPLOAD
# =========================================================

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


# =========================================================
# INDIVIDUAL SHAP EXPLANATION FUNCTION
# =========================================================

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

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

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

    st.markdown("### Welcome to FraudShield AI")

    st.write(
        "FraudShield AI is an explainable machine learning system "
        "that identifies potentially fraudulent e-commerce transactions "
        "using a trained XGBoost classification model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Input Features", len(FEATURES))

    col2.metric(
        "Decision Threshold",
        f"{THRESHOLD * 100:.0f}%"
    )

    col3.metric("Machine Learning Model", "XGBoost")

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
            "for the selected transaction."
        )


# =========================================================
# PAGE 3: EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    page_header(
        "Explainable AI – SHAP Analysis",
        "Understand which features contributed most to the model's predictions."
    )

    st.markdown("### Global Feature Importance")

    st.write(
        "These features have the strongest overall influence "
        "on the model's predictions."
    )

    importance_path = result_file(
        "shap_feature_importance.csv"
    )

    if importance_path.exists():

        importance = pd.read_csv(importance_path)

        if "Mean_Absolute_SHAP" in importance.columns:

            importance = importance.sort_values(
                "Mean_Absolute_SHAP",
                ascending=False
            ).head(15)

            st.bar_chart(
                importance.set_index("Feature")[
                    "Mean_Absolute_SHAP"
                ],
                horizontal=True
            )

        else:

            st.dataframe(
                importance,
                use_container_width=True
            )

        st.caption(
            "Mean absolute SHAP values measure average feature "
            "contribution magnitude. They do not establish causation."
        )

    else:

        st.warning("Global SHAP feature importance file not found.")

    col1, col2 = st.columns(2)

    summary_path = result_file("shap_summary.png")
    waterfall_path = result_file("shap_waterfall.png")

    with col1:

        st.markdown("### SHAP Summary Plot")

        if summary_path.exists():

            st.image(
                str(summary_path),
                use_container_width=True
            )

        else:

            st.info("SHAP summary plot not found.")

    with col2:

        st.markdown("### SHAP Waterfall Plot")

        if waterfall_path.exists():

            st.image(
                str(waterfall_path),
                use_container_width=True
            )

        else:

            st.info("SHAP waterfall plot not found.")

    st.divider()

    st.markdown("### Individual Transaction Explanation")

    st.write(
        "Calculate how individual transaction features influence "
        "the prediction made by your trained XGBoost model."
    )

    # -----------------------------------------------------
    # NEW FEATURE: SELECT CSV TRANSACTION FOR SHAP
    # -----------------------------------------------------

    csv_results = st.session_state.get("csv_prediction_results")

    if csv_results is not None and not csv_results.empty:

        st.markdown("#### Select a Transaction to Explain")

        def transaction_option_label(row_index):

            row = csv_results.iloc[row_index]

            probability = float(
                row["Fraud_Probability"]
            ) * 100

            prediction = row["Prediction"]

            return (
                f"Row {row_index + 1} | "
                f"Fraud Probability: {probability:.2f}% | "
                f"{prediction}"
            )

        selected_row = st.selectbox(
            "Choose a transaction:",
            options=list(range(len(csv_results))),
            format_func=transaction_option_label,
            key="shap_csv_row_selector"
        )

        current_transaction = prepare_input(
            csv_results.iloc[[selected_row]]
        )

        source = f"CSV Upload - Row {selected_row + 1}"

    else:

        current_transaction = get_current_transaction()

        source = st.session_state.get(
            "current_source",
            "Saved Sample Transaction"
        )

    # -----------------------------------------------------
    # INDIVIDUAL SHAP EXPLANATION
    # -----------------------------------------------------

    if current_transaction is None:

        st.warning(
            "No transaction is available. Submit a transaction "
            "from the Fraud Prediction page first."
        )

    else:

        st.info(
            f"Explaining: **{source}**. "
            "This explanation is calculated from the selected transaction."
        )

        with st.expander("View selected transaction"):

            st.dataframe(
                current_transaction,
                use_container_width=True
            )

        try:

            probabilities, predictions = get_prediction(
                current_transaction
            )

            probability = float(probabilities[0])

            st.markdown("#### Individual Model Prediction")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Prediction",
                prediction_label(probability)
            )

            col3.metric(
                "Decision Threshold",
                f"{THRESHOLD * 100:.0f}%"
            )

            st.divider()

            with st.spinner(
                "Calculating SHAP values for the selected transaction..."
            ):

                explanation = calculate_individual_shap(
                    current_transaction
                )

            st.success(
                "Individual SHAP explanation calculated successfully."
            )

            positive = explanation[
                explanation["SHAP_Value"] > 0
            ].sort_values(
                "SHAP_Value",
                ascending=False
            )

            negative = explanation[
                explanation["SHAP_Value"] < 0
            ].sort_values(
                "SHAP_Value",
                ascending=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("#### Positive SHAP Contributions")

                st.caption(
                    "These features push the model output towards fraud."
                )

                if not positive.empty:

                    st.dataframe(
                        positive[
                            ["Feature", "SHAP_Value"]
                        ].head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.write("No positive contributions.")

            with col2:

                st.markdown("#### Negative SHAP Contributions")

                st.caption(
                    "These features push the model output towards legitimate."
                )

                if not negative.empty:

                    st.dataframe(
                        negative[
                            ["Feature", "SHAP_Value"]
                        ].head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.write("No negative contributions.")

            st.divider()

            st.markdown("### Top 10 Feature Contributions")

            top_features = explanation.head(10).copy()

            chart_data = top_features[
                ["Feature", "SHAP_Value"]
            ].set_index("Feature")

            st.bar_chart(
                chart_data,
                horizontal=True
            )

            st.caption(
                "Positive SHAP values push the model output towards fraud, "
                "while negative values push it towards legitimate. "
                "These values explain the XGBoost raw output (log-odds), "
                "not direct percentage-point changes in probability."
            )

            with st.expander("View complete SHAP contribution data"):

                st.dataframe(
                    explanation[
                        ["Feature", "SHAP_Value", "Absolute_SHAP"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            st.markdown("### Understanding SHAP Contributions")

            st.write(
                "**Positive SHAP value:** Pushes the model output towards fraud."
            )

            st.write(
                "**Negative SHAP value:** Pushes the model output towards legitimate."
            )

            st.write(
                "**Larger absolute value:** Indicates a stronger contribution."
            )

            st.write(
                "**Global importance:** Shows average contribution magnitude across samples."
            )

            st.write(
                "SHAP explains how the trained model uses features to make "
                "predictions. It does not establish that a feature causes fraud."
            )

        except Exception as error:

            st.error(
                f"Unable to calculate individual SHAP values: {error}"
            )

    if importance_path.exists():

        with st.expander("Feature Importance Data"):

            st.dataframe(
                pd.read_csv(importance_path),
                use_container_width=True,
                hide_index=True
            )

# =========================================================
# PAGE 4: WHAT-IF SIMULATOR
# =========================================================

elif page == "🎛️ What-if Simulator":

    page_header(
        "What-if Transaction Simulator",
        "Experiment with hypothetical transaction amount changes "
        "and observe model predictions."
    )

    current_transaction = get_current_transaction()

    if current_transaction is None:

        st.warning("No transaction is available.")

    elif "TransactionAmt" not in current_transaction.columns:

        st.error(
            "TransactionAmt feature is not available in the current model input."
        )

    else:

        original_transaction = current_transaction.copy()

        original_amount = float(
            original_transaction["TransactionAmt"].iloc[0]
        )

        original_probability = float(
            get_prediction(original_transaction)[0][0]
        )

        st.markdown("### Original Transaction")

        col1, col2 = st.columns(2)

        col1.metric(
            "Original Transaction Amount",
            f"${original_amount:.2f}"
        )

        col2.metric(
            "Original Fraud Probability",
            f"{original_probability * 100:.2f}%"
        )

        st.divider()

        st.markdown("### Change Transaction Amount")

        new_amount = st.slider(
            "Select a hypothetical transaction amount",
            min_value=0.0,
            max_value=float(max(1000, original_amount * 5)),
            value=float(original_amount),
            step=1.0
        )

        hypothetical_transaction = original_transaction.copy()

        hypothetical_transaction.loc[
            hypothetical_transaction.index[0],
            "TransactionAmt"
        ] = new_amount

        hypothetical_probability = float(
            get_prediction(hypothetical_transaction)[0][0]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "New Amount",
            f"${new_amount:.2f}"
        )

        col2.metric(
            "New Fraud Probability",
            f"{hypothetical_probability * 100:.2f}%"
        )

        col3.metric(
            "Probability Change",
            f"{(hypothetical_probability - original_probability) * 100:+.2f} pp"
        )

        if hypothetical_probability >= THRESHOLD:

            st.error(
                "The hypothetical transaction is classified as potential fraud."
            )

        else:

            st.success(
                "The hypothetical transaction is classified as legitimate risk."
            )

        comparison = pd.DataFrame({
            "Transaction": [
                "Original",
                "Hypothetical"
            ],
            "Fraud Probability (%)": [
                original_probability * 100,
                hypothetical_probability * 100
            ]
        })

        st.markdown("### Prediction Comparison")

        st.bar_chart(
            comparison.set_index("Transaction")
        )

        st.caption(
            "Only TransactionAmt is changed. All other transaction "
            "features remain fixed. This is a hypothetical model experiment, "
            "not a guarantee of real-world fraud behaviour."
        )


# =========================================================
# PAGE 5: BURST DETECTION
# =========================================================

elif page == "🚨 Burst Detection":

    page_header(
        "Suspicious Transaction Burst Detection",
        "Identify unusual concentrations of suspicious transactions "
        "within short time windows."
    )

    burst_path = result_file("burst_detection_results.csv")

    alert_path = result_file("suspicious_burst_alerts.csv")

    if burst_path.exists():

        burst_data = pd.read_csv(burst_path)

        total_windows = len(burst_data)

        if "Burst_Alert" in burst_data.columns:

            alerts = burst_data[
                burst_data["Burst_Alert"].astype(str).str.lower().isin(
                    ["true", "1", "yes"]
                )
            ]

        else:

            alerts = pd.DataFrame()

        alert_count = len(alerts)

        suspicious_count = 0

        if "Suspicious_Transactions" in burst_data.columns:

            suspicious_count = int(
                burst_data["Suspicious_Transactions"].sum()
            )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Time Windows", total_windows)

        col2.metric("Burst Alert Windows", alert_count)

        col3.metric("Suspicious Transactions", suspicious_count)

        st.divider()

        st.markdown("### Transaction Burst Analysis")

        st.dataframe(
            burst_data,
            use_container_width=True
        )

        if "Total_Transactions" in burst_data.columns:

            st.markdown("### Transactions per Time Window")

            st.line_chart(
                burst_data["Total_Transactions"]
            )

        if "Suspicious_Transactions" in burst_data.columns:

            st.markdown("### Suspicious Transactions per Window")

            st.bar_chart(
                burst_data["Suspicious_Transactions"]
            )

    else:

        st.warning("Burst detection results file not found.")

    if alert_path.exists():

        st.divider()

        st.markdown("### Suspicious Burst Alerts")

        alert_data = pd.read_csv(alert_path)

        st.dataframe(
            alert_data,
            use_container_width=True
        )

        st.download_button(
            "Download Burst Alerts",
            data=alert_data.to_csv(index=False).encode("utf-8"),
            file_name="suspicious_burst_alerts.csv",
            mime="text/csv"
        )

    st.info(
        "This page displays the existing historical burst detection "
        "analysis. It is not a live transaction monitoring service."
    )


# =========================================================
# PAGE 6: MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    page_header(
        "Model Performance",
        "Evaluate and compare machine learning models used "
        "for e-commerce fraud detection."
    )

    comparison_path = result_file("model_comparison.csv")

    final_metrics_path = result_file("final_test_metrics.csv")

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.markdown("### Model Comparison")

    if comparison_path.exists():

        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if "Algorithm" in comparison.columns:

            metric_options = [
                metric for metric in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score",
                    "ROC-AUC",
                    "PR-AUC"
                ]
                if metric in comparison.columns
            ]

            if metric_options:

                selected_metric = st.selectbox(
                    "Select evaluation metric",
                    metric_options
                )

                chart_data = comparison[
                    ["Algorithm", selected_metric]
                ].set_index("Algorithm")

                st.markdown(
                    f"### Algorithm Comparison – {selected_metric}"
                )

                st.bar_chart(chart_data)

    else:

        st.warning("Model comparison file not found.")

    st.divider()

    # -----------------------------------------------------
    # FINAL TEST PERFORMANCE
    # -----------------------------------------------------

    st.markdown("### Final XGBoost Test Performance")

    if final_metrics_path.exists():

        final_metrics = pd.read_csv(final_metrics_path)

        st.dataframe(
            final_metrics,
            use_container_width=True,
            hide_index=True
        )

        if len(final_metrics) > 0:

            row = final_metrics.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{row['Accuracy'] * 100:.2f}%"
            )

            col2.metric(
                "Precision",
                f"{row['Precision'] * 100:.2f}%"
            )

            col3.metric(
                "Recall",
                f"{row['Recall'] * 100:.2f}%"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "F1-Score",
                f"{row['F1-score'] * 100:.2f}%"
            )

            col2.metric(
                "ROC-AUC",
                f"{row['ROC-AUC']:.4f}"
            )

            col3.metric(
                "PR-AUC",
                f"{row['PR-AUC']:.4f}"
            )

    else:

        st.warning("Final test metrics file not found.")

    st.caption(
        "Model comparison metrics are from the comparison evaluation. "
        "Final XGBoost metrics are from the separate held-out test set."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:13px;">
        FraudShield AI | E-Commerce Fraud Detection |
        Powered by Machine Learning and XGBoost
    </div>
    """,
    unsafe_allow_html=True
)
