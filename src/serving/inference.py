"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.
"""

import glob
import os
import mlflow
import pandas as pd

# === MODEL LOADING CONFIGURATION ===
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"⚠️ Failed to load model from {MODEL_DIR}: {e}")
    try:
        # Fallback for local development: Find the latest model run inside mlruns
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns folder")
    except Exception as fallback_error:
        raise Exception(
            f"Failed to load model: {e}. Fallback failed: {fallback_error}"
        )

# === FEATURE TRANSFORMATION CONSTANTS ===
# Hardcoded configuration arrays guarantee consistent layout across serving requests
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Adjusted numeric targets (TotalCharges and PhoneService dropped due to VIF)
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "Age",
    "SupportTickets",
    "SatisfactionScore",
    "UsageHoursPerWeek",
]

# Exact feature array sequence used by your final optimized XGBoost booster
FEATURE_COLS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PaperlessBilling",
    "MonthlyCharges",
    "Age",
    "SupportTickets",
    "SatisfactionScore",
    "UsageHoursPerWeek",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "OnlineSecurity_Yes",
    "OnlineBackup_Yes",
    "DeviceProtection_Yes",
    "TechSupport_Yes",
    "StreamingTV_Yes",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "SubscriptionType_Enterprise",
    "SubscriptionType_Premium",
    "No_internet_service",
]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw user dictionary payload inputs into clean arrays for XGBoost."""
    df = df.copy()

    # Clean header spaces
    df.columns = df.columns.str.strip()

    # === STEP 1: Numeric Type Coercion ===
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Convert SeniorCitizen flag safely
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # === STEP 2: Binary Feature Encoding ===
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    obj_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "SubscriptionType",
    ]

    # Convert specified categorical columns into dummy columns
    df = pd.get_dummies(
        df, columns=[c for c in obj_cols if c in df.columns], drop_first=True
    )

    # === STEP 4: Collapse Redundant Internet Dummies ===
    no_internet_cols = [col for col in df.columns if "No internet service" in col]
    if no_internet_cols:
        df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_internet_cols)

    # === STEP 5: Boolean to Integer Conversion ===
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 6: Feature Alignment and Padding ===
    # For single-row inputs, ensure any missing columns are filled with 0
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> str:
    """Main function called by FastAPI and Gradio to make a live prediction.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply matching feature engineering pipeline
    3. Run model probability engine using your custom THRESHOLD sensitivity
    """
    try:
        # Convert dictionary input data row into a single-row DataFrame
        df = pd.DataFrame([input_dict])

        # Transform inputs to match training layouts
        X_live = _serve_transform(df)

        # Get Churn probability percentage (Class 1)
        # We use predict_proba to enforce your custom 0.3 sensitivity line
        churn_probability = model.predict_proba(X_live)[0, 1]

        # Use the custom threshold (0.3) established during calibration
        THRESHOLD = 0.3
        if churn_probability >= THRESHOLD:
            return f"⚠️ Likely to Churn (Probability: {churn_probability*100:.1f}%)"
        else:
            return f"✅ Not Likely to Churn (Probability: {churn_probability*100:.1f}%)"

    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"
