import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Production-ready cleaning for the optimized Telco churn model.
    - Trims column spaces
    - Maps binary text categories to 0/1 integers
    - COLLAPSES redundant 'No internet service' columns exactly like training
    - DROPS high-VIF copycat columns (TotalCharges, PhoneService, etc.)
    - Handles One-Hot Encoding format matching the model shape
    """
    # 1. Clean up header spaces
    df.columns = df.columns.str.strip()

    # 2. Drop ID columns if they exist
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3. Convert target column to numbers if present (only applies during training)
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # 4. Map binary categories to 0/1 integers matching your training steps
    binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = ["gender", "Partner", "Dependents", "PaperlessBilling"]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().map(binary_mapping)

    # Ensure correct data type format for SeniorCitizen
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # 5. One-Hot Encode categorical columns with more than 2 options
    multi_cat_cols = [
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
        "SubscriptionType",  # <-- Your new custom feature category
    ]

    # Create dummy column structures matching your exact pd.get_dummies training shape
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # 6. COLLAPSE copycat tech service columns exactly like your training notebook!
    no_internet_cols = [col for col in df.columns if "No internet service" in col]
    if no_internet_cols:
        df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_internet_cols)

    # 7. CRITICAL: Drop the high-VIF duplicate features your model does not use
    # If these columns are passed to the model, it will throw an error and crash!
    high_vif_drops = [
        "TotalCharges",
        "PhoneService",
        "InternetService_No",
        "No_phone_service",
        "InternetService_Fiber optic",
        "MultipleLines_No phone service",
    ]

    df = df.drop(columns=[col for col in high_vif_drops if col in df.columns])

    # 8. Clean up any leftover empty data spaces safely
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
