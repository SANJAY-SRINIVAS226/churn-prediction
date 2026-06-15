import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """Apply deterministic binary encoding to 2-category features.

    This maps Yes/No and Male/Female text strings cleanly into 0/1 integers.
    """
    # Clean up edge whitespaces from strings
    s_clean = s.astype(str).str.strip()

    # Match exact string patterns safely
    if set(s_clean.dropna().unique()).issubset({"Yes", "No", "1", "0", "1.0", "0.0"}):
        return s_clean.map({"No": 0, "Yes": 1, "0": 0, "1": 1}).astype("Int64")

    if set(s_clean.dropna().unique()).issubset({"Male", "Female"}):
        return s_clean.map({"Female": 0, "Male": 1}).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Apply complete feature engineering pipeline for training and serving data.

    This function transforms raw inputs into the exact shape and columns expected
    by your optimized XGBoost model.
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[0]} rows...")

    # === STEP 1: Define Static Feature Configurations ===
    # Hardcoded lists prevent single-row inputs from shifting column classifications
    binary_cols = ["gender", "Partner", "Dependents", "PaperlessBilling"]

    multi_cols = [
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

    # === STEP 2: Apply Binary Encoding ===
    for c in binary_cols:
        if c in df.columns:
            df[c] = _map_binary_series(df[c])
            df[c] = df[c].fillna(0).astype(int)
            print(f"      ✅ {c} → binary integer (0/1)")

    # Handle SeniorCitizen column safely as an integer
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # === STEP 3: One-Hot Encoding for Multi-Category Features ===
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to multi-category columns...")
        df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns], drop_first=True)

    # Convert true/false boolean flags to 0/1 integers for XGBoost
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 4: Collapse Redundant Tech Service Dummies ===
    # Groups those 6 redundant "No internet service" flags into one single metric
    no_internet_cols = [col for col in df.columns if "No internet service" in col]
    if no_internet_cols:
        df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_internet_cols)

    # === STEP 5: Drop High-VIF Copycat Columns ===
    # These columns were removed during training to fix multicollinearity. 
    # They must be dropped here so they don't break the model interface shapes.
    high_vif_drops = [
        "TotalCharges",
        "PhoneService",
        "InternetService_No",
        "No_phone_service",
        "InternetService_Fiber optic",
        "MultipleLines_No phone service",
    ]
    df = df.drop(columns=[col for col in high_vif_drops if col in df.columns])

    # === STEP 6: Ensure All Expected Training Columns Exist ===
    # For single-row production serving, pd.get_dummies might skip columns.
    # This block fills missing one-hot dummies with 0 to match the exact model blueprint.
    expected_features = [
        "SeniorCitizen", "Partner", "Dependents", "tenure", "PaperlessBilling", 
        "MonthlyCharges", "Age", "SupportTickets", "SatisfactionScore", "UsageHoursPerWeek",
        "MultipleLines_No phone service", "MultipleLines_Yes", "OnlineSecurity_Yes", 
        "OnlineBackup_Yes", "DeviceProtection_Yes", "TechSupport_Yes", "StreamingTV_Yes", 
        "StreamingMovies_Yes", "Contract_One year", "Contract_Two year", 
        "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", 
        "PaymentMethod_Mailed check", "SubscriptionType_Enterprise", "SubscriptionType_Premium",
        "No_internet_service"
    ]
    
    # Don't try to add back the target variable if we are just serving an inference request
    for feature in expected_features:
        if feature not in df.columns and feature != target_col:
            df[feature] = 0

    # Sort columns to ensure the array matches the exact sequence XGBoost expects
    final_cols = [c for c in expected_features if c in df.columns]
    if target_col in df.columns:
        final_cols.append(target_col)
        
    df = df[final_cols]

    print(f"✅ Feature engineering complete: {df.shape[1]} final features verified.")
    return df
