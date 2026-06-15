import os
import sys
import pandas as pd

# Make src importable by looking one folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# 🛠️ Updated to point to your exact updated feature file location
RAW = r"C:\Users\Admin\churn-prediction\Dataset\telco_with_all_features.csv"
OUT = r"C:\Users\Admin\churn-prediction\Dataset\telco_churn_processed.csv"

# 1) Load data with updated features
df = pd.read_csv(RAW)

# 2) Preprocess (cleans formatting, handles numeric casting)
df = preprocess_data(df, target_col="Churn")

# 3) Ensure target is 0/1 integers only
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

# Handle any missing target values safely before modeling steps
df["Churn"] = df["Churn"].fillna(0).astype(int)

# Safety sanity checks
assert df["Churn"].isna().sum() == 0, "Error: Churn has NaNs after preprocessing"
assert set(df["Churn"].unique()) <= {0, 1}, "Error: Churn values must be only 0 or 1"

# 4) Transform features (Applies One-Hot Encoding and VIF column drops)
df_processed = build_features(df, target_col="Churn")

# 5) Save the clean processed table to your dataset folder
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)

print("\n" + "=" * 50)
print(f"Preprocessing pipeline complete!")
print(f" Processed file saved to: {OUT}")
print(f" Final Dimensions: {df_processed.shape[0]} rows x {df_processed.shape[1]} columns")
print("=" * 50)
