# test_pipeline_phase1.py
import os
import pandas as pd
import sys

# Make sure Python can find your src package correctly by stepping up if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("src"))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# === CONFIG ===
# 🛠️ FIXED: Pointing directly to your updated custom features file path!
DATA_PATH = (
    r"C:\Users\Admin\churn-prediction\Dataset\telco_with_all_features.csv"
)
TARGET_COL = "Churn"


def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Error: Cannot find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"✅ Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"✅ Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n🎉 Phase 1 pipeline completed successfully!")


if __name__ == "__main__":
    main()
