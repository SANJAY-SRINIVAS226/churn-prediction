#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → training → evaluation
============================================================================================
Optimized for the custom hybrid dataset with Windows path formatting fixes.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Configure logging with UTF-8 encoding support for Windows terminal stability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../project_logs.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core project module imports
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):
    """Main orchestrator function running the complete machine learning life cycle."""

    # === FIXED: Windows File URI Setup using Pathlib ===
    project_root = Path(os.getcwd()).parent.absolute()
    mlruns_path = args.mlflow_uri or project_root.joinpath("mlruns").as_uri()

    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    logger.info("📡 Launching full model pipeline run under MLflow tracking...")

    with mlflow.start_run():
        # Log system hyper-parameters
        mlflow.log_param("model_architecture", "xgboost")
        mlflow.log_param("classification_threshold", args.threshold)
        mlflow.log_param("test_split_ratio", args.test_size)

        # === STAGE 1: Data Loading & Validation ===
        logger.info(f"🔄 Extracting data file from: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found at: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")

        # Data Quality Validation using Great Expectations
        logger.info("🔍 Auditing data schema and ranges with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass_flag", int(is_valid))

        if not is_valid:
            mlflow.log_text(
                json.dumps(failed, indent=2),
                artifact_file="failed_expectations.json",
            )
            raise ValueError(
                f"❌ Data quality validation failed. See failed_expectations.json for details."
            )
        logger.info("✅ Data validation sweeps passed safely.")

        # === STAGE 2: Data Preprocessing ===
        logger.info("🔧 Cleansing data and converting types...")
        df = preprocess_data(df, target_col=args.target)

        # Save local processed copy
        processed_path = project_root.joinpath(
            "data", "processed", "telco_churn_processed.csv"
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info(f"✅ Preprocessed checkpoint data saved to: {processed_path}")

        # === STAGE 3: Feature Engineering ===
        logger.info("🛠️  Running feature transformation pipeline...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target variable column '{target}' missing!")

        df_enc = build_features(df, target_col=target)
        logger.info(
            f"✅ Feature engineering complete. Total feature count: {df_enc.shape[1] - 1}"
        )

        # Save feature columns index metadata for production consistency
        artifacts_dir = project_root.joinpath("artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # Save preprocessing deployment pickle package
        preprocessing_artifact = {"feature_columns": feature_cols, "target": target}
        joblib.dump(
            preprocessing_artifact,
            os.path.join(artifacts_dir, "preprocessing.pkl"),
        )
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        # === STAGE 4: Train/Test Split ===
        logger.info("📊 Executing Stratified Train-Test data split...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        logger.info(
            f"✅ Split completed. Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows"
        )

        # Balance minority class weight calculation
        scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()

        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        logger.info("🤖 Constructing optimized XGBoost framework...")

        # FIXED: Injected your precise 93.9% recall Optuna hyperparameter array values
        model = XGBClassifier(
            n_estimators=341,
            learning_rate=0.010097714230991773,
            max_depth=3,
            subsample=0.8917761743152861,
            colsample_bytree=0.6890709730326446,
            min_child_weight=6,
            gamma=2.9918211528566943,
            reg_alpha=3.7833634210377407,
            reg_lambda=2.885061057240918,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight_value,
            eval_metric="logloss",
        )

        # Train model and record calculation runtime
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("model_training_duration", train_time)
        logger.info(f"✅ Model fitting execution completed in {train_time:.2f} seconds.")

        # === STAGE 6: Model Evaluation (Restored Complete Block) ===
        logger.info("📊 Evaluating model performance against test dataset...")
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)

        # Compute evaluation scores
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, proba)

        # Log metric scores to MLflow server metrics logs
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # Upload model configuration to Artifacts
        mlflow.xgboost.log_model(model, "model")

        logger.info("============================================================")
        logger.info("🎉 COMPLETE PIPELINE RUN LOGGED SUCCESSFULLY TO MLFLOW!")
        logger.info("============================================================")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run End-to-End Telco Churn Training Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        # Default points directly to your updated custom features file path!
        default=r"C:\Users\Admin\churn-prediction\Dataset\telco_with_all_features.csv",
        help="Path to input raw CSV file",
    )
    parser.add_argument(
        "--target", type=str, default="Churn", help="Name of the target column"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in test split",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Custom inference sensitivity classification threshold",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn - EndToEnd Pipeline",
        help="MLflow experiment group name",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="Override default local file tracking path URI",
    )

    parsed_args = parser.parse_args(
        sys.argv[1:] if "ipykernel" not in sys.argv[0] else []
    )
    main(parsed_args)
