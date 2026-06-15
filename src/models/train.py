import logging
import time
from xgboost import XGBClassifier
import mlflow
import mlflow.data
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, target_col: str = "Churn", threshold: float = 0.3):
    """
    Trains the final optimized XGBoost model with class imbalance correction,
    custom thresholding, system logging, and comprehensive MLflow tracking.
    """
    logger.info("🎬 Initializing final model training workflow...")

    # 1. Separate Features from Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Balanced Train-Test Split matching your project configuration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"    Split sizes -> Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # 3. Compute scale_pos_weight dynamically to handle imbalanced churn groups
    scale_pos_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # 4. Inject your winning Optuna hyperparameter set
    best_params = {
        "n_estimators": 341,
        "learning_rate": 0.010097714230991773,
        "max_depth": 3,
        "subsample": 0.8917761743152861,
        "colsample_bytree": 0.6890709730326446,
        "min_child_weight": 6,
        "gamma": 2.9918211528566943,
        "reg_alpha": 3.7833634210377407,
        "reg_lambda": 2.885061057240918,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight_ratio,
        "eval_metric": "logloss",
    }

    # Initialize model structure
    model = XGBClassifier(**best_params)

    # 5. Open MLflow Tracking Run
    with mlflow.start_run():
        logger.info(" Connection established. Logging metrics to MLflow UI...")
        
        # Log all configuration parameters at once
        mlflow.log_params(best_params)
        mlflow.log_param("custom_inference_threshold", threshold)

        # 6. Fit the Model with training timers
        logger.info(" Fitting XGBoost structures... Please wait...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        mlflow.log_metric("training_duration_seconds", elapsed_time)
        logger.info(f"   Training process complete in {elapsed_time:.2f} seconds.")

        # 7. Generate Predictions using your custom THRESHOLD dial
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        # Calculate final evaluation scores
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)

        # Log main performance criteria to server dashboard
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        
        # 8. Save the raw trained booster binary to Artifact Storage
        mlflow.xgboost.log_model(model, "model")
        logger.info("   Model files packaged and uploaded to MLflow artifact storage.")

        # 9. Register Dataset metadata to MLflow UI profile
        # Uses the modern mlflow.data integration to snapshot the training file profile
        try:
            train_ds = mlflow.data.from_pandas(df, source="clean_hybrid_dataset")
            mlflow.log_input(train_ds, context="training")
            logger.info("   Dataset profile footprint successfully logged into run timeline.")
        except Exception as e:
            logger.warning(f"Could not log dataset profile context: {str(e)}")

        # Print the beautiful classification report to screen
        print("\n" + "=" * 60)
        print(" MODEL PRODUCTION RUN LOGGED SUCCESSFULLY TO MLFLOW!")
        print("=" * 60)
        print(classification_report(y_test, y_pred, digits=3))
        
    return model
