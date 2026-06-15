import logging
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

# Silence messy Optuna logging updates to keep your notebook clean
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def tune_model(X: pd.DataFrame, y: pd.Series, threshold: float = 0.3) -> dict:
    """
    Tunes an XGBoost model using Optuna using Stratified Cross-Validation
    while strictly accounting for class imbalance and custom inference thresholds.
    """
    logger.info(" Initializing hyperparameter optimization workspace...")

    # Calculate class weight imbalance ratio once outside the optimization loop
    scale_pos_weight_ratio = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        # Optimized search spaces to prevent overfitting and save computer processing time
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 4),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight_ratio,
            "eval_metric": "logloss"
        }
        
        model = XGBClassifier(**params)
        
        # 3-Fold Stratified Cross-Validation: Maintains precise label proportions
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit on training fold, predict probabilities on validation fold
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            
            # Apply your custom sensitivity threshold to calculate the true score
            preds = (proba >= threshold).astype(int)
            scores.append(recall_score(y_val, preds, pos_label=1, zero_division=0))
            
        return np.mean(scores)

    # Launch Optuna optimization engine
    study = optuna.create_study(direction="maximize")
    
    logger.info(" Commencing hyperparameter trial sweeps (20 trials)...")
    study.optimize(objective, n_trials=20)
    
    print("\n" + "=" * 50)
    print(" OPTUNA TUNING WORK COMPLETE!")
    print("=" * 50)
    print("Best Params Found:", study.best_params)
    print(f"Best Cross-Validated Recall Score: {study.best_value:.4f}")
    
    return study.best_params
