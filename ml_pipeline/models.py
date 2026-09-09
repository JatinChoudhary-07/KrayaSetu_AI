import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

def train_escalation_model(historical_defects: pd.DataFrame, features: list, target: str) -> CalibratedClassifierCV:
    """
    Train XGBoost classifier for escalation risk prediction.
    Must use time-sliced splits (shuffle=False) and handle imbalance using scale_pos_weight.
    Includes calibration to ensure probabilities reflect true risk rates.
    """
    X = historical_defects[features]
    y = historical_defects[target]
    
    # Time-sliced split to prevent temporal leakage
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Calculate scale_pos_weight for imbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        objective="binary:logistic"
    )
    
    base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Calibrate the classifier
    # Using isotonic if enough positives, else sigmoid/Platt scaling. Using prefit since base_model is trained.
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)
    
    return calibrated_model

def train_duration_model(historical_jobs: pd.DataFrame, features: list, target: str) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor for P80 duration prediction.
    """
    X = historical_jobs[features]
    y = historical_jobs[target]
    
    # Time-sliced split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:quantileerror",
        quantile_alpha=0.80
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model
