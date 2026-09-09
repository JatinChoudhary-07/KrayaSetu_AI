import xgboost as xgb
import pandas as pd

MIN_POSSESSION_MINUTES = 15

def train_duration_model(historical_jobs: pd.DataFrame, quantile: float = 0.80) -> xgb.XGBRegressor:
    X = historical_jobs[["defect_type_encoded", "gang_id_encoded", "machine_type_encoded",
                          "track_type_encoded", "crew_size", "is_night_shift"]]
    y = historical_jobs["actual_duration_minutes"]
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile,
        n_estimators=250, max_depth=4, learning_rate=0.05,
    )
    model.fit(X, y)
    return model

def predict_duration_minutes(model, job_features: dict, static_default_minutes: int) -> dict:
    try:
        df = pd.DataFrame([job_features])
        pred = float(model.predict(df)[0])
        dur = max(round(pred), MIN_POSSESSION_MINUTES)
        return {"duration_minutes": dur, "duration_source": "model_p80"}
    except Exception:
        return {"duration_minutes": static_default_minutes, "duration_source": "static_fallback"}
