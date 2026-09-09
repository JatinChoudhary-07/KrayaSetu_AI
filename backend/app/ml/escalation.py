import xgboost as xgb
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

FEATURES = [
    "track_age_years", "gmt_since_renewal", "defect_type_encoded",
    "rail_wear_mm", "rainfall_7d_mm", "traffic_density_trains_per_day",
    "defect_recurrence_count", "days_since_last_inspection"
]

LOW_CONFIDENCE_BAND = (0.40, 0.60)

def train_escalation_model(historical_defects: pd.DataFrame) -> xgb.XGBClassifier:
    X = historical_defects[FEATURES]
    y = historical_defects["escalated_within_7d"]
    # time-sliced training splits strictly avoid temporal leakage
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def predict_escalation_risk(model, defect_features: dict, rule_based_fallback_R: int) -> dict:
    try:
        df = pd.DataFrame([defect_features])
        proba = model.predict_proba(df)[0][1]
        
        # Pillar 3 Explainability using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        
        # Extract top 3 features
        feature_importance = list(zip(FEATURES, shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_risk_factors = [f[0] for f in feature_importance[:3]]
        
    except Exception:
        # Model unavailable, malformed record, or a defect_type/feature combo the
        # model has never seen (cold start)
        return {
            "escalation_risk_R": rule_based_fallback_R,
            "risk_confidence": "fallback_rule_based",
            "top_risk_factors": []
        }

    if LOW_CONFIDENCE_BAND[0] <= proba <= LOW_CONFIDENCE_BAND[1]:
        return {
            "escalation_risk_R": rule_based_fallback_R,
            "risk_confidence": "low_model_confidence",
            "top_risk_factors": top_risk_factors
        }

    return {
        "escalation_risk_R": round(proba * 100),
        "risk_confidence": "high",
        "top_risk_factors": top_risk_factors
    }
