import xgboost as xgb
import shap
import pandas as pd
import numpy as np

# Minimum meaningful possession window duration per spec
MIN_POSSESSION_MINUTES = 15.0

def get_shap_explainer(model):
    """
    Create SHAP TreeExplainer from the XGBoost model.
    Handles extraction from CalibratedClassifierCV if necessary.
    """
    # If the model is wrapped in CalibratedClassifierCV, get the base XGBoost estimator
    base_model = getattr(model, 'estimator', model)
    if hasattr(base_model, 'get_booster'):
        return shap.TreeExplainer(base_model)
    return shap.TreeExplainer(model)

def predict_escalation_risk(model, explainer, feature_dict: dict, feature_names: list, rule_based_score: float, model_version: str = "v1.0") -> dict:
    """
    Predict escalation risk probability.
    Enforces the LOW_CONFIDENCE_BAND (0.40 - 0.60) fallback.
    Enforces cold-start try/except fallback.
    Extracts top risk factors using shap.TreeExplainer.
    """
    try:
        df = pd.DataFrame([feature_dict], columns=feature_names)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            proba = float(model.predict_proba(df)[0][1])
        else:
            proba = float(model.predict(df)[0])
            
        # SHAP Explainability
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list): # For multiclass/binary, shap might return a list
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]
            
        # Extract top 3 risk factors (by highest absolute influence driving risk up)
        top_indices = np.argsort(shap_vals)[-3:][::-1]
        top_factors = [feature_names[i] for i in top_indices if shap_vals[i] > 0]
        if not top_factors:
            # If no features actively pushed the risk up, just list the highest magnitude ones
            top_factors = [feature_names[i] for i in np.argsort(np.abs(shap_vals))[-3:][::-1]]

        # Low-Confidence Band Fallback
        if 0.40 <= proba <= 0.60:
            return {
                "escalation_risk_probability": proba,
                "escalation_risk_R": float(rule_based_score),
                "risk_confidence": "low_model_confidence",
                "top_risk_factors": top_factors,
                "model_version": model_version
            }
            
        return {
            "escalation_risk_probability": proba,
            "escalation_risk_R": proba * 100.0,
            "risk_confidence": "high",
            "top_risk_factors": top_factors,
            "model_version": model_version
        }
    except Exception:
        # Cold-start or missing data fallback
        return {
            "escalation_risk_probability": 0.0,
            "escalation_risk_R": float(rule_based_score),
            "risk_confidence": "fallback_rule_based",
            "top_risk_factors": ["cold_start_fallback"],
            "model_version": model_version
        }

def predict_duration(model, feature_dict: dict, feature_names: list, static_fallback_duration: float) -> dict:
    """
    Predict duration at P80 quantile.
    Clamps duration to max(pred, MIN_POSSESSION_MINUTES).
    """
    try:
        df = pd.DataFrame([feature_dict], columns=feature_names)
        pred_duration = float(model.predict(df)[0])
        
        final_duration = max(pred_duration, MIN_POSSESSION_MINUTES)
        
        return {
            "predicted_duration_minutes": final_duration,
            "duration_quantile": "p80",
            "duration_source": "model_p80"
        }
    except Exception:
        # Cold-start fallback
        return {
            "predicted_duration_minutes": float(max(static_fallback_duration, MIN_POSSESSION_MINUTES)),
            "duration_quantile": "p50_static",
            "duration_source": "static_fallback"
        }

def generate_ml_enrichment(
    escalation_model,
    duration_model,
    explainer,
    feature_dict: dict,
    feature_names: list,
    rule_based_score: float,
    static_fallback_duration: float,
    cluster_id: str = None,
    model_version: str = "v1.0"
) -> dict:
    """
    Generates the complete MLEnrichment dictionary according to contract.ts.
    """
    risk_info = predict_escalation_risk(
        escalation_model, explainer, feature_dict, feature_names, rule_based_score, model_version
    )
    
    duration_info = predict_duration(
        duration_model, feature_dict, feature_names, static_fallback_duration
    )
    
    # Merge dictionaries
    ml_enrichment = {**risk_info, **duration_info}
    if cluster_id is not None:
        ml_enrichment["cluster_id"] = str(cluster_id)
        
    return ml_enrichment
