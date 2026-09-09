import pytest
import pandas as pd
import numpy as np

from app.ml.escalation import train_escalation_model, predict_escalation_risk, FEATURES
from app.ml.duration import train_duration_model, predict_duration_minutes, MIN_POSSESSION_MINUTES
from app.ml.clustering import cluster_defects_into_blocks

# Fixtures for training models
@pytest.fixture
def escalation_data():
    np.random.seed(42)
    N = 200
    data = {f: np.random.rand(N) * 10 for f in FEATURES}
    # Create an artificial signal
    escalation_prob = (data["track_age_years"] + data["gmt_since_renewal"]) / 20
    data["escalated_within_7d"] = (escalation_prob > 0.5).astype(int)
    return pd.DataFrame(data)
    
@pytest.fixture
def escalation_model(escalation_data):
    return train_escalation_model(escalation_data)

@pytest.fixture
def duration_data():
    np.random.seed(42)
    N = 100
    data = {
        "defect_type_encoded": np.random.randint(0, 5, N),
        "gang_id_encoded": np.random.randint(0, 3, N),
        "machine_type_encoded": np.random.randint(0, 2, N),
        "track_type_encoded": np.random.randint(0, 2, N),
        "crew_size": np.random.randint(4, 10, N),
        "is_night_shift": np.random.randint(0, 2, N),
    }
    # Durations between 20 and 120
    data["actual_duration_minutes"] = np.random.randint(20, 120, N)
    return pd.DataFrame(data)

@pytest.fixture
def duration_model(duration_data):
    return train_duration_model(duration_data, quantile=0.80)

def test_escalation_fallback(escalation_model):
    # 1. Test cold start fallback
    # Pass features the model won't like (e.g. string instead of float where model expects numeric)
    # The model expects exactly FEATURES
    res = predict_escalation_risk(escalation_model, {"bad_feature": "yes"}, rule_based_fallback_R=88)
    assert res["escalation_risk_R"] == 88
    assert res["risk_confidence"] == "fallback_rule_based"
    
    # 2. Test low confidence
    # We will just stub the model's predict_proba to force a 0.5 probability
    class StubModel:
        def predict_proba(self, df):
            return np.array([[0.5, 0.5]])
        # Mock tree explainer to avoid shap crash on stub
        # We'll just trigger the main logic
            
    # Well, patching predict_proba is easier by using unittest.mock, but let's test cold-start explicitly
    # To test confidence band, let's just make a very ambiguous input, or use the stub and disable shap temporarily
    # Actually, if we mock predict_proba we also must mock SHAP or handle the exception.
    # Instead, let's just monkeypatch the low confidence band
    from unittest.mock import patch
    with patch("app.ml.escalation.LOW_CONFIDENCE_BAND", (0.0, 1.0)):
        # Any prediction will now fall into low confidence
        valid_features = {f: 5.0 for f in FEATURES}
        res_low = predict_escalation_risk(escalation_model, valid_features, rule_based_fallback_R=77)
        assert res_low["escalation_risk_R"] == 77
        assert res_low["risk_confidence"] == "low_model_confidence"
        assert len(res_low["top_risk_factors"]) > 0  # SHAP should still extract features

def test_duration_floor(duration_model):
    # 1. Test cold start fallback
    res = predict_duration_minutes(duration_model, {"bad_feat": 1}, static_default_minutes=90)
    assert res["duration_minutes"] == 90
    assert res["duration_source"] == "static_fallback"
    
    # 2. Test that P80 predictions never undercut the 15 minute floor
    # We'll pass a valid feature set but monkeypatch predict to return 5.0 minutes
    valid_feat = {
        "defect_type_encoded": 1, "gang_id_encoded": 1, "machine_type_encoded": 1,
        "track_type_encoded": 1, "crew_size": 5, "is_night_shift": 0
    }
    class StubDurModel:
        def predict(self, df):
            return np.array([5.0])
            
    res_floor = predict_duration_minutes(StubDurModel(), valid_feat, static_default_minutes=90)
    assert res_floor["duration_minutes"] == 15
    assert res_floor["duration_source"] == "model_p80"

def test_dbscan_clustering():
    # 1. Geographically adjacent, overlapping windows
    defects = [
        {"id": "d1", "chainage_km": 10.0, "block_section_id": "BS-1", "earliest_start_min": 0, "latest_start_min": 100, "dur_i": 20, "priority_score": 10},
        {"id": "d2", "chainage_km": 11.5, "block_section_id": "BS-1", "earliest_start_min": 10, "latest_start_min": 90, "dur_i": 30, "priority_score": 20},
    ]
    # EPS=2.0, so d1 and d2 (dist 1.5) will cluster. Windows [0,100] and [10,90] overlap.
    blocks = cluster_defects_into_blocks(defects)
    assert len(blocks) == 1
    assert "d1" in blocks[0]["job_ids"]
    assert "d2" in blocks[0]["job_ids"]
    assert blocks[0]["total_duration"] == 20 + 30 + 20 # 20 is setup default
    
    # 2. Geographically adjacent, completely disjoint windows
    defects_disjoint = [
        {"id": "d1", "chainage_km": 10.0, "block_section_id": "BS-1", "earliest_start_min": 0, "latest_start_min": 100, "dur_i": 20, "priority_score": 10},
        {"id": "d3", "chainage_km": 11.5, "block_section_id": "BS-1", "earliest_start_min": 200, "latest_start_min": 300, "dur_i": 30, "priority_score": 20},
    ]
    # Dist 1.5 (clusters), but windows [0,100] and [200,300] do not overlap.
    blocks_disjoint = cluster_defects_into_blocks(defects_disjoint)
    assert len(blocks_disjoint) == 2
    
    # 3. Geographically distant
    defects_far = [
        {"id": "d1", "chainage_km": 10.0, "block_section_id": "BS-1", "earliest_start_min": 0, "latest_start_min": 100, "dur_i": 20, "priority_score": 10},
        {"id": "d4", "chainage_km": 15.0, "block_section_id": "BS-1", "earliest_start_min": 0, "latest_start_min": 100, "dur_i": 30, "priority_score": 20},
    ]
    # Dist 5.0 > EPS, so they don't cluster despite overlapping windows.
    blocks_far = cluster_defects_into_blocks(defects_far)
    assert len(blocks_far) == 2
