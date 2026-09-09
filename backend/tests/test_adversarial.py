import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.contracts import ScheduleRequest

client = TestClient(app)

def generate_adversarial_payload() -> dict:
    """
    Generates a highly dense payload designed to break CP-SAT if the
    timeout or soft-mandatory bounds are missing.
    50 jobs, 10 mandatory, all clustered on BS-X, tight window, 1 crew pool.
    50 trains blocking BS-X at regular intervals.
    """
    import copy
    from app.data.producers import MOCK_PAYLOAD
    
    payload = copy.deepcopy(MOCK_PAYLOAD)
    
    # 1. 50 trains heavily fragmenting track time
    payload["train_movements"] = []
    for i in range(50):
        # 10 minutes every 20 minutes from 00:00 to 16:40
        h = (i * 20) // 60
        m = (i * 20) % 60
        h2 = (i * 20 + 10) // 60
        m2 = (i * 20 + 10) % 60
        payload["train_movements"].append({
            "train_id": f"TRN-{i}",
            "block_section_id": "BS-101",
            "direction": "UP",
            "scheduled_entry": f"2026-09-08T{h:02d}:{m:02d}:00+05:30",
            "scheduled_exit": f"2026-09-08T{h2:02d}:{m2:02d}:00+05:30",
            "headway_before_minutes": 2,
            "headway_after_minutes": 2,
            "flexibility": "fixed",
            "max_shift_minutes": 0,
            "train_priority_class": "A"
        })
        
    # 2. 50 maintenance requests all wanting BS-101 in a short window
    payload["maintenance_requests"] = []
    for i in range(50):
        # Window: 00:00 to 06:00
        # Duration: 30 to 60 mins
        # Crew: All P-Way_gang
        # Mandatory: First 10
        payload["maintenance_requests"].append({
            "work_id": f"WP-ADV-{i}",
            "source_system": "SMMS",
            "defect_type": "typeA",
            "block_sections_required": ["BS-101"],
            "duration_minutes": 45,
            "earliest_start": "2026-09-08T00:00:00+05:30",
            "latest_start": "2026-09-08T06:00:00+05:30",
            "mandatory": i < 10,
            "crew_type": "P-Way_gang",
            "crew_size": 4, # Shift capacity is 8, so 2 can fit at a time, but track is fragmented
            "priority_score": 100.0 - i,
            "setup_minutes": 10
        })
        
    payload["solver_config"] = {
        "max_time_in_seconds": 8.0,
        "num_search_workers": 4,
        "random_seed": 42
    }
    
    return payload

def test_adversarial_stress():
    payload = generate_adversarial_payload()
    
    import time
    start_time = time.time()
    
    response = client.post("/plan", json=payload)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Assert HTTP 200
    assert response.status_code == 200
    
    data = response.json()
    
    # Assert solve completed within tight bounds (~8 seconds CP-SAT + ~2 seconds overhead max)
    assert elapsed < 12.0
    
    # It must drop mandatory items because 10 mandatory jobs * 45 minutes = 450 minutes of work.
    # But window is 6 hours (360 minutes), and half of it is blocked by trains.
    # Therefore, it is mathematically impossible to fit all 10.
    assert data["kpis"]["mandatory_items_dropped"] > 0
    
    # Deferred items should have reason codes
    for deferred in data["deferred_work_packages"]:
        assert deferred["reason_code"] in ["NO_FEASIBLE_WINDOW", "CREW_CAPACITY_EXCEEDED", "PLANT_CAPACITY_EXCEEDED", "LOWER_PRIORITY"]
        
def test_macro_planner():
    import copy
    from app.data.producers import MOCK_PAYLOAD
    payload = copy.deepcopy(MOCK_PAYLOAD)
    
    # Provide daily capacity
    payload["daily_capacity_calendar"] = [
      {
        "date": "2026-09-08T00:00:00+05:30",
        "crew_hours": { "P-Way_gang": 2, "tamping_gang": 0 }, # Very low capacity
        "plant_hours": { "tamping_machine": 0 },
        "corridor_possession_hours": { "BS-101": 2, "BS-102": 2, "BS-103": 2 }
      }
    ]
    
    response = client.post("/macro-plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # With 0 tamping plant hours and 0 tamping gang hours, the MW-0100 job should be deferred
    assert "unassignable_before_deadline" in data
    unassignable_ids = [u["work_id"] for u in data["unassignable_before_deadline"]]
    assert "MW-0100" in unassignable_ids
