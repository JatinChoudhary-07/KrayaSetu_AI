import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data.producers import MOCK_PAYLOAD
from app.data.railradar import RailRadarClient

client = TestClient(app)

def test_plan_endpoint():
    response = client.post("/plan", json=MOCK_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "schedule_id" in data
    assert data["solver_status"] in ["OPTIMAL", "FEASIBLE", "FALLBACK_HEURISTIC"]

@pytest.mark.asyncio
async def test_railradar_timeout():
    # Attempt to connect to a non-existent port to simulate timeout/connection error
    rr = RailRadarClient(base_url="http://localhost:9999")
    res = await rr.get_live_telemetry("BS-101")
    assert res["status"] == "FALLBACK"
    assert res["occupancy"] == "unknown"

def test_override_endpoint():
    response = client.post("/override", json={
        "work_id": "WP-9999",
        "override_type": "FORCE_INCLUDE",
        "parameters": {}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["work_id"] == "WP-9999"

def test_reason_code_engine_no_feasible_window():
    import copy
    from app.engine.reason_codes import determine_reason_code
    from app.schemas.contracts import ScheduleRequest
    from datetime import datetime
    
    payload = copy.deepcopy(MOCK_PAYLOAD)
    
    # Let's create a job that cannot fit due to NO_FEASIBLE_WINDOW
    # Train 12301 blocks BS-101 from 05:12 to 05:19 + 5 mins headways = 05:10 to 05:22 (min 310 to 322)
    # Give the job bounds exactly from 05:00 (300) to 05:30 (330), duration 25.
    # Gap before: 300 to 310 (10 mins). Gap after: 322 to 330 (8 mins).
    # Job needs 25, so it can't fit.
    
    req = ScheduleRequest.model_validate(payload)
    start_dt = datetime.fromisoformat(req.planning_horizon.start_datetime)
    
    # We create a dummy MaintenanceRequest
    job = next(j for j in req.maintenance_requests if j.work_id == "WP-4521")
    job.earliest_start = "2026-09-08T05:00:00+05:30"
    job.latest_start = "2026-09-08T05:30:00+05:30"
    job.duration_minutes = 25
    
    code, detail = determine_reason_code(job, req, [], [], start_dt)
    assert code == "NO_FEASIBLE_WINDOW"
    
def test_reason_code_engine_capacity_exceeded():
    import copy
    from app.engine.reason_codes import determine_reason_code
    from app.schemas.contracts import ScheduleRequest
    from datetime import datetime
    
    payload = copy.deepcopy(MOCK_PAYLOAD)
    req = ScheduleRequest.model_validate(payload)
    start_dt = datetime.fromisoformat(req.planning_horizon.start_datetime)
    
    # We create a dummy MaintenanceRequest with crazy capacity demand
    job = next(j for j in req.maintenance_requests if j.work_id == "WP-4521")
    job.crew_type = "P-Way_gang"
    job.crew_size = 999  # More than 8 available
    
    code, detail = determine_reason_code(job, req, [], [], start_dt)
    assert code == "CREW_CAPACITY_EXCEEDED"
