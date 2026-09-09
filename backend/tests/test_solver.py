import pytest
from pydantic import ValidationError
from app.schemas.contracts import ScheduleRequest
from app.solver.cp_sat_core import solve_plan
from app.solver.greedy_heuristic import greedy_heuristic

FIXTURE_PAYLOAD = {
  "planning_horizon": {
    "start_datetime": "2026-09-08T00:00:00+05:30",
    "horizon_minutes": 1440,
    "timezone": "Asia/Kolkata"
  },
  "block_sections": [
    {
      "block_section_id": "BS-101",
      "name": "Test Section",
      "division": "Delhi",
      "adjacent_section_ids": [],
      "track_type": "single"
    }
  ],
  "train_movements": [
    {
      "train_id": "T1",
      "block_section_id": "BS-101",
      "direction": "UP",
      "scheduled_entry": "2026-09-08T02:00:00+05:30",
      "scheduled_exit": "2026-09-08T04:00:00+05:30",
      "headway_before_minutes": 0,
      "headway_after_minutes": 0,
      "flexibility": "fixed",
      "max_shift_minutes": 0,
      "train_priority_class": "A"
    }
  ],
  "maintenance_requests": [
    {
      "work_id": "M1",
      "source_system": "SMMS",
      "defect_type": "typeA",
      "job_type": "single",
      "block_sections_required": ["BS-101"],
      "duration_minutes": 120,
      "earliest_start": "2026-09-08T00:00:00+05:30",
      "latest_start": "2026-09-08T05:00:00+05:30",
      "mandatory": True,
      "crew_type": "C1",
      "crew_size": 1,
      "priority_score": 90.0,
      "setup_minutes": 0
    },
    {
      "work_id": "M2",
      "source_system": "SMMS",
      "defect_type": "typeA",
      "job_type": "single",
      "block_sections_required": ["BS-101"],
      "duration_minutes": 120,
      "earliest_start": "2026-09-08T00:00:00+05:30",
      "latest_start": "2026-09-08T05:00:00+05:30",
      "mandatory": True,
      "crew_type": "C1",
      "crew_size": 1,
      "priority_score": 80.0,
      "setup_minutes": 0
    }
  ],
  "resource_capacity": {
    "crew_pools": [
      { "crew_type": "C1", "shifts": [{ "start": "00:00", "end": "23:59", "capacity": 1 }] }
    ],
    "plant_pools": []
  },
  "solver_config": {
    "max_time_in_seconds": 2.0,
    "num_search_workers": 1,
    "random_seed": 42
  }
}

def test_greedy_heuristic_fallback():
    request = ScheduleRequest.model_validate(FIXTURE_PAYLOAD)
    result = greedy_heuristic(request)
    
    # M1 and M2 cannot fit at the same time due to crew capacity and track conflict
    # One should be assigned, one deferred
    assert len(result["assigned"]) == 1
    assert len(result["deferred"]) == 1
    
    # M1 has higher priority so it should be assigned
    assert result["assigned"][0]["work_id"] == "M1"
    assert result["deferred"][0]["work_id"] == "M2"

def test_cpsat_soft_mandatory():
    request = ScheduleRequest.model_validate(FIXTURE_PAYLOAD)
    
    # Run the solver. The two mandatory jobs overlap in a tiny window disrupted by T1.
    # T1 blocks 02:00 to 04:00.
    # M1 and M2 each need 120 mins in a 4-hour window (01:00 to 05:00) where 2 hours are blocked.
    # It is physically impossible to schedule both.
    
    response = solve_plan(request)
    
    # Should resolve gracefully (FEASIBLE or OPTIMAL), NOT INFEASIBLE/Crash
    assert response.solver_status in ["FEASIBLE", "OPTIMAL", "FALLBACK_HEURISTIC"]
    
    # We expect 1 job scheduled, 1 job deferred
    assert len(response.selected_work_packages) == 1
    assert len(response.deferred_work_packages) == 1
    
    # The dropped mandatory item should be flagged in KPIs
    assert response.kpis.mandatory_items_dropped == 1
