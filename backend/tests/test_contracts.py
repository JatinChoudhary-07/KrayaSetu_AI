import pytest
from pydantic import ValidationError
from app.schemas.contracts import ScheduleRequest, ScheduleResponse, SolverStatus

INPUT_PAYLOAD = {
  "planning_horizon": {
    "start_datetime": "2026-09-08T00:00:00+05:30",
    "horizon_minutes": 1440,
    "timezone": "Asia/Kolkata"
  },
  "block_sections": [
    {
      "block_section_id": "BS-101",
      "name": "GZB-MTC UP Line",
      "division": "Delhi",
      "adjacent_section_ids": ["BS-102"],
      "track_type": "double_line"
    }
  ],
  "train_movements": [
    {
      "train_id": "12301",
      "block_section_id": "BS-101",
      "direction": "UP",
      "scheduled_entry": "2026-09-08T05:12:00+05:30",
      "scheduled_exit": "2026-09-08T05:19:00+05:30",
      "headway_before_minutes": 2,
      "headway_after_minutes": 3,
      "flexibility": "fixed",
      "max_shift_minutes": 0,
      "train_priority_class": "A_superfast"
    }
  ],
  "maintenance_requests": [
    {
      "work_id": "WP-4521",
      "source_system": "SMMS",
      "defect_type": "rail_fracture_risk",
      "block_sections_required": ["BS-101"],
      "duration_minutes": 90,
      "earliest_start": "2026-09-08T22:00:00+05:30",
      "latest_start": "2026-09-09T04:00:00+05:30",
      "mandatory": True,
      "crew_type": "P-Way_gang",
      "crew_size": 6,
      "plant_type": "tamping_machine",
      "plant_qty": 1,
      "isolation_group": "ISO-GZB-01",
      "precedence_after": [],
      "priority_inputs": {
        "severity_S": 92,
        "escalation_risk_R": 70,
        "criticality_C": 85,
        "age_A": 40,
        "opportunity_O": 60
      },
      "priority_score": 78.4
    },
    {
      "work_id": "MW-0100",
      "source_system": "SMMS",
      "defect_type": "tamping_corridor",
      "job_type": "traveling",
      "route_legs": [
        { "leg_id": 1, "block_section_id": "BS-101", "duration_minutes": 70 },
        { "leg_id": 2, "block_section_id": "BS-102", "duration_minutes": 45 },
        { "leg_id": 3, "block_section_id": "BS-103", "duration_minutes": 90 }
      ],
      "max_wait_minutes": 20,
      "setup_minutes": 20,
      "earliest_start": "2026-09-08T22:00:00+05:30",
      "latest_start": "2026-09-09T05:00:00+05:30",
      "mandatory": False,
      "crew_type": "tamping_gang",
      "crew_size": 5,
      "plant_type": "tamping_machine",
      "plant_qty": 1,
      "priority_score": 74.0
    }
  ],
  "resource_capacity": {
    "crew_pools": [
      { "crew_type": "P-Way_gang", "shifts": [{ "start": "22:00", "end": "06:00", "capacity": 8 }] }
    ],
    "plant_pools": [
      { "plant_type": "tamping_machine", "total_qty": 2 }
    ]
  },
  "solver_config": {
    "max_time_in_seconds": 8,
    "num_search_workers": 8,
    "random_seed": 42
  }
}

OUTPUT_PAYLOAD = {
  "schedule_id": "PLAN-20260908-0001",
  "generated_at": "2026-09-07T18:03:11+05:30",
  "solver_status": "FEASIBLE",
  "objective_value": 812.6,
  "best_bound": 845.0,
  "optimality_gap_pct": 3.85,
  "solve_time_ms": 7940,
  "selected_work_packages": [
    {
      "work_id": "WP-4521",
      "selected": True,
      "assigned_start": "2026-09-08T22:00:00+05:30",
      "assigned_end": "2026-09-08T23:30:00+05:30",
      "block_sections_used": ["BS-101"],
      "crew_assigned": { "crew_type": "P-Way_gang", "crew_size": 6 },
      "plant_assigned": { "plant_type": "tamping_machine", "plant_qty": 1 },
      "bundle_group_id": "BUNDLE-01",
      "priority_score": 78.4
    }
  ],
  "traveling_job_schedules": [
    {
      "work_id": "MW-0100",
      "selected": True,
      "legs": [
        { "block_section_id": "BS-101", "start": "2026-09-08T22:00:00+05:30", "end": "2026-09-08T23:30:00+05:30" },
        { "block_section_id": "BS-102", "start": "2026-09-08T23:35:00+05:30", "end": "2026-09-09T00:20:00+05:30" },
        { "block_section_id": "BS-103", "start": "2026-09-09T00:22:00+05:30", "end": "2026-09-09T01:52:00+05:30" }
      ],
      "crew_assigned": { "crew_type": "tamping_gang", "crew_size": 5 },
      "plant_assigned": { "plant_type": "tamping_machine", "plant_qty": 1 },
      "priority_score": 74.0
    }
  ],
  "deferred_work_packages": [
    {
      "work_id": "WP-4589",
      "selected": False,
      "reason_code": "CREW_CAPACITY_EXCEEDED",
      "reason_detail": "P-Way_gang capacity (8) fully consumed by WP-4521, WP-4530 in window 22:00-02:00",
      "priority_score": 41.2,
      "suggested_next_window": "2026-09-09T22:00:00+05:30"
    }
  ],
  "gantt_tasks": [
    {
      "id": "WP-4521",
      "text": "Rail fracture repair - BS-101",
      "start_date": "2026-09-08 22:00",
      "end_date": "2026-09-08 23:30",
      "parent": "BS-101",
      "type": "maintenance",
      "priority_color": "red",
      "conflict": False
    }
  ],
  "conflicts": [
    {
      "conflict_id": "CFX-01",
      "type": "resource_capacity",
      "involved_work_ids": ["WP-4589", "WP-4521", "WP-4530"],
      "block_section_id": None,
      "description": "Combined crew demand exceeds P-Way_gang shift capacity"
    }
  ],
  "kpis": {
    "train_delay_minutes_total": 0,
    "work_packages_included_pct": 84.7,
    "unused_block_minutes": 55,
    "mandatory_items_dropped": 0
  }
}

def test_deserialize_input_payload():
    request = ScheduleRequest.model_validate(INPUT_PAYLOAD)
    assert request.planning_horizon.horizon_minutes == 1440
    assert len(request.maintenance_requests) == 2
    assert request.maintenance_requests[1].job_type == "traveling"
    assert len(request.maintenance_requests[1].route_legs) == 3

def test_serialize_output_payload():
    response = ScheduleResponse.model_validate(OUTPUT_PAYLOAD)
    serialized = response.model_dump()
    assert serialized["solver_status"] == "FEASIBLE"
    assert len(serialized["traveling_job_schedules"]) == 1
    assert serialized["traveling_job_schedules"][0]["legs"][0]["block_section_id"] == "BS-101"
    assert serialized["deferred_work_packages"][0]["reason_code"] == "CREW_CAPACITY_EXCEEDED"

def test_solver_status_validation():
    valid_statuses = ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "FALLBACK_HEURISTIC"]
    
    for status in valid_statuses:
        payload = OUTPUT_PAYLOAD.copy()
        payload["solver_status"] = status
        response = ScheduleResponse.model_validate(payload)
        assert response.solver_status == status

    # Invalid status should raise ValidationError
    payload = OUTPUT_PAYLOAD.copy()
    payload["solver_status"] = "UNKNOWN_STATUS"
    with pytest.raises(ValidationError):
        ScheduleResponse.model_validate(payload)
