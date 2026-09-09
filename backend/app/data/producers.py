from app.schemas.contracts import ScheduleRequest

MOCK_PAYLOAD = {
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
    },
    {
      "block_section_id": "BS-102",
      "name": "GZB-MTC UP Line 2",
      "division": "Delhi",
      "adjacent_section_ids": ["BS-101", "BS-103"],
      "track_type": "double_line"
    },
    {
      "block_section_id": "BS-103",
      "name": "GZB-MTC UP Line 3",
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
      "priority_score": 78.4,
      "ml_enrichment": {
        "escalation_risk_probability": 0.8,
        "escalation_risk_R": 70,
        "risk_confidence": "high",
        "top_risk_factors": ["age", "tonnage"],
        "predicted_duration_minutes": 90,
        "duration_quantile": "p80",
        "duration_source": "model_p80",
        "cluster_id": "BUNDLE-01",
        "model_version": "v2"
      }
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
      { "crew_type": "P-Way_gang", "shifts": [{ "start": "22:00", "end": "06:00", "capacity": 8 }] },
      { "crew_type": "tamping_gang", "shifts": [{ "start": "22:00", "end": "06:00", "capacity": 10 }] }
    ],
    "plant_pools": [
      { "plant_type": "tamping_machine", "total_qty": 2 }
    ]
  },
  "solver_config": {
    "max_time_in_seconds": 2.0,
    "num_search_workers": 1,
    "random_seed": 42
  }
}

def generate_mock_fixture() -> ScheduleRequest:
    return ScheduleRequest.model_validate(MOCK_PAYLOAD)
