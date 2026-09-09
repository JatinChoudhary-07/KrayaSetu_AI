from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class PlanningHorizon(BaseModel):
    start_datetime: str
    horizon_minutes: int
    timezone: str

class BlockSection(BaseModel):
    block_section_id: str
    name: str
    division: str
    adjacent_section_ids: List[str]
    track_type: str

class TrainMovement(BaseModel):
    train_id: str
    block_section_id: str
    direction: str
    scheduled_entry: str
    scheduled_exit: str
    headway_before_minutes: int
    headway_after_minutes: int
    flexibility: Literal["fixed", "semi_flexible"]
    max_shift_minutes: int
    train_priority_class: str

class PriorityInputs(BaseModel):
    severity_S: float
    escalation_risk_R: float
    criticality_C: float
    age_A: float
    opportunity_O: float

class RouteLeg(BaseModel):
    leg_id: int
    block_section_id: str
    duration_minutes: int

class MLEnrichment(BaseModel):
    escalation_risk_probability: float
    escalation_risk_R: float
    risk_confidence: Literal["high", "low_model_confidence", "fallback_rule_based"]
    top_risk_factors: List[str]
    predicted_duration_minutes: int
    duration_quantile: str
    duration_source: Literal["model_p80", "static_fallback"]
    cluster_id: Optional[str] = None
    model_version: str

class MaintenanceRequest(BaseModel):
    work_id: str
    source_system: str
    defect_type: str
    job_type: Optional[Literal["single", "traveling"]] = None
    block_sections_required: Optional[List[str]] = None
    duration_minutes: Optional[int] = None
    route_legs: Optional[List[RouteLeg]] = None
    max_wait_minutes: Optional[int] = None
    setup_minutes: Optional[int] = None
    earliest_start: str
    latest_start: str
    mandatory: bool
    crew_type: Optional[str] = None
    crew_size: Optional[int] = None
    plant_type: Optional[str] = None
    plant_qty: Optional[int] = None
    isolation_group: Optional[str] = None
    precedence_after: Optional[List[str]] = Field(default_factory=list)
    priority_inputs: Optional[PriorityInputs] = None
    priority_score: float
    ml_enrichment: Optional[MLEnrichment] = None

class CrewShift(BaseModel):
    start: str
    end: str
    capacity: int

class CrewPool(BaseModel):
    crew_type: str
    shifts: List[CrewShift]

class PlantPool(BaseModel):
    plant_type: str
    total_qty: int

class ResourceCapacity(BaseModel):
    crew_pools: List[CrewPool]
    plant_pools: List[PlantPool]

class SolverConfig(BaseModel):
    max_time_in_seconds: float = 8.0
    num_search_workers: int = 8
    random_seed: int = 42
    max_number_of_conflicts: int = 100000

class DailyCapacityCalendar(BaseModel):
    date: str
    crew_hours: Dict[str, int]
    plant_hours: Dict[str, int]
    corridor_possession_hours: Dict[str, int]

class ScheduleRequest(BaseModel):
    planning_horizon: PlanningHorizon
    block_sections: List[BlockSection]
    train_movements: List[TrainMovement]
    maintenance_requests: List[MaintenanceRequest]
    resource_capacity: ResourceCapacity
    solver_config: SolverConfig = Field(default_factory=SolverConfig)
    daily_capacity_calendar: Optional[List[DailyCapacityCalendar]] = None

class SolverStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"

class SelectedWorkPackage(BaseModel):
    work_id: str
    selected: bool
    assigned_start: str
    assigned_end: str
    block_sections_used: List[str]
    crew_assigned: Optional[Dict[str, Any]] = None
    plant_assigned: Optional[Dict[str, Any]] = None
    bundle_group_id: Optional[str] = None
    priority_score: float

class TravelingJobLegSchedule(BaseModel):
    block_section_id: str
    start: str
    end: str

class TravelingJobSchedule(BaseModel):
    work_id: str
    selected: bool
    legs: List[TravelingJobLegSchedule]
    crew_assigned: Optional[Dict[str, Any]] = None
    plant_assigned: Optional[Dict[str, Any]] = None
    priority_score: float

class ReasonCode(str, Enum):
    NO_FEASIBLE_WINDOW = "NO_FEASIBLE_WINDOW"
    CREW_CAPACITY_EXCEEDED = "CREW_CAPACITY_EXCEEDED"
    PLANT_CAPACITY_EXCEEDED = "PLANT_CAPACITY_EXCEEDED"
    LOWER_PRIORITY = "LOWER_PRIORITY"
    NO_CAPACITY_BEFORE_DEADLINE = "NO_CAPACITY_BEFORE_DEADLINE"

class DeferredWorkPackage(BaseModel):
    work_id: str
    selected: bool
    reason_code: ReasonCode
    reason_detail: str
    priority_score: float
    suggested_next_window: Optional[str] = None

class GanttTask(BaseModel):
    id: str
    text: str
    start_date: str
    end_date: str
    parent: str
    type: str
    priority_color: str
    conflict: bool

class Conflict(BaseModel):
    conflict_id: str
    type: str
    involved_work_ids: List[str]
    block_section_id: Optional[str] = None
    description: str

class KPIs(BaseModel):
    train_delay_minutes_total: int
    work_packages_included_pct: float
    unused_block_minutes: int
    mandatory_items_dropped: int

class ScheduleResponse(BaseModel):
    schedule_id: str
    generated_at: str
    solver_status: SolverStatus
    objective_value: Optional[float] = None
    best_bound: Optional[float] = None
    optimality_gap_pct: Optional[float] = None
    solve_time_ms: int
    selected_work_packages: List[SelectedWorkPackage]
    traveling_job_schedules: List[TravelingJobSchedule]
    deferred_work_packages: List[DeferredWorkPackage]
    gantt_tasks: List[GanttTask]
    conflicts: List[Conflict]
    kpis: KPIs

class DayAllocation(BaseModel):
    date: str
    assigned_work_ids: List[str]

class UnassignableWork(BaseModel):
    work_id: str
    reason_code: ReasonCode
    mandatory: bool

class MacroAllocationCalendar(BaseModel):
    generated_at: str
    horizon_days: int
    days: List[DayAllocation]
    unassignable_before_deadline: List[UnassignableWork]
