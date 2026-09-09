export interface PlanningHorizon {
    start_datetime: string;
    horizon_minutes: number;
    timezone: string;
}

export interface BlockSection {
    block_section_id: string;
    name: string;
    division: string;
    adjacent_section_ids: string[];
    track_type: string;
}

export interface TrainMovement {
    train_id: string;
    block_section_id: string;
    direction: string;
    scheduled_entry: string;
    scheduled_exit: string;
    headway_before_minutes: number;
    headway_after_minutes: number;
    flexibility: "fixed" | "semi_flexible";
    max_shift_minutes: number;
    train_priority_class: string;
}

export interface PriorityInputs {
    severity_S: number;
    escalation_risk_R: number;
    criticality_C: number;
    age_A: number;
    opportunity_O: number;
}

export interface RouteLeg {
    leg_id: number;
    block_section_id: string;
    duration_minutes: number;
}

export interface MLEnrichment {
    escalation_risk_probability: number;
    escalation_risk_R: number;
    risk_confidence: "high" | "low_model_confidence" | "fallback_rule_based";
    top_risk_factors: string[];
    predicted_duration_minutes: number;
    duration_quantile: string;
    duration_source: "model_p80" | "static_fallback";
    cluster_id?: string;
    model_version: string;
}

export interface MaintenanceRequest {
    work_id: string;
    source_system: string;
    defect_type: string;
    job_type?: "single" | "traveling";
    block_sections_required?: string[];
    duration_minutes?: number;
    route_legs?: RouteLeg[];
    max_wait_minutes?: number;
    setup_minutes?: number;
    earliest_start: string;
    latest_start: string;
    mandatory: boolean;
    crew_type?: string;
    crew_size?: number;
    plant_type?: string;
    plant_qty?: number;
    isolation_group?: string;
    precedence_after?: string[];
    priority_inputs?: PriorityInputs;
    priority_score: number;
    ml_enrichment?: MLEnrichment;
}

export interface CrewShift {
    start: string;
    end: string;
    capacity: number;
}

export interface CrewPool {
    crew_type: string;
    shifts: CrewShift[];
}

export interface PlantPool {
    plant_type: string;
    total_qty: number;
}

export interface ResourceCapacity {
    crew_pools: CrewPool[];
    plant_pools: PlantPool[];
}

export interface SolverConfig {
    max_time_in_seconds?: number;
    num_search_workers?: number;
    random_seed?: number;
    max_number_of_conflicts?: number;
}

export interface DailyCapacityCalendar {
    date: string;
    crew_hours: Record<string, number>;
    plant_hours: Record<string, number>;
    corridor_possession_hours: Record<string, number>;
}

export interface ScheduleRequest {
    planning_horizon: PlanningHorizon;
    block_sections: BlockSection[];
    train_movements: TrainMovement[];
    maintenance_requests: MaintenanceRequest[];
    resource_capacity: ResourceCapacity;
    solver_config?: SolverConfig;
    daily_capacity_calendar?: DailyCapacityCalendar[];
}

export type SolverStatus = 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'FALLBACK_HEURISTIC';

export interface SelectedWorkPackage {
    work_id: string;
    selected: boolean;
    assigned_start: string;
    assigned_end: string;
    block_sections_used: string[];
    crew_assigned?: Record<string, any>;
    plant_assigned?: Record<string, any>;
    bundle_group_id?: string;
    priority_score: number;
}

export interface TravelingJobLegSchedule {
    block_section_id: string;
    start: string;
    end: string;
}

export interface TravelingJobSchedule {
    work_id: string;
    selected: boolean;
    legs: TravelingJobLegSchedule[];
    crew_assigned?: Record<string, any>;
    plant_assigned?: Record<string, any>;
    priority_score: number;
}

export type ReasonCode = 'NO_FEASIBLE_WINDOW' | 'CREW_CAPACITY_EXCEEDED' | 'PLANT_CAPACITY_EXCEEDED' | 'LOWER_PRIORITY' | 'NO_CAPACITY_BEFORE_DEADLINE';

export interface DeferredWorkPackage {
    work_id: string;
    selected: boolean;
    reason_code: ReasonCode;
    reason_detail: string;
    priority_score: number;
    suggested_next_window?: string;
}

export interface GanttTask {
    id: string;
    text: string;
    start_date: string;
    end_date: string;
    parent: string;
    type: string;
    priority_color: string;
    conflict: boolean;
}

export interface Conflict {
    conflict_id: string;
    type: string;
    involved_work_ids: string[];
    block_section_id?: string;
    description: string;
}

export interface KPIs {
    train_delay_minutes_total: number;
    work_packages_included_pct: number;
    unused_block_minutes: number;
    mandatory_items_dropped: number;
}

export interface ScheduleResponse {
    schedule_id: string;
    generated_at: string;
    solver_status: SolverStatus;
    objective_value?: number;
    best_bound?: number;
    optimality_gap_pct?: number;
    solve_time_ms: number;
    selected_work_packages: SelectedWorkPackage[];
    traveling_job_schedules: TravelingJobSchedule[];
    deferred_work_packages: DeferredWorkPackage[];
    gantt_tasks: GanttTask[];
    conflicts: Conflict[];
    kpis: KPIs;
}

export interface DayAllocation {
    date: string;
    assigned_work_ids: string[];
}

export interface UnassignableWork {
    work_id: string;
    reason_code: ReasonCode;
    mandatory: boolean;
}

export interface MacroAllocationCalendar {
    generated_at: string;
    horizon_days: number;
    days: DayAllocation[];
    unassignable_before_deadline: UnassignableWork[];
}
