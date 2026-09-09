from typing import Tuple, List, Dict
from datetime import datetime
from app.schemas.contracts import ScheduleRequest, SelectedWorkPackage, TravelingJobSchedule
from app.solver.greedy_heuristic import parse_time_to_minutes

def determine_reason_code(
    job,
    request: ScheduleRequest,
    selected_wps: List[SelectedWorkPackage],
    traveling_wps: List[TravelingJobSchedule],
    start_dt: datetime
) -> Tuple[str, str]:
    """
    Deterministic 3-check fallback logic for explaining why a job was dropped.
    Order: NO_FEASIBLE_WINDOW -> CAPACITY_EXCEEDED -> LOWER_PRIORITY
    """
    # Parse bounds
    s_min = parse_time_to_minutes(job.earliest_start, start_dt)
    e_max = parse_time_to_minutes(job.latest_start, start_dt)
    dur = job.duration_minutes if job.duration_minutes is not None else 60
    
    required_blocks = job.block_sections_required or []
    if job.job_type == "traveling":
        required_blocks = [leg.block_section_id for leg in (job.route_legs or [])]
        dur = sum(leg.duration_minutes for leg in (job.route_legs or [])) + (job.setup_minutes if job.setup_minutes is not None else 20)
        
    # Check 1: NO_FEASIBLE_WINDOW (Track Exclusivity)
    # Check if there is physically any continuous block of `dur` minutes in [s_min, e_max] 
    # that is free from fixed trains for the required blocks.
    train_intervals = []
    for t in request.train_movements:
        if t.flexibility == "fixed" and t.block_section_id in required_blocks:
            t_s = parse_time_to_minutes(t.scheduled_entry, start_dt) - t.headway_before_minutes
            t_e = parse_time_to_minutes(t.scheduled_exit, start_dt) + t.headway_after_minutes
            train_intervals.append((t_s, t_e))
            
    # Sort and merge train intervals
    train_intervals.sort(key=lambda x: x[0])
    merged_trains = []
    for t_s, t_e in train_intervals:
        if not merged_trains:
            merged_trains.append([t_s, t_e])
        else:
            prev_s, prev_e = merged_trains[-1]
            if t_s <= prev_e:
                merged_trains[-1][1] = max(prev_e, t_e)
            else:
                merged_trains.append([t_s, t_e])
                
    # Find max free gap in bounds
    max_gap = 0
    curr_t = s_min
    for t_s, t_e in merged_trains:
        if t_s > curr_t:
            gap = min(t_s, e_max) - curr_t
            if gap > max_gap:
                max_gap = gap
        curr_t = max(curr_t, t_e)
        if curr_t >= e_max:
            break
            
    if curr_t < e_max:
        gap = e_max - curr_t
        if gap > max_gap:
            max_gap = gap
            
    if max_gap < dur:
        return ("NO_FEASIBLE_WINDOW", f"No continuous {dur}-min track window available between {job.earliest_start} and {job.latest_start} due to fixed trains")

    # Check 2: CAPACITY_EXCEEDED
    # Check if crew capacity is exhausted
    if job.crew_type and job.crew_size:
        crew_cap = 0
        for pool in request.resource_capacity.crew_pools:
            if pool.crew_type == job.crew_type:
                crew_cap = max((s.capacity for s in pool.shifts), default=0)
                break
        if job.crew_size > crew_cap:
            return ("CREW_CAPACITY_EXCEEDED", f"{job.crew_type} demand ({job.crew_size}) exceeds absolute shift capacity ({crew_cap})")
            
    if job.plant_type and job.plant_qty:
        plant_cap = 0
        for pool in request.resource_capacity.plant_pools:
            if pool.plant_type == job.plant_type:
                plant_cap = pool.total_qty
                break
        if job.plant_qty > plant_cap:
            return ("PLANT_CAPACITY_EXCEEDED", f"{job.plant_type} demand ({job.plant_qty}) exceeds total pool capacity ({plant_cap})")

    # Check 3: LOWER_PRIORITY
    # It fit, but CP-SAT prioritized something else
    competing_jobs = []
    for w in selected_wps:
        if any(b in required_blocks for b in w.block_sections_used):
            comp_s = parse_time_to_minutes(w.assigned_start, start_dt)
            comp_e = parse_time_to_minutes(w.assigned_end, start_dt)
            if comp_s < e_max and comp_e > s_min:
                if w.priority_score >= job.priority_score:
                    competing_jobs.append(w.work_id)
                    
    for tw in traveling_wps:
        legs_blocks = [l.block_section_id for l in tw.legs]
        if any(b in required_blocks for b in legs_blocks):
            # Check overlap
            comp_s = parse_time_to_minutes(tw.legs[0].start, start_dt)
            comp_e = parse_time_to_minutes(tw.legs[-1].end, start_dt)
            if comp_s < e_max and comp_e > s_min:
                if tw.priority_score >= job.priority_score:
                    competing_jobs.append(tw.work_id)

    if competing_jobs:
        winners = ", ".join(competing_jobs[:3])
        if len(competing_jobs) > 3:
            winners += f" and {len(competing_jobs)-3} others"
        return ("LOWER_PRIORITY", f"Deferred in favor of higher-priority scheduled work ({winners})")
        
    return ("LOWER_PRIORITY", "Deferred due to objective optimization constraints")
