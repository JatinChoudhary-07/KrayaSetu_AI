from datetime import datetime, timedelta
import zoneinfo
from typing import Dict, Any
from app.schemas.contracts import ScheduleRequest

def parse_time_to_minutes(time_str: str, start_dt: datetime) -> int:
    dt = datetime.fromisoformat(time_str)
    return int((dt - start_dt).total_seconds() / 60)

def greedy_heuristic(request: ScheduleRequest) -> Dict[str, Any]:
    """
    A fast greedy heuristic to pack maintenance requests into available windows.
    Serves as a warm-start for CP-SAT and a fallback if it returns INFEASIBLE.
    """
    start_dt = datetime.fromisoformat(request.planning_horizon.start_datetime)
    
    forbidden_intervals = {b.block_section_id: [] for b in request.block_sections}
    
    for train in request.train_movements:
        s = parse_time_to_minutes(train.scheduled_entry, start_dt)
        e = parse_time_to_minutes(train.scheduled_exit, start_dt)
        
        s_padded = s - train.headway_before_minutes
        e_padded = e + train.headway_after_minutes
        
        if train.block_section_id in forbidden_intervals:
            forbidden_intervals[train.block_section_id].append((s_padded, e_padded))
            
    def job_urgency(job):
        return (job.mandatory, job.priority_score)
        
    sorted_jobs = sorted(request.maintenance_requests, key=job_urgency, reverse=True)
    
    assigned_jobs = []
    deferred_jobs = []
    
    for job in sorted_jobs:
        if job.job_type == "traveling":
            # Simplified greedy: treat traveling job as a monolithic block across all its route_legs for the total duration
            # In a real heuristic, we'd stagger them, but for warm-start this is okay if it finds a window.
            dur = sum(leg.duration_minutes for leg in (job.route_legs or [])) + (job.setup_minutes if job.setup_minutes is not None else 0)
            blocks_req = [leg.block_section_id for leg in (job.route_legs or [])]
        else:
            dur = job.duration_minutes if job.duration_minutes is not None else 60
            blocks_req = job.block_sections_required or []
            
        s_min = parse_time_to_minutes(job.earliest_start, start_dt)
        e_max = parse_time_to_minutes(job.latest_start, start_dt)
        
        placed = False
        for t in range(s_min, e_max - dur + 1, 5):
            window_end = t + dur
            
            overlap = False
            for bs in blocks_req:
                for f_s, f_e in forbidden_intervals.get(bs, []):
                    if not (window_end <= f_s or t >= f_e):
                        overlap = True
                        break
                if overlap: break
                
            if not overlap:
                placed = True
                
                if job.job_type == "traveling":
                    legs = []
                    curr_t = t + (job.setup_minutes if job.setup_minutes is not None else 0)
                    for leg in (job.route_legs or []):
                        legs.append({
                            "block_section_id": leg.block_section_id,
                            "start_min": curr_t,
                            "end_min": curr_t + leg.duration_minutes
                        })
                        curr_t += leg.duration_minutes
                        
                    assigned_jobs.append({
                        "work_id": job.work_id,
                        "selected": True,
                        "job_type": "traveling",
                        "legs": legs,
                        "priority_score": job.priority_score,
                        "mandatory": job.mandatory,
                        "assigned_start": t,
                        "assigned_end": window_end
                    })
                else:
                    assigned_jobs.append({
                        "work_id": job.work_id,
                        "selected": True,
                        "job_type": "single",
                        "assigned_start": t,
                        "assigned_end": window_end,
                        "block_sections_used": blocks_req,
                        "priority_score": job.priority_score,
                        "mandatory": job.mandatory
                    })
                    
                for bs in blocks_req:
                    forbidden_intervals[bs].append((t, window_end))
                break
                
        if not placed:
            deferred_jobs.append({
                "work_id": job.work_id,
                "selected": False,
                "reason_code": "NO_FEASIBLE_WINDOW",
                "reason_detail": "No valid window found in greedy search",
                "priority_score": job.priority_score,
                "mandatory": job.mandatory
            })
            
    return {
        "assigned": assigned_jobs,
        "deferred": deferred_jobs
    }
