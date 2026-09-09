from collections import defaultdict
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

from app.schemas.contracts import (
    ScheduleRequest, MacroAllocationCalendar, DayAllocation, UnassignableWork, ReasonCode
)
from app.solver.greedy_heuristic import parse_time_to_minutes

URGENCY_BOOST = 40  # tunable: how hard SLA proximity can override raw priority

class MacroItem:
    def __init__(self, job, start_dt: datetime):
        self.id = job.work_id
        self.priority_score = job.priority_score
        self.mandatory = job.mandatory
        self.crew_type = job.crew_type
        self.crew_hours = (job.crew_size or 1) if self.crew_type else 0
        self.plant_type = job.plant_type
        self.plant_hours = (job.plant_qty or 1) if self.plant_type else 0
        
        # Calculate start/end days
        s_min = parse_time_to_minutes(job.earliest_start, start_dt)
        e_max = parse_time_to_minutes(job.latest_start, start_dt)
        self.release_day = s_min // 1440
        self.deadline_day = e_max // 1440
        
        # Calculate rough duration and blocks
        if job.job_type == "traveling":
            self.corridor_id = job.route_legs[0].block_section_id if job.route_legs else "UNKNOWN"
            dur_mins = sum(leg.duration_minutes for leg in (job.route_legs or [])) + (job.setup_minutes if job.setup_minutes is not None else 0)
        else:
            self.corridor_id = job.block_sections_required[0] if job.block_sections_required else "UNKNOWN"
            dur_mins = job.duration_minutes if job.duration_minutes is not None else 60
            
        self.rough_duration_hours = max(1, dur_mins // 60) # round up roughly or min 1 hr

def macro_bucket_plan(request: ScheduleRequest) -> MacroAllocationCalendar:
    start_dt = datetime.fromisoformat(request.planning_horizon.start_datetime)
    horizon_days = max(1, request.planning_horizon.horizon_minutes // 1440)
    
    # 1. Parse jobs into MacroItem
    backlog = []
    for job in request.maintenance_requests:
        backlog.append(MacroItem(job, start_dt))
        
    # 2. Build Daily Capacity
    # Default to empty if not provided, assuming no capacity constraints if omitted (though in practice it should be passed)
    daily_cap = {}
    for d in range(horizon_days):
        daily_cap[d] = defaultdict(int)
        
    if request.daily_capacity_calendar:
        for cal in request.daily_capacity_calendar:
            cal_dt = datetime.fromisoformat(cal.date)
            # Find which relative day this maps to
            delta_days = (cal_dt.date() - start_dt.date()).days
            if 0 <= delta_days < horizon_days:
                for k, v in cal.crew_hours.items():
                    daily_cap[delta_days][("crew", k)] += v
                for k, v in cal.plant_hours.items():
                    daily_cap[delta_days][("plant", k)] += v
                for k, v in cal.corridor_possession_hours.items():
                    daily_cap[delta_days][("corridor", k)] += v

    # 3. Urgency Function
    # Assume "today" is day 0 for this run
    today = 0
    def urgency(item: MacroItem):
        days_left = max(item.deadline_day - today, 1)
        boost = URGENCY_BOOST * max(0, 1 - days_left / 14)
        return (item.mandatory, item.priority_score + boost)
        
    # 4. FFD Assignment
    calendar_assignments = defaultdict(list)
    unassignable = []
    
    def fits(item: MacroItem, cap: dict) -> bool:
        if item.crew_type and item.crew_hours > cap.get(("crew", item.crew_type), 0):
            return False
        if item.plant_type and item.plant_hours > cap.get(("plant", item.plant_type), 0):
            return False
        if item.corridor_id and item.rough_duration_hours > cap.get(("corridor", item.corridor_id), 0):
            return False
        return True
        
    def deduct(item: MacroItem, cap: dict):
        if item.crew_type:
            cap[("crew", item.crew_type)] -= item.crew_hours
        if item.plant_type:
            cap[("plant", item.plant_type)] -= item.plant_hours
        if item.corridor_id:
            cap[("corridor", item.corridor_id)] -= item.rough_duration_hours

    for item in sorted(backlog, key=urgency, reverse=True):
        placed = False
        start_day = max(item.release_day - today, 0)
        end_day = min(item.deadline_day - today, horizon_days - 1)
        
        for day in range(start_day, end_day + 1):
            if fits(item, daily_cap[day]):
                deduct(item, daily_cap[day])
                calendar_assignments[day].append(item.id)
                placed = True
                break
                
        if not placed:
            unassignable.append(UnassignableWork(
                work_id=item.id,
                reason_code=ReasonCode.NO_CAPACITY_BEFORE_DEADLINE,
                mandatory=item.mandatory
            ))
            
    # 5. Format Output
    days_out = []
    for d in range(horizon_days):
        day_date = (start_dt + timedelta(days=d)).date().isoformat()
        if d in calendar_assignments:
            days_out.append(DayAllocation(date=day_date, assigned_work_ids=calendar_assignments[d]))
            
    return MacroAllocationCalendar(
        generated_at=datetime.now().isoformat(),
        horizon_days=horizon_days,
        days=days_out,
        unassignable_before_deadline=unassignable
    )
