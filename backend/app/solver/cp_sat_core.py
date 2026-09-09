import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any

from ortools.sat.python import cp_model
from app.schemas.contracts import (
    ScheduleRequest, ScheduleResponse, SolverStatus,
    SelectedWorkPackage, TravelingJobSchedule, TravelingJobLegSchedule,
    DeferredWorkPackage, GanttTask, KPIs
)
from app.solver.greedy_heuristic import greedy_heuristic, parse_time_to_minutes
from app.engine.reason_codes import determine_reason_code

def minutes_to_datetime(minutes: int, start_dt: datetime) -> str:
    return (start_dt + timedelta(minutes=minutes)).isoformat()

def solve_plan(request: ScheduleRequest) -> ScheduleResponse:
    start_time_ms = int(time.time() * 1000)
    
    start_dt = datetime.fromisoformat(request.planning_horizon.start_datetime)
    HORIZON = request.planning_horizon.horizon_minutes
    
    model = cp_model.CpModel()
    
    train_iv = {}
    by_block = defaultdict(list)
    delay_terms = []
    
    for t in request.train_movements:
        s = parse_time_to_minutes(t.scheduled_entry, start_dt)
        e = parse_time_to_minutes(t.scheduled_exit, start_dt)
        dur = max(0, e - s)
        padded_dur = dur + t.headway_before_minutes + t.headway_after_minutes
        s_padded = s - t.headway_before_minutes
        
        if t.flexibility == "fixed" or t.max_shift_minutes <= 0:
            iv = model.NewIntervalVar(s_padded, padded_dur, s_padded + padded_dur, f"trn_{t.train_id}")
        else:
            shift = model.NewIntVar(0, t.max_shift_minutes, f"shift_{t.train_id}")
            shift_s = model.NewIntVar(s_padded, s_padded + t.max_shift_minutes, f"s_{t.train_id}")
            model.Add(shift_s == s_padded + shift)
            iv = model.NewIntervalVar(shift_s, padded_dur, shift_s + padded_dur, f"trn_{t.train_id}")
            delay_terms.append(shift)
            
        train_iv[t.train_id] = iv
        by_block[t.block_section_id].append(iv)

    clusters = defaultdict(list)
    for w in request.maintenance_requests:
        c_id = w.ml_enrichment.cluster_id if w.ml_enrichment and w.ml_enrichment.cluster_id else w.work_id
        clusters[c_id].append(w)
        
    x = {} 
    start = {}
    end = {}
    iv = {}
    priority = {}
    drop_penalty = {}
    
    crew_load = defaultdict(list)
    plant_load = defaultdict(list)
    
    y_b = {}
    for c_id, jobs in clusters.items():
        if any(j.job_type == "traveling" for j in jobs):
            continue
            
        block_earliest = min(parse_time_to_minutes(j.earliest_start, start_dt) for j in jobs)
        block_latest = max(parse_time_to_minutes(j.latest_start, start_dt) for j in jobs)
        setup_mins = max((j.setup_minutes if j.setup_minutes is not None else 20) for j in jobs)
        total_dur = sum((j.duration_minutes if j.duration_minutes is not None else 60) for j in jobs) + setup_mins
        
        y_b[c_id] = model.NewBoolVar(f"block_{c_id}")
        S_b = model.NewIntVar(block_earliest, max(block_earliest, block_latest - total_dur), f"bstart_{c_id}")
        E_b = model.NewIntVar(0, HORIZON * 2, f"bend_{c_id}")
        model.Add(E_b == S_b + total_dur)
        
        for job in jobs:
            dur = job.duration_minutes if job.duration_minutes is not None else 60
            x[job.work_id] = y_b[c_id]
            priority[job.work_id] = int(job.priority_score)
            
            s = model.NewIntVar(block_earliest, block_latest, f"start_{job.work_id}")
            e = model.NewIntVar(0, HORIZON * 2, f"end_{job.work_id}")
            v = model.NewOptionalIntervalVar(s, dur, e, y_b[c_id], f"iv_{job.work_id}")
            
            start[job.work_id] = s
            end[job.work_id] = e
            iv[job.work_id] = v
            
            model.Add(s >= S_b + setup_mins).OnlyEnforceIf(y_b[c_id])
            model.Add(e <= E_b).OnlyEnforceIf(y_b[c_id])
            
            for bs in (job.block_sections_required or []):
                by_block[bs].append(v)
            if job.crew_type:
                crew_load[job.crew_type].append((v, job.crew_size or 1))
            if job.plant_type:
                plant_load[job.plant_type].append((v, job.plant_qty or 1))
                
        if len(jobs) > 1:
            model.AddNoOverlap([iv[job.work_id] for job in jobs])

    traveling_jobs = [j for j in request.maintenance_requests if j.job_type == "traveling"]
    for job in traveling_jobs:
        presence = model.NewBoolVar(f"present_{job.work_id}")
        x[job.work_id] = presence
        priority[job.work_id] = int(job.priority_score)
        
        s_min = parse_time_to_minutes(job.earliest_start, start_dt)
        e_max = parse_time_to_minutes(job.latest_start, start_dt)
        
        prev_end = None
        job_setup = job.setup_minutes if job.setup_minutes is not None else 20
        max_wait = job.max_wait_minutes if job.max_wait_minutes is not None else 20
        
        legs_vars = []
        for k, leg in enumerate(job.route_legs or []):
            dur = leg.duration_minutes + (job_setup if k == 0 else 0)
            s = model.NewIntVar(s_min, e_max, f"start_{job.work_id}_{k}")
            e = model.NewIntVar(s_min, e_max + dur, f"end_{job.work_id}_{k}")
            v = model.NewOptionalIntervalVar(s, dur, e, presence, f"iv_{job.work_id}_{k}")
            
            legs_vars.append((leg, s, e, v))
            by_block[leg.block_section_id].append(v)
            
            if job.crew_type:
                crew_load[job.crew_type].append((v, job.crew_size or 1))
            if job.plant_type:
                plant_load[job.plant_type].append((v, job.plant_qty or 1))
                
            if prev_end is not None:
                model.Add(s >= prev_end).OnlyEnforceIf(presence)
                model.Add(s - prev_end <= max_wait).OnlyEnforceIf(presence)
            else:
                start[job.work_id] = s
            prev_end = e
            
        end[job.work_id] = prev_end
        iv[job.work_id] = legs_vars

    all_jobs = request.maintenance_requests
    for j in all_jobs:
        if j.mandatory:
            d_pen = model.NewBoolVar(f"dropped_{j.work_id}")
            model.Add(d_pen == 1 - x[j.work_id])
            drop_penalty[j.work_id] = d_pen

    for b, ivs in by_block.items():
        if len(ivs) > 1:
            model.AddNoOverlap(ivs)

    crew_caps = {}
    for pool in request.resource_capacity.crew_pools:
        cap = max(s.capacity for s in pool.shifts) if pool.shifts else 0
        crew_caps[pool.crew_type] = cap
        
    plant_caps = {}
    for pool in request.resource_capacity.plant_pools:
        plant_caps[pool.plant_type] = pool.total_qty
        
    for k, pairs in crew_load.items():
        if k in crew_caps:
            ivs_c, demands_c = zip(*pairs)
            model.AddCumulative(list(ivs_c), list(demands_c), crew_caps[k])
            
    for m, pairs in plant_load.items():
        if m in plant_caps:
            ivs_p, demands_p = zip(*pairs)
            model.AddCumulative(list(ivs_p), list(demands_p), plant_caps[m])

    # Warm Start
    greedy_res = greedy_heuristic(request)
    greedy_assigned = {j["work_id"]: j for j in greedy_res["assigned"]}
    
    # Hint block vars y_b instead of x directly for block jobs
    for c_id, jobs in clusters.items():
        if any(j.job_type == "traveling" for j in jobs):
            continue
        # Hint block if all jobs in block are assigned in greedy
        all_assigned = all(j.work_id in greedy_assigned for j in jobs)
        model.AddHint(y_b[c_id], 1 if all_assigned else 0)
        
    for job in traveling_jobs:
        model.AddHint(x[job.work_id], 1 if job.work_id in greedy_assigned else 0)

    # Lexicographic Solve: Pass 1 (Maximize Priority)
    LAMBDA_SAFETY = 1_000_000
    priority_expr = sum(priority[jid] * x[jid] for jid in x)
    safety_expr = sum(LAMBDA_SAFETY * drop_penalty[jid] for jid in drop_penalty)
    
    model.Maximize(priority_expr - safety_expr)
    
    solver = cp_model.CpSolver()
    max_time = request.solver_config.max_time_in_seconds if request.solver_config else 8.0
    solver.parameters.max_time_in_seconds = max_time * (5.0 / 8.0) # 5s for pass 1
    solver.parameters.num_search_workers = request.solver_config.num_search_workers if request.solver_config else 8
    solver.parameters.random_seed = request.solver_config.random_seed if request.solver_config else 42
    solver.parameters.max_number_of_conflicts = request.solver_config.max_number_of_conflicts if request.solver_config else 100000
    
    status = solver.Solve(model)
    
    def generate_fallback_response():
        # Build response from greedy_res
        sel_wps = []
        tr_schedules = []
        def_wps = []
        for a in greedy_res["assigned"]:
            if a.get("job_type") == "traveling":
                tr_schedules.append(TravelingJobSchedule(
                    work_id=a["work_id"],
                    selected=True,
                    legs=[TravelingJobLegSchedule(block_section_id=l["block_section_id"], start=minutes_to_datetime(l["start_min"], start_dt), end=minutes_to_datetime(l["end_min"], start_dt)) for l in a["legs"]],
                    priority_score=a["priority_score"]
                ))
            else:
                sel_wps.append(SelectedWorkPackage(
                    work_id=a["work_id"],
                    selected=True,
                    assigned_start=minutes_to_datetime(a["assigned_start"], start_dt),
                    assigned_end=minutes_to_datetime(a["assigned_end"], start_dt),
                    block_sections_used=a["block_sections_used"],
                    priority_score=a["priority_score"]
                ))
                
        for d in greedy_res["deferred"]:
            def_wps.append(DeferredWorkPackage(
                work_id=d["work_id"],
                selected=False,
                reason_code=d["reason_code"],
                reason_detail=d["reason_detail"],
                priority_score=d["priority_score"]
            ))
            
        return ScheduleResponse(
            schedule_id="PLAN-FALLBACK",
            generated_at=datetime.now().isoformat(),
            solver_status=SolverStatus.FALLBACK_HEURISTIC,
            solve_time_ms=int(time.time() * 1000) - start_time_ms,
            selected_work_packages=sel_wps,
            traveling_job_schedules=tr_schedules,
            deferred_work_packages=def_wps,
            gantt_tasks=[],
            conflicts=[],
            kpis=KPIs(train_delay_minutes_total=0, work_packages_included_pct=0, unused_block_minutes=0, mandatory_items_dropped=sum(1 for d in def_wps if d.mandatory))
        )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return generate_fallback_response()

    # Pass 2 (Minimize Delay)
    best_priority_val = solver.Value(priority_expr)
    best_safety_val = solver.Value(safety_expr)
    
    # Restrict objective degradation (allow 2% drop in priority, no drop in safety)
    min_priority = int(best_priority_val * 0.98)
    model.Add(priority_expr >= min_priority)
    model.Add(safety_expr <= best_safety_val)
    
    model.Minimize(sum(delay_terms))
    
    solver.parameters.max_time_in_seconds = max_time * (3.0 / 8.0) # 3s for pass 2
    status2 = solver.Solve(model)
    
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fall back to pass 1 result or greedy if weirdly infeasible
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Pass 1 was fine, we just didn't solve Pass 2 in time or something
            # Note: CP-SAT doesn't lose the first solution, we can use it
            # But the Python API resets. Let's just use generate_fallback_response for simplicity if Pass 2 crashes
            return generate_fallback_response()

    # 7. Construct Response
    sel_wps = []
    tr_schedules = []
    def_wps = []
    
    # Pass 1: Collect selected jobs
    deferred_jobs = []
    for j in all_jobs:
        selected = solver.Value(x[j.work_id]) == 1
        if selected:
            if j.job_type == "traveling":
                legs_out = []
                for (leg_req, s_var, e_var, v_var) in iv[j.work_id]:
                    legs_out.append(TravelingJobLegSchedule(
                        block_section_id=leg_req.block_section_id,
                        start=minutes_to_datetime(solver.Value(s_var), start_dt),
                        end=minutes_to_datetime(solver.Value(e_var), start_dt)
                    ))
                tr_schedules.append(TravelingJobSchedule(
                    work_id=j.work_id,
                    selected=True,
                    legs=legs_out,
                    crew_assigned={"crew_type": j.crew_type, "crew_size": j.crew_size} if j.crew_type else None,
                    plant_assigned={"plant_type": j.plant_type, "plant_qty": j.plant_qty} if j.plant_type else None,
                    priority_score=j.priority_score
                ))
            else:
                sel_wps.append(SelectedWorkPackage(
                    work_id=j.work_id,
                    selected=True,
                    assigned_start=minutes_to_datetime(solver.Value(start[j.work_id]), start_dt),
                    assigned_end=minutes_to_datetime(solver.Value(end[j.work_id]), start_dt),
                    block_sections_used=j.block_sections_required or [],
                    crew_assigned={"crew_type": j.crew_type, "crew_size": j.crew_size} if j.crew_type else None,
                    plant_assigned={"plant_type": j.plant_type, "plant_qty": j.plant_qty} if j.plant_type else None,
                    priority_score=j.priority_score
                ))
        else:
            deferred_jobs.append(j)
            
    # Pass 2: Determine reason codes for deferred jobs
    for j in deferred_jobs:
        code, detail = determine_reason_code(j, request, sel_wps, tr_schedules, start_dt)
        def_wps.append(DeferredWorkPackage(
            work_id=j.work_id,
            selected=False,
            reason_code=code,
            reason_detail=detail,
            priority_score=j.priority_score
        ))
            
    # Compile KPIs
    total_delay = sum(solver.Value(term) for term in delay_terms)
    mand_dropped = sum(1 for j in all_jobs if j.mandatory and solver.Value(x[j.work_id]) == 0)
    incl_pct = (len(sel_wps) + len(tr_schedules)) / max(len(all_jobs), 1) * 100
    
    response = ScheduleResponse(
        schedule_id="PLAN-OPT",
        generated_at=datetime.now().isoformat(),
        solver_status=SolverStatus.OPTIMAL if status2 == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
        objective_value=float(solver.Value(priority_expr)),
        solve_time_ms=int(time.time() * 1000) - start_time_ms,
        selected_work_packages=sel_wps,
        traveling_job_schedules=tr_schedules,
        deferred_work_packages=def_wps,
        gantt_tasks=[],
        conflicts=[],
        kpis=KPIs(
            train_delay_minutes_total=int(total_delay),
            work_packages_included_pct=round(incl_pct, 1),
            unused_block_minutes=0,
            mandatory_items_dropped=mand_dropped
        )
    )
    
    return response
