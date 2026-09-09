# SIH26027 — Integrated Block Bundling Engine
### CP-SAT Formulation · Data Contract · Red-Team Review

---

## Executive Summary

SIH26027 is not a machine-learning scheduling system. It is a **constraint-satisfaction execution engine that consumes machine-learning predictions** — a distinction that is the single most important architectural decision in this document, and the one most naive submissions to this problem statement get backwards.

The obvious failure mode for a rail block-planning system is to train an end-to-end ML model — a sequence model, a reinforcement-learning agent, an LLM prompted to "generate a schedule" — and ask it to output start times directly. That approach cannot survive contact with a signaling engineer, because nothing about it can *prove* that two trains never occupy the same block section at the same instant. A neural network can be trained to be right 99.9% of the time; a railway cannot operate on a 99.9% collision-avoidance guarantee. This architecture is built on the opposite premise, expressed in four pillars.

**1. Absolute Mathematical Safety — ML predicts, CP-SAT decides.**
Every number a machine-learning model produces in this system — an escalation-risk probability, a predicted task duration, a cluster assignment — is consumed strictly as an *input coefficient* into a symbolic constraint-satisfaction model (Google OR-Tools CP-SAT), never as a scheduling decision in its own right. Track exclusivity is enforced by `AddNoOverlap` over exact-integer interval variables — a combinatorial guarantee, checked by a SAT-style solver, not a statistical one. No ML model in this stack ever touches a `start_i` or `end_i` value directly, and none sits in the causal path of a go/no-go occupancy decision. Delete every prediction in Part 5 and replace it with the static defaults from the original spec, and the system degrades in *quality* — worse prioritization, worse duration estimates — never in *safety*. That property is the whole point of the design.

**2. Graceful Degradation — the solver is structurally incapable of returning nothing.**
A hard `x_i == 1` constraint on every mandatory safety job is a live landmine: the instant two mandatory jobs cannot physically coexist, the entire model returns `INFEASIBLE` — no plan, no partial result, and to a judge or a dispatcher, "the system crashed" (Part 3.4). This spec converts every mandatory constraint into a **soft constraint carrying a massive drop penalty** — orders of magnitude above the normal 0–100 priority scale — so the solver will move every other lever before it drops a safety-critical job, but will never refuse to produce *a* schedule outright. Paired with a millisecond greedy fallback served verbatim if the solver ever returns `INFEASIBLE` or `UNKNOWN`, the demo — and the production system — never shows an empty screen.

**3. Explainable AI — silence is a bug, not a feature.**
An optimizer that drops a defect from the plan without saying why is functionally indistinguishable, to the person relying on it, from an optimizer that lost the defect. This system treats a `reason_code` on every deferred work package as a first-class output, not a nice-to-have: `NO_FEASIBLE_WINDOW`, `CAPACITY_EXCEEDED`, `LOWER_PRIORITY` are computed against the already-solved model with three cheap, deterministic checks (Part 3.6) — no black-box re-solve, no shrug. The same discipline extends into the ML layer: every escalation-risk and duration prediction in Part 5 carries its contributing factors and a confidence label, so "why did the model call this defect high-risk" has a real, auditable answer instead of a re-run of a black box.

**4. Macro/Micro Decoupling — the right tool at the right time horizon.**
Requirement #4 asks for weekly and monthly block plans; Requirement #3 asks for exact, safety-critical minute-level scheduling. These are not the same problem at different resolutions — they are different problems with different stakes, and this architecture refuses to conflate them. The **micro layer** (Parts 1–3) is exact-minute CP-SAT, trustworthy only over the near-term window where the real train timetable is actually known. The **macro layer** (Part 4) is a greedy, day-granularity bucket-assigner that captures the large majority of an optimizer's value on a 30-day backlog in milliseconds, with none of the combinatorial risk of running an exact solver over a horizon it cannot honestly reason about. CP-SAT is reserved for what it is actually built for — 24-hour, minute-level, safety-exclusive execution — and nothing upstream of it is ever asked to pretend to a precision it doesn't have.

Everything that follows — the CP-SAT formulation (Part 1), the data contract (Part 2), the red-team stress test (Part 3), the macro bucket-planning layer (Part 4), the ML prediction pipeline (Part 5), and the full engineering tech stack (Part 6) — is written in service of these four pillars, not the other way around.

---

### Requirement coverage

| Problem statement requirement | Where it's solved |
|---|---|
| 1. Integrate TMS/SMMS/TDMS defect data with COA corridor/timetable and goods-train forecast | Part 2 — data contract (`maintenance_requests`, `train_movements`, `block_sections`) |
| 2. AI/ML prioritization by criticality, urgency, impact | Part 1.2 (SRCAO formula) + Part 5 (trained escalation-risk model for the `R_i` term) |
| 3. Optimize scheduling, maximize uptime, coordinate multi-department activities | Part 1 (CP-SAT: `NoOverlap`, `Cumulative`, block bundling) + Part 3 (scale, robustness, demo safety) |
| 4. Block plans over weekly and monthly horizons | Part 4 — macro bucket-assignment layer, feeding the Part 1 micro solver on a rolling basis |

---

## PART 1 — THE MATH

### 1.1 Problem framing

This is **not** a calendar problem. It's a resource-constrained scheduling problem with one exclusive-resource layer (track) and one shared-capacity layer (crews/plant), plus an inclusion decision (which defects get bundled in). Concretely, it's a **job-shop / RCPSP hybrid with optional jobs**:

- **Machines (exclusive resources):** block sections. At most one occupant (a train *or* a possession) at any instant → `NoOverlap`.
- **Renewable resources (shared capacity):** crew pools, plant/machines. Multiple jobs can run concurrently up to a cap → `Cumulative`.
- **Jobs:** train movements (mostly fixed) and maintenance work packages (optional — the solver chooses whether to include them, `x_i`).
- **The "bundling" part** isn't a separate constraint type — it's a *reward* in the objective for co-locating compatible work in the same possession, not a new constraint.

**One more thing this framing needs, up front:** everything in Part 1 is a *micro* layer — exact, minute-level, safety-critical, and only trustworthy against a train timetable you actually know precisely, which in practice means "the next day or two." Requirement #4 of the problem statement (weekly/monthly plans) needs a separate *macro* layer sitting above this one, at day-level granularity, deciding *which day* work happens rather than *which minute* — see Part 4. The two are not the same model at different resolutions; they're different problems with different stakes, and conflating them is how you'd end up trying to run this exact CP-SAT formulation over a 30-day horizon and watch it fall over.

### 1.2 Sets & parameters

| Symbol | Meaning |
|---|---|
| `B` | block sections |
| `T` | train movements, each tied to one `b(t) ∈ B` |
| `W` | maintenance work packages (from SMMS) |
| `K` | crew pools (P-Way gang, S&T team, electrical isolation team, …) |
| `M` | plant/machine pools (tamper, crane, ballast train, …) |
| `P_i` | priority score for work package `i` (your 0.35S+0.25R+0.20C+0.10A+0.10O formula) |
| `dur_i` | duration of work package `i`, minutes |
| `req_i` | set of block sections work package `i` needs simultaneously |
| `iso(i)` | isolation/setup group of `i` (used for bundling reward) |

### 1.3 Decision variables

```
x_i        ∈ {0,1}                for i in W          # is this work package selected?
start_i    ∈ [earliest_i, latest_i - dur_i]            # only meaningful if x_i = 1
end_i      = start_i + dur_i
iv_i       = OptionalIntervalVar(start_i, dur_i, end_i, presence=x_i)

shift_t    ∈ [0, max_shift_t]     for t in T if t is "semi-flexible"   # minutes of allowed slip
train_iv_t = IntervalVar(sched_start_t + shift_t, dur_t, sched_end_t + shift_t)  # mandatory, not optional
```

Fixed trains get a plain `IntervalVar` with constant bounds — they are **not** decision variables, they're forbidden zones. Only semi-flexible trains and maintenance work get real freedom. This distinction matters a lot for Part 3.

### 1.4 Constraints

**(a) Track exclusivity — `NoOverlap` per block section**
For each `b ∈ B`, collect every interval (train or maintenance) that touches `b`, and forbid overlap:

```
for b in block_sections:
    ivs = [train_iv[t] for t in trains if t.block_section == b] \
        + [iv[i] for i in work_packages if b in i.req]
    model.AddNoOverlap(ivs)
```

Pad each interval's duration by the required headway/clearance buffer *before* building it — CP-SAT's `NoOverlap` has no native "gap" concept, so the gap has to be baked into `dur_i`.

**Careful how you pad, though.** If you pad *both* neighbors — train A's `headway_after` and train B's `headway_before` — the enforced gap between them is the *sum* of the two, not the max. That's the safe direction to be wrong in (you'll never under-space two occupants), but it's more conservative than real operating rules usually require, and it eats directly into how much maintenance time the model thinks a block section has. Confirm with your signaling rules whether headway is additive or "whichever is larger governs" — if it's the latter, pad only one side of each pair, not both.

**(b) Crew / plant capacity — `Cumulative` per pool**

```
for k in crew_pools:
    ivs, demands = [], []
    for i in work_packages:
        if i.crew_type == k:
            ivs.append(iv[i]); demands.append(i.crew_size)
    model.AddCumulative(ivs, demands, capacity[k])
```

Same pattern for plant pools. **Caveat:** `AddCumulative`'s capacity argument is one constant (or one `IntVar`) for the *entire* horizon — it can't step up/down at shift boundaries on its own. If crew capacity is 8 on the night shift and 0 during the day, either (i) run separate `Cumulative` constraints per shift window, or (ii) add a synthetic "phantom" interval that occupies the off-shift hours and consumes the full pool capacity, forcing everything else to zero during that stretch.

**Second caveat, and this one will actually bite you:** a night shift like `22:00–06:00` crosses midnight, but the horizon in Part 2.1 is `1440` minutes — one calendar day, minute 0 to minute 1440. Converted naively, "22:00" → 1320 and "06:00" → 360, and 360 < 1320, which is nonsense for an interval. Your phantom-interval and shift-capacity math will silently misbehave right at the boundary that matters most — the night block is *the* possession window. Fix it at the ETL layer, not the solver: when converting a shift's end time to absolute minutes, if `end < start`, add 1440 before subtracting ("06:00 next day" → 360 + 1440 = 1800). And size `planning_horizon.horizon_minutes` generously enough to actually contain the shift you're scheduling into — e.g. anchor the horizon at 18:00 the day before and run 36 hours (2160 minutes), rather than assuming a bare single calendar day covers a possession that starts near its end. If a crew pool has more than one shift with gaps between them, compute "off-shift" as the horizon minus the *union* of all listed shift windows, not just "before the one shift starts."

**(c) Precedence (inspection before renewal, etc.)** — only binding if both jobs are actually selected:

```
model.Add(end[a] <= start[b]).OnlyEnforceIf([x[a], x[b]])
```

**(d) Mandatory safety work** — see Part 3.4 before you write this the naive way:

```
model.Add(x[i] == 1)   # DON'T do this unconditionally — read the red team section
```

**(e) Bundling reward (reification, not a hard constraint)**
For candidate pairs `(i, j)` that are geographically adjacent *and* share an isolation group (pre-filter this list with NetworkX — do not generate all `O(n²)` pairs):

```
b_ij = model.NewBoolVar(f"bundle_{i}_{j}")
model.AddBoolAnd([x[i], x[j]]).OnlyEnforceIf(b_ij)
model.AddBoolOr([x[i].Not(), x[j].Not()]).OnlyEnforceIf(b_ij.Not())
# b_ij == 1 iff both i and j are selected → add bundle_value_ij * b_ij to the objective
```

*Gap worth naming if you keep this path:* unlike the block-based approach below, nothing here gives an individually-selected job a setup/isolation time — `dur_i` is just the work duration. If you use this pairwise path instead of the block default, bake each job's setup time into its own `dur_i` at the ETL layer, or the solver will happily schedule the crew to start cutting rail the instant they arrive on site.

**(e′) Revised recommendation: precomputed candidate blocks (MVP default)**

The pairwise reification above is more flexible but it's also more search space for no real MVP payoff. A cleaner formulation: run the defect-clustering/priority engine *before* CP-SAT ever sees the problem, and have it emit a set of candidate possession blocks `b ∈ B` — each one a pre-grouped bundle of jobs on one block section, with its own setup overhead. Make the block itself the optional interval, and tie job selection to block selection with a plain equality instead of a reified `AND`:

```
y_b   = model.NewBoolVar(f"block_{b.id}")
S_b   = model.NewIntVar(b.earliest_start, b.latest_start - b.total_duration, f"bstart_{b.id}")
E_b   = S_b + b.total_duration
biv   = model.NewOptionalIntervalVar(S_b, b.total_duration, E_b, y_b, f"biv_{b.id}")

for job_id in b.job_ids:
    model.Add(x[job_id] == y_b)                                    # block selected ⇒ every job in it is selected
    model.Add(start[job_id] >= S_b + b.setup_minutes).OnlyEnforceIf(y_b)
    model.Add(end[job_id]   <= E_b).OnlyEnforceIf(y_b)

model.AddNoOverlap([iv[job_id] for job_id in b.job_ids])   # enforces the "sequential execution" claimed in 1.1/1.2
```

*Don't skip that last line.* 1.1 and 1.2 both describe execution inside a block as sequential — that's how `D_b = σ_b + Σd_i` was sized — but nothing above actually stops two jobs in the same block from being scheduled at the same time if crew/plant capacity happens to allow it; the containment constraints only bound each job *within* the block window, not against each other. Add the `NoOverlap` above, or, if parallel execution inside one possession genuinely is fine operationally (two independent crews sharing one isolation), say so explicitly and drop "sequential" from 1.1/1.2 instead. Pick one — right now the prose promises something the constraints didn't deliver.

This turns "which subset of 150 jobs should be bundled" — the actual NP-hard part — into "which of ~50–100 precomputed blocks should be selected," a much smaller and better-behaved search problem, at the cost of losing the solver's ability to invent a bundling combination the clustering step didn't think of. That's the right trade for a hackathon MVP: the clustering engine already exists in the architecture, so let it do the combinatorial work outside the solver, and let CP-SAT do what it's actually good at — arbitrating scarce track/crew/plant time between blocks that are already formed. **Use this as the default; treat the pairwise version above as a post-MVP upgrade** if you later want the solver to discover bundlings the clusterer missed.

Middle ground, if you want some of that flexibility back cheaply: generate 2–3 *overlapping* candidate blocks per cluster (e.g. "just the two most severe defects" vs. "all four defects in this cluster") as mutually-exclusive alternatives, and let the solver pick between them. Cheap to add, and it recovers some lost flexibility without reopening the full pairwise search.

### 1.5 Objective

```
maximize  Σ P_i · x_i
        − λ1 · Σ shift_t              (train disruption, minutes)
        − λ2 · Σ overtime_k           (crew overtime beyond shift capacity)
        + λ3 · Σ bundle_value_ij · b_ij   (reward for shared-isolation bundling)
```

### 1.6 Full pseudocode skeleton

```python
from ortools.sat.python import cp_model
from collections import defaultdict

model = cp_model.CpModel()
HORIZON = 1440  # minutes in the planning day

# ---- 1. Fixed / semi-flexible train intervals ----
train_iv = {}
by_block = defaultdict(list)
for t in trains:
    if t.flexibility == "fixed":
        iv = model.NewIntervalVar(t.start_min, t.padded_dur, t.start_min + t.padded_dur, f"trn_{t.id}")
    else:
        shift = model.NewIntVar(0, t.max_shift_min, f"shift_{t.id}")
        s = model.NewIntVar(t.start_min, t.start_min + t.max_shift_min, f"s_{t.id}")
        model.Add(s == t.start_min + shift)
        iv = model.NewIntervalVar(s, t.padded_dur, s + t.padded_dur, f"trn_{t.id}")
        delay_terms.append(shift)
    train_iv[t.id] = iv
    by_block[t.block_section_id].append(iv)

# ---- 2. Optional maintenance intervals (single/simultaneous-section jobs) ----
x, start, end, iv, priority = {}, {}, {}, {}, {}
crew_load, plant_load = defaultdict(list), defaultdict(list)

for w in work_packages:
    x[w.id] = model.NewBoolVar(f"sel_{w.id}")
    s = model.NewIntVar(w.earliest_min, w.latest_min - w.dur_min, f"start_{w.id}")
    e = model.NewIntVar(0, HORIZON, f"end_{w.id}")
    v = model.NewOptionalIntervalVar(s, w.dur_min, e, x[w.id], f"iv_{w.id}")
    start[w.id], end[w.id], iv[w.id], priority[w.id] = s, e, v, w.priority_score

    for b in w.block_sections_required:
        by_block[b].append(v)
    if w.crew_type:
        crew_load[w.crew_type].append((v, w.crew_size))
    if w.plant_type:
        plant_load[w.plant_type].append((v, w.plant_qty))

# ---- 2.5. Traveling jobs (multi-section; full derivation in Part 1.7) ----
# One shared presence literal per job, reused across every leg's interval — that's
# what makes atomicity automatic instead of something checked after the fact.
for job in traveling_jobs:
    presence = model.NewBoolVar(f"present_{job.id}")
    x[job.id] = presence
    priority[job.id] = job.priority_score
    prev_end = None
    for k, leg in enumerate(job.route_legs):
        dur = leg.duration_minutes + (job.setup_minutes if k == 0 else 0)
        s = model.NewIntVar(job.release_min, job.deadline_min, f"start_{job.id}_{k}")
        e = model.NewIntVar(job.release_min, job.deadline_min, f"end_{job.id}_{k}")
        v = model.NewOptionalIntervalVar(s, dur, e, presence, f"iv_{job.id}_{k}")

        by_block[leg.block_section_id].append(v)          # same NoOverlap pool as everything else
        if job.crew_type:
            crew_load[job.crew_type].append((v, job.crew_size))
        if job.plant_type:
            plant_load[job.plant_type].append((v, job.plant_qty))

        if prev_end is not None:
            model.Add(s >= prev_end).OnlyEnforceIf(presence)
            model.Add(s - prev_end <= job.max_wait_minutes).OnlyEnforceIf(presence)
        else:
            start[job.id] = s                              # job-level start = first leg (for precedence/objective)
        prev_end = e
    end[job.id] = prev_end                                 # job-level end = last leg

all_jobs = list(work_packages) + list(traveling_jobs)       # for anything below that isn't per-leg

# ---- 3. NoOverlap per block section ----
for b, ivs in by_block.items():
    model.AddNoOverlap(ivs)

# ---- 4. Cumulative per crew / plant pool ----
for k, pairs in crew_load.items():
    ivs, demands = zip(*pairs)
    model.AddCumulative(list(ivs), list(demands), crew_capacity[k])
for m, pairs in plant_load.items():
    ivs, demands = zip(*pairs)
    model.AddCumulative(list(ivs), list(demands), plant_capacity[m])

# ---- 5. Precedence ----
for a_id, b_id in precedence_pairs:
    model.Add(end[a_id] <= start[b_id]).OnlyEnforceIf([x[a_id], x[b_id]])

# ---- 6. Safety-critical work: SOFT, not hard (see Part 3.4) ----
# Iterates all_jobs, not just work_packages — a mandatory traveling job (an urgent
# multi-section renewal) needs the same protection as a mandatory single-section one.
drop_penalty = {}
for j in all_jobs:
    if j.mandatory:
        drop_penalty[j.id] = model.NewBoolVar(f"dropped_{j.id}")
        model.Add(drop_penalty[j.id] == 1 - x[j.id])

# ---- 7. Bundling reification (adjacency-filtered pairs only) ----
bundle_terms = []
for i, j, val in candidate_bundle_pairs:   # pre-filtered via NetworkX adjacency
    b_ij = model.NewBoolVar(f"bundle_{i}_{j}")
    model.AddBoolAnd([x[i], x[j]]).OnlyEnforceIf(b_ij)
    model.AddBoolOr([x[i].Not(), x[j].Not()]).OnlyEnforceIf(b_ij.Not())
    bundle_terms.append(val * b_ij)

# ---- 8. Objective ----
# Iterate over x directly, not work_packages — that's what actually includes
# traveling jobs' priority. Missing this is a silent bug, not a loud one: the
# model still solves fine, it just never has a reason to select any traveling job.
model.Maximize(
    sum(priority[jid] * x[jid] for jid in x)
    - LAMBDA1 * sum(delay_terms)
    - LAMBDA_SAFETY * sum(1_000_000 * drop_penalty[k] for k in drop_penalty)  # huge, not infinite
    + LAMBDA3 * sum(bundle_terms)
)

# ---- 9. Warm start (see Part 3.3) ----
for w_id, val in greedy_initial_solution.items():
    model.AddHint(x[w_id], val)

# ---- 10. Solve on a hard wall-clock budget ----
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 8.0
solver.parameters.num_search_workers = 8
solver.parameters.random_seed = 42
status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    # build response from solver.Value(x[w.id]), solver.Value(start[w.id]), etc.
    pass
else:
    # status == INFEASIBLE should now be structurally impossible, because
    # mandatory work is soft (step 6). If you still hit INFEASIBLE, the bug
    # is in your bounds, not your data.
    pass
```

### 1.7 Extension — multi-section traveling jobs

You confirmed this is real: tamping, ballast trains, OHE inspection cars — some jobs physically move through several block sections in one continuous run, not just one. The flat `req_i` / single `dur_i` model in 1.2–1.3 doesn't cover that. Modeled naively it either over-claims (one job locks every section it touches for its *entire* duration, blocking section 3 for two hours before the machine has even left section 1) or gets faked as several unrelated jobs and hoped into alignment — which is exactly the "shredding" failure mode worth avoiding: nothing then stops the solver scheduling leg 2 without leg 1, or out of physical order.

The right primitive is simpler than it sounds: give the *whole job* one shared presence literal, and reuse it across every leg's optional interval. Atomicity falls out for free — no `parent_work_id` post-hoc check needed, because `NewOptionalIntervalVar` with the same Boolean for every leg means all legs are present together or none are, by construction.

**Added parameters**

- A traveling job `j` has an ordered list of legs `(r_1, d_1) … (r_n, d_n)` — block section and duration per leg, in physical travel order. Durations come from your ETL layer (track length, machine speed, historical rates), not from CP-SAT.
- `wait_j`: max allowed idle time between one leg ending and the next starting — waiting for a train to clear the next section, a signal, a point to be set. Get the real ceiling from your signaling/ops mentors; don't ship a guess. 15–20 minutes is a reasonable placeholder to unblock development.
- Setup/isolation time `σ_j` applies once, to leg 1 only — once the crew and machine are working under an established block protection, moving into the next section along the same corridor doesn't repeat the full initial isolation.

**Added constraints** — one shared presence `x_j`, and for each leg `k`:

```
s_{j,k}   ∈ [release_j, deadline_j]
dur_{j,k} = d_{j,k} + (σ_j if k == 0 else 0)
iv_{j,k}  = OptionalIntervalVar(s_{j,k}, dur_{j,k}, e_{j,k}, presence=x_j)     # same x_j for every leg

# sequencing between consecutive legs, only binding if the job is selected at all:
s_{j,k+1} >= e_{j,k}                  )
s_{j,k+1} - e_{j,k} <= wait_j         )   both .OnlyEnforceIf(x_j)
```

Each `iv_{j,k}` drops into the *same* per-section `NoOverlap` list as every other train and maintenance interval — there's no separate "corridor job" category and nothing to reconcile between two competing constraint systems. One list per section; everything that can occupy it goes in that list, whether it's a single-section defect repair, a leg of a traveling job, or a train. `x_j` slots directly into the existing `x[...]`, `crew_load[...]`, `plant_load[...]`, mandatory-soft (3.4), and objective machinery — a traveling job isn't a new kind of thing to the rest of the model, just a job whose interval happens to be several intervals sharing one on/off switch.

The implementation lives in the master pseudocode now, not here: it's **step 2.5** in Part 1.6, folded in alongside ordinary work packages so it populates the same `x`, `start`, `end`, `crew_load`, `plant_load`, and `priority` structures. That's what fixed three real bugs a separate bolted-on loop actually had — see Part 3.7: it silently never got selected by the objective, silently skipped the mandatory-soft check, and would `KeyError` on any precedence reference to it. Keeping one copy of the code instead of two is deliberate — an appendix snippet and a master snippet quietly drifting apart is exactly how bugs like that survive a review.

Two things worth carrying over from the injected critique a few turns back, on their own engineering merits — not because of the fabricated backstory they arrived with:

- **Minimum possession floor.** If leg durations are computed from track length ÷ machine speed, clamp with `max(computed_time, MIN_POSSESSION_MINUTES)` at the ETL layer. Signaling paperwork and point-setting overhead make a sub-15-minute possession operationally meaningless even if the arithmetic says six.
- **First-leg-only setup padding.** Already folded into the pseudocode above (`σ_j` added only when `k == 0`) — same idea as the critique proposed, applied cleanly instead of as a patch bolted onto a separate corridor-job system.

**Warm-start note:** hint `x_j` once per job, not once per leg — every leg shares the same literal, so hinting them differently is a contradiction the solver just discards. Same lesson as the block-vs-`x_i` hinting note in 3.3(4).

**Scale note, since this adds variables:** a traveling job now costs one Boolean plus roughly `2 × legs` integer vars instead of one Boolean plus 2 — a 4-leg job is about as expensive as 4 ordinary work packages. If a meaningful share of your 150 maintenance requests are multi-leg, revisit the 3.1 scale estimate upward before assuming the 8-second cap is comfortable.

---

## PART 2 — THE DATA CONTRACT

### 2.1 Input JSON

```json
{
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
      "mandatory": true,
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
      "mandatory": false,
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
```

### 2.2 Output JSON

```json
{
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
      "selected": true,
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
      "selected": true,
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
      "selected": false,
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
      "conflict": false
    }
  ],
  "conflicts": [
    {
      "conflict_id": "CFX-01",
      "type": "resource_capacity",
      "involved_work_ids": ["WP-4589", "WP-4521", "WP-4530"],
      "block_section_id": null,
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
```

### 2.3 Schema notes

- **Timestamps in, integers out.** Convert every ISO8601 timestamp to minutes-since-`planning_horizon.start_datetime` at the API boundary. CP-SAT should never see a string or a datetime object — only small integers. Convert back to ISO8601 only when building `gantt_tasks`.
- **`solver_status` is not optional to expose, and it needs a fourth value.** DHTMLX shouldn't just render a schedule — the UI needs to visibly distinguish `OPTIMAL` from `FEASIBLE` (best-effort, time-limited) from `INFEASIBLE`. Add a fourth explicit value, `FALLBACK_HEURISTIC`, for the path in 3.3(4) where CP-SAT returned nothing usable and you served the greedy plan instead — silently labeling that `FEASIBLE` misrepresents what actually produced the schedule, exactly at the one moment this field's honesty matters most. Leave `objective_value` / `best_bound` / `optimality_gap_pct` null under `FALLBACK_HEURISTIC` rather than reporting numbers that don't mean what they normally mean.
- **`reason_code` on every deferred item is what makes this "explainable" rather than "a black box that dropped my defect."** CP-SAT will not generate this for you — see Part 3.6.
- **`mandatory_items_dropped` in `kpis` must always be present and always be checked by the frontend**, because you're about to make "mandatory" soft (Part 3.4) — this counter is your safety net for noticing when that happens.
- **Split headway into `headway_before_minutes` / `headway_after_minutes`** rather than one combined buffer — approach clearance and departure clearance are governed by different rules in practice, and a single number forces you to either over- or under-pad one side. Whichever shape you use, remember the field alone doesn't enforce anything: it has to be baked into `dur_i` before the interval is built (Part 1.4a), or it's just documentation.
- **Consider also surfacing *why a block was accepted*, not just why work was deferred** — a short list of the binding constraints a selected block satisfied (e.g. "respects headway with train 12301", "track-machine capacity not exceeded") is a nice audit trail for judges. Scope it to blocks near a resource limit, though — generating this for every accepted item at 500-train scale will bury the one deferral explanation anyone actually wants to read.
- **Multi-section jobs use `route_legs` instead of `block_sections_required` + one `duration_minutes`** (see Part 1.7) — a job with `route_legs` is a traveling job with one shared inclusion decision across all legs; a job without it is an ordinary single- (or simultaneous-multi-) section job as originally defined. `traveling_job_schedules` in the output mirrors `selected_work_packages` but reports one entry per leg instead of one start/end.

---

## PART 3 — THE RED TEAM ATTACK

### 3.1 Verdict, up front

At 500 trains + 150 maintenance requests: **it will not reliably blow the 5-minute demo window if you build it right, and it absolutely can if you build it the way the doc above describes it in Part 1.5 without the guardrails in 3.3.** The honest answer isn't a number of seconds — it's "put a hard wall-clock cap on it regardless, because you cannot prove in advance that your demo data won't hit an adversarial case."

Here's why the two halves of your problem are not equally dangerous:

- **500 trains are cheap.** If your timetable is fixed (which it is, for most of them), each train is a constant `IntervalVar` — a static forbidden zone, not a search decision. CP-SAT's propagation handles hundreds of fixed intervals almost for free. This number in your problem statement is scarier than it actually is.
- **150 maintenance requests are where the real combinatorics live.** Each one is a *Boolean inclusion decision* × a *continuous start-time domain* × a *shared-resource contention* with every other request competing for the same crew pool or the same block section. This is a genuine bin-packing-with-sequencing problem, which is NP-hard, and CP-SAT solves it by search + propagation, not by a closed-form formula.
- **Rough scale check, so "cheap" and "where the combinatorics live" aren't just vibes:** 500 trains × ~6–8 block sections each ≈ 3,000–4,000 fixed interval variables — trivial for CP-SAT's presolve, which collapses constant-domain intervals almost immediately. The 150 maintenance requests add ~150 Boolean presence vars plus ~300 bounded integer start/end vars whose domains span the *whole* horizon — that's what's actually driving search depth. Total variable count (a few thousand) isn't what should worry you; domain *width* and *resource-contention density* are.

### 3.2 Where the explosion actually comes from

Two specific design choices in the source plan will hurt you, and neither is visible until you stress-test:

1. **Unfiltered bundling pairs.** "Optional intervals + Boolean literals" is fine. Naively considering *every pair* of 150 work packages for a bundling reward is `150·149/2 ≈ 11,175` reified `AND` constraints. Most of those pairs are geographically nonsensical (a defect in Ghaziabad bundled with one in Mathura). The doc already tells you to use NetworkX for topology — use it to pre-filter bundling candidates down to genuinely adjacent, same-isolation-group pairs *before* they ever reach CP-SAT. That should cut the pair count by 1–2 orders of magnitude.
2. **Minute-level continuous start times over a 1440-minute horizon.** A `start_i` domain of `[earliest_i, latest_i]` spanning hundreds of minutes gives the solver a huge number of positions to consider for every one of 150 optional intervals, all interacting through `NoOverlap` and `Cumulative`. CP-SAT is very good at this, but "very good" is not "instant," and the failure mode isn't smooth degradation — it's fine on 47 of 50 test runs and then one adversarial or near-tied instance chews through your entire time budget trying to *prove* optimality on a solution it already found in the first two seconds.

That last point is the crux: **finding a good feasible solution is fast. Proving it's optimal is what can run long.** You almost never need the proof for a demo.

### 3.3 Mitigations, in priority order

1. **Hard wall-clock cap, no exceptions.** `solver.parameters.max_time_in_seconds = 8` (or whatever your latency budget is) and *always* accept `FEASIBLE`, not just `OPTIMAL`. This is a five-line change and it is non-negotiable — everything else on this list is about making that time budget produce a *good* answer, not about avoiding the cap.
2. **Slot-based time discretization for the demo build.** Instead of a free-floating `start_i` over 1440 minutes, offer the solver a small menu of pre-defined possession windows per block section (e.g., 4–6 candidate slots covering the night block). This turns a continuous scheduling problem into something much closer to an assignment problem, which CP-SAT chews through far faster. You can loosen this back to continuous time post-hackathon once you have a working baseline. *Lighter middle ground, if full slots feel too rigid for your timeline:* just coarsen the resolution — 5-minute buckets instead of 1-minute cuts the horizon from 1,440 steps to 288 and shrinks every domain 5×, for far less redesign than committing to fixed slots.
3. **Pre-filter before the model exists, not inside it.** Use NetworkX (as the doc already proposes) to (a) restrict bundling pairs to real adjacency, and (b) drop any work package whose required block sections have zero feasible windows outright, so CP-SAT never even builds a variable for it.
4. **Warm-start with `AddHint`.** Run a trivial greedy heuristic (sort by priority score, pack into available windows) in milliseconds, and hand it to the solver as a hint. This doesn't change worst-case behavior, but it means that even if you hit the time cap, the "best found" solution is at least as good as your fallback heuristic — you never demo something worse than what a spreadsheet would've given you. Keep that same greedy output on hand as a literal return value, too: if `solver.Solve()` ever comes back `INFEASIBLE` or `UNKNOWN` (shouldn't happen once mandatory work is soft, per 3.4, but don't bet a live demo on "shouldn't"), serve the greedy plan instead of an error page. The demo must never show an empty screen. One more thing to get right: hint at whatever variable is actually free in the model. On the block-based default (1.4e′), that means hinting `y_b` per block, not individual `x_i` — those are tied by a hard equality (`x_i == y_b`), so a heuristic that hints them inconsistently just produces a hint the solver mostly discards.
5. **Decompose by connected component.** Two work packages that share no block section, no crew pool, and no plant pool are independent — solving them in one monolithic CP-SAT model is strictly harder than solving them as separate smaller models. Build a "shares-a-resource" graph (again, NetworkX) and solve components separately. For a division-wide problem this can turn one hard 150-item model into several easy 20–40 item models.
6. **`num_search_workers = 8`** (or however many cores your demo machine has) — CP-SAT's parallel portfolio search meaningfully cuts wall-clock time on the same model for free.
7. **Wrap the solve in a server-side hard timeout too — but know what it actually protects.** Don't call the solver synchronously inside a FastAPI request handler; run it off the event loop. Be honest about what this buys you, though: `asyncio.wait_for` timing out does **not** kill a running thread. Python can't forcibly interrupt a thread mid-execution, and `Future.cancel()` is a no-op once the task has started — so if CP-SAT ever failed to honor its own time limit, this pattern wouldn't stop it, it would just stop *waiting* for it, while the thread (and a `with ThreadPoolExecutor() as executor` block, which blocks on exactly that thread finishing) keeps running in the background regardless. What it actually buys you: it bounds how long the HTTP response takes, and keeps a hung solve from blocking the event loop itself. The real hard bound on the solve is `max_time_in_seconds` (item 1) — that's the one CP-SAT actually checks internally and reliably returns from. Use the executor for response latency, not as a kill switch:

   ```python
   import asyncio, concurrent.futures

   async def solve_plan(request):
       loop = asyncio.get_running_loop()
       executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
       try:
           return await asyncio.wait_for(
               loop.run_in_executor(executor, run_solver, request),
               timeout=45.0,
           )
       finally:
           executor.shutdown(wait=False)   # don't block the response on a thread that's already CP-SAT-bounded
   ```

   If you genuinely need to kill a runaway process outright — not just stop waiting on it — that needs a `ProcessPoolExecutor` and `process.terminate()`, not a thread. More overhead, and your model/data have to cross a process boundary, so reach for it only if item 1 alone hasn't been enough in testing.
8. **Add `max_number_of_conflicts` as a second, independent stop condition** alongside `max_time_in_seconds` (e.g. `solver.parameters.max_number_of_conflicts = 100_000`) — belt-and-suspenders in case wall-clock enforcement ever behaves unexpectedly on the demo machine.

### 3.4 The infeasibility trap you've already written into the plan

`model.Add(x_i == 1)` for every mandatory safety job is a landmine. The moment two mandatory jobs can't both physically fit — same block section, overlapping windows, insufficient crew — the *entire model* returns `INFEASIBLE`, with no schedule, no partial result, and (by default) no explanation of which constraint caused it. In front of judges, "the solver returned nothing" reads as "the system is broken," not "the system correctly detected a genuine safety conflict."

Fix: make "mandatory" **soft with a very large penalty** (as in the pseudocode above — `drop_penalty` weighted at, say, 1,000,000 vs. your normal priority scores of 0–100), not hard. The solver will still refuse to drop a mandatory item unless it is *physically impossible* to include it, but now you always get a schedule back, and `mandatory_items_dropped > 0` becomes the exact signal you flash red in the UI: "these safety-critical items could not be accommodated — here's why." That's a much stronger demo moment than a crash.

### 3.5 The weighted-objective trap

```
maximize Σ P_i x_i − λ1(delay) − λ2(setup) − λ3(overtime)
```

This is the classic scalarization bug: you're adding terms with completely different natural units (a priority score capped at 100, raw minutes of delay that could run into the hundreds, raw overtime hours) into one linear objective. Whichever term has the largest raw magnitude will dominate the solver's behavior regardless of what your λ's were "supposed" to mean, and it will *look* like it's working on your first few test cases and then do something inexplicable on a case with slightly different numbers. Two ways out, pick one before you tune anything:

- **Normalize every term to a common 0–100 scale** before applying weights, so λ's actually control relative importance.
- **Or go lexicographic:** solve once to maximize `Σ P_i x_i` alone, fix the objective within, say, 2% of that optimum as a new constraint, then solve again minimizing disruption. Slower (two solves), but it eliminates the "which term secretly dominates" debugging spiral entirely — genuinely worth it if you have the time budget, since your demo scenarios (3.6) will exercise this repeatedly. If you go this route, split your existing time cap across both solves rather than giving each the full budget (e.g. 5s + 3s inside an 8s total) — two solves at `max_time_in_seconds = 8` each is a 16-second worst case nobody budgeted for at the API-timeout layer in item 7.

### 3.6 The demo-day checklist

- **Test against adversarial data, not your happy-path fixtures.** Your actual stress case isn't "150 random defects" — it's 40–50 defects all clustered on the same 2–3 block sections with tight, overlapping windows and one shared crew pool. That's the instance that will run long. Build it and time it *before* judging day.
- **Fix `random_seed`.** CP-SAT's parallel search is not fully deterministic run-to-run without it; you want the demo to behave identically every time you rehearse it.
- **Never show a spinner with no fallback.** If the solve hits its time cap, the UI should already be designed to say "best plan found in 8s (97% of estimated optimum)" — not freeze, not error.
- **Instrument the "why not" answer, because CP-SAT won't give it to you for free.** `reason_code` / `reason_detail` on deferred items (Part 2.2) has to be computed by *you*, typically with a cheap heuristic rather than a second solver call per rejected item — a full re-solve-to-explain-one-rejection loop is its own way to blow your time budget. A workable order of checks against the solution you already have, no re-solve needed: (1) does the job have *any* window at all against just the fixed trains, ignoring every other maintenance job? → `NO_FEASIBLE_WINDOW` if not. (2) Would adding its crew/plant demand to what's already selected in its window exceed pool capacity? → `CAPACITY_EXCEEDED`, name the pool. (3) Otherwise it lost on priority to whatever *did* get selected in the same slot → `LOWER_PRIORITY`, name the winner. Three cheap checks, in that order, covers the large majority of real deferrals.
- **Demonstrate exactly the three scenarios your own doc proposes** (normal day, late-running train, emergency high-severity defect) — and make sure the emergency-defect scenario is the one where you show the mandatory-item soft-constraint behavior from 3.4 actually triggering and being visibly flagged, not just silently succeeding. That's the scenario most likely to expose whether your "explainability" is real or decorative.

### 3.7 Implementation audit — this pass

A systems-level pass over the whole spec, done because "should work" and "does work" are different claims. Findings, worst first:

1. **The objective never included traveling jobs' priority.** The solver had zero incentive to ever select one — a silent bug, not a loud one; the model still solved fine, it just never picked a traveling job. Fixed by iterating `x` directly in the objective (step 8) instead of `work_packages` alone.
2. **The mandatory-soft loop (step 6) skipped traveling jobs entirely.** A `mandatory: true` traveling job had no enforcement of any kind — not hard, not soft, just ignored. Fixed by iterating `all_jobs` (work packages + traveling jobs).
3. **`priority` was referenced in the objective but never populated anywhere** — present since the very first draft, unrelated to the traveling-jobs extension. Fixed by populating it alongside `x` / `start` / `end` in step 2.
4. **Traveling jobs never populated `start[]` / `end[]`**, so any precedence constraint referencing one by id would `KeyError`. Fixed by recording the first leg's start and last leg's end as the job-level values.
5. **"Sequential execution inside a block" (1.1, 1.2) was asserted but never enforced.** Nothing stopped two jobs in the same candidate block from being scheduled simultaneously if crew/plant capacity allowed it, even though `D_b` was sized assuming they wouldn't. Fixed with an explicit `NoOverlap` over each block's own job intervals (1.4e′).
6. **Night shifts crossing midnight break under a naive 1440-minute single-day horizon** (`22:00` → 1320, `06:00` → 360, and 360 < 1320). Fixed with an explicit ETL conversion rule and a wider recommended horizon (1.4b).
7. **The FastAPI timeout wrapper (3.3 item 7) didn't actually kill a stuck solve** — Python can't interrupt a running thread, and the blocking `with` form waits for it regardless. Reframed as what it actually protects (response latency, not a kill switch), with `max_time_in_seconds` named as the real bound and a process-pool alternative noted for anyone who needs a true hard kill.
8. **`solver_status` had no way to honestly represent the greedy-fallback path** — it would've been mislabeled `FEASIBLE`. Added `FALLBACK_HEURISTIC` as an explicit fourth value (2.3).
9. Two smaller items folded in inline rather than getting their own numbers: headway padding can double-count if applied to both neighbors (1.4a), and multi-shift capacity needs the *union* of shift windows, not just one (1.4b).

Nothing here changes the architecture from the last few turns — the block-based bundling default, the soft-mandatory pattern, the traveling-job extension are all still the right calls. This pass is about the gap between "the design is right" and "the code that implements the design does what the design says," which is usually where a hackathon build actually breaks, not in the modeling choices.

---

## PART 4 — THE MACRO LAYER: WEEKLY/MONTHLY BUCKET PLANNING

### 4.1 Why this is a separate layer, not an extension of Part 1

Requirement #4 asks for block plans over weekly and monthly horizons. Part 1–3 deliberately don't do this — running the exact-minute, safety-exclusive CP-SAT model over a 30-day horizon isn't just slow, it's answering a question you can't actually know the answer to yet: you don't have a reliable minute-level train timetable 20 days out, so any "exact" schedule that far ahead is false precision, not a stronger plan.

The two layers have genuinely different jobs and genuinely different stakes, which is why they get different rigor:

- **Micro (Part 1–3):** *which minute*, exact, safety-exclusive, only trustworthy for the near-term window where the real timetable is known. Getting this wrong risks a real conflict. Worth a CP-SAT model, worth the audit in 3.7.
- **Macro (this part):** *which day*, coarse, advisory, running weeks or a month out against estimated capacity. Getting this wrong just means re-shuffling a day — the micro layer still enforces every real safety constraint when that day actually arrives. Not worth a second solver.

That asymmetry is also the justification for the answer you picked: a greedy first-fit-decreasing heuristic gets the large majority of an optimizer's value on a large backlog, in milliseconds, with none of the modeling and tuning cost of a second CP-SAT model — appropriate, because a suboptimal day assignment here is a minor inefficiency, not a safety incident.

### 4.2 The algorithm

Sort the full maintenance backlog — everything from TMS/SMMS/TDMS, not just what's relevant today — by mandatory status first, then by priority score boosted for SLA proximity (so a low-priority item doesn't sit unassigned forever just because higher-priority work keeps outbidding it — the boost term exists specifically to prevent starvation). Walk the backlog in that order; for each item, walk the days from its earliest permissible day to its deadline, and place it on the first day with enough remaining crew-hours, plant-hours, and corridor possession-hours. If it doesn't fit anywhere before its deadline, it's `NO_CAPACITY_BEFORE_DEADLINE` — surfaced, never silently dropped, same discipline as the mandatory-soft pattern in 3.4, just at day-level instead of a solver constraint.

This consumes the *same* `maintenance_requests` / `traveling_jobs` records from Part 2.1 — not a second parallel schema. `rough_duration_hours` is just `duration_minutes / 60` (or, for a traveling job, the sum of its legs' durations plus setup, divided by 60); `release_day` / `deadline_day` come from converting `earliest_start` / `latest_start` to calendar dates. One data contract, one source of truth — the same reason step 2.5 got folded into the master pseudocode instead of living as a separate copy in 3.7.

```python
from collections import defaultdict

URGENCY_BOOST = 40   # tunable: how hard SLA proximity can override raw priority

def macro_bucket_plan(backlog, daily_capacity, horizon_days, today):
    """
    backlog: MaintenanceRequest / TravelingJob records reduced to
             (id, priority_score, release_day, deadline_day, rough_duration_hours,
              crew_type, crew_hours, plant_type, plant_hours, corridor_id, mandatory)
    daily_capacity[day][(kind, key)] -> available hours, for day in [0, horizon_days)
    """
    def urgency(item):
        days_left = max(item.deadline_day - today, 1)
        boost = URGENCY_BOOST * max(0, 1 - days_left / 14)   # ramps up inside a 2-week SLA window
        return (item.mandatory, item.priority_score + boost)

    remaining = {d: dict(daily_capacity[d]) for d in range(horizon_days)}
    calendar = defaultdict(list)
    unassignable = []

    for item in sorted(backlog, key=urgency, reverse=True):
        placed = False
        start_day = max(item.release_day - today, 0)
        end_day = min(item.deadline_day - today, horizon_days - 1)
        for day in range(start_day, end_day + 1):
            if fits(item, remaining[day]):
                deduct(item, remaining[day])
                calendar[day].append(item.id)
                placed = True
                break
        if not placed:
            unassignable.append({
                "id": item.id, "reason_code": "NO_CAPACITY_BEFORE_DEADLINE", "mandatory": item.mandatory,
            })

    return calendar, unassignable

def fits(item, cap):
    return (item.crew_hours          <= cap.get(("crew", item.crew_type), 0)
        and item.plant_hours         <= cap.get(("plant", item.plant_type), 0)
        and item.rough_duration_hours <= cap.get(("corridor", item.corridor_id), 0))

def deduct(item, cap):
    cap[("crew", item.crew_type)]       -= item.crew_hours
    cap[("plant", item.plant_type)]     -= item.plant_hours
    cap[("corridor", item.corridor_id)] -= item.rough_duration_hours
```

### 4.3 Data contract additions

New input — a coarse capacity calendar, since `resource_capacity` in 2.1 only describes one day's shifts:

```json
"daily_capacity_calendar": [
  {
    "date": "2026-09-09",
    "crew_hours": { "P-Way_gang": 48, "tamping_gang": 24 },
    "plant_hours": { "tamping_machine": 16 },
    "corridor_possession_hours": { "BS-101": 4, "BS-102": 4, "BS-103": 3 }
  }
]
```

New output — the allocation calendar, referencing the same `work_id`s from `maintenance_requests` / `traveling_jobs`, nothing new invented:

```json
"macro_allocation_calendar": {
  "generated_at": "2026-09-08T09:00:00+05:30",
  "horizon_days": 30,
  "days": [
    { "date": "2026-09-09", "assigned_work_ids": ["WP-4521", "MW-0100"] },
    { "date": "2026-09-10", "assigned_work_ids": ["WP-4589"] }
  ],
  "unassignable_before_deadline": [
    { "work_id": "WP-4777", "reason_code": "NO_CAPACITY_BEFORE_DEADLINE", "mandatory": true }
  ]
}
```

### 4.4 The rolling hand-off to the micro layer

Each morning (or before that night's block, if you're planning ahead): pull `assigned_work_ids` for today out of the macro calendar, look those ids up in the full `maintenance_requests` / `traveling_jobs` records, and feed exactly that subset into the Part 1 CP-SAT model as today's problem instance — same contract, same code, nothing new to build there. **The loop only works if the return path exists too:** if the micro solver defers any of those ids (`deferred_work_packages`, 2.2), re-inject them into tomorrow's macro backlog with `release_day = tomorrow` before the next nightly macro run. An item the micro layer couldn't fit today isn't resolved — it's still owed a day. Skipping this makes the macro calendar decorative: a plan that never learns what actually happened stops being a plan.

Re-run the macro assigner on a schedule (nightly is reasonable) — TMS/SMMS/TDMS feeds are live, and a bucket plan computed once at project kickoff goes stale as fast as the backlog it was built from.

### 4.5 Where a greedy macro layer can quietly fail

- **Capacity estimates need to be pessimistic, not average.** `corridor_possession_hours` is an *estimate* — you don't have day+20's real train timetable. If it's built from a historical mean, roughly half your days will actually deliver less than promised, and the micro layer will defer things the macro layer told the backlog were handled. Use something like the 80th-percentile *used* capacity from historical data, not the mean — better to under-promise and have the micro layer occasionally do better than expected than to over-promise and generate a steady stream of surprise deferrals.
- **The feedback loop in 4.4 is the actual safety valve here, not a nice-to-have.** Without it, a day that turns out tighter than estimated just silently loses whatever got deferred.
- **Starvation is a real risk with pure greedy-by-priority**, which is why `urgency()` boosts SLA-proximity independent of raw priority — without that term, a backlog with a steady stream of high-priority items can leave a moderate-priority item sitting unassigned indefinitely even though nothing ever explicitly rejected it.
- **This is advisory, not authoritative — say so in the UI.** The macro calendar is a planning aid for coordinators, not a commitment the way the micro layer's output is. Label it as such, or a coordinator will reasonably read "assigned to Sept 22" as a promise the system isn't actually able to make that far out.

---

## PART 5 — THE ML ESCALATION-RISK PREDICTOR & SPATIOTEMPORAL CLUSTERING

### 5.0 Where this layer sits, and where it categorically does not

Everything in this Part runs **offline (training) and once-per-ETL-pass (inference)** — never inside the CP-SAT solve loop, and never as a candidate for a `model.Add(...)` call. Concretely, the pipeline slots into the "ETL / Score" stage that already exists conceptually in Part 2's data contract, one step before the NetworkX pre-filter in Part 3.3:

```
raw TMS/SMMS/TDMS logs
    → feature engineering (Pandas/Polars)
    → [5.1] escalation-risk model  → R_i  (feeds the existing 0.35S+0.25R+0.20C+0.10A+0.10O formula, 1.2)
    → [5.2] duration model         → dur_i (replaces the static per-defect-type default, 1.2/2.1)
    → NetworkX adjacency + connected-component split (3.3 item 5)
    → [5.3] spatiotemporal clustering (within each component) → candidate blocks (1.4e′)
    → CP-SAT (Part 1) — sees only plain integers, exactly as specified in 2.1
```

This ordering matters: clustering (5.3) runs **after** the adjacency/component split, not before, because bundling defects that share no NetworkX-adjacent block section is geographically meaningless regardless of what a clustering algorithm's distance metric says. The two pre-filtering mechanisms are complementary, not redundant — NetworkX answers "which defects are physically eligible to be considered together," and DBSCAN answers "which of those eligible defects should actually be grouped into one possession."

**The boundary that must never be crossed:** nothing in Part 5 outputs a `start_i`, an `x_i`, or any variable CP-SAT treats as a decision. Every output is a plain number — a risk score 0–100, a duration in minutes, a cluster/block ID — written into the same JSON fields Part 2.1 already defines. If every model described below were deleted tomorrow and replaced with the static defaults from the original spec, Part 1's `NoOverlap`/`Cumulative` guarantees would not change by one bit. That is by design, and it is the concrete mechanism behind Pillar 1 of the Executive Summary.

### 5.1 Predicting defect escalation risk (`R_i`)

**Why this needs a model at all.** `R_i` is currently a hand-scored or rule-table-derived 0–100 input in `priority_inputs` (see the `WP-4521` example in 2.1: `"escalation_risk_R": 70`). A threshold table is defensible but coarse — it can't capture the nonlinear interactions that actually drive escalation, e.g. old rail *combined with* high cumulative tonnage *combined with* a recent monsoon is disproportionately more dangerous than any one factor alone. A trained model captures that interaction; a lookup table has to be hand-tuned to approximate it.

**Target definition.** Binary label: does this defect escalate to a critical-severity fault within 7 days of being logged? Label this from historical TMS resolution/re-inspection records, joined back to the original defect entry by `defect_id`. **Censoring matters here** — a defect logged 3 days before the end of your historical data window, and not yet resolved, is *not* a confirmed negative; it's unresolved-status-unknown and should be excluded from training rather than silently coded as "did not escalate."

| Feature | Source |
|---|---|
| `track_age_years` | Track asset registry — years since last rail renewal |
| `gmt_since_renewal` | Gross Million Tonnes carried since last renewal (fatigue proxy) |
| `defect_type` | TMS/SMMS defect taxonomy (categorical, one-hot or target-encoded) |
| `rail_wear_mm` | Latest ultrasonic/visual inspection reading |
| `rainfall_7d_mm` | Weather archive, preceding 7 days at that km marker |
| `traffic_density_trains_per_day` | COA timetable, trains touching that block section per day |
| `defect_recurrence_count` | Prior defects logged at the same km marker |
| `days_since_last_inspection` | TMS inspection log |
| `division` / `zone` | Categorical — different divisions have different baseline risk profiles |

**Model.** Gradient-boosted trees (`XGBClassifier`, Random Forest as a fallback baseline) producing a calibrated probability, not a direct 0–100 regression — probability is the more honest target, and rescaling `proba × 100` into `R_i` at inference time keeps the model's actual output interpretable on its own terms.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

FEATURES = [
    "track_age_years", "gmt_since_renewal", "defect_type_encoded",
    "rail_wear_mm", "rainfall_7d_mm", "traffic_density_trains_per_day",
    "defect_recurrence_count", "days_since_last_inspection",
]

def train_escalation_model(historical_defects: pd.DataFrame) -> xgb.XGBClassifier:
    # historical_defects must be TIME-SLICED, not row-shuffled: train on defects
    # opened before CUTOFF_DATE, validate on defects opened after it. A random
    # k-fold split silently leaks corridor-wide degradation trends across the
    # train/validation boundary — the temporal-leakage trap named below.
    X = historical_defects[FEATURES]
    y = historical_defects["escalated_within_7d"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),  # escalations are rare
        eval_metric="aucpr",   # PR-AUC, not accuracy — class imbalance makes accuracy meaningless here
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
```

**Two leakage traps, named explicitly because they're the ones that silently produce a model that looks great in validation and useless in production:**

- **Outcome leakage.** Never include a feature derived from what happened *after* the 7-day escalation window closes — "was this defect eventually renewed under emergency protocol" is downstream of the label, not a predictor of it. If it's only knowable in hindsight, it doesn't belong in `FEATURES`.
- **Temporal leakage.** Defect escalation is spatially and temporally correlated — an entire stretch of rail degrades together during one bad monsoon. A random train/test split lets the model see next month's version of this month's defect and "predict" it trivially. Always split by time, as in the snippet above (`shuffle=False`), never by random row sampling.

**Evaluation.** Report PR-AUC and a calibration curve (Brier score), not raw accuracy — with escalation events rare, a model that predicts "never escalates" for everything can still score 95%+ accuracy while being worthless. Calibration matters specifically because `R_i` gets rescaled into a 0–100 score that's *added* to other terms in the SRCAO formula — an uncalibrated model that's systematically over- or under-confident will silently skew every priority score it touches.

**Confidence band and fallback — this is where Pillar 1 actually shows up in Part 5, not just in the Executive Summary.** A prediction near the decision boundary is barely better than a coin flip; serving it as a confident 0–100 number misrepresents what the model actually knows. Route low-confidence and cold-start cases back to the original rule-based scoring instead of trusting the model blindly — the same "never let one component's failure produce a blank result" discipline as the solver's greedy fallback in 3.3(4), applied one layer upstream of the solver:

```python
LOW_CONFIDENCE_BAND = (0.40, 0.60)   # near-coin-flip predictions aren't trustworthy enough to act on alone

def predict_escalation_risk(model, defect_features: dict, rule_based_fallback_R: int) -> dict:
    try:
        proba = model.predict_proba(pd.DataFrame([defect_features]))[0][1]
    except Exception:
        # Model unavailable, malformed record, or a defect_type/feature combo the
        # model has never seen (cold start) — fall back, don't guess.
        return {"escalation_risk_R": rule_based_fallback_R, "risk_confidence": "fallback_rule_based"}

    if LOW_CONFIDENCE_BAND[0] <= proba <= LOW_CONFIDENCE_BAND[1]:
        return {"escalation_risk_R": rule_based_fallback_R, "risk_confidence": "low_model_confidence"}

    return {"escalation_risk_R": round(proba * 100), "risk_confidence": "high"}
```

**Explainability.** Attach SHAP values at inference time and surface the top 2–3 contributing features per prediction (e.g. "high GMT since renewal, track age above division median, rainfall spike in the last 7 days") into the audit trail alongside the score. This is Pillar 3 extended one layer upstream: a judge — or a track engineer — asking "why did the system flag this defect as high-risk" gets a real, model-grounded answer instead of "the model said so."

**Retraining cadence.** Batch retrain weekly or monthly against the growing historical log, not real-time — escalation patterns don't shift hour to hour, and continuous retraining on a live production feed just adds instability for no benefit. Tag every prediction with a `model_version` so a prediction can always be traced back to the model that produced it.

### 5.2 Predicting task duration (`dur_i`)

**Why this is the higher-stakes of the two models.** Part 1.2's `dur_i` and the `duration_minutes: 90` field in 2.1 are currently static per-defect-type estimates. Real durations vary with gang experience, machine type and age, weather, and section accessibility — but the reason this matters more than it might first appear is **safety, not just accuracy**: if `dur_i` is under-estimated, the schedule CP-SAT *proves* feasible via `NoOverlap`/`Cumulative` is not actually feasible in the field. Pillar 1's "absolute mathematical safety" is a guarantee about the model, and a model is only as trustworthy as the numbers fed into it. A wrong `dur_i` is the one class of ML error in this entire architecture that can propagate into an operational safety problem rather than just a quality one — which is exactly why it gets the most conservative treatment of any prediction in this spec.

| Feature | Source |
|---|---|
| `defect_type` | Same taxonomy as 5.1 |
| `gang_id` | Historical mean/variance of completion time for that specific gang |
| `machine_type` | Tamping machine / crane / ballast train model and age |
| `track_type` | Single line, double line, curvature/gradient if available |
| `crew_size` | Planned crew size vs. that job type's historical standard |
| `is_night_shift` | Night possessions historically run slower — lighting, visibility, fatigue |
| `section_accessibility` | Distance from depot / access-road quality |

**Model and the one design decision that matters most in this subsection: predict a conservative quantile, not the mean.** Use a quantile-loss objective (`reg:quantileerror` in XGBoost, or LightGBM's native quantile objective) to predict the **P80** duration — not the expected value. Feed that P80 figure into `dur_i`, not the mean.

```python
def train_duration_model(historical_jobs: pd.DataFrame, quantile: float = 0.80) -> xgb.XGBRegressor:
    # Predicting the mean would mean roughly HALF of all jobs run over their
    # allotted possession window by construction — which quietly breaks the
    # safety guarantee NoOverlap is supposed to provide, since NoOverlap only
    # proves consistency against whatever dur_i it's handed, not against reality.
    X = historical_jobs[["defect_type_encoded", "gang_id_encoded", "machine_type_encoded",
                          "track_type_encoded", "crew_size", "is_night_shift"]]
    y = historical_jobs["actual_duration_minutes"]
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile,
        n_estimators=250, max_depth=4, learning_rate=0.05,
    )
    model.fit(X, y)
    return model

def predict_duration_minutes(model, job_features: dict,
                              static_default_minutes: int, min_possession_minutes: int) -> dict:
    try:
        pred = float(model.predict(pd.DataFrame([job_features]))[0])
        dur = max(round(pred), min_possession_minutes)   # 1.7's MIN_POSSESSION_MINUTES floor still applies
        return {"duration_minutes": dur, "duration_source": "model_p80"}
    except Exception:
        return {"duration_minutes": static_default_minutes, "duration_source": "static_fallback"}
```

**Two things carried over deliberately from earlier parts, because a duration model doesn't get to relax them:**

- The `MIN_POSSESSION_MINUTES` floor from 1.7 still applies to a model-predicted duration exactly as it does to a computed one — a thinly-populated feature combination can produce an anomalously low quantile prediction, and the floor is what stops that from becoming an operationally meaningless sub-15-minute possession.
- Cold start (a new gang, a newly commissioned machine with no track record) falls back to the existing static default, tagged `duration_source: "static_fallback"` — same explainability discipline as 5.1, so nothing downstream has to guess whether a duration came from a trained model or a placeholder.

### 5.3 Spatiotemporal clustering for candidate-block generation

**What this is actually for.** Part 1.4e′ specifies the MVP default: run a "defect-clustering/priority engine" *before* CP-SAT ever sees the problem, and hand the solver pre-grouped candidate possession blocks instead of 150 individual jobs to bundle combinatorially. This subsection is that engine, specified in full. It is the concrete mechanism behind Pillar 4: the combinatorial work of *deciding what could plausibly be bundled* happens here, outside the solver, so CP-SAT is left to do what it's actually good at — arbitrating scarce track/crew/plant time between blocks that already exist.

**Why unsupervised clustering, not just the NetworkX adjacency filter that already exists.** Adjacency (3.2/3.3) answers "which defect pairs are geographically eligible to share a possession." It doesn't answer "which of those eligible defects should actually be grouped together tonight." Clustering answers the second question, deterministically and in milliseconds, instead of leaving it to a hand-written rule or a wider CP-SAT search space.

**Distance metric: corridor chainage, not raw lat/long.** A rail defect lives on a 1-dimensional corridor (distance along a specific line), not a 2D plane — two defects can be geographically close as the crow flies while sitting on different lines that share no block section. Cluster on **chainage distance along the corridor**, restricted to defects already inside one NetworkX-adjacent connected component (3.3 item 5) — never across a whole division.

**Algorithm: DBSCAN, not K-Means.** DBSCAN is the right choice here for three concrete reasons specific to this problem, not as a general preference: (1) it doesn't require pre-specifying the number of clusters, and the "right" number of possession groupings genuinely varies night to night with the backlog; (2) it naturally leaves sparse or isolated defects as noise points, which become valid single-defect candidate blocks rather than being force-fit into a cluster that doesn't suit them; (3) rail defect distributions along a corridor are elongated, not spherical, which is exactly the shape K-Means's centroid-based clustering handles poorly.

- `eps` (max chainage distance to bundle, e.g. 2 km): set from the operational constraint — how far one crew and its machinery can reasonably traverse within a single possession — not fit statistically. There's no ground-truth cluster count to optimize against.
- `min_samples = 1`: a lone defect with no eligible neighbor is still a valid candidate block on its own.

**Temporal compatibility is a second, separate filter — geography alone isn't sufficient.** Two defects can be a few hundred metres apart and still have permissible windows (`earliest_start`/`latest_start`) that barely overlap. Run DBSCAN geographically first, then split each resulting cluster further by interval-overlap before emitting final blocks.

```python
from sklearn.cluster import DBSCAN
import numpy as np

EPS_KM = 2.0
MIN_SAMPLES = 1
SETUP_MINUTES_DEFAULT = 20

def cluster_defects_into_blocks(defects: list[dict]) -> list[dict]:
    """
    defects: already restricted to ONE NetworkX-adjacent connected component
    (3.3 item 5) — running this across a whole division is not geographically
    meaningful. Each dict needs: id, chainage_km, block_section_id,
    earliest_start_min, latest_start_min, dur_i (from 5.2), priority_score.
    """
    coords = np.array([[d["chainage_km"]] for d in defects])
    labels = DBSCAN(eps=EPS_KM, min_samples=MIN_SAMPLES).fit_predict(coords)

    blocks = []
    for cluster_id in set(labels):
        members = [d for d, lbl in zip(defects, labels) if lbl == cluster_id]
        for group in _split_by_window_overlap(members):   # geography ≠ compatibility; check both
            blocks.append({
                "block_id": f"CLUSTER-{group[0]['block_section_id']}-{cluster_id}-{len(blocks)}",
                "job_ids": [m["id"] for m in group],
                "earliest_start": max(m["earliest_start_min"] for m in group),
                "latest_start": min(m["latest_start_min"] for m in group),
                "total_duration": sum(m["dur_i"] for m in group) + SETUP_MINUTES_DEFAULT,
            })
    return blocks

def _split_by_window_overlap(members: list[dict]) -> list[list[dict]]:
    members = sorted(members, key=lambda m: m["earliest_start_min"])
    groups, current, window_end = [], [], None
    for m in members:
        if not current or m["earliest_start_min"] <= window_end:
            current.append(m)
            window_end = min(window_end, m["latest_start_min"]) if window_end else m["latest_start_min"]
        else:
            groups.append(current)
            current, window_end = [m], m["latest_start_min"]
    if current:
        groups.append(current)
    return groups
```

**Recovering some lost flexibility cheaply.** As 1.4e′ already notes for the block-based approach generally: emit 2–3 overlapping candidate groupings per cluster (the full cluster vs. just its two highest-priority members) as mutually exclusive alternatives, and let CP-SAT — which is genuinely good at this kind of arbitration — pick between them. This recovers some of the combinatorial flexibility lost by pre-committing to fixed blocks, without reopening the full `O(n²)` pairwise search this design deliberately avoids.

**Output shape — no new contract on the CP-SAT side.** Each candidate block emits exactly the fields 1.4e′'s pseudocode already consumes (`b.job_ids`, `b.earliest_start`, `b.latest_start`, `b.total_duration`). Everything in Part 5 is upstream of Part 1; nothing about the solver's inputs changes.

### 5.4 Where this ML layer can quietly fail — red-team, same discipline as Part 3

- **Overconfident predictions in sparse regions of feature space.** A defect type / gang combination rarely or never seen in training is exactly where a tree-based model will still confidently output *something* — it doesn't know what it doesn't know. This is precisely what the confidence-band fallback in 5.1 and the cold-start fallback in 5.2 exist to catch; treat both as load-bearing, not optional polish.
- **Feature drift.** Track conditions, gang composition, and machinery change across a season — a model trained on last year's monsoon behavior can misjudge this year's. This needs a monitoring signal (predicted-vs-actual residuals over time) and a scheduled retrain, not a train-once-and-forget deployment.
- **The duration model must never be allowed to silently undercut the safety floor.** An inadequately regularized quantile model can produce anomalously low predictions for rare, thin-data feature combinations. The `max(pred, min_possession_minutes)` clamp in the inference snippet above is not defensive boilerplate — it's the one line standing between a bad prediction and an operationally unsafe schedule.
- **Feedback-loop bias in the escalation-risk label itself.** Once this system starts influencing which defects get fixed first, the defects that *don't* escalate may increasingly be so specifically because the system already deprioritized the easy, low-risk-looking ones — which selectively removes exactly the ambiguous cases the model most needs to keep learning from. A hackathon MVP won't fully solve this, but the retraining pipeline should be built aware of it (e.g. periodically holding out a small randomized sample from prioritization to preserve an unbiased training signal), rather than pretending the problem doesn't exist.
- **A failed or missing ML call must never block the pipeline.** Every model call is wrapped in try/except with a deterministic fallback (5.1, 5.2) — the same "the demo must never show an empty screen" discipline as the solver's greedy fallback in 3.3(4), applied one layer upstream of the solver instead of at it.

### 5.5 Data contract additions

The ML layer's outputs attach to each `maintenance_requests` record during ETL, before it reaches Part 1's model construction. Nothing new reaches CP-SAT — `escalation_risk_R` and `duration_minutes` are the same fields 2.1 already defines; this is where their values come from, plus an audit trail alongside them:

```json
"ml_enrichment": {
  "escalation_risk_probability": 0.70,
  "escalation_risk_R": 70,
  "risk_confidence": "high",
  "top_risk_factors": ["gmt_since_renewal", "track_age_years", "rainfall_7d_mm"],
  "predicted_duration_minutes": 96,
  "duration_quantile": "p80",
  "duration_source": "model_p80",
  "cluster_id": "CLUSTER-BS101-03",
  "model_version": "escalation-xgb-2026.08.2"
}
```

`priority_inputs.escalation_risk_R` and `maintenance_requests[].duration_minutes` in 2.1 are populated from this object at ETL time. `cluster_id` maps a defect to the candidate block that 1.4e′'s pseudocode consumes as `b.job_ids`. CP-SAT's pseudocode in 1.6 needs zero changes — it already only ever sees plain integers.

### 5.6 The boundary, restated

Nothing in Part 5 selects, schedules, or excludes a single job. Every model here produces one thing: a better number for an input CP-SAT already expected. The escalation-risk model makes `R_i` more accurate than a threshold table. The duration model makes `dur_i` more accurate than a flat per-defect-type default, with a conservative bias built in specifically because that number feeds a safety-critical constraint. The clustering step makes the candidate blocks fed to 1.4e′ smarter than a static adjacency rule. In every case, the thing that actually decides which trains and which maintenance jobs occupy which block section at which minute remains exactly what it was in Part 1: `AddNoOverlap`, `AddCumulative`, and a solver that proves its answer rather than predicting it.

---

## PART 6 — TECH STACK BLUEPRINT

This part is the complete, install-ready parts list for the system described in Parts 1–5, built from the CP-SAT spec itself, the curated resource pack, and the ML pipeline in Part 5. Every entry carries one of three status tags:

- **[SPEC]** — required by the math itself (Parts 1–3, 5). Not a choice.
- **[RESOURCE]** — recommended by the project's own curated resource pack.
- **[GAP]** — load-bearing for the system to actually run end to end, but not named in either source above. Flagged honestly, with the reason spelled out, so skipping it (if you do) is a decision and not an oversight.

### 6.0 How the pieces fit together

```
Ingest            Mock COA/TMS/SMMS/TDMS producers + RailRadar live feed
     │
     ▼
ETL / Score       Pandas·Polars — SRCAO priority scoring, midnight-safe minute conversion
     │
     ▼
ML Predict        [Part 5] XGBoost escalation-risk classifier → R_i
                   [Part 5] XGBoost quantile duration model   → dur_i
     │
     ▼
Pre-filter        NetworkX adjacency + connected-component split (3.3 item 5)
                   → [Part 5.3] scikit-learn DBSCAN spatiotemporal clustering → candidate blocks (1.4e′)
     │
     ▼
Solve             OR-Tools CP-SAT — 8s hard wall-clock cap, warm-started from a greedy fallback
     │
     ▼
API               FastAPI — Part 2 JSON contract in, Part 2 JSON contract out
     │
     ▼
Render            React + DHTMLX Gantt

Dispatcher override (lock/reject a block)
     │
     ▼
Persist decision (SQLite)
     │
     ▼
re-enters Solve as a fixed constraint on the next run
```

### 6.1 Core Optimization Engine

**Google OR-Tools (`ortools.sat.python.cp_model`)** — `[SPEC]`
The substrate the entire Part 1 formulation is written against, not a choice among alternatives. `NewOptionalIntervalVar` models optional maintenance jobs and candidate blocks (1.3, 1.4e′); plain `NewIntervalVar` models fixed trains as forbidden zones rather than decision variables; `AddNoOverlap` enforces track exclusivity per block section and, in the block-based default, sequencing within a block (1.4a, 1.4e′); `AddCumulative` caps crew/plant pools (1.4b); `OnlyEnforceIf` gates precedence (1.4c), traveling-job leg sequencing (1.7), and the soft-mandatory drop penalty (3.4); `AddHint` is the warm-start mechanism (1.6 step 9); `solver.parameters.{max_time_in_seconds, num_search_workers, random_seed, max_number_of_conflicts}` are the demo-safety guardrails from 3.3.
`pip install ortools` · Python 3.10–3.12

**NetworkX** — `[SPEC]`
Named twice as the pre-filter layer: once to restrict bundling pairs to real geographic/isolation adjacency before they become thousands of reified `AND` constraints (3.2), and again to decompose the problem by connected component so independent work packages sharing no resource aren't solved in one monolithic model (3.3 item 5). Build the graph directly from `block_sections[].adjacent_section_ids`, use `connected_components` for the decomposition, and a custom edge weight (chainage distance + `isolation_group` equality) to generate bundling candidates — and, as of Part 5.3, to scope the DBSCAN clustering pass to one component at a time.
`pip install networkx` · a few hundred nodes at this scale — no GPU or compiled backend needed

**cpsat-utils** — `[RESOURCE]`
Its real value here is `assert_hint_feasible` — catching the exact bug 3.3 item 4 calls out: hinting `x_i` individually when the model actually ties jobs to a block via a hard equality means the solver mostly discards the hint. Use in tests, not on the runtime path.
`pip install cpsat-utils` · single-maintainer package — dev/test dependency only

**cpsat-logutils** — `[GAP]`
Turning on `log_search_progress` and parsing it tells you whether a slow solve is stuck in presolve versus genuinely searching — the difference between "add a better hint" and "your `Cumulative` bound is wrong." Directly serves 3.6's requirement to test and time the adversarial case before judging day. Dev-time only.
`pip install cpsat-logutils`

**Greedy fallback heuristic (plain Python)** — `[SPEC]`
Required first-class code, not a script: sort by priority score, pack into available windows, run in milliseconds. Serves two jobs — seeds `AddHint` for the real solve, and is served verbatim as the `FALLBACK_HEURISTIC` response if the solver ever returns `INFEASIBLE`/`UNKNOWN` (3.3 item 4). This is the mechanism that guarantees Pillar 2 (Graceful Degradation) holds in practice, not just on paper.
No library — bespoke logic against the project's own data model

**Reason-code / explainability engine (plain Python + enums)** — `[SPEC]`
CP-SAT will not generate `reason_code` for you. The exact cheap check order, read against the already-solved model with no re-solve: (1) no feasible window against fixed trains alone → `NO_FEASIBLE_WINDOW`; (2) adding demand would exceed pool capacity → `CAPACITY_EXCEEDED`, name the pool; (3) otherwise it lost on priority → `LOWER_PRIORITY`, name the winner (3.6).
No library — a Pydantic enum for the `reason_code` literal, plain Python for the check order

**stdlib `datetime` / `zoneinfo`** — `[SPEC]`
2.3 mandates timestamps-in, integers-out at the API boundary; 1.4b spells out exactly why: a night shift `22:00→06:00` naively converts to `1320→360` on a 1440-minute horizon, and `360 < 1320` breaks the interval. Use `zoneinfo.ZoneInfo("Asia/Kolkata")` on every incoming timestamp, convert to horizon-relative integer minutes, and apply the explicit fix — if `end < start`, add 1440 before subtracting. Stdlib is exactly sufficient here; no need for `pendulum` or `arrow`.
Standard library — zero extra dependency

### 6.2 ML Prediction & Clustering Layer (Part 5)

**scikit-learn** — `[SPEC]`
Directly required by Part 5.3's `DBSCAN` clustering and by standard preprocessing (`StandardScaler`, `train_test_split` with time-respecting `shuffle=False`) used to prepare features for both the escalation-risk and duration models in 5.1/5.2. This is the foundation the rest of the ML layer builds on, not an optional convenience.
`pip install scikit-learn`

**XGBoost** — `[SPEC]`
The model family behind both 5.1 (escalation-risk classifier, `scale_pos_weight` + `aucpr` for the class-imbalance problem) and 5.2 (quantile-loss duration regressor via `reg:quantileerror`). Chosen over a deep-learning approach deliberately — see 6.7 for why that's a decision, not an oversight, for a tabular, feature-engineered problem of this size.
`pip install xgboost`

**Pandas or Polars** — `[RESOURCE]`
Does double duty: named in the original resource pack for TMS/SMMS/TDMS normalization (joining raw maintenance requests against block sections, building each job's `req_i` set, computing the SRCAO formula as a vectorized column op), and now also the feature-engineering layer feeding 5.1/5.2's `FEATURES` tables (track age, GMT, rainfall joins, historical gang-performance aggregates). Either library satisfies both jobs — the spec doesn't prefer one.
`pip install pandas` — or — `pip install polars`

**SHAP** — `[GAP]`
Not in the original resource pack, but directly required to keep Part 5's explainability promise (5.1, 5.5's `top_risk_factors` field) honest — without it, "why is this defect rated high-risk" has no real answer beyond "the model said so," which is precisely the failure mode Pillar 3 exists to prevent. Compute per-prediction SHAP values at inference time; this is cheap enough to run per-request at this scale (150 maintenance requests, not millions).
`pip install shap`

**joblib** — `[GAP]`
Neither source names a model-serialization strategy, but the escalation-risk and duration models have to be trained offline and loaded once at API startup — retraining per request would blow the 8-second solve budget before CP-SAT ever runs. `joblib.dump`/`joblib.load` is the standard, low-friction way to persist a fitted scikit-learn/XGBoost model to disk.
`pip install joblib`

**imbalanced-learn** — `[GAP]`, optional
`scale_pos_weight` in 5.1's XGBoost config is usually sufficient for the class imbalance in escalation labels. If validation shows it isn't (very rare positive rate, or a small dataset where reweighting alone underperforms), `imbalanced-learn`'s SMOTE variants are the natural next step. Listed as a documented fallback, not a default dependency — don't add complexity you haven't confirmed you need.
`pip install imbalanced-learn` · optional, add only if 5.1's validation metrics call for it

### 6.3 Backend & API Layer

**Python 3.11+** — `[GAP]`
OR-Tools' Python bindings, FastAPI, and XGBoost/scikit-learn all target modern Python, and 3.11+'s general speedups indirectly help the 8-second solve budget. Not named directly in either source, but it's the runtime everything else in this stack assumes.

**FastAPI** — `[RESOURCE]`
Named directly in the resource pack for ingestion. The spec's own async wrapper (3.3 item 7) is written as an `async def` route using `asyncio.get_running_loop()` — FastAPI/Starlette's execution model verbatim, not a generic suggestion. Implied endpoints from the data contract: `POST /plan` (2.1 in → 2.2 out) and an override/re-solve endpoint for locked dispatcher decisions.
`pip install fastapi`

**Uvicorn** — `[GAP]`
FastAPI is a framework, not a server — it needs an ASGI server to run at all.
`pip install "uvicorn[standard]"` — the `[standard]` extra pulls in uvloop/httptools for real performance

**Pydantic v2** — `[RESOURCE]`
Named for the priority engine, but its bigger job is the entire Part 2 data contract — `BlockSection`, `TrainMovement`, `MaintenanceRequest`, `TravelingJob` (with `route_legs`), `CrewPool`/`PlantPool`, `SolverConfig` in; `ScheduleResponse`, `DeferredWorkPackage`, `Conflict`, `KPIs`, `GanttTask` out — plus, as of Part 5.5, the `ml_enrichment` object. This is also exactly where `solver_status` becomes a strict `Literal` with all four values, including `FALLBACK_HEURISTIC`, instead of an unvalidated string.
`pip install pydantic`

**pydantic-settings** — `[GAP]`
`solver_config` (max_time_in_seconds, num_search_workers, random_seed) in 2.1 is a settings object that also needs safe env-var defaults and per-request overrides — a judge's laptop might have 4 cores, not 8. Also the natural home for the ML layer's `model_version` / model-path configuration from Part 5.
`pip install pydantic-settings`

**httpx** — `[GAP]`
RailRadar is named as the live data source, but nothing in the source docs names an HTTP client to call it. It has to be async — `requests` is synchronous and would block the event loop inside a FastAPI route, undermining the exact async pattern 3.3 item 7 sets up.
`pip install httpx`

**SQLAlchemy + SQLite** — `[GAP]`
Neither source names a database, but the MVP path requires one: if a dispatcher locks a block or rejects a task, that decision has to survive to inform the *next* solve as a fixed constraint (`model.Add(x[locked_id] == 1)` or an exclusion). Without a store, an override is a parameter that vanishes the moment the HTTP response returns. SQLite is enough for a hackathon.
`pip install sqlalchemy` · `sqlite3` is stdlib

**pytest + pytest-asyncio** — `[GAP]`
3.6's demo-day checklist — test the 40–50-defect, single-block-section adversarial case before judging day; fix `random_seed` so the demo behaves identically every rehearsal — is a testing requirement even though no framework is named. Also the natural home for a regression test locking in the midnight-crossing conversion (3.7 item 6), one asserting `solver_status` is never mislabeled under the fallback path (3.7 item 8), and, as of Part 5, one asserting the ML confidence-band fallback actually triggers on an out-of-distribution feature vector rather than silently serving a low-confidence prediction.
`pip install pytest pytest-asyncio`

**Pyomo** — `[RESOURCE]`, not installed
Named in the resource pack as a "secondary modeling layer," but both the resource pack and the spec's own architecture converge on CP-SAT alone as faster to model and simpler to deploy for this exact problem shape. Listed so skipping it is a decision, not a gap.
skip for MVP

### 6.4 Data & Integration Layer

**Part 2 JSON contract as source of truth** — `[SPEC]`
Every entity in 2.1/2.2 becomes one Pydantic model on the backend and, ideally, one TypeScript interface on the frontend: `PlanningHorizon`, `BlockSection`, `TrainMovement`, `MaintenanceRequest`, `TravelingJob`, `CrewPool`, `PlantPool`, `SolverConfig` in; `ScheduleResponse`, `SelectedWorkPackage`, `TravelingJobSchedule`, `DeferredWorkPackage`, `GanttTask`, `Conflict`, `KPIs` out. Treat this JSON as the actual interface spec between every layer of the system, not documentation of one — Part 5's `ml_enrichment` object extends it without breaking it.

**Mock COA / TMS / SMMS / TDMS producers** — `[RESOURCE]`
There is no public REST/GraphQL API for COA or TMS. The only real option is Pydantic-validated fixture generators shaped like what the CAG/CRIS documentation describes — train movements, section occupancy, defect records, km markers, severity, last inspection date. This is an explicit, first-class "mock data service" module, not a stub to apologize for — and, per Part 5.1, it's also the only realistic source of a *labeled* historical training set for the escalation-risk model at hackathon scale, since there's no live TMS resolution history available to train against.

**RailRadar API** — `[RESOURCE]`
The one real, documented, live external data source. Maps directly onto the late-running-train demo scenario (3.6). Base `https://api.railradar.in/v1`, key endpoint `GET /v1/trains/{number}/live` returning delay, current location, segment progress, next halt — feed a real delay into `shift_t`/`max_shift_t` for a semi-flexible train for a live-data-driven disruption instead of a hand-typed one. Treat as enrichment: the solver must work fully on mocked COA/TMS data alone.
reference: `https://railradar.in/docs`

**Unofficial IR API repos / IRCTC-via-RapidAPI** — `[RESOURCE]`, reference only
Listed in the resource pack with its own caveat: "for inspiration, not production." Don't depend on these for anything judged live — rate limits, uptime, and terms-of-service status are all outside your control on demo day.

### 6.5 Frontend & Visualization

**React** — `[RESOURCE]`
Named directly as the wrapping framework for the Gantt view — a React-based Gantt displays train paths, block windows, engineering possessions, conflicts, and defect priority. Not an open choice.

**DHTMLX Gantt v10+ Community Edition (`dhtmlx-gantt`)** — `[RESOURCE]`
Recommended in the resource pack, and independently validated against the output schema — `gantt_tasks` in 2.2 (`id`, `text`, `start_date`, `end_date`, `parent`, `type`, `priority_color`, `conflict`) is shaped as DHTMLX's native task-parse format almost field-for-field, so the API emits DHTMLX's format directly rather than translating into it. v10 Community Edition ships under MIT (`npm install dhtmlx-gantt`; `gantt.license` returns `"mit"` at runtime); undo/redo, the "today" marker, and multi-task drag-select are gated to the paid PRO tier.
`npm install dhtmlx-gantt` · lanes: Track/Block Section, Train paths (non-editable), Proposed blocks, Maintenance bundles, Conflict markers, Priority color — via `gantt.templates` hooks reading `priority_color`/`conflict` straight off the contract

**Frappe Gantt** — `[RESOURCE]`, optional
A legitimate secondary, read-only "before vs. after" comparison view — e.g. unoptimized manual plan vs. solver output — cheap to add without committing DHTMLX's full interaction surface to it. Skip if time is tight; DHTMLX alone satisfies every rendering requirement.
`npm install frappe-gantt` · optional

**TypeScript** — `[GAP]`
DHTMLX ships TypeScript definitions, and the data contract is a strict, branching JSON shape — `route_legs` vs. `block_sections_required` depending on `job_type`, a four-way `solver_status` enum, and now `ml_enrichment` alongside it — exactly the kind of thing that silently breaks in plain JS on a backend field rename. Given how much of Part 3.7's audit was silent, non-crashing bugs, type-checking the frontend's consumption of the contract is cheap insurance in the same spirit.
`npm install -D typescript`

**@tanstack/react-query** — `[GAP]`
The dispatcher lock/re-solve loop is a real round trip with loading/error states, and 3.6 is explicit: "never show a spinner with no fallback" — the UI must distinguish `OPTIMAL` from `FEASIBLE` from `FALLBACK_HEURISTIC`, and needs a visible in-flight state for a solve that can legitimately take 8+ seconds. React Query gives you that state machine without hand-rolling it.
`npm install @tanstack/react-query`

**Vite** — `[GAP]`
Needed to actually run a React + TypeScript + DHTMLX app in dev and build it for the demo. Not named because it's assumed tooling — listed so a clean checkout is never missing a piece.
`npm create vite@latest`

### 6.6 Vibe Coding Toolchain — Antigravity

**Antigravity as grounding context, not a one-time prompt** — `[GAP]`
Google's agent-first IDE (a VS Code fork) with two surfaces — an Editor view for hands-on work, and a Manager view for spawning and orchestrating multiple agents in parallel across workspaces. Agents produce "Artifacts" — task lists, implementation plans, screenshots, browser recordings — as verifiable deliverables instead of raw tool calls. Part 3.7's audit already found and fixed nine silent bugs in an earlier draft of this exact system; feed the spec and JSON contract into the workspace as shared context so every agent inherits the corrected math instead of re-deriving the naive, already-broken version of it.

**Manager view — split along the spec's own seams** — `[GAP]`
Because every layer's interface is the fixed Part 2 JSON contract (now extended by Part 5's `ml_enrichment`), agents can run in parallel without stepping on each other: one on the solver core (Part 1 pseudocode → real `cp_model` code), one on the ML pipeline (Part 5 — training scripts, inference wrappers, the confidence-band fallback), one on the API/backend (FastAPI routes, Pydantic contract models, mock data generators), one on the frontend (React + DHTMLX wired to the exact 2.2 shape), one on data/ETL and explainability (SRCAO scoring, reason-code engine, RailRadar client, SHAP integration). The contract is the seam that makes true parallelism safe.

**Browser control as the acceptance test** — `[GAP]`
The hardest parts of this system to verify from code alone are visual — do conflict markers actually render, does the priority color show up on the right task. Point an agent at the running Vite dev server, POST the 40–50-defect adversarial payload from 3.6 to `/plan`, and screenshot the resulting Gantt.

**Encode the demo-day checklist as task-list Artifacts** — `[GAP]`
3.6's checklist — fixed `random_seed`, the three named scenarios (normal / late-train / emergency-defect), no-spinner-without-fallback UI states, the adversarial stress fixture timed before judging day, plus Part 5's confidence-band and cold-start fallback tests — is the actual definition of done. Turn it into a checkable, agent-visible task list instead of something you hope you remember on demo morning.

### 6.7 Consolidated Dependency Manifest

**Python (backend + ML)**

| Package | Role | Status |
|---|---|---|
| `ortools` | CP-SAT solver | SPEC |
| `networkx` | Adjacency pre-filter, component decomposition, clustering scope | SPEC |
| `scikit-learn` | DBSCAN clustering, preprocessing, train/test splitting | SPEC |
| `xgboost` | Escalation-risk classifier, quantile duration regressor | SPEC |
| `fastapi` | API framework | RESOURCE |
| `uvicorn[standard]` | ASGI server to actually run FastAPI | GAP |
| `pydantic` | Part 2 data contract, request/response models | RESOURCE |
| `pydantic-settings` | `solver_config` + ML model-path layering (env + per-request) | GAP |
| `pandas` *or* `polars` | ETL, SRCAO scoring, ML feature engineering | RESOURCE |
| `shap` | Per-prediction explainability for the ML layer | GAP |
| `joblib` | Model persistence — train offline, load once at API startup | GAP |
| `imbalanced-learn` | Optional class-imbalance handling for escalation-risk training | GAP, optional |
| `httpx` | Async HTTP client for RailRadar (non-blocking) | GAP |
| `sqlalchemy` + `sqlite` | Persist dispatcher-locked decisions between solves | GAP |
| `cpsat-utils` | Hint validation / completion (dev-time only) | RESOURCE |
| `cpsat-logutils` | Parse solver logs to diagnose slow solves (dev-time) | GAP |
| `pytest`, `pytest-asyncio` | Adversarial fixture, contract + ML fallback regression tests | GAP |
| `python` (3.11+) | Runtime | GAP |

**JavaScript / TypeScript (frontend)**

| Package | Role | Status |
|---|---|---|
| `react`, `react-dom` | UI framework | RESOURCE |
| `dhtmlx-gantt` | Gantt rendering (v10+, MIT, Community Edition) | RESOURCE |
| `frappe-gantt` | Optional lightweight "before/after" view | RESOURCE |
| `typescript` | Type-safe consumption of the Part 2 contract | GAP |
| `vite`, `@vitejs/plugin-react` | Dev server / build tool | GAP |
| `@tanstack/react-query` | Loading/error/stale state for the solve round trip | GAP |

**Explicitly not installed**

| Package | Why it's listed but skipped |
|---|---|
| `pyomo` | Named in the resource pack as a secondary modeling layer, but both the resource pack and the spec converge on CP-SAT alone as sufficient and faster to ship. Listed so skipping it is a decision, not an oversight. |
| TensorFlow / PyTorch | Part 5's escalation-risk and duration problems are structured, tabular, feature-engineered prediction tasks at a few-hundred-row-per-training-run scale — exactly where gradient-boosted trees (XGBoost) match or beat deep learning on accuracy while being faster to train, easier to explain via SHAP, and far simpler to deploy without a GPU dependency. A deep-learning framework would add deployment complexity for no accuracy benefit at this problem's scale and data volume. |

**Primary sources referenced:** OR-Tools CP-SAT guide, OR-Tools `scheduling.md`, NetworkX docs, `cpsat-utils` (PyPI), CP-SAT Primer, scikit-learn `DBSCAN` docs, XGBoost documentation (quantile regression objective), SHAP documentation, DHTMLX Gantt docs, DHTMLX Gantt v10.0.0 release notes, Frappe Gantt, RailRadar API docs, Antigravity launch materials.
