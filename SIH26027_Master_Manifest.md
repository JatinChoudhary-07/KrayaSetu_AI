# SIH26027 — Integrated Block Bundling Engine
## Master Tech Stack & Resource Manifest (Antigravity Grounding Context)

**Authority order used to resolve every conflict below:** `SIH26027_Master_Spec_v2.md` > Production research pack > earlier Perplexity research > preliminary HTML blueprint. Where the earlier documents floated options (Pyomo vs. OR-Tools, pairwise bundling vs. block-based bundling, DHTMLX vs. Frappe), the spec's explicit decision is treated as final and the alternative is recorded only as a rejected option, not a live choice.

There were no real contradictions to resolve — the later Production Research Pack explicitly validates the spec's architecture rather than disputing it. The one place worth flagging: the spec's Part 6 already contains a tech-stack blueprint. This manifest is that blueprint, reorganized into the exact structure requested and written as standalone grounding context (no need to re-read spec.md to use it), with the MCP/agent layer added since spec.md's Part 6.6 only sketches it.

---

## 1. The Definitive Tech Stack

### Solver Core
| Component | Decision | Why |
|---|---|---|
| Solver | **Google OR-Tools CP-SAT** (`ortools.sat.python.cp_model`) | Not a choice among alternatives — the entire Part 1 formulation is written against it. `NewOptionalIntervalVar` = optional maintenance jobs / candidate blocks; `NewIntervalVar` = fixed trains as forbidden zones; `AddNoOverlap` = track exclusivity; `AddCumulative` = crew/plant capacity; `OnlyEnforceIf` = precedence, traveling-job leg sequencing, soft-mandatory drop penalty; `AddHint` = warm start. |
| Python | **3.11+** | Modern OR-Tools/FastAPI/XGBoost target; speedups matter against the 8s solve budget. |
| Pyomo | **Explicitly rejected for MVP.** | Both research pack and spec converge on CP-SAT alone as faster to model, simpler to deploy for this exact problem shape. Do not install. |
| Bundling model | **Precomputed candidate blocks (spec 1.4e′), not pairwise reification (1.4d/1.4e).** | Pairwise is `150·149/2 ≈ 11,175` reified AND constraints — too much search space for MVP payoff. Blocks turn "which subset of 150 jobs bundle" into "which of ~50–100 precomputed blocks get selected." Optionally emit 2–3 overlapping candidate blocks per cluster to recover some flexibility. |
| Mandatory jobs | **Soft constraint + 1,000,000-point drop penalty**, never hard `x_i == 1`. | Hard equality makes two conflicting mandatory jobs return `INFEASIBLE` with zero explanation — a live landmine at demo time. |
| Objective | **Lexicographic two-pass** (maximize priority → fix within tolerance → minimize disruption), splitting the 8s budget (e.g. 5s + 3s). | Avoids the scalarization bug where mismatched units (priority 0–100 vs. delay-minutes vs. overtime-hours) let one term silently dominate. |
| Time model | **5-minute discretization**, minutes-since-horizon-start integers only inside the model; ISO8601 only at the API boundary. | CP-SAT never sees a string or datetime. |
| Dev/test only | `cpsat-utils` (hint validation), `cpsat-logutils` (search-progress diagnosis) | Neither ships to production; both catch exactly the bugs the spec's red-team pass (3.7) found. |
| Solver guardrails | `max_time_in_seconds=8`, `num_search_workers=8`, `random_seed=42`, `max_number_of_conflicts=100_000` | Non-negotiable demo-safety parameters (spec 3.3). |

### ML & Pre-Filtering
| Component | Decision | Why |
|---|---|---|
| Escalation-risk model | **XGBoost `XGBClassifier`**, `objective="binary:logistic"`, `eval_metric="aucpr"`, `scale_pos_weight` for imbalance, time-sliced train/val split (`shuffle=False`) | Feeds `R_i` in the SRCAO formula. Calibrated probability, not raw regression — rescale `proba × 100` at inference. |
| Duration model | **XGBoost `XGBRegressor`**, `objective="reg:quantileerror"`, `quantile_alpha=0.80` (P80) | Predicting the mean would mean ~half of all jobs overrun their possession window — the one ML error class in this entire system with a direct safety consequence. Clamp with `max(pred, MIN_POSSESSION_MINUTES)` always. |
| Explainability | **SHAP** (`shap.TreeExplainer`), top 2–3 contributing factors surfaced per prediction | Not in the original resource pack but load-bearing for Pillar 3 (explainability) — without it "why is this high-risk" has no real answer. |
| Calibration | `sklearn.calibration.CalibratedClassifierCV` (isotonic if enough positives, else Platt/sigmoid) | Escalation-risk gets *added* into SRCAO — an over/under-confident model silently skews every priority score. |
| Confidence fallback | Custom low-confidence band (0.40–0.60 proba) and cold-start `try/except` → route to rule-based static score | Same "never a blank result" discipline as the solver's greedy fallback, one layer upstream. |
| Clustering (candidate blocks) | **scikit-learn `DBSCAN`** on 1-D corridor chainage, `eps` set operationally (~2km), `min_samples=1`, scoped to one NetworkX-connected-component at a time, then split further by time-window overlap | DBSCAN chosen specifically because cluster count isn't known in advance, isolated defects should remain valid singleton blocks, and defect distributions are elongated (corridor-shaped), not spherical — K-Means is the wrong tool here. |
| Topology / pre-filter | **NetworkX** — build graph from `block_sections[].adjacent_section_ids`; `connected_components` for problem decomposition; custom edge weights (chainage distance + isolation-group equality) for bundling-candidate adjacency | Used twice: restrict bundling pairs to real adjacency before CP-SAT (cuts pair count 1–2 orders of magnitude), and split the model into independent sub-problems by shared resource. |
| Tabular ops / feature eng | **Pandas or Polars** (spec doesn't prefer one) | SRCAO scoring, TMS/SMMS/TDMS normalization, ML feature tables (GMT, rainfall joins, gang-performance aggregates). |
| Model persistence | **joblib** | Train offline, load once at API startup — retraining per request would blow the 8s solve budget before CP-SAT even runs. |
| Optional imbalance handling | `imbalanced-learn` (SMOTE variants) | Only if `scale_pos_weight` alone underperforms in validation — documented fallback, not a default dependency. |
| Explicitly rejected | **TensorFlow / PyTorch** | Structured, tabular, few-hundred-row-per-run prediction tasks — gradient-boosted trees match or beat deep learning here while being faster to train, SHAP-explainable, and GPU-free. |

### Backend & API
| Component | Decision | Why |
|---|---|---|
| Framework | **FastAPI** | Named directly in the resource pack; the spec's own async-timeout wrapper (3.3 item 7) is written as an `async def` FastAPI/Starlette route verbatim. |
| Server | **Uvicorn** (`uvicorn[standard]` for uvloop/httptools) | FastAPI needs an ASGI server; not named in either source but load-bearing. |
| Schema/validation | **Pydantic v2** | The entire Part 2 data contract becomes Pydantic models: `PlanningHorizon`, `BlockSection`, `TrainMovement`, `MaintenanceRequest`, `TravelingJob` (with `route_legs`), `CrewPool`/`PlantPool`, `SolverConfig` in; `ScheduleResponse`, `DeferredWorkPackage`, `Conflict`, `KPIs`, `GanttTask` out; plus `ml_enrichment` (Part 5.5). `solver_status` is a strict `Literal["OPTIMAL","FEASIBLE","INFEASIBLE","FALLBACK_HEURISTIC"]` — the fourth value is mandatory, not optional. |
| Config | `pydantic-settings` | `solver_config` needs safe env-var defaults + per-request override (a judge's laptop may have 4 cores, not 8); also the home for ML `model_version`/model-path config. |
| Async HTTP client | **httpx** | RailRadar client must be async — `requests` would block the event loop inside the async wall-clock-timeout wrapper the spec requires. |
| Persistence | **SQLAlchemy + SQLite** | Dispatcher lock/reject decisions have to survive to the *next* solve as a fixed constraint or exclusion — without a store, an override vanishes when the HTTP response returns. SQLite is sufficient for hackathon scale. |
| Testing | **pytest + pytest-asyncio** | Demo-day checklist (3.6) is a testing requirement even though no framework is named in the spec: adversarial 40–50-defect fixture, fixed-seed determinism, midnight-crossing regression test, `FALLBACK_HEURISTIC` mislabeling test, ML confidence-band fallback test. |
| Reason-code engine | Plain Python + Pydantic enum, no library | Three cheap deterministic checks against the already-solved model (no re-solve): `NO_FEASIBLE_WINDOW` → `CAPACITY_EXCEEDED` (name the pool) → `LOWER_PRIORITY` (name the winner). |
| Greedy fallback heuristic | Plain Python, bespoke | Sorts by priority, packs into windows, runs in milliseconds. Seeds `AddHint`; served verbatim as `FALLBACK_HEURISTIC` if CP-SAT ever returns `INFEASIBLE`/`UNKNOWN`. |
| Time handling | stdlib `datetime` / `zoneinfo.ZoneInfo("Asia/Kolkata")` | Sufficient — no need for `pendulum`/`arrow`. Implements the explicit midnight-crossing fix: if `end < start`, add 1440 before subtracting. |

### Frontend
| Component | Decision | Why |
|---|---|---|
| Framework | **React** | Named directly in the resource pack as the Gantt-wrapping framework — not an open choice. |
| Gantt library | **DHTMLX Gantt v10+ Community Edition** (`npm install dhtmlx-gantt`) | MIT-licensed (verify at runtime: `gantt.license === "mit"`). `gantt_tasks` in the Part 2 output schema (`id`, `text`, `start_date`, `end_date`, `parent`, `type`, `priority_color`, `conflict`) is shaped almost field-for-field as DHTMLX's native task-parse format — the API should emit DHTMLX's format directly rather than translating into it. Undo/redo, "today" marker, multi-task drag-select are PRO-gated; skip them. |
| Secondary Gantt (optional) | **Frappe Gantt** (`npm install frappe-gantt`) | Only for a lightweight read-only "manual plan vs. solver output" comparison view. Skip if time is tight — DHTMLX alone satisfies every rendering requirement. |
| Language | **TypeScript** | DHTMLX ships type defs; the contract has real branching complexity (`route_legs` vs. `block_sections_required`, 4-way `solver_status`, `ml_enrichment`) — exactly what silently breaks in plain JS on a backend field rename. |
| Data fetching | **@tanstack/react-query** | The lock/re-solve round trip can legitimately take 8+ seconds; spec 3.6 explicitly forbids "spinner with no fallback" — needs a real loading/error/stale state machine. |
| Build tool | **Vite** (`npm create vite@latest`) | Standard React+TS+DHTMLX dev/build tooling; not named because it's assumed, listed so a clean checkout is never missing it. |
| Explicitly rejected | Old DHTMLX v9-and-earlier tutorials (GPL-licensed) | Use only v10+ Community docs to stay MIT. |

---

## 2. Master Resource & Integration Registry

### APIs & Telemetry

**Live train telemetry — RailRadar (the only real, documented, live external source):**
- Docs: `https://railradar.in/docs`
- Base URL: `https://api.railradar.in/v1`
- Live status: `GET /v1/trains/{number}/live` → delay, current section, segment progress, next halt, diversion/cancellation — `https://railradar.in/docs/live-train-status`
- Train details/timetable: `https://railradar.in/docs/get-train-details`
- Trains between stations: `https://railradar.in/docs/trains-between-stations`
- **Integration rule (non-negotiable):** treat every RailRadar response as a delayed, cached, external enrichment signal — never an authority for block occupancy. Feed observed delay into `shift_t`/`max_shift_t` for a semi-flexible train (the "late-running train" demo scenario), bounded, with a freshness TTL, retry/circuit-breaker, stale-data indicator, and fallback to static COA data. The solver must work fully on mocked COA/TMS data alone with RailRadar disconnected.

**Unofficial IR API wrappers (reference/fixture generation only — never production/demo-critical):**
- `https://github.com/AniCrad/indian-rail-api`
- `https://github.com/sachinB94/railway-api`
- `https://github.com/topics/indian-railways-api`
- Rate limits, uptime, and ToS status are outside your control on demo day — do not depend on these live.

**TMS / COA / SMMS / TDMS — no public API exists. This is confirmed across every research document.**
- There is no REST/GraphQL endpoint for TMS, COA, SMMS, or TDMS. Build a **mock producer module** (Pydantic-validated fixture generators) shaped from the schemas described in the official/semi-official references below — this is a first-class module, not a stub.
- Reference schemas to build the mocks from:
  - CAG audit report on COA/ICMS: `https://cag.gov.in/uploads/download_audit_report/2016/Union_Report_32_2016_Railways_on_ICMS_in_Indian_Railways.pdf`
  - RDPMS Functional Requirement Spec (SMMS asset/telemetry integration patterns): `https://ecr.indianrailways.gov.in/uploads/files/1757402904205-ECR-HQ0S&T (RBCT) 1 2024 Dt- 25.06.2025 FUNCTIONAL REQUIREMENT SPECIFICATION FOR Remote Diagnostic and Predictive Maintenance System (RDPMS)(Version 1.0)..pdf`
- This mock layer doubles as your only realistic source of a *labeled historical training set* for the escalation-risk model (Part 5.1) — there's no live TMS resolution history to train against at hackathon scale.

### Datasets (download now, static/structural)

| Dataset | URL | Use |
|---|---|---|
| IR Train Time Table (OGD) | `https://www.data.gov.in/catalog/indian-railways-train-time-table` | Base station/route/timing structure for mock COA fixtures |
| IR Time Table CSV (reservation trains snapshot) | `https://www.data.gov.in/resource/indian-railways-time-table-trains-available-reservation-03082015` | Same, ~7.7MB CSV |
| Operations of Indian Railways (aggregate stats) | `https://www.data.gov.in/catalog/operations-indian-railways` | High-level capacity context only, not per-block |
| datameet/railways | `https://github.com/datameet/railways` | JSON stations/routes/schedules — best base for block-section graph construction |
| Route-graph tutorial + cleaned CSV | `https://anuj-seth.github.io/postgresql/2020/10/07/railways_data_model_part_1.html` | Shows modeling OGD timetable as a graph — direct pattern for the NetworkX corridor graph |
| Kaggle IR Schedule Dataset | `https://www.kaggle.com/datasets/dhanashrishirsath/railyatri-railway-schedule-dataset` | Small CSV, fast prototyping |

Use OGD timetable + `datameet/railways` together to approximate block-section topology and populate `block_sections[].adjacent_section_ids` for the mock COA layer.

### Reference Repositories (patterns to clone/adapt, not turnkey solutions — no single repo matches this spec)

**CP-SAT / OR-Tools:**
- `https://github.com/google/or-tools` — clone this; it's the actual dependency, not just a reference
- Scheduling docs (optional intervals, NoOverlap/Cumulative semantics): `https://github.com/google/or-tools/blob/stable/ortools/sat/docs/scheduling.md`
- Flexible job-shop example: `https://github.com/google/or-tools/blob/master/examples/python/flexible_job_shop_sat.py`
- RCPSP example: `examples/python/rcpsp_sat.py` in the OR-Tools repo — directly analogous to crew/plant-pooled block sections
- Single-machine setup/release/due-date example: `examples/python/single_machine_scheduling_with_setup_release_due_dates_sat.py` — maps to possession windows + isolation overhead
- CP-SAT Primer (patterns, warm starts, parameters): `https://d-krupke.github.io/cpsat-primer/`
- Scheduling benchmark harness: `https://scheduleopt.github.io/benchmarks/`

**Topological pre-filtering:**
- NetworkX reference: `https://networkx.org/documentation/stable/reference/index.html`
- Connected components: `https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html`

**ML pipeline:**
- XGBoost quantile regression worked example: `https://xgboost.readthedocs.io/en/release_3.2.0/python/examples/quantile_regression.html`
- XGBoost parameter reference: `https://xgboost.readthedocs.io/en/stable/parameter.html`
- SHAP TreeExplainer docs: `https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html`
- scikit-learn DBSCAN: `https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html`

**Gantt frontend:**
- DHTMLX Gantt docs: `https://docs.dhtmlx.com/gantt/`
- DHTMLX v10 MIT release notes: `https://dhtmlx.com/blog/dhtmlx-gantt-10-0/`

### What genuinely does not exist publicly (mock or MoU only)
- Official TMS/SMMS/TDMS defect schemas and live feeds
- Official COA section-occupancy logs
- Any end-to-end "railway maintenance bundling" repo matching this spec's shape — you are assembling primitives from OR-Tools + XGBoost + SHAP + NetworkX + your own data layer, not adapting one existing project.

---

## 3. Vibe Coding & Agentic Architecture (Antigravity IDE)

### Why the Part 2 JSON contract is the thing that makes parallel agents safe
Every layer of this system — solver, ML pipeline, API, frontend — communicates exclusively through the Pydantic-validated Part 2/5.5 JSON contract (`MaintenanceRequest`, `TravelingJob`, `ScheduleResponse`, `ml_enrichment`, etc.). Because the contract is fixed and versioned, agents working on different layers can run genuinely in parallel without stepping on each other's code — the seam between them is the schema, not a shared module. Load the full `spec.md` and this manifest into the Antigravity workspace as shared context **before** spawning any agent, so every agent inherits the corrected math from the spec's Part 3.7 red-team pass (soft-mandatory, block-based bundling, midnight-safe horizon, traveling-job atomicity) instead of re-deriving — and re-breaking — the naive version.

### MCP surface the backend must expose

**Resources (read-only context):**
```
rail://corridors/{corridor_id}/topology
rail://plans/{plan_id}/gantt
rail://backlog/current
rail://trains/{train_id}/movements
rail://solver/{run_id}/diagnostics
```

**Tools (typed, some approval-gated):**
| Tool | Access |
|---|---|
| `solve_micro_schedule` | Direct — runs Part 1 CP-SAT against today's backlog subset |
| `generate_macro_plan` | Direct — runs Part 4 greedy FFD bucket assignment |
| `explain_deferral` | Direct — returns the 3-check reason-code result for a given `work_id` |
| `render_gantt_snapshot` | Direct — renders current plan as DHTMLX task JSON |
| `validate_schedule` | Direct — structural assertion pass (no overlaps, every selected item has section+resource, every deferred item has a reason code) |
| `run_adversarial_test` | Direct — fires the 40–50-defect dense-corridor fixture from spec 3.6 |
| `lock_possession_block` | **Proposal only** → requires human/dispatcher approval before persisting |
| `reject_candidate_block` | **Proposal only** → requires human approval |
| `apply_dispatcher_override` | **Typed payload, dual approval, immutable audit event** — never accepts free-form text |

**Prompts:**
- "Explain why this mandatory task was deferred."
- "Compare the current plan with the previous dispatch-approved plan."
- "Prepare a safe override proposal without applying it."

**Authorization rule:** `apply_dispatcher_override` must receive a structured payload only (`plan_id`, `override_type`, `block_id`, `reason`, `approved_by`, `approval_reference`) — an agent must never be allowed to call it with arbitrary text, per the spec's own MCP design section.

Use the official SDKs directly:
- Python SDK: `https://github.com/modelcontextprotocol/python-sdk`
- Spec: `https://modelcontextprotocol.io/specification/2025-06-18`

### Agent Task Breakdown (sequential prompting plan to avoid context collision)

Spawn these as **separate Antigravity agents/workspaces**, each grounded only in `spec.md` + this manifest + its own layer's slice of the contract — not the other agents' in-progress code.

**Agent 1 — Data Contract Foundation (run first, alone; everything else depends on it)**
> Build every Pydantic v2 model from Part 2.1/2.2 and Part 5.5 of spec.md: `PlanningHorizon`, `BlockSection`, `TrainMovement`, `MaintenanceRequest` (including the `route_legs`-bearing traveling-job variant), `CrewPool`/`PlantPool`, `SolverConfig` on input; `ScheduleResponse`, `SelectedWorkPackage`, `TravelingJobSchedule`, `DeferredWorkPackage`, `GanttTask`, `Conflict`, `KPIs`, `ml_enrichment` on output. Make `solver_status` a strict 4-value `Literal` including `FALLBACK_HEURISTIC`. Emit matching TypeScript interfaces for the frontend. Do not write solver or ML logic yet.

**Agent 2 — CP-SAT Solver Core**
> Implement Part 1 of spec.md exactly as pseudocoded in 1.6, including the block-based bundling default (1.4e′), soft-mandatory drop penalties (3.4), traveling-job shared-presence-literal legs (1.7), the lexicographic two-pass objective (3.5) with 5s/3s budget split, `AddHint` warm start from a greedy fallback, and all guardrails from 3.3 (wall-clock cap, `num_search_workers`, `random_seed`, `max_number_of_conflicts`). Cross-check against every fix listed in the Part 3.7 audit table before declaring done — those nine bugs are the actual acceptance criteria.

**Agent 3 — ML Pipeline (Part 5)**
> Build the escalation-risk classifier (5.1) and P80 quantile duration regressor (5.2) with time-sliced splits, calibration, SHAP explainability, and the confidence-band/cold-start fallback to rule-based scoring. Build the DBSCAN spatiotemporal clustering step (5.3), scoped to one NetworkX connected component at a time, emitting candidate blocks in the exact shape Agent 2's `b.job_ids`/`b.earliest_start`/`b.latest_start`/`b.total_duration` expects. Never write a `start_i` or `x_i` — outputs are plain numbers into `ml_enrichment` only.

**Agent 4 — Backend API & Data Layer**
> Build FastAPI routes (`POST /plan`, override/re-solve endpoint) wired to Agent 1's models, the async timeout wrapper from spec 3.3 item 7, the reason-code engine (3.6's 3-check order), the mock COA/TMS/SMMS/TDMS fixture generators, the RailRadar httpx client with the bounded-enrichment pattern, and SQLAlchemy persistence for dispatcher overrides. Also implement the Part 4 macro FFD bucket planner and its nightly rolling hand-off (4.4) into Agent 2's micro solver.

**Agent 5 — Frontend**
> Build the React + TypeScript + DHTMLX Gantt app wired to Agent 1's exact contract shape. Implement the lanes from the spec/resource pack (Track/Block Section, Train paths, Proposed blocks, Maintenance bundles, Conflict markers, Priority color), the 4-way `solver_status` visual distinction, and @tanstack/react-query loading/error states that never show a bare spinner.

**Agent 6 — MCP Server & Explainability/Audit Layer**
> Expose the MCP resources and tools listed above using the official Python SDK, wired to Agent 4's API. Wire dual-approval logic for `lock_possession_block`, `reject_candidate_block`, and `apply_dispatcher_override`. Implement `explain_deferral` and `run_adversarial_test` as thin wrappers over Agent 4's reason-code engine and the 40–50-defect fixture.

**Agent 7 — Acceptance & Demo-Day Verification (run last, against the integrated system)**
> Encode spec 3.6's demo-day checklist as a runnable, agent-visible task list: fixed `random_seed` determinism check, the three named scenarios (normal / late-running-train / emergency-defect with visible mandatory-drop-and-alert), no-spinner-without-fallback UI states, the adversarial dense-corridor stress fixture timed before judging day, and Part 5.4's ML fallback tests (confidence-band trigger, cold-start trigger). Use browser control to POST the adversarial payload to `/plan`, screenshot the resulting Gantt, and assert structurally — no overlapping bars on the same section, every deferred item has a visible reason code, `FEASIBLE`/`OPTIMAL`/`FALLBACK_HEURISTIC` are visually distinct.

**Order matters:** 1 → (2, 3, 4, 5 in parallel, all reading only Agent 1's finished contract) → 6 → 7. Don't let 2–5 start before 1 is merged — that's the collision Antigravity's Manager view is specifically good at avoiding once the seam is fixed.
