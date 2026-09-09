# KrayaSetu AI

## Integrated Railway Maintenance Block Bundling

KrayaSetu AI is a railway maintenance planning and block bundling system
built around a typed scheduling contract, an optimization oriented
backend architecture, an ML enrichment layer, and a React based Gantt
dashboard.

The repository currently contains the implemented frontend, shared data
contracts, ML pipeline, specification and architecture documents, and
automated acceptance tests. The documented target architecture uses
Google OR Tools CP SAT for constrained scheduling, XGBoost and SHAP for
ML enrichment, NetworkX and DBSCAN for topology and maintenance
grouping, FastAPI for the API layer, and React with DHTMLX Gantt for
visualization.

## What the system is designed to solve

Railway maintenance requests may compete for the same block sections,
track possession windows, crews, plant, and operating capacity. The
intended scheduling engine consolidates compatible maintenance work into
candidate blocks while accounting for train movements, resource limits,
priorities, precedence and travel legs.

The shared contract models:

-   Planning horizons and timezone information
-   Block sections and their adjacency
-   Train movements, headways and flexibility
-   Maintenance requests and priority inputs
-   Crew and plant capacity
-   Solver configuration
-   Selected and deferred work packages
-   Traveling job schedules
-   Conflicts and KPIs
-   Gantt tasks for visualization
-   ML enrichment containing risk, duration, confidence and cluster
    information

These interfaces are defined centrally in `contracts.ts`.

## Repository structure

``` text
KrayaSetu_AI/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── GanttDashboard.tsx
│   │   ├── StatusBanner.tsx
│   │   ├── main.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── types/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig*.json
│
├── ml_pipeline/
│   ├── clustering.py
│   ├── models.py
│   └── predict.py
│
├── contracts.ts
├── tests/
├── test-results/
├── artifacts/
├── playwright.config.ts
├── package.json
├── package-lock.json
├── B_BLOCK_COMPLETION.md
├── SIH26027_Master_Manifest.md
└── SIH26027_Master_Spec_v2.md
```

The repository root currently contains the ML pipeline, frontend,
contracts, tests and specification artifacts. `node_modules` is also
present in the repository tree.

## Core architecture

``` text
                    Maintenance / Railway Data
                              |
                +-------------+-------------+
                |                           |
                v                           v
        Topology + Clustering          ML Enrichment
       NetworkX + DBSCAN         XGBoost + SHAP + P80
                |                           |
                +-------------+-------------+
                              |
                              v
                       Scheduling Engine
                         OR Tools CP SAT
                              |
                 +------------+------------+
                 |                         |
                 v                         v
         Selected Work              Deferred Work
         + Resources               + Reason Codes
                 |                         |
                 +------------+------------+
                              |
                              v
                         ScheduleResponse
                              |
                              v
                    React + DHTMLX Gantt
```

The master manifest specifies OR Tools CP SAT as the solver, precomputed
candidate blocks as the preferred bundling approach, a two pass
lexicographic objective, five minute discretization, and explicit solver
guardrails including an eight second solve limit, eight workers,
deterministic seed 42 and a conflict limit. These are documented
architecture decisions rather than all being implemented in the files
currently inspected.

## Frontend

The frontend is a React application written in TypeScript and built with
Vite. Its package configuration uses:

-   React 19
-   React DOM
-   TypeScript
-   Vite
-   TanStack React Query
-   DHTMLX Gantt
-   Oxlint

The available npm scripts are:

``` bash
npm run dev
npm run build
npm run lint
npm run preview
```

### Dashboard

`frontend/src/App.tsx` creates the dashboard and uses TanStack React
Query to POST to `/plan`.

The UI expects a `ScheduleResponse` and displays:

-   A `SIH26027 Block Bundling Dashboard` heading
-   A solver status banner
-   A large Gantt area
-   Gantt tasks when returned by the `/plan` response

Loading and error states are explicitly handled in the component.

### Gantt visualization

`frontend/src/GanttDashboard.tsx` renders the returned `GanttTask[]`
through DHTMLX Gantt.

Configured features include:

-   Tree based task or block section hierarchy
-   Start and end columns
-   Hourly time scale
-   Read only display mode
-   Priority based task styling
-   Explicit conflict markers
-   Cleanup of the Gantt instance on component unmount
-   Strictly typed dispatcher override payloads

The dashboard recognizes the following override types:

``` text
LOCK_POSSESSION
REJECT_CANDIDATE
```

The override function is currently a typed hook that logs the payload.
The code marks backend dispatch as a TODO, so the frontend hook should
not be described as a completed dispatcher write path.

### Solver status

`StatusBanner.tsx` translates solver states into human readable
messages:

``` text
OPTIMAL             -> Optimal Plan Found
FEASIBLE            -> Feasible Plan (Time Limit Reached)
INFEASIBLE         -> Infeasible Plan (Cannot Satisfy Constraints)
FALLBACK_HEURISTIC -> Fallback Heuristic Active
```

This gives the operator explicit visibility into whether the displayed
plan is optimal, merely feasible, infeasible, or produced by the
fallback heuristic.

## Shared data contract

`contracts.ts` is the central TypeScript boundary for the application.

Important interfaces include:

### Input

`ScheduleRequest` contains:

-   `PlanningHorizon`
-   `BlockSection[]`
-   `TrainMovement[]`
-   `MaintenanceRequest[]`
-   `ResourceCapacity`
-   optional `SolverConfig`
-   optional `DailyCapacityCalendar[]`

### Maintenance

`MaintenanceRequest` supports:

-   source system and defect type
-   single or traveling jobs
-   required block sections
-   duration, route legs and setup information
-   earliest and latest start
-   mandatory flag
-   crew and plant requirements
-   isolation groups
-   precedence constraints
-   priority inputs and score
-   optional ML enrichment

### Output

`ScheduleResponse` exposes:

-   schedule metadata
-   solver status
-   objective information
-   solve time
-   selected work packages
-   traveling job schedules
-   deferred work packages
-   Gantt tasks
-   conflicts
-   KPIs

The contract also defines reason codes for deferred work:

``` text
NO_FEASIBLE_WINDOW
CREW_CAPACITY_EXCEEDED
PLANT_CAPACITY_EXCEEDED
LOWER_PRIORITY
NO_CAPACITY_BEFORE_DEADLINE
```

## ML pipeline

The `ml_pipeline/` directory contains three focused components:

``` text
clustering.py
models.py
predict.py
```

The repository's Block B completion record identifies these as the
delivered ML foundation.

### Defect clustering

`clustering.py` uses DBSCAN on one dimensional corridor chainage.

Default parameters:

``` text
eps = 2.0
min_samples = 1
```

The implementation groups defects using `chainage_km` and returns
cluster IDs mapped to maintenance `work_id`s.

The documented architecture scopes this operation to one NetworkX
connected component at a time and uses clustering as part of maintenance
grouping and candidate block generation.

### Escalation risk model

`models.py` trains an XGBoost classifier for escalation risk.

The implementation uses:

-   Time sliced train and validation data with `shuffle=False`
-   `scale_pos_weight` for class imbalance
-   Binary logistic XGBoost classification
-   Precision recall AUC as the evaluation metric
-   Isotonic calibration through `CalibratedClassifierCV`

Current classifier settings include 300 estimators, depth 4 and learning
rate 0.05.

### Duration model

The same module trains an XGBoost regressor targeting the P80 duration
quantile.

The model uses:

``` text
objective = reg:quantileerror
quantile_alpha = 0.80
```

A time sliced split is also used.

### Explainable prediction

`predict.py` combines the trained models with SHAP.

For escalation risk it:

1.  Builds a model input DataFrame
2.  Predicts risk probability
3.  Calculates SHAP values
4.  Identifies the strongest risk factors
5.  Applies the low confidence rule
6.  Applies a cold start fallback if inference fails

The low confidence band is:

``` text
0.40 <= probability <= 0.60
```

In this band the code uses the provided rule based risk score and labels
the result `low_model_confidence`.

### Duration safeguards

The duration predictor enforces:

``` python
MIN_POSSESSION_MINUTES = 15.0
```

The final model duration is therefore never lower than 15 minutes.

If the model fails, the implementation falls back to a supplied static
duration and marks the source as `static_fallback`.

### ML enrichment output

`generate_ml_enrichment()` combines risk and duration information into
the ML contract.

The enrichment can include:

``` text
escalation_risk_probability
escalation_risk_R
risk_confidence
top_risk_factors
predicted_duration_minutes
duration_quantile
duration_source
cluster_id
model_version
```

## Optimization architecture

The master manifest defines Google OR Tools CP SAT as the intended
scheduling solver.

The documented formulation uses:

-   Optional intervals for maintenance jobs and candidate blocks
-   Fixed intervals for train movements
-   `AddNoOverlap` for track exclusivity
-   `AddCumulative` for crew and plant capacity
-   Conditional constraints for precedence and other optional
    relationships
-   Warm starts through `AddHint`

The architecture explicitly prefers precomputed candidate blocks instead
of large pairwise reification. Mandatory jobs are intended to be modeled
as soft constraints with a large drop penalty so that conflicts can be
reported rather than producing an unexplained infeasible result.

The documented solve guardrails are:

``` text
max_time_in_seconds = 8
num_search_workers = 8
random_seed = 42
max_number_of_conflicts = 100000
```

The master manifest describes these as non negotiable demo safety
parameters.

## Scheduling outputs and explainability

The response contract is designed to support operational review rather
than returning a single schedule only.

A response can expose:

-   Which work packages were selected
-   Where and when they were scheduled
-   Which traveling job legs were assigned
-   Which jobs were deferred and why
-   Conflicts involving work packages
-   Gantt rendering data
-   Aggregate KPIs
-   Solver quality information
-   ML risk and duration enrichment

This allows the planning interface to show both a proposed plan and the
reasons surrounding difficult scheduling decisions.

## Testing and acceptance verification

The repository includes Playwright based end to end acceptance tests
under `tests/`.

`B_BLOCK_COMPLETION.md` records a successful acceptance run against the
React dashboard using an adversarial payload containing 50 tight
maintenance requests across 3 block sections.

The recorded checks include:

-   DHTMLX Gantt rendering under stress
-   Zero overlapping Gantt bars on the same block section track lane
-   Correct handling of the `FALLBACK_HEURISTIC` state
-   Avoidance of infinite loading behavior

The completion record also states that the ML layer was mapped to
`ml_pipeline/` and that its outputs align with the `MLEnrichment`
boundary in `contracts.ts`.

## Running the frontend

From the repository root:

``` bash
cd frontend
npm install
npm run dev
```

For a production build:

``` bash
npm run build
```

To preview the production build:

``` bash
npm run preview
```

The frontend expects a backend endpoint:

``` text
POST /plan
```

The current frontend code sends an empty JSON body and expects the
backend to return a `ScheduleResponse`.

## Current implementation boundaries

The repository contains a substantial frontend and ML foundation, but
the inspected files do not establish every part of the full target
architecture as production ready.

Notably:

-   The frontend `/plan` endpoint is assumed rather than implemented in
    the inspected frontend files.
-   Dispatcher override submission is typed but the backend dispatch
    call is marked TODO.
-   The repository contract defines the scheduling model, but the
    inspected code does not include a complete CP SAT implementation in
    the repository tree shown here.
-   The ML files provide training and inference functions, but the
    inspected repository does not establish a complete model training
    dataset, model registry or startup model loading process.
-   The master manifest describes FastAPI, SQLAlchemy, SQLite, RailRadar
    enrichment and several other components as architecture decisions,
    but those should be considered planned or repository documented
    unless corresponding implementation files are present.

## Technology stack

  Layer                    Technology
  ------------------------ -----------------------------
  Frontend                 React 19 + TypeScript
  Build                    Vite
  Data fetching            TanStack React Query
  Timeline                 DHTMLX Gantt
  Validation / contracts   TypeScript interfaces
  Optimization target      Google OR Tools CP SAT
  ML classification        XGBoost XGBClassifier
  ML regression            XGBoost quantile regression
  Explainability           SHAP TreeExplainer
  Clustering               scikit learn DBSCAN
  Topology architecture    NetworkX
  Linting                  Oxlint
  E2E testing              Playwright

The stack above combines technologies visible in the repository with the
documented target architecture in `SIH26027_Master_Manifest.md`.

## Project documents

`SIH26027_Master_Spec_v2.md` contains the detailed system specification.

`SIH26027_Master_Manifest.md` records the consolidated technology and
integration decisions.

`B_BLOCK_COMPLETION.md` records Block B acceptance verification and the
delivered ML layer.

## Project status

The repository currently demonstrates a working frontend visualization
layer, typed scheduling contract, ML prediction and clustering
components, and recorded Playwright acceptance verification.

The larger system is structured around an optimization driven
maintenance planning engine whose final integration depends on
connecting the backend solver, data ingestion, persistence, ML models
and frontend API boundary described in the repository specification.
