import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any, Dict

from app.schemas.contracts import ScheduleRequest, ScheduleResponse
from app.solver.cp_sat_core import solve_plan
from app.db.database import SessionLocal, DispatcherOverride
from app.data.producers import generate_mock_fixture
from app.planner.macro_planner import macro_bucket_plan
from app.schemas.contracts import MacroAllocationCalendar

app = FastAPI(title="SIH26027 Integrated Block Bundling Engine")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Bounded Executor for CP-SAT
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/plan", response_model=ScheduleResponse)
async def create_plan(request: ScheduleRequest):
    loop = asyncio.get_running_loop()
    try:
        # Wrap the synchronous CP-SAT solver in an asyncio executor with a hard timeout
        # max_time_in_seconds internally handles the clean exit, but wait_for ensures
        # the HTTP request doesn't hang forever if the thread hangs.
        response = await asyncio.wait_for(
            loop.run_in_executor(executor, solve_plan, request),
            timeout=request.solver_config.max_time_in_seconds + 5.0  # pad for overhead
        )
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Solver timeout exceeded")

class OverrideRequest(BaseModel):
    work_id: str
    override_type: str
    parameters: Dict[str, Any]

@app.post("/override")
def create_override(req: OverrideRequest, db: Session = Depends(get_db)):
    db_ovr = db.query(DispatcherOverride).filter(DispatcherOverride.work_id == req.work_id).first()
    if db_ovr:
        db_ovr.override_type = req.override_type
        db_ovr.parameters = req.parameters
    else:
        db_ovr = DispatcherOverride(
            work_id=req.work_id,
            override_type=req.override_type,
            parameters=req.parameters
        )
        db.add(db_ovr)
    db.commit()
    db.refresh(db_ovr)
    return {"status": "success", "work_id": db_ovr.work_id}

@app.get("/mock/request", response_model=ScheduleRequest)
def get_mock_request():
    return generate_mock_fixture()

@app.post("/macro-plan", response_model=MacroAllocationCalendar)
def create_macro_plan(request: ScheduleRequest):
    return macro_bucket_plan(request)

@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=False)
