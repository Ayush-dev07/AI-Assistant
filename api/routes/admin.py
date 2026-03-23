from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

class HealthResponse(BaseModel):
    status:       str
    version:      str
    uptime_s:     float
    bus_mode:     str
    worker_count: int
    queue_size:   int

class AuditVerifyResponse(BaseModel):
    intact:          bool
    entries_checked: int
    broken_at:       str | None
    error:           str | None
    verified_at:     str

class QueueStatsResponse(BaseModel):
    qsize:            int
    maxsize:          int
    workers:          int
    active_workers:   int
    results_cached:   int
    running:          bool

class CircuitResetResponse(BaseModel):
    previous_state: str
    current_state:  str
    message:        str

class SecurityCounters(BaseModel):
    session_id:               str
    tool_calls_per_minute:    float
    failed_calls_per_minute:  float
    cost_per_hour_usd:        float
    memory_writes_per_minute: float
    circuit_state:            str
    circuit_open_reason:      str

@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness probe",
    description    = (
        "Returns HTTP 200 if the server is running. "
        "No authentication required. "
        "Used by systemd, nginx, and load balancer health checks."
    ),
)
async def health_check() -> HealthResponse:
    from api.main import _startup_time, _bus, _task_queue
    import os

    uptime = 0.0
    if _startup_time is not None:
        uptime = (datetime.now(tz=timezone.utc) - _startup_time).total_seconds()

    return HealthResponse(
        status       = "ok",
        version      = "2.0.0",
        uptime_s     = round(uptime, 1),
        bus_mode     = _bus.mode if _bus else "not_ready",
        worker_count = int(os.getenv("WORKER_COUNT", "2")),
        queue_size   = _task_queue._q.qsize() if _task_queue else 0,
    )

@router.get(
    "/audit/verify",
    response_model = AuditVerifyResponse,
    summary        = "Verify SHA-256 audit chain integrity",
    description    = (
        "Recomputes the SHA-256 chain across all audit log entries. "
        "Returns intact=true if no tampering is detected. "
        "Returns intact=false and broken_at (first 16 chars of the "
        "tampered entry_hash) if the chain is broken. "
        "This is an O(n) scan — may be slow for very large audit logs."
    ),
)
async def verify_audit_chain(
    session_id: str | None = Query(default=None,
                                   description="Verify only entries for this session."),
) -> AuditVerifyResponse:
    from api.main import get_security_stack
    sec   = get_security_stack()
    audit = sec.get("audit")

    if audit is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "AuditLog not configured (Supabase not available).",
        )

    result = await audit.verify_chain(session_id=session_id)
    log.info(
        "api_audit_verify",
        intact   = result.intact,
        entries  = result.entries_checked,
        session  = session_id,
    )
    return AuditVerifyResponse(
        intact          = result.intact,
        entries_checked = result.entries_checked,
        broken_at       = result.broken_at,
        error           = result.error,
        verified_at     = datetime.now(tz=timezone.utc).isoformat(),
    )

@router.get(
    "/queue/stats",
    response_model = QueueStatsResponse,
    summary        = "Task queue statistics",
    description    = "Current queue depth, active worker count, and cached result count.",
)
async def queue_stats() -> QueueStatsResponse:
    from api.main import get_task_queue
    queue = get_task_queue()
    s     = queue.queue_stats()
    return QueueStatsResponse(
        qsize          = s["qsize"],
        maxsize        = s["maxsize"],
        workers        = s["workers"],
        active_workers = s["active_workers"],
        results_cached = s["results_cached"],
        running        = s["running"],
    )

@router.post(
    "/circuit/reset",
    response_model = CircuitResetResponse,
    summary        = "Manually reset the anomaly circuit breaker",
    description    = (
        "Reset the circuit breaker from OPEN or HALF_OPEN back to CLOSED. "
        "Use after reviewing the anomaly that triggered the trip. "
        "Does NOT reset counters — see POST /admin/circuit/reset-session/{id} "
        "to also clear the sliding window counters for a session."
    ),
)
async def reset_circuit_breaker() -> CircuitResetResponse:
    from api.main import get_security_stack
    sec      = get_security_stack()
    anomaly  = sec.get("anomaly")

    if anomaly is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "AnomalyDetector not configured.",
        )

    prev_state = anomaly.circuit_state.value
    anomaly.reset_circuit()
    curr_state = anomaly.circuit_state.value

    log.info("api_circuit_reset", prev=prev_state, curr=curr_state)
    return CircuitResetResponse(
        previous_state = prev_state,
        current_state  = curr_state,
        message        = f"Circuit reset from {prev_state} → {curr_state}.",
    )

@router.post(
    "/circuit/reset-session/{session_id}",
    summary     = "Clear anomaly counters for a session",
    description = (
        "Clear the sliding window counters for a specific session. "
        "Use before intentional high-activity operations (bulk imports, evals) "
        "that would otherwise trip the circuit."
    ),
)
async def reset_session_counters(session_id: str) -> dict[str, str]:
    from api.main import get_security_stack
    sec     = get_security_stack()
    anomaly = sec.get("anomaly")

    if anomaly is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "AnomalyDetector not configured.",
        )

    await anomaly.reset_session(session_id)
    return {"session_id": session_id, "status": "counters_cleared"}

@router.get(
    "/security/counters",
    response_model = SecurityCounters,
    summary        = "Current anomaly counters for a session",
    description    = (
        "Read current sliding window counter values without triggering "
        "a threshold check.  Useful for monitoring dashboards."
    ),
)
async def get_security_counters(
    session_id: str = Query(default="default", description="Session to inspect."),
) -> SecurityCounters:
    from api.main import get_security_stack
    sec     = get_security_stack()
    anomaly = sec.get("anomaly")

    if anomaly is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "AnomalyDetector not configured.",
        )

    counters = await anomaly.get_counters(session_id)
    return SecurityCounters(
        session_id                = session_id,
        tool_calls_per_minute     = counters.get("tool_calls_per_minute",    0.0),
        failed_calls_per_minute   = counters.get("failed_calls_per_minute",  0.0),
        cost_per_hour_usd         = counters.get("cost_per_hour_usd",        0.0),
        memory_writes_per_minute  = counters.get("memory_writes_per_minute", 0.0),
        circuit_state             = anomaly.circuit_state.value,
        circuit_open_reason       = anomaly.open_reason,
    )