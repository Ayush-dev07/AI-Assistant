from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    try:
        from jose import jwt as pyjwt
        _JWT_AVAILABLE = True
    except ImportError:
        _JWT_AVAILABLE = False

from core.main_configuration import settings
from core.logging import get_logger
from agents.bus import MessageBus, get_bus
from agents.protocol import Channels
from workers.queue import TaskQueue, make_agent_factory, Priority, TaskQueueFull

log = get_logger(__name__)

_task_queue:       TaskQueue | None = None
_bus:              MessageBus | None = None
_episodic:         Any | None = None   
_short_term_cache: dict[str, Any] = {}   
_long_memory:      Any | None = None   
_security_stack:   dict[str, Any] = {}
_startup_time:     datetime | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_queue, _bus, _episodic, _long_memory, _security_stack, _startup_time

    log.info("api_startup_begin", version="2.0.0")
    _startup_time = datetime.now(tz=timezone.utc)

    from core.llm.router import create_provider_from_env
    llm = create_provider_from_env()
    log.info("api_llm_ready", provider=llm.model_name)
 
    try:
        from memory.episodic import EpisodicMemory
        _episodic = EpisodicMemory()
        await _episodic.initialize()
        log.info("api_episodic_memory_ready")
    except Exception as exc:
        log.warning("api_episodic_memory_unavailable", error=str(exc))
        _episodic = None

    try:
        from memory.long_term import LongTermMemory
        _long_memory = LongTermMemory()
        log.info("api_long_memory_ready")
    except Exception as exc:
        log.warning("api_long_memory_unavailable", error=str(exc))
        _long_memory = None

    from Security import build_security_stack
    _security_stack = build_security_stack(llm=llm, long_memory=_long_memory)
  
    audit = _security_stack.get("audit")
    if audit is not None:
        try:
            await audit.initialize()
            log.info("api_audit_log_ready")
        except Exception as exc:
            log.warning("api_audit_log_init_failed", error=str(exc))

    _bus = get_bus()
    log.info("api_message_bus_ready", mode=_bus.mode)

    _task_queue = TaskQueue(
        episodic     = _episodic,
        bus          = _bus,
        worker_count = int(os.getenv("WORKER_COUNT", "2")),
    )
    agent_factory = make_agent_factory(llm)
    await _task_queue.start(agent_factory)

    recovered = await _task_queue.recover_pending_tasks()
    if recovered:
        log.info("api_tasks_recovered", count=recovered)

    log.info(
        "api_startup_complete",
        workers   = int(os.getenv("WORKER_COUNT", "2")),
        bus_mode  = _bus.mode,
        episodic  = _episodic is not None,
        long_mem  = _long_memory is not None,
    )
    yield  
    log.info("api_shutdown_begin")
    if _task_queue is not None:
        await _task_queue.stop(drain_timeout_s=30.0)
    if audit is not None:
        await audit.close()
    if _bus is not None:
        await _bus.close()
    if _long_memory is not None:
        await _long_memory.close()

    log.info("api_shutdown_complete")

app = FastAPI(
    title       = "SuperAgent API",
    version     = "2.0.0",
    description = (
        "Production AI agent API. POST /tasks to submit goals, "
        "poll GET /tasks/{id} for results, "
        "stream events via WebSocket /ws/{session_id}."
    ),
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    openapi_url = "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

_JWT_ALGORITHM = "HS256"
_bearer        = HTTPBearer(auto_error=False)

def _get_secret() -> str:
    return settings.secret_key.get_secret_value()

def create_access_token(
    session_id:   str,
    role:         str  = "user",
    expires_mins: int  = 60 * 24,   
) -> str:
    if not _JWT_AVAILABLE:
        raise RuntimeError(
            "JWT library not installed. "
            "Run: poetry add 'python-jose[cryptography]'"
        )
    payload = {
        "sub":  session_id,
        "role": role,
        "exp":  datetime.now(tz=timezone.utc) + timedelta(minutes=expires_mins),
        "iat":  datetime.now(tz=timezone.utc),
    }
    return pyjwt.encode(payload, _get_secret(), algorithm=_JWT_ALGORITHM)

def _decode_token(token: str) -> dict[str, Any]:
    if not _JWT_AVAILABLE:
        
        log.warning("jwt_lib_missing_auth_disabled")
        return {"sub": "default", "role": "user"}
    try:
        return pyjwt.decode(token, _get_secret(), algorithms=[_JWT_ALGORITHM])
    except Exception:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or expired token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

async def get_current_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    token_query: str | None = Query(default=None, alias="token"),
) -> str:
    raw_token: str | None = None

    if credentials is not None:
        raw_token = credentials.credentials
    elif token_query is not None:
        raw_token = token_query

    if raw_token is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Missing authentication token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_token(raw_token)
    return payload.get("sub", "default")

async def get_admin_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Missing authentication token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Admin role required.",
        )
    return payload.get("sub", "admin")

def get_task_queue() -> TaskQueue:
    if _task_queue is None:
        raise HTTPException(status_code=503, detail="Task queue not initialised.")
    return _task_queue

def get_bus_dep() -> MessageBus:
    if _bus is None:
        raise HTTPException(status_code=503, detail="Message bus not initialised.")
    return _bus

def get_episodic() -> Any:
    if _episodic is None:
        raise HTTPException(status_code=503, detail="Episodic memory not configured.")
    return _episodic

def get_long_memory() -> Any:
    if _long_memory is None:
        raise HTTPException(status_code=503, detail="Long-term memory not configured.")
    return _long_memory

def get_security_stack() -> dict[str, Any]:
    return _security_stack

def get_short_term_memory(session_id: str) -> Any:
    if session_id not in _short_term_cache:
        try:
            from memory.short_term import ShortTermMemory
            _short_term_cache[session_id] = ShortTermMemory(session_id=session_id)
        except Exception:
            return None
    return _short_term_cache.get(session_id)

from api.routes import tasks as _tasks_module
from api.routes import sessions as _sessions_module
from api.routes import memory as _memory_module
from api.routes import admin as _admin_module

app.include_router(_tasks_module.router,    prefix="/tasks",    tags=["Tasks"])
app.include_router(_sessions_module.router, prefix="/sessions", tags=["Sessions"])
app.include_router(_memory_module.router,   prefix="/memory",   tags=["Memory"])
app.include_router(_admin_module.router,    prefix="/admin",    tags=["Admin"])

@app.websocket("/ws/{session_id}")
async def websocket_stream(
    ws:         WebSocket,
    session_id: str,
    token:      str | None = Query(default=None),
) -> None:
    if token is None:
        await ws.close(code=4401, reason="Missing token.")
        return
    try:
        payload = _decode_token(token)
        token_session = payload.get("sub", "default")
    except HTTPException:
        await ws.close(code=4401, reason="Invalid token.")
        return

    if token_session != session_id and payload.get("role") != "admin":
        await ws.close(code=4403, reason="Token session mismatch.")
        return

    await ws.accept()
    log.info("ws_client_connected", session_id=session_id)

    channel = Channels.agent_events(session_id)
    bus     = get_bus_dep()

    try:
        async for message in bus.subscribe(channel):
            try:
                await ws.send_text(message.model_dump_json())
            except WebSocketDisconnect:
                break
            except Exception as exc:
                log.debug("ws_send_error", error=str(exc))
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("ws_stream_error", session_id=session_id, error=str(exc))
    finally:
        log.info("ws_client_disconnected", session_id=session_id)

from pydantic import BaseModel as _BaseModel

class ApprovalDecision(_BaseModel):
    approved:   bool
    reason:     str  = ""
    decided_by: str  = "human"

@app.post(
    "/approvals/{request_id}",
    summary     = "Submit a HITL approval decision",
    description = (
        "Called by dashboard, Telegram bot, or Slack bot when the human "
        "clicks Approve or Reject. Publishes the decision to the Upstash "
        "approval:responses channel so the waiting ApprovalGate receives it."
    ),
    tags = ["HITL"],
)
async def post_approval(
    request_id:  str,
    body:        ApprovalDecision,
    session_id:  str = Depends(get_current_session),
) -> dict[str, str]:
    from agents.protocol import AgentMessage, ApprovalResponse

    bus = get_bus_dep()
    response = ApprovalResponse(
        request_id = request_id,
        approved   = body.approved,
        reason     = body.reason,
        decided_by = body.decided_by,
        session_id = session_id,
    )
    msg = response.to_message(from_agent=f"api:{session_id}") 
    await bus.publish("approval:responses", msg)

    log.info(
        "approval_decision_delivered",
        request_id = request_id,
        approved   = body.approved,
        decided_by = body.decided_by,
    )
    return {"status": "delivered", "request_id": request_id}

class TokenRequest(_BaseModel):
    session_id:   str  = "default"
    role:         str  = "user"
    expires_mins: int  = 1440   

@app.post(
    "/auth/token",
    summary     = "Generate a JWT for a session (development helper)",
    description = (
        "Generates a signed JWT for a session_id. "
        "In production, replace with your identity provider. "
        "Protected by the DEV_TOKEN env var — set it to a strong secret "
        "and pass it as ?dev_key=... to prevent unauthorised token minting."
    ),
    tags = ["Auth"],
)
async def create_token(
    body:    TokenRequest,
    dev_key: str | None = Query(default=None),
) -> dict[str, str]:
    expected = os.getenv("DEV_TOKEN")
    if not expected:
        raise HTTPException(
            status_code = 404,
            detail      = "Token endpoint disabled. Set DEV_TOKEN in .env to enable.",
        )
    if dev_key != expected:
        raise HTTPException(
            status_code = 401,
            detail      = "Invalid dev_key.",
        )
    token = create_access_token(
        session_id   = body.session_id,
        role         = body.role,
        expires_mins = body.expires_mins,
    )
    return {
        "access_token": token,
        "token_type":   "bearer",
        "session_id":   body.session_id,
        "expires_mins": str(body.expires_mins),
    }