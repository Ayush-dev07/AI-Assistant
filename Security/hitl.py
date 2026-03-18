from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agents.bus import MessageBus, get_bus
from agents.protocol import AgentMessage, ApprovalRequest, ApprovalResponse, Channels
from core.logging import get_logger

log = get_logger(__name__)

ACTIONS_REQUIRING_APPROVAL: dict[str, list[str]] = {
    "api_caller":    ["POST", "PUT", "DELETE", "PATCH"],
    "filesystem":    ["write", "append", "delete", "move"],
    "code_executor": ["*"],
    "install_skill": ["*"],
}

PRE_APPROVED_ACTIONS: dict[str, list[str]] = {
    "api_caller":  ["GET", "HEAD", "OPTIONS"],
    "filesystem":  ["read", "list", "info", "parse"],
    "browser":     ["*"],
}


def requires_approval(tool_name: str, action: str) -> bool:
    pre = PRE_APPROVED_ACTIONS.get(tool_name, [])
    if "*" in pre or action.upper() in [p.upper() for p in pre]:
        return False

    required = ACTIONS_REQUIRING_APPROVAL.get(tool_name, [])
    if not required:
        return False

    if "*" in required:
        return True

    return action.upper() in [r.upper() for r in required]

@dataclass
class ApprovalAuditEntry:
    request_id:   str
    tool_name:    str
    action:       str
    session_id:   str
    agent_type:   str
    risk_level:   str
    approved:     bool
    decided_by:   str      
    reason:       str
    wait_ms:      int     
    timestamp:    str      = field(default_factory=lambda: _iso_now())


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()

class PolicyEngine:

    def __init__(self) -> None:
        self._rules: list[tuple[str, str, str | None, bool]] = []

    def allow(self, tool_name: str, action: str, path_prefix: str | None = None) -> None:
        """Add an auto-approve rule."""
        self._rules.append((tool_name.lower(), action.upper(), path_prefix, True))

    def deny(self, tool_name: str, action: str, path_prefix: str | None = None) -> None:
        """Add an auto-reject rule."""
        self._rules.append((tool_name.lower(), action.upper(), path_prefix, False))

    def evaluate(self, req: ApprovalRequest) -> bool | None:
        for rule_tool, rule_action, rule_prefix, decision in self._rules:
            if rule_tool != req.tool_name.lower():
                continue
            if rule_action != "*" and rule_action != req.action.upper():
                continue
            if rule_prefix is not None:
                path = str(req.parameters.get("path", ""))
                if not path.startswith(rule_prefix):
                    continue
            return decision
        return None

class ApprovalGate:

    def __init__(
        self,
        bus:                MessageBus | None = None,
        policy:             PolicyEngine | None = None,
        session_id:         str = "default",
        default_timeout_s:  float = 60.0,
    ) -> None:
        self._bus             = bus or get_bus()
        self._policy          = policy or PolicyEngine()
        self._session_id      = session_id
        self._default_timeout = default_timeout_s

        self._pending:   dict[str, asyncio.Event] = {}
        self._decisions: dict[str, ApprovalResponse] = {}

        self._audit_log: list[ApprovalAuditEntry] = []

        self._listener_task: asyncio.Task | None = None

    async def request_approval(
        self,
        req: ApprovalRequest,
    ) -> bool:
        start_ms = time.monotonic()
        timeout  = req.auto_reject_after_s or self._default_timeout

        policy_decision = self._policy.evaluate(req)
        if policy_decision is not None:
            decided_by = "auto_policy"
            self._write_audit(
                req        = req,
                approved   = policy_decision,
                decided_by = decided_by,
                reason     = "Matched policy rule.",
                wait_ms    = 0,
            )
            log.info(
                "hitl_policy_decision",
                request_id = req.request_id,
                approved   = policy_decision,
                tool       = req.tool_name,
                action     = req.action,
            )
            return policy_decision

        event = asyncio.Event()
        self._pending[req.request_id] = event

        message = req.to_message(from_agent=f"hitl-{self._session_id}")
        await self._bus.publish(Channels.approval_requests(), message)

        log.info(
            "hitl_approval_requested",
            request_id   = req.request_id,
            tool         = req.tool_name,
            action       = req.action,
            risk_level   = req.risk_level,
            timeout_s    = timeout,
            session_id   = self._session_id,
            agent_type   = req.agent_type,
        )

        await self._ensure_listener()

        approved   = False
        decided_by = "auto_timeout"
        reason     = ""

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            response   = self._decisions.get(req.request_id)
            approved   = response.approved   if response else False
            decided_by = response.decided_by if response else "auto_timeout"
            reason     = response.reason     if response else ""
        except asyncio.TimeoutError:
            approved   = False
            decided_by = "auto_timeout"
            reason     = f"No response within {timeout:.0f}s — auto-rejected."
            log.warning(
                "hitl_approval_timeout",
                request_id = req.request_id,
                timeout_s  = timeout,
                session_id = self._session_id,
            )
        finally:
            self._pending.pop(req.request_id, None)

        wait_ms = int((time.monotonic() - start_ms) * 1000)

        self._write_audit(
            req        = req,
            approved   = approved,
            decided_by = decided_by,
            reason     = reason,
            wait_ms    = wait_ms,
        )

        log.info(
            "hitl_approval_resolved",
            request_id = req.request_id,
            approved   = approved,
            decided_by = decided_by,
            wait_ms    = wait_ms,
            session_id = self._session_id,
        )

        return approved

    async def respond(
        self,
        request_id: str,
        approved:   bool,
        reason:     str = "",
        decided_by: str = "human",
    ) -> bool:
        response = ApprovalResponse(
            request_id = request_id,
            approved   = approved,
            reason     = reason,
            decided_by = decided_by,
            session_id = self._session_id,
        )
        self._decisions[request_id] = response

        event = self._pending.get(request_id)
        if event is None:
            log.warning(
                "hitl_respond_no_pending",
                request_id = request_id,
                approved   = approved,
            )
            return False

        event.set()
        return True

    async def _ensure_listener(self) -> None:
        if (
            self._listener_task is not None
            and not self._listener_task.done()
        ):
            return

        self._listener_task = asyncio.create_task(
            self._response_listener(),
            name=f"hitl-listener-{self._session_id}",
        )

    async def _response_listener(self) -> None:
        channel = "approval:responses"
        log.debug("hitl_listener_started", channel=channel, session_id=self._session_id)

        try:
            async for msg in self._bus.subscribe(channel):
                if msg.message_type != "approval_response":
                    continue

                try:
                    response = ApprovalResponse(**msg.payload)
                except Exception as exc:
                    log.warning(
                        "hitl_listener_bad_response",
                        error = str(exc),
                        raw   = str(msg.payload)[:120],
                    )
                    continue

                if response.request_id in self._pending:
                    await self.respond(
                        request_id = response.request_id,
                        approved   = response.approved,
                        reason     = response.reason,
                        decided_by = response.decided_by,
                    )

        except Exception as exc:
            log.warning(
                "hitl_listener_error",
                error      = str(exc),
                session_id = self._session_id,
            )

    def pending_requests(self) -> list[str]:
        return list(self._pending.keys())

    def is_pending(self, request_id: str) -> bool:
        return request_id in self._pending

    def get_audit_log(self) -> list[ApprovalAuditEntry]:
        return list(self._audit_log)

    def get_session_audit(self, session_id: str) -> list[ApprovalAuditEntry]:
        return [e for e in self._audit_log if e.session_id == session_id]

    def _write_audit(
        self,
        req:        ApprovalRequest,
        approved:   bool,
        decided_by: str,
        reason:     str,
        wait_ms:    int,
    ) -> None:
        entry = ApprovalAuditEntry(
            request_id = req.request_id,
            tool_name  = req.tool_name,
            action     = req.action,
            session_id = req.session_id,
            agent_type = req.agent_type,
            risk_level = req.risk_level,
            approved   = approved,
            decided_by = decided_by,
            reason     = reason,
            wait_ms    = wait_ms,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    async def close(self) -> None:
        for request_id in list(self._pending.keys()):
            await self.respond(
                request_id = request_id,
                approved   = False,
                reason     = "Gate closed — auto-rejected.",
                decided_by = "auto_timeout",
            )

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

def build_request(
    tool_name:    str,
    action:       str,
    parameters:   dict[str, Any],
    session_id:   str   = "default",
    agent_type:   str   = "",
    goal_context: str   = "",
    timeout_s:    int   = 60,
    risk_level:   str | None = None,
) -> ApprovalRequest:
    safe_params = sanitize_parameters(parameters)
    inferred    = risk_level or _assess_risk(tool_name, action, parameters)

    return ApprovalRequest(
        request_id          = str(uuid.uuid4()),
        action              = action,
        tool_name           = tool_name,
        parameters          = safe_params,
        risk_level          = inferred,
        auto_reject_after_s = timeout_s,
        session_id          = session_id,
        agent_type          = agent_type,
        goal_context        = goal_context[:500],   # cap length
    )

_SECRET_KEY_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(api[_\-]?key|apikey)"),
    re.compile(r"(?i)(secret|token|password|passwd|pwd)"),
    re.compile(r"(?i)(authorization|auth[_\-]?header)"),
    re.compile(r"(?i)(private[_\-]?key|client[_\-]?secret)"),
    re.compile(r"(?i)(access[_\-]?key|access[_\-]?secret)"),
]

_SECRET_VALUE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^sk-[A-Za-z0-9]{20,}$"),              # OpenAI / Anthropic keys
    re.compile(r"^AIza[A-Za-z0-9_\-]{35}$"),            # Google API keys
    re.compile(r"^gsk_[A-Za-z0-9]{50,}$"),              # Groq keys
    re.compile(r"^Bearer\s+\S{20,}$"),                   # Bearer tokens
    re.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"), # JWT
    re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$"),          # Base64 blobs
]


def sanitize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    return _redact_recursive(copy.deepcopy(params), "")


def _redact_recursive(obj: Any, key_name: str) -> Any:
    if isinstance(obj, dict):
        return {k: _redact_recursive(v, k) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_redact_recursive(item, "") for item in obj]

    if isinstance(obj, str):
        for pat in _SECRET_KEY_PATTERNS:
            if pat.search(key_name):
                return "[REDACTED]"
        if len(obj) >= 20:
            for pat in _SECRET_VALUE_PATTERNS:
                if pat.search(obj):
                    return "[REDACTED]"

    return obj

_HIGH_RISK_ACTIONS: frozenset[str] = frozenset({
    "DELETE", "delete",       # data deletion
    "DROP",   "drop",         # database operations
    "*",                      # wildcard (code_executor)
})

_MEDIUM_RISK_TOOLS: frozenset[str] = frozenset({
    "api_caller",   # external side effects
    "filesystem",   # local file changes
    "install_skill", # modifies the running agent
})


def _assess_risk(tool_name: str, action: str, parameters: dict[str, Any]) -> str:
    action_upper = action.upper()

    # HIGH: destructive actions or arbitrary code
    if action_upper in ("DELETE", "DROP", "*") or action in ("*",):
        return "high"
    if tool_name in ("code_executor", "install_skill"):
        return "high"

    # HIGH: external API calls to unknown domains
    if tool_name == "api_caller":
        url = str(parameters.get("url", ""))
        if action_upper in ("POST", "PUT", "PATCH"):
            return "medium"
        return "low"

    # MEDIUM: filesystem writes
    if tool_name == "filesystem":
        if action_upper in ("WRITE", "APPEND", "MOVE"):
            return "medium"
        if action_upper == "DELETE":
            return "high"
        return "low"

    # LOW: read-only operations
    return "low"

_gate_instance: ApprovalGate | None = None


def get_gate(
    bus:        MessageBus | None = None,
    policy:     PolicyEngine | None = None,
    session_id: str = "default",
) -> ApprovalGate:
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = ApprovalGate(
            bus        = bus or get_bus(),
            policy     = policy,
            session_id = session_id,
        )
    return _gate_instance


def reset_gate() -> None:
    global _gate_instance
    _gate_instance = None