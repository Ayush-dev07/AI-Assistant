from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())

class AgentMessage(BaseModel):

    message_id:     str             = Field(default_factory=_new_id)
    from_agent:     str             = Field(..., description="Sender identifier.")
    to_agent:       str             = Field(default="*", description="'*' = broadcast.")
    message_type:   str             = Field(..., description="Payload shape tag.")
    payload:        dict[str, Any]  = Field(default_factory=dict)
    timestamp:      str             = Field(default_factory=_now_iso)
    session_id:     str             = Field(default="default")
    correlation_id: str | None      = Field(default=None)

    model_config = {"extra": "ignore"}

    @field_validator("message_type")
    @classmethod
    def validate_message_type(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message_type cannot be empty.")
        return v.lower().strip()

    def reply(
        self,
        from_agent:   str,
        message_type: str,
        payload:      dict[str, Any],
        session_id:   str | None = None,
    ) -> "AgentMessage":
        return AgentMessage(
            from_agent     = from_agent,
            to_agent       = self.from_agent,
            message_type   = message_type,
            payload        = payload,
            session_id     = session_id or self.session_id,
            correlation_id = self.message_id,
        )

class TaskRequest(BaseModel):

    task_id:              str             = Field(default_factory=_new_id)
    goal:                 str
    context:              dict[str, Any]  = Field(default_factory=dict)
    priority:             int             = Field(default=5, ge=1, le=10)
    timeout_seconds:      int             = Field(default=300, ge=10)
    require_capabilities: list[str]       = Field(default_factory=list)
    session_id:           str             = Field(default="default")
    node_id:              str             = Field(default="")

    def to_message(self, from_agent: str) -> AgentMessage:
        return AgentMessage(
            from_agent   = from_agent,
            message_type = "task_request",
            payload      = self.model_dump(),
            session_id   = self.session_id,
        )

class TaskResult(BaseModel):

    task_id:      str            = Field(default_factory=_new_id)
    node_id:      str            = Field(default="")
    agent_type:   str            = Field(default="")
    session_id:   str            = Field(default="default")
    success:      bool           = True
    output:       Any            = None
    error:        str | None     = None
    confidence:   float          = Field(default=1.0, ge=0.0, le=1.0)
    duration_ms:  int            = 0
    total_tokens: int            = 0
    cost_usd:     float          = 0.0
    iterations:   int            = 0

    def to_message(self, from_agent: str) -> AgentMessage:
        return AgentMessage(
            from_agent   = from_agent,
            message_type = "task_result",
            payload      = self.model_dump(),
            session_id   = self.session_id,
        )

    @classmethod
    def from_agent_result(
        cls,
        result:   Any,  
        node_id:  str = "",
    ) -> "TaskResult":
        return cls(
            task_id      = getattr(result, "task_id",      ""),
            node_id      = node_id,
            agent_type   = getattr(result, "agent_type",   ""),
            session_id   = getattr(result, "session_id",   "default"),
            success      = getattr(result, "success",       False),
            output       = getattr(result, "output",        None),
            error        = getattr(result, "error",         None),
            confidence   = getattr(result, "confidence",    0.0),
            duration_ms  = getattr(result, "duration_ms",  0),
            total_tokens = getattr(result, "total_tokens", 0),
            cost_usd     = getattr(result, "cost_usd",     0.0),
            iterations   = getattr(result, "iterations",   0),
        )

class AgentCapability(BaseModel):

    agent_id:         str        = Field(default_factory=_new_id)
    agent_type:       str        = ""
    capabilities:     list[str]  = Field(default_factory=list)
    is_busy:          bool       = False
    tasks_completed:  int        = 0
    session_id:       str        = "default"
    max_iterations:   int        = 6

    def to_message(self, from_agent: str) -> AgentMessage:
        return AgentMessage(
            from_agent   = from_agent,
            message_type = "capability",
            payload      = self.model_dump(),
            session_id   = self.session_id,
        )

    def mark_busy(self) -> "AgentCapability":
        return self.model_copy(update={"is_busy": True})

    def mark_free(self) -> "AgentCapability":
        return self.model_copy(
            update={"is_busy": False, "tasks_completed": self.tasks_completed + 1}
        )

class ApprovalRequest(BaseModel):

    request_id:          str             = Field(default_factory=_new_id)
    action:              str             = ""
    tool_name:           str             = ""
    parameters:          dict[str, Any]  = Field(default_factory=dict)
    risk_level:          str             = "medium"
    auto_reject_after_s: int             = Field(default=60, ge=5, le=3600)
    session_id:          str             = "default"
    agent_type:          str             = ""
    goal_context:        str             = ""

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        v = v.lower()
        return v if v in allowed else "medium"

    def to_message(self, from_agent: str) -> AgentMessage:
        return AgentMessage(
            from_agent   = from_agent,
            to_agent     = "human",
            message_type = "approval_request",
            payload      = self.model_dump(),
            session_id   = self.session_id,
        )

    def channel(self) -> str:
        return "approval:requests"

    def response_channel(self) -> str:
        return f"approval:response:{self.request_id}"


class ApprovalResponse(BaseModel):
    request_id:  str  = ""
    approved:    bool = False
    reason:      str  = ""
    decided_by:  str  = "human"
    session_id:  str  = "default"

    def to_message(self, from_agent: str) -> AgentMessage:
        return AgentMessage(
            from_agent     = from_agent,
            to_agent       = "approval_gate",
            message_type   = "approval_response",
            payload        = self.model_dump(),
            session_id     = self.session_id,
            correlation_id = self.request_id,
        )

class Channels:

    @staticmethod
    def agent_events(session_id: str) -> str:
        return f"agent:events:{session_id}"

    @staticmethod
    def agent_capabilities() -> str:
        return "agent:capabilities"

    @staticmethod
    def agent_tasks(agent_type: str) -> str:
        return f"agent:tasks:{agent_type}"

    @staticmethod
    def approval_requests() -> str:
        return "approval:requests"

    @staticmethod
    def approval_response(request_id: str) -> str:
        return f"approval:response:{request_id}"