from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.logging import METRICS, get_logger
from core.loop import AgentContext, AgentLoop, AgentLoopError
from memory.short_term import ShortTermMemory

if TYPE_CHECKING:
    from core.llm.base import LLMProvider

log = get_logger(__name__)


class _InstrumentedAgentLoop(AgentLoop):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_iterations:   int   = 0
        self.last_total_tokens: int   = 0
        self.last_cost_usd:     float = 0.0

    async def run(
        self,
        goal: str,
        context_override: dict[str, Any] | None = None,
    ) -> str:
        # Reset stats
        self.last_iterations   = 0
        self.last_total_tokens = 0
        self.last_cost_usd     = 0.0

        ctx_kwargs: dict[str, Any] = {
            "goal":           goal,
            "session_id":     self._session_id,
            "max_iterations": self._max_iterations,
        }
        if context_override:
            ctx_kwargs.update(context_override)

        result = await super().run(goal, context_override=context_override)

        return result

@dataclass
class AgentResult:
    success:      bool
    output:       Any
    error:        str | None     = None
    agent_type:   str            = ""
    session_id:   str            = ""
    task_id:      str            = ""
    iterations:   int            = 0
    total_tokens: int            = 0
    cost_usd:     float          = 0.0
    duration_ms:  int            = 0
    confidence:   float          = 1.0
    raw_output:   str            = ""
    metadata:     dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        if not self.success:
            return f"[{self.agent_type} FAILED: {self.error or 'unknown error'}]"
        if isinstance(self.output, dict):
            import json
            text = json.dumps(self.output, ensure_ascii=False, indent=2)
        else:
            text = str(self.output)
        if len(text) > 3000:
            text = text[:3000] + "\n...[truncated — full result in Supabase]"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":      self.success,
            "output":       (
                self.output if isinstance(self.output, (str, dict, list, type(None)))
                else str(self.output)
            ),
            "error":        self.error,
            "agent_type":   self.agent_type,
            "session_id":   self.session_id,
            "task_id":      self.task_id,
            "iterations":   self.iterations,
            "total_tokens": self.total_tokens,
            "cost_usd":     self.cost_usd,
            "duration_ms":  self.duration_ms,
            "confidence":   self.confidence,
        }

class BaseAgent(ABC):
    agent_type:    str
    capabilities:  list[str]
    default_tools: list[str]
    system_prompt: str

    max_iterations:      int   = 6
    default_temperature: float = 0.3

    def __init__(
        self,
        llm:            "LLMProvider",
        tools:          dict[str, Any] | None = None,
        session_id:     str | None = None,
        memory:         ShortTermMemory | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self._llm        = llm
        self._session_id = session_id or f"{self.agent_type}-{str(uuid.uuid4())[:8]}"
        self._max_iter   = max_iterations if max_iterations is not None else self.max_iterations

        self._tools = tools if tools is not None else self._build_tools()

        self._memory = memory or ShortTermMemory(
            session_id=self._session_id,
            window_size=20,
        )

        self._loop = _InstrumentedAgentLoop(
            llm            = self._llm,
            tools          = self._tools,
            memory         = self._memory,
            max_iterations = self._max_iter,
            session_id     = self._session_id,
        )

        log.debug(
            "agent_initialized",
            agent_type = self.agent_type,
            session_id = self._session_id,
            llm_model  = llm.model_name,
            tools      = list(self._tools.keys()),
            max_iter   = self._max_iter,
        )

    async def run(
        self,
        task:    str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        task_id = str(uuid.uuid4())[:8]
        start   = time.monotonic()

        goal = self._build_goal(task, context)

        log.info(
            "agent_run_start",
            agent_type = self.agent_type,
            session_id = self._session_id,
            task_id    = task_id,
            goal_chars = len(goal),
            tools      = list(self._tools.keys()),
            llm        = self._llm.model_name,
        )
        METRICS.tasks_total.labels(status="started").inc()

        try:
            raw_answer  = await self._loop.run(goal)
            duration_ms = int((time.monotonic() - start) * 1000)

            parsed, confidence = self._parse_output(raw_answer)

            METRICS.tasks_total.labels(status="success").inc()
            log.info(
                "agent_run_complete",
                agent_type  = self.agent_type,
                session_id  = self._session_id,
                task_id     = task_id,
                duration_ms = duration_ms,
                confidence  = confidence,
            )

            return AgentResult(
                success     = True,
                output      = parsed,
                raw_output  = raw_answer,
                agent_type  = self.agent_type,
                session_id  = self._session_id,
                task_id     = task_id,
                duration_ms = duration_ms,
                confidence  = confidence,
            )

        except (AgentLoopError, Exception) as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            METRICS.tasks_total.labels(status="failed").inc()
            log.error(
                "agent_run_failed",
                agent_type  = self.agent_type,
                session_id  = self._session_id,
                task_id     = task_id,
                error       = str(exc),
                duration_ms = duration_ms,
            )
            return AgentResult(
                success     = False,
                output      = None,
                raw_output  = "",
                error       = str(exc),
                agent_type  = self.agent_type,
                session_id  = self._session_id,
                task_id     = task_id,
                duration_ms = duration_ms,
                confidence  = 0.0,
            )

    @abstractmethod
    def _parse_output(self, raw: str) -> tuple[Any, float]:
        """
        Parse the raw answer string from the ReAct loop.

        Returns:
            (structured_output, confidence_0_to_1)
        """

    def _build_goal(
        self,
        task:    str,
        context: dict[str, Any] | None,
    ) -> str:
        parts: list[str] = []

        parts.append(self.system_prompt)
        parts.append("")

        if context:
            parts.append("--- CONTEXT FROM PREVIOUS STEPS ---")
            for node_id, result_val in context.items():
                result_str = (
                    result_val.to_context_string()
                    if isinstance(result_val, AgentResult)
                    else str(result_val)
                )
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n...[truncated]"
                parts.append(f"[{node_id}]:\n{result_str}")
            parts.append("--- END CONTEXT ---")
            parts.append("")

        parts.append(f"Task: {task}")
        return "\n".join(parts)

    def _build_tools(self) -> dict[str, Any]:
        from tools.base import ToolRegistry
        available  = set(ToolRegistry.list_all())
        tools_dict: dict[str, Any] = {}

        for tool_name in self.default_tools:
            if tool_name in available:
                tools_dict[tool_name] = ToolRegistry.get_callable(tool_name)
            else:
                log.warning(
                    "agent_tool_not_registered",
                    agent_type = self.agent_type,
                    tool_name  = tool_name,
                    available  = sorted(available),
                )
        return tools_dict

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    @property
    def llm(self) -> "LLMProvider":
        return self._llm

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"type={self.agent_type!r} "
            f"llm={self._llm.model_name!r} "
            f"tools={list(self._tools.keys())}>"
        )

def try_parse_json(raw: str) -> tuple[dict | None, bool]:
    """
    Try to parse a string as JSON, stripping markdown code fences first.
    """
    import json
    import re

    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    for candidate in (stripped, raw.strip()):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except (json.JSONDecodeError, ValueError):
            continue
    return None, False


def extract_confidence(parsed: dict) -> float:
    raw = parsed.get("confidence")
    if raw is None:
        return 1.0
    try:
        val = float(raw)
        return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))
    except (TypeError, ValueError):
        return 1.0