from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError, field_validator

from core.logging import METRICS, get_logger
from core.llm.base import LLMMessage

if TYPE_CHECKING:
    from core.llm.base import LLMProvider
    from agents.base import BaseAgent, AgentResult

log = get_logger(__name__)

class TaskNode(BaseModel):
    id:          str
    goal:        str
    agent_type:  str       = "research"
    depends_on:  list[str] = Field(default_factory=list)
    status:      str       = "pending"
    result:      Any       = None   
    retry_count: int       = 0

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        allowed = {"research", "coding", "communication", "data"}
        v = v.lower().strip()
        if v not in allowed:
            for a in allowed:
                if a in v:
                    return a
            return "research" 
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"pending", "running", "done", "failed"}
        return v if v in allowed else "pending"

@dataclass
class OrchestratorResult:
    success:       bool
    final_answer:  str
    error:         str | None          = None
    session_id:    str                 = ""
    task_id:       str                 = ""
    node_results:  dict[str, Any]      = field(default_factory=dict)
    nodes_total:   int                 = 0
    nodes_done:    int                 = 0
    nodes_failed:  int                 = 0
    total_tokens:  int                 = 0
    cost_usd:      float               = 0.0
    duration_ms:   int                 = 0
    confidence:    float               = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":      self.success,
            "final_answer": self.final_answer,
            "error":        self.error,
            "session_id":   self.session_id,
            "task_id":      self.task_id,
            "nodes_total":  self.nodes_total,
            "nodes_done":   self.nodes_done,
            "nodes_failed": self.nodes_failed,
            "total_tokens": self.total_tokens,
            "cost_usd":     self.cost_usd,
            "duration_ms":  self.duration_ms,
            "confidence":   self.confidence,
        }

class Orchestrator:
    """
    Decomposes a goal into a parallel task DAG and runs it.
    """

    def __init__(
        self,
        llm:                       "LLMProvider",
        agents:                    dict[str, "BaseAgent"],
        session_id:                str | None = None,
        max_retries_per_node:      int = 1,
        synthesis_temperature:     float = 0.5,
        decomposition_max_nodes:   int = 8,
    ) -> None:
        self._llm                     = llm
        self._agents                  = agents
        self._session_id              = session_id or f"orch-{str(uuid.uuid4())[:8]}"
        self._max_retries             = max_retries_per_node
        self._synthesis_temp          = synthesis_temperature
        self._decomp_max_nodes        = decomposition_max_nodes

    async def run(self, goal: str) -> OrchestratorResult:
        task_id = str(uuid.uuid4())[:8]
        start   = time.monotonic()

        log.info(
            "orchestrator_run_start",
            session_id = self._session_id,
            task_id    = task_id,
            goal_len   = len(goal),
            agents     = list(self._agents.keys()),
        )
        METRICS.tasks_total.labels(status="started").inc()

        try:
            # ── 1. Decompose goal into task DAG 
            dag = await self._decompose(goal)
            log.info(
                "orchestrator_dag_created",
                session_id  = self._session_id,
                task_id     = task_id,
                node_count  = len(dag),
                node_ids    = [n.id for n in dag],
                agent_types = [n.agent_type for n in dag],
            )

            # ── 2. Execute DAG 
            completed_results: dict[str, "AgentResult"] = {}
            await self._execute_dag(dag, completed_results, task_id)

            # ── 3. Synthesise results 
            duration_ms   = int((time.monotonic() - start) * 1000)
            nodes_done    = sum(1 for n in dag if n.status == "done")
            nodes_failed  = sum(1 for n in dag if n.status == "failed")

            if nodes_done == 0:
                METRICS.tasks_total.labels(status="failed").inc()
                return OrchestratorResult(
                    success      = False,
                    final_answer = "",
                    error        = (
                        f"All {len(dag)} sub-tasks failed. "
                        "Check agent logs for individual error details."
                    ),
                    session_id   = self._session_id,
                    task_id      = task_id,
                    nodes_total  = len(dag),
                    nodes_done   = 0,
                    nodes_failed = nodes_failed,
                    duration_ms  = duration_ms,
                    confidence   = 0.0,
                )

            final_answer, confidence = await self._synthesise(
                goal=goal,
                results=completed_results,
            )

            METRICS.tasks_total.labels(status="success").inc()
            log.info(
                "orchestrator_run_complete",
                session_id   = self._session_id,
                task_id      = task_id,
                nodes_done   = nodes_done,
                nodes_failed = nodes_failed,
                duration_ms  = duration_ms,
                confidence   = confidence,
            )

            return OrchestratorResult(
                success       = True,
                final_answer  = final_answer,
                session_id    = self._session_id,
                task_id       = task_id,
                node_results  = {k: v.to_dict() for k, v in completed_results.items()},
                nodes_total   = len(dag),
                nodes_done    = nodes_done,
                nodes_failed  = nodes_failed,
                duration_ms   = duration_ms,
                confidence    = confidence,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            METRICS.tasks_total.labels(status="failed").inc()
            log.error(
                "orchestrator_run_error",
                session_id  = self._session_id,
                task_id     = task_id,
                error       = str(exc),
                duration_ms = duration_ms,
            )
            return OrchestratorResult(
                success      = False,
                final_answer = "",
                error        = str(exc),
                session_id   = self._session_id,
                task_id      = task_id,
                duration_ms  = duration_ms,
                confidence   = 0.0,
            )

    # ─── DAG Execution 

    async def _execute_dag(
        self,
        dag:       list[TaskNode],
        results:   dict[str, "AgentResult"],
        task_id:   str,
    ) -> None:
        max_iterations = len(dag) + 1 

        for _iteration in range(max_iterations):
            ready = [
                n for n in dag
                if n.status == "pending"
                and all(results.get(dep_id) is not None for dep_id in n.depends_on)
            ]

            if not ready:
                pending_count = sum(1 for n in dag if n.status == "pending")
                if pending_count > 0:
                    for node in dag:
                        if node.status == "pending":
                            node.status = "failed"
                            node.result = None
                            log.warning(
                                "orchestrator_node_deadlocked",
                                node_id    = node.id,
                                depends_on = node.depends_on,
                                session_id = self._session_id,
                            )
                break

            for node in ready:
                node.status = "running"
                log.debug(
                    "orchestrator_node_starting",
                    node_id    = node.id,
                    agent_type = node.agent_type,
                    session_id = self._session_id,
                    task_id    = task_id,
                )

            node_tasks = [
                self._run_node(node, results)
                for node in ready
            ]
            node_outcomes = await asyncio.gather(*node_tasks, return_exceptions=True)

            for node, outcome in zip(ready, node_outcomes):
                if isinstance(outcome, Exception):
                    log.error(
                        "orchestrator_node_exception",
                        node_id    = node.id,
                        error      = str(outcome),
                        session_id = self._session_id,
                    )
                    node.status = "failed"
                elif outcome is None:
                    node.status = "failed"
                else:
                    node.result = outcome
                    if outcome.success:
                        node.status = "done"
                        results[node.id] = outcome
                        log.info(
                            "orchestrator_node_done",
                            node_id     = node.id,
                            agent_type  = node.agent_type,
                            confidence  = outcome.confidence,
                            duration_ms = outcome.duration_ms,
                            session_id  = self._session_id,
                        )
                    else:
                        if node.retry_count < self._max_retries:
                            node.retry_count += 1
                            node.status = "pending"  
                            log.info(
                                "orchestrator_node_retry",
                                node_id     = node.id,
                                retry       = node.retry_count,
                                error       = outcome.error,
                                session_id  = self._session_id,
                            )
                        else:
                            node.status = "failed"
                            log.warning(
                                "orchestrator_node_failed",
                                node_id    = node.id,
                                agent_type = node.agent_type,
                                error      = outcome.error,
                                session_id = self._session_id,
                            )

    async def _run_node(
        self,
        node:    TaskNode,
        results: dict[str, "AgentResult"],
    ) -> "AgentResult | None":
        agent = self._agents.get(node.agent_type)
        if agent is None:
            log.error(
                "orchestrator_no_agent_for_type",
                agent_type  = node.agent_type,
                available   = list(self._agents.keys()),
                session_id  = self._session_id,
            )
            from agents.base import AgentResult
            return AgentResult(
                success    = False,
                output     = None,
                error      = (
                    f"No agent registered for type '{node.agent_type}'. "
                    f"Available: {sorted(self._agents.keys())}."
                ),
                agent_type = node.agent_type,
                session_id = self._session_id,
            )

        context: dict[str, Any] = {}
        for dep_id in node.depends_on:
            dep_result = results.get(dep_id)
            if dep_result is not None:
                context[dep_id] = dep_result

        temperature = 0.9 if node.retry_count > 0 else agent.default_temperature

        try:
            if node.retry_count > 0:
                result = await agent.run(
                    task=node.goal + "\n\n[RETRY: previous attempt failed — try a different approach]",
                    context=context,
                )
            else:
                result = await agent.run(task=node.goal, context=context)
        except Exception as exc:
            from agents.base import AgentResult
            return AgentResult(
                success    = False,
                output     = None,
                error      = f"Agent raised unexpected exception: {exc}",
                agent_type = node.agent_type,
                session_id = self._session_id,
            )

        return result

    # ─── Decomposition 

    async def _decompose(self, goal: str) -> list[TaskNode]:
        prompt = _build_decomposition_prompt(
            goal            = goal,
            available_agents = list(self._agents.keys()),
            max_nodes        = self._decomp_max_nodes,
        )

        try:
            response = await self._llm.complete(
                messages    = [LLMMessage(role="user", content=prompt)],
                system      = _DECOMPOSITION_SYSTEM_PROMPT,
                temperature = 0.2,   
                max_tokens  = 2048,
            )
        except Exception as exc:
            log.warning(
                "orchestrator_decompose_llm_error",
                error      = str(exc),
                session_id = self._session_id,
            )
            return self._single_node_fallback(goal)

        raw = response.content.strip()
        nodes = _parse_decomposition_response(raw)

        if not nodes:
            log.warning(
                "orchestrator_decompose_parse_failed",
                raw_len    = len(raw),
                raw_preview = raw[:200],
                session_id = self._session_id,
            )
            return self._single_node_fallback(goal)

        all_ids = {n.id for n in nodes}
        for node in nodes:
            valid_deps = [dep for dep in node.depends_on if dep in all_ids]
            if len(valid_deps) != len(node.depends_on):
                invalid = set(node.depends_on) - all_ids
                log.warning(
                    "orchestrator_invalid_deps",
                    node_id     = node.id,
                    invalid_ids = sorted(invalid),
                    session_id  = self._session_id,
                )
                node.depends_on = valid_deps

        return nodes

    def _single_node_fallback(self, goal: str) -> list[TaskNode]:
        for preferred in ("research", "data", "coding", "communication"):
            if preferred in self._agents:
                agent_type = preferred
                break
        else:
            agent_type = next(iter(self._agents), "research")

        log.info(
            "orchestrator_single_node_fallback",
            agent_type = agent_type,
            session_id = self._session_id,
        )
        return [TaskNode(id="t1", goal=goal, agent_type=agent_type)]

    # ─── Synthesis 

    async def _synthesise(
        self,
        goal:    str,
        results: dict[str, "AgentResult"],
    ) -> tuple[str, float]:
        if len(results) == 1:
            only_result = next(iter(results.values()))
            text        = only_result.to_context_string()
            return text, only_result.confidence

        results_text = "\n\n".join(
            f"[{node_id} — {r.agent_type}]:\n{r.to_context_string()}"
            for node_id, r in results.items()
        )

        conflicts      = _detect_numeric_conflicts(results)
        conflict_text  = (
            "CONFLICTS DETECTED — acknowledge uncertainty for these values:\n"
            + "\n".join(f"  ⚠ {c}" for c in conflicts)
            if conflicts
            else "No numeric conflicts detected."
        )

        prompt = (
            f"Synthesise the following sub-agent results into a single coherent answer.\n\n"
            f"Original goal: {goal}\n\n"
            f"Sub-agent results:\n{results_text}\n\n"
            f"Conflict analysis:\n{conflict_text}\n\n"
            f"Instructions:\n"
            f"- Produce a complete, well-structured final answer.\n"
            f"- If conflicts exist, state the discrepancy and the range of values — "
            f"  do NOT silently choose one value.\n"
            f"- Cite which sub-agent provided each key fact.\n"
            f"- Start your response with: FINAL ANSWER:"
        )

        try:
            response = await self._llm.complete(
                messages    = [LLMMessage(role="user", content=prompt)],
                temperature = self._synthesis_temp,
                max_tokens  = 4096,
            )
            text = response.content.strip()
            if text.upper().startswith("FINAL ANSWER:"):
                text = text[len("FINAL ANSWER:"):].strip()
        except Exception as exc:
            log.warning(
                "orchestrator_synthesis_failed",
                error      = str(exc),
                session_id = self._session_id,
            )
            text = "\n\n".join(
                f"## {node_id} ({r.agent_type})\n{r.to_context_string()}"
                for node_id, r in results.items()
            )

        confidences = [r.confidence for r in results.values() if r.success]
        confidence  = sum(confidences) / len(confidences) if confidences else 0.0
        if conflicts:
            confidence *= 0.85  

        return text, round(confidence, 3)


_DECOMPOSITION_SYSTEM_PROMPT = """\
You are a task decomposition engine for an AI agent system.
Your job is to break complex goals into parallel sub-tasks that specialist agents can execute.
You MUST respond with valid JSON only — no prose, no markdown fences, just the JSON array.
"""

_AGENT_DESCRIPTIONS = {
    "research":      "Searches the web, reads pages, summarises information, checks facts.",
    "coding":        "Writes Python code, executes it, reads error output, fixes bugs.",
    "data":          "Reads CSV/Excel files, runs analysis code, produces statistical summaries.",
    "communication": "Sends emails, creates calendar events, formats professional messages.",
}


def _build_decomposition_prompt(
    goal:             str,
    available_agents: list[str],
    max_nodes:        int,
) -> str:
    agent_list = "\n".join(
        f'  "{a}": {_AGENT_DESCRIPTIONS.get(a, "General purpose agent.")}'
        for a in available_agents
    )
    return f"""\
Decompose this goal into {max_nodes} or fewer sub-tasks for specialist agents.

GOAL: {goal}

AVAILABLE AGENTS:
{agent_list}

RULES:
1. Each sub-task must be achievable by exactly ONE agent type.
2. Express dependencies using node IDs in the depends_on list.
3. Nodes with empty depends_on run in parallel immediately.
4. Keep each goal string specific and self-contained.
5. Use the minimum number of nodes needed — do not over-decompose.

Respond with ONLY a JSON array. Example format:
[
  {{"id": "t1", "goal": "Research X", "agent_type": "research", "depends_on": []}},
  {{"id": "t2", "goal": "Analyse the data from t1", "agent_type": "data", "depends_on": ["t1"]}}
]

JSON array for the goal above:"""


def _parse_decomposition_response(raw: str) -> list[TaskNode]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    array_str = cleaned[start : end + 1]

    try:
        raw_list = json.loads(array_str)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_list, list):
        return []

    nodes: list[TaskNode] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            node = TaskNode(**item)
            nodes.append(node)
        except (ValidationError, TypeError) as exc:
            log.debug("orchestrator_node_parse_skip", item=item, error=str(exc))
            continue

    return nodes

def _detect_numeric_conflicts(results: dict[str, "AgentResult"]) -> list[str]:
    pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    number_sets: dict[str, set[str]] = {}

    for node_id, result in results.items():
        text = result.to_context_string()
        number_sets[node_id] = set(pattern.findall(text))

    conflicts: list[str] = []
    ids = list(number_sets.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b    = ids[i], ids[j]
            diff_a = {n for n in number_sets[a] - number_sets[b] if float(n.replace(",", "")) > 9}
            diff_b = {n for n in number_sets[b] - number_sets[a] if float(n.replace(",", "")) > 9}
            if diff_a or diff_b:
                conflicts.append(
                    f"{a} has {sorted(diff_a)} that {b} does not; "
                    f"{b} has {sorted(diff_b)} that {a} does not"
                )

    return conflicts

def _assign_agent_type(goal: str, available: list[str]) -> str:
    lower = goal.lower()
    rules = [
        ({"write code", "implement", "debug", "function", "script",
          "algorithm", "fix the", "refactor"},       "coding"),
        ({"analyse", "analyze", "csv", "excel", "xlsx",
          "statistics", "chart", "plot", "dataset"},  "data"),
        ({"send email", "email", "calendar", "meeting",
          "notify", "message", "slack"},              "communication"),
        ({"research", "find", "search", "look up",
          "summarise", "summarize", "what is",
          "who is", "latest", "recent"},              "research"),
    ]
    for keywords, agent_type in rules:
        if any(kw in lower for kw in keywords):
            if agent_type in available:
                return agent_type
    return "research" if "research" in available else next(iter(available), "research")