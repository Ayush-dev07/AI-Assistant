from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

from core.logging import METRICS, get_logger

if TYPE_CHECKING:
    from memory.episodic import EpisodicMemory
    from agents.bus import MessageBus

log = get_logger(__name__)

class Priority:
    CRITICAL   = 1   # HITL-gated actions, urgent retries
    DEFAULT    = 5   # Normal user-submitted tasks
    BACKGROUND = 10  # Skill installs, eval runs, memory consolidation

class TaskStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    CANCELLED = "cancelled"

@dataclass(order=True)
class QueuedTask:
    priority:     int
    task_id:      str            = dataclasses.field(compare=False, default_factory=lambda: str(uuid.uuid4())[:8])
    goal:         str            = dataclasses.field(compare=False, default="")
    agent_type:   str            = dataclasses.field(compare=False, default="orchestrator")
    session_id:   str            = dataclasses.field(compare=False, default="default")
    submitted_at: float          = dataclasses.field(compare=False, default_factory=time.monotonic)
    metadata:     dict[str, Any] = dataclasses.field(compare=False, default_factory=dict)

    @property
    def queue_wait_ms(self) -> int:
        return int((time.monotonic() - self.submitted_at) * 1000)

@dataclass
class TaskResult:
    task_id:      str
    session_id:   str
    status:       str          
    goal:         str
    final_answer: str            = ""
    error:        str            = ""
    agent_type:   str            = ""
    total_tokens: int            = 0
    cost_usd:     float          = 0.0
    duration_ms:  int            = 0
    queue_wait_ms: int           = 0
    completed_at: str            = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    metadata:     dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id":      self.task_id,
            "session_id":   self.session_id,
            "status":       self.status,
            "goal":         self.goal[:200],
            "final_answer": self.final_answer,
            "error":        self.error,
            "agent_type":   self.agent_type,
            "total_tokens": self.total_tokens,
            "cost_usd":     self.cost_usd,
            "duration_ms":  self.duration_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "completed_at": self.completed_at,
        }

AgentFactory = Callable[[str, str], Any]

class TaskQueue:

    def __init__(
        self,
        episodic:       "EpisodicMemory | None" = None,
        bus:            "MessageBus | None"     = None,
        worker_count:   int                     = 2,
        max_queue_size: int                     = 100,
        result_ttl_s:   int                     = 3600,
    ) -> None:
        self._episodic       = episodic
        self._bus            = bus
        self._worker_count   = worker_count
        self._result_ttl_s   = result_ttl_s

        # The actual priority queue
        self._q: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        self._results:      dict[str, TaskResult] = {}
        self._result_times: dict[str, float]      = {}

        self._worker_tasks: list[asyncio.Task] = []

        # Running state
        self._running = False
        self._started = False

    async def start(self, agent_factory: AgentFactory) -> None:
        if self._started:
            log.warning("task_queue_already_started")
            return

        self._running = True
        self._started = True

        for i in range(self._worker_count):
            task = asyncio.create_task(
                self._worker(agent_factory, worker_id=i),
                name=f"task-queue-worker-{i}",
            )
            self._worker_tasks.append(task)

        log.info(
            "task_queue_started",
            workers    = self._worker_count,
            has_episodic = self._episodic is not None,
            has_bus      = self._bus is not None,
        )

    async def stop(self, drain_timeout_s: float = 30.0) -> None:
        if not self._running:
            return

        self._running = False
        log.info("task_queue_stopping", queued=self._q.qsize())

        try:
            await asyncio.wait_for(self._q.join(), timeout=drain_timeout_s)
            log.info("task_queue_drained")
        except asyncio.TimeoutError:
            log.warning(
                "task_queue_drain_timeout",
                timeout_s = drain_timeout_s,
                remaining = self._q.qsize(),
            )

        # Cancel workers
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._worker_tasks.clear()
        log.info("task_queue_stopped")

    async def submit(
        self,
        goal:       str,
        agent_type: str   = "orchestrator",
        session_id: str   = "default",
        priority:   int   = Priority.DEFAULT,
        metadata:   dict[str, Any] | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())[:8]

        qt = QueuedTask(
            priority   = priority,
            task_id    = task_id,
            goal       = goal,
            agent_type = agent_type,
            session_id = session_id,
            metadata   = metadata or {},
        )

        if self._episodic is not None:
            try:
                supabase_id = await self._episodic.record_start(
                    session_id = session_id,
                    goal       = goal,
                )
                qt.metadata["_supabase_task_id"] = supabase_id
            except Exception as exc:
                log.warning(
                    "task_queue_supabase_start_failed",
                    task_id = task_id,
                    error   = str(exc),
                )

        try:
            self._q.put_nowait(qt)
        except asyncio.QueueFull as exc:
            raise TaskQueueFull(
                f"Task queue is full ({self._q.maxsize} tasks). "
                "Try again later or increase max_queue_size."
            ) from exc

        METRICS.tasks_total.labels(status="queued").inc()
        log.info(
            "task_queue_submitted",
            task_id    = task_id,
            agent_type = agent_type,
            session_id = session_id,
            priority   = priority,
            qsize      = self._q.qsize(),
        )

        # Publish submission event to bus
        if self._bus is not None:
            await self._publish_event(
                session_id   = session_id,
                task_id      = task_id,
                event_type   = "task_queued",
                payload      = {
                    "task_id":    task_id,
                    "agent_type": agent_type,
                    "priority":   priority,
                    "qsize":      self._q.qsize(),
                },
            )

        return task_id

    def get_result_sync(self, task_id: str) -> TaskResult | None:
        self._evict_expired_results()
        return self._results.get(task_id)

    async def get_result(
        self,
        task_id:   str,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.5,
    ) -> TaskResult | None:
        """
        Wait for a task result with polling.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.get_result_sync(task_id)
            if result is not None:
                return result
            await asyncio.sleep(poll_interval_s)
        return None

    def queue_stats(self) -> dict[str, Any]:
        return {
            "qsize":        self._q.qsize(),
            "maxsize":      self._q.maxsize,
            "workers":      self._worker_count,
            "active_workers": sum(
                1 for t in self._worker_tasks if not t.done()
            ),
            "results_cached": len(self._results),
            "running":      self._running,
        }

    async def _worker(
        self,
        agent_factory: AgentFactory,
        worker_id:     int,
    ) -> None:
        """
        Worker coroutine — runs in an infinite loop until _running=False.
        """
        log.debug("task_queue_worker_started", worker_id=worker_id)

        while self._running:
            try:
                try:
                    qt = await asyncio.wait_for(self._q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

            except asyncio.CancelledError:
                break

            queue_wait = qt.queue_wait_ms
            start      = time.monotonic()
            supabase_id = qt.metadata.get("_supabase_task_id")

            log.info(
                "task_queue_worker_executing",
                worker_id   = worker_id,
                task_id     = qt.task_id,
                agent_type  = qt.agent_type,
                session_id  = qt.session_id,
                priority    = qt.priority,
                queue_wait_ms = queue_wait,
            )

            if supabase_id and self._episodic is not None:
                await self._safe_supabase(
                    self._episodic._client.update(
                        self._episodic.TABLE,
                        data  = {"status": TaskStatus.RUNNING},
                        match = {"id": supabase_id},
                    )
                )

            if self._bus is not None:
                await self._publish_event(
                    session_id = qt.session_id,
                    task_id    = qt.task_id,
                    event_type = "task_running",
                    payload    = {"task_id": qt.task_id, "worker_id": worker_id},
                )

            status       = TaskStatus.FAILED
            final_answer = ""
            error_msg    = ""
            total_tokens = 0
            cost_usd     = 0.0

            try:
                agent  = agent_factory(qt.agent_type, qt.session_id)
                result = await agent.run(qt.goal)

                success = getattr(result, "success", False)
                if success:
                    status = TaskStatus.SUCCESS
                    output = getattr(result, "output", None)
                    final_answer = (
                        getattr(result, "final_answer", None)
                        or (result.to_context_string() if hasattr(result, "to_context_string") else str(output or ""))
                    )
                else:
                    status    = TaskStatus.FAILED
                    error_msg = getattr(result, "error", "Agent returned failure") or ""
                    final_answer = ""

                total_tokens = getattr(result, "total_tokens", 0)
                cost_usd     = getattr(result, "cost_usd",     0.0)

            except asyncio.CancelledError:
                try:
                    await self._q.put(qt)
                except asyncio.QueueFull:
                    pass
                self._q.task_done()
                raise

            except Exception as exc:
                status    = TaskStatus.FAILED
                error_msg = f"{type(exc).__name__}: {exc}"
                log.error(
                    "task_queue_worker_error",
                    worker_id  = worker_id,
                    task_id    = qt.task_id,
                    error      = error_msg,
                )

            duration_ms = int((time.monotonic() - start) * 1000)

            task_result = TaskResult(
                task_id      = qt.task_id,
                session_id   = qt.session_id,
                status       = status,
                goal         = qt.goal,
                final_answer = final_answer[:10000],
                error        = error_msg,
                agent_type   = qt.agent_type,
                total_tokens = total_tokens,
                cost_usd     = cost_usd,
                duration_ms  = duration_ms,
                queue_wait_ms = queue_wait,
            )
            self._results[qt.task_id]      = task_result
            self._result_times[qt.task_id] = time.monotonic()

            if supabase_id and self._episodic is not None:
                await self._safe_supabase(
                    self._episodic.record_completion(
                        task_id      = supabase_id,
                        final_answer = final_answer,
                        status       = status,
                        total_tokens = total_tokens,
                        cost_usd     = cost_usd,
                        duration_ms  = duration_ms,
                    )
                )

            prom_status = "success" if status == TaskStatus.SUCCESS else "failed"
            METRICS.tasks_total.labels(status=prom_status).inc()

            if self._bus is not None:
                await self._publish_event(
                    session_id = qt.session_id,
                    task_id    = qt.task_id,
                    event_type = "task_completed" if status == TaskStatus.SUCCESS else "task_failed",
                    payload    = task_result.to_dict(),
                )

            log.info(
                "task_queue_worker_done",
                worker_id    = worker_id,
                task_id      = qt.task_id,
                status       = status,
                duration_ms  = duration_ms,
                queue_wait_ms = queue_wait,
            )

            self._q.task_done()

        log.debug("task_queue_worker_stopped", worker_id=worker_id)

    async def recover_pending_tasks(self) -> int:
        """
        Re-queue tasks that were "running" when the process last died.
        """
        if self._episodic is None:
            return 0

        try:
            running_tasks = await self._episodic.get_recent_tasks(
                status="running",
                limit=50,
            )
        except Exception as exc:
            log.warning("task_queue_recovery_failed", error=str(exc))
            return 0

        recovered = 0
        for record in running_tasks:
            qt = QueuedTask(
                priority   = Priority.DEFAULT,
                task_id    = record.id,
                goal       = record.goal,
                agent_type = "orchestrator", 
                session_id = record.session_id,
                metadata   = {"_supabase_task_id": record.id, "_recovered": True},
            )
            try:
                self._q.put_nowait(qt)
                recovered += 1
                log.info(
                    "task_queue_task_recovered",
                    task_id    = record.id,
                    session_id = record.session_id,
                )
            except asyncio.QueueFull:
                log.warning(
                    "task_queue_recovery_full",
                    task_id = record.id,
                )
                break

        return recovered

    async def _publish_event(
        self,
        session_id: str,
        task_id:    str,
        event_type: str,
        payload:    dict[str, Any],
    ) -> None:
        if self._bus is None:
            return
        try:
            from agents.protocol import AgentMessage, Channels
            msg = AgentMessage(
                from_agent   = "task_queue",
                message_type = event_type,
                payload      = payload,
                session_id   = session_id,
            )
            await self._bus.publish(Channels.agent_events(session_id), msg)
        except Exception as exc:
            log.debug("task_queue_bus_publish_failed", error=str(exc))

    @staticmethod
    async def _safe_supabase(coro: Any) -> None:
        try:
            await coro
        except Exception as exc:
            log.warning("task_queue_supabase_op_failed", error=str(exc))

    def _evict_expired_results(self) -> None:
        now     = time.monotonic()
        cutoff  = now - self._result_ttl_s
        expired = [
            tid for tid, ts in self._result_times.items()
            if ts < cutoff
        ]
        for tid in expired:
            self._results.pop(tid, None)
            self._result_times.pop(tid, None)

class TaskQueueFull(RuntimeError):
    """Raised when submit() is called and the queue is at capacity."""


class TaskQueueNotStarted(RuntimeError):
    """Raised when submit() is called before start()."""

def make_agent_factory(llm: Any) -> AgentFactory:
    def factory(agent_type: str, session_id: str) -> Any:
        from agents.specialists import (
            CodingAgent, CommsAgent, DataAgent, ResearchAgent,
        )
        from agents.orchestrator import Orchestrator
        from agents import build_default_agents

        specialist_map = {
            "research":      lambda: ResearchAgent(llm=llm, session_id=session_id),
            "coding":        lambda: CodingAgent(llm=llm,   session_id=session_id),
            "communication": lambda: CommsAgent(llm=llm,    session_id=session_id),
            "data":          lambda: DataAgent(llm=llm,     session_id=session_id),
        }

        creator = specialist_map.get(agent_type.lower())
        if creator is not None:
            return creator()

        agents = build_default_agents(llm)
        return Orchestrator(
            llm        = llm,
            agents     = agents,
            session_id = session_id,
        )

    return factory