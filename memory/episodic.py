from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from core.logging import get_logger
from dotenv import load_dotenv
load_dotenv()

log = get_logger(__name__)


# Data Models 

class TaskRecord(BaseModel):
    """A complete record of one agent task execution."""

    id: str
    session_id: str
    goal: str
    final_answer: str = ""
    status: str = "running"
    iterations: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None


class TaskSummary(BaseModel):
    """Aggregated statistics across multiple tasks."""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_duration_ms: float
    avg_cost_usd: float
    total_cost_usd: float
    total_tokens: int
    most_used_tools: list[dict[str, Any]]


# ─── Supabase REST Client ─────────────────────────────────────────────────────

class SupabaseClient:
    def __init__(self, url: str, service_key: str) -> None:
        self._url = url.rstrip("/")
        self._rest_url = f"{self._url}/rest/v1"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",  # Return the inserted/updated row
        }
        self._http = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,
        )

    async def insert(self, table: str, data: dict) -> dict | None:
        response = await self._http.post(
            f"{self._rest_url}/{table}",
            json=data,
        )
        if response.status_code not in (200, 201):
            log.error(
                "supabase_insert_failed",
                table=table,
                status=response.status_code,
                body=response.text[:200],
            )
            return None
        result = response.json()
        return result[0] if isinstance(result, list) and result else result

    async def update(
        self,
        table: str,
        data: dict,
        match: dict,
    ) -> list[dict]:
        # Build PostgREST filter query string
        # Format: ?column=eq.value (eq = equals)
        params = {k: f"eq.{v}" for k, v in match.items()}

        response = await self._http.patch(
            f"{self._rest_url}/{table}",
            params=params,
            json=data,
        )
        if response.status_code not in (200, 204):
            log.error(
                "supabase_update_failed",
                table=table,
                status=response.status_code,
                body=response.text[:200],
            )
            return []
        return response.json() if response.text else []

    async def select(
        self,
        table: str,
        columns: str = "*",
        match: dict | None = None,
        order: str | None = None,
        limit: int | None = None,
        ilike: dict | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"select": columns}

        if match:
            for k, v in match.items():
                params[k] = f"eq.{v}"

        if ilike:
            for k, v in ilike.items():
                params[k] = f"ilike.*{v}*"

        if order:
            params["order"] = order

        if limit:
            params["limit"] = limit

        response = await self._http.get(
            f"{self._rest_url}/{table}",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def rpc(self, function_name: str, params: dict | None = None) -> Any:
        response = await self._http.post(
            f"{self._url}/rest/v1/rpc/{function_name}",
            json=params or {},
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()


# Episodic Memory 
class EpisodicMemory:

    TABLE = "agent_tasks"

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ) -> None:
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase credentials are required. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file. "
                "Get them from: app.supabase.com → Project Settings → API"
            )

        self._client = SupabaseClient(url, key)
        log.info(
            "episodic_memory_initialized",
            backend="supabase",
            project_url=url,
        )

    async def initialize(self) -> None:
        try:
            await self._client.select(self.TABLE, columns="id", limit=1)
            log.info("episodic_memory_connection_verified", table=self.TABLE)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Table '{self.TABLE}' not found in Supabase. "
                    f"Run the CREATE TABLE SQL in the Supabase SQL Editor first. "
                    f"SQL is in the CREATE_TABLE_SQL string at the top of memory/episodic.py"
                ) from e
            raise

    async def record_start(
        self,
        session_id: str,
        goal: str,
    ) -> str:
        task_id = str(uuid.uuid4())
        row = {
            "id": task_id,
            "session_id": session_id,
            "goal": goal[:2000],  # Truncate very long goals
            "status": "running",
        }
        result = await self._client.insert(self.TABLE, row)
        if not result:
            log.error("episodic_memory_start_failed", goal=goal[:80])
            # Return the ID anyway — the task can still run without logging
            return task_id

        log.info("episodic_memory_task_started", task_id=task_id, goal=goal[:80])
        return task_id

    async def record_completion(
        self,
        task_id: str,
        final_answer: str,
        status: str = "success",
        iterations: int = 0,
        tool_calls: list[dict[str, Any]] | None = None,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> None:
        updates = {
            "final_answer": final_answer[:10000],  # Cap at 10K chars
            "status": status,
            "iterations": iterations,
            "tool_calls": json.dumps(tool_calls or []),  # Serialize to JSON string
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 8),
            "duration_ms": duration_ms,
            "completed_at": datetime.utcnow().isoformat(),
        }

        await self._client.update(
            self.TABLE,
            data=updates,
            match={"id": task_id},
        )

        log.info(
            "episodic_memory_task_completed",
            task_id=task_id,
            status=status,
            iterations=iterations,
            cost_usd=round(cost_usd, 6),
            duration_ms=duration_ms,
        )

    async def get_task(self, task_id: str) -> TaskRecord | None:
        rows = await self._client.select(
            self.TABLE,
            match={"id": task_id},
            limit=1,
        )
        if not rows:
            return None
        return self._row_to_record(rows[0])

    async def get_recent_tasks(
        self,
        session_id: str | None = None,
        limit: int = 20,
        status: str | None = None,
    ) -> list[TaskRecord]:
        match: dict[str, Any] = {}
        if session_id:
            match["session_id"] = session_id
        if status:
            match["status"] = status

        rows = await self._client.select(
            self.TABLE,
            match=match or None,
            order="created_at.desc",
            limit=limit,
        )
        return [self._row_to_record(row) for row in rows]

    async def search_by_goal(
        self,
        keyword: str,
        limit: int = 10,
    ) -> list[TaskRecord]:
        rows = await self._client.select(
            self.TABLE,
            ilike={"goal": keyword},
            order="created_at.desc",
            limit=limit,
        )
        return [self._row_to_record(row) for row in rows]

    async def get_analytics(self, days: int = 7) -> TaskSummary:
        try:
            # Try server-side aggregation via RPC (faster for large datasets)
            result = await self._client.rpc(
                "get_task_analytics",
                {"days_back": days},
            )
            if result:
                return TaskSummary(**result)
        except Exception:
            pass  # Fall through to client-side aggregation

        all_tasks = await self._client.select(
            self.TABLE,
            order="created_at.desc",
            limit=1000,  # Cap at 1000 for client-side processing
        )

        # Filter to last N days in Python
        cutoff = datetime.utcnow().timestamp() - (days * 86400)
        recent = []
        for row in all_tasks:
            if row.get("created_at"):
                try:
                    ts = datetime.fromisoformat(
                        row["created_at"].replace("Z", "+00:00")
                    ).timestamp()
                    if ts >= cutoff:
                        recent.append(row)
                except (ValueError, AttributeError):
                    recent.append(row)

        # Aggregate
        total = len(recent)
        successful = sum(1 for r in recent if r.get("status") == "success")
        failed = sum(1 for r in recent if r.get("status") == "failed")
        total_cost = sum(float(r.get("cost_usd", 0)) for r in recent)
        total_tokens = sum(int(r.get("total_tokens", 0)) for r in recent)
        durations = [int(r.get("duration_ms", 0)) for r in recent if r.get("duration_ms")]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_cost = total_cost / total if total > 0 else 0.0

        # Count tool usage across all tasks
        tool_counts: dict[str, dict[str, int]] = {}
        for row in recent:
            calls_raw = row.get("tool_calls", "[]")
            try:
                calls = json.loads(calls_raw) if isinstance(calls_raw, str) else calls_raw
                for call in (calls or []):
                    tool = call.get("tool", "unknown")
                    if tool not in tool_counts:
                        tool_counts[tool] = {"calls": 0, "successes": 0}
                    tool_counts[tool]["calls"] += 1
                    if call.get("success"):
                        tool_counts[tool]["successes"] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        most_used = sorted(
            [{"tool_name": k, **v} for k, v in tool_counts.items()],
            key=lambda x: x["calls"],
            reverse=True,
        )[:10]

        return TaskSummary(
            total_tasks=total,
            successful_tasks=successful,
            failed_tasks=failed,
            avg_duration_ms=avg_duration,
            avg_cost_usd=avg_cost,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            most_used_tools=most_used,
        )

    def _row_to_record(self, row: dict) -> TaskRecord:
        tool_calls = row.get("tool_calls", [])
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                tool_calls = []

        return TaskRecord(
            id=row["id"],
            session_id=row["session_id"],
            goal=row["goal"],
            final_answer=row.get("final_answer", ""),
            status=row.get("status", "unknown"),
            iterations=row.get("iterations", 0),
            tool_calls=tool_calls or [],
            total_tokens=row.get("total_tokens", 0),
            cost_usd=float(row.get("cost_usd", 0.0)),
            duration_ms=row.get("duration_ms", 0),
            created_at=(
                datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                if row.get("created_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"].replace("Z", "+00:00"))
                if row.get("completed_at") else None
            ),
        )

    async def close(self) -> None:
        """Close the HTTP client. Call at application shutdown."""
        await self._client.close()
        log.info("episodic_memory_closed")