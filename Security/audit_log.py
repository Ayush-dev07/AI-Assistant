from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger

log = get_logger(__name__)

class ActionType:
    TOOL_CALL       = "tool_call"        # Agent called a tool
    SCAN_RESULT     = "scan_result"      # InjectionDetector verdict
    APPROVAL        = "approval"         # HITL approval request/decision
    MEMORY_WRITE    = "memory_write"     # GuardedMemory.store() call
    CIRCUIT_BREAKER = "circuit_breaker"  # AnomalyDetector state change
    TASK_START      = "task_start"       # Agent task began
    TASK_COMPLETE   = "task_complete"    # Agent task finished
    ERROR           = "error"            # Unhandled error in agent loop

class AuditEntry(BaseModel):

    session_id:           str             = Field(default="default")
    action_type:          str             = Field(..., description="Use ActionType.xxx constants")
    tool_name:            str | None      = Field(default=None)
    parameters_sanitized: dict[str, Any]  = Field(default_factory=dict)
    result_summary:       str             = Field(default="")
    threat_level:         str             = Field(default="safe")

    prev_hash:            str             = Field(default="")
    entry_hash:           str             = Field(default="")
    timestamp:            str             = Field(default="")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        known = {
            ActionType.TOOL_CALL, ActionType.SCAN_RESULT,
            ActionType.APPROVAL, ActionType.MEMORY_WRITE,
            ActionType.CIRCUIT_BREAKER, ActionType.TASK_START,
            ActionType.TASK_COMPLETE, ActionType.ERROR,
        }
        if v not in known:
            log.debug("audit_unknown_action_type", action_type=v)
        return v

    @field_validator("threat_level")
    @classmethod
    def validate_threat_level(cls, v: str) -> str:
        if v not in ("safe", "suspicious", "dangerous"):
            return "safe"
        return v

    def _fields_for_hash(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"entry_hash"})
        d["result_summary"] = d["result_summary"][:500]
        return d

    @classmethod
    def for_tool_call(
        cls,
        session_id:         str,
        tool_name:          str,
        parameters:         dict[str, Any],
        result_summary:     str,
        threat_level:       str = "safe",
    ) -> "AuditEntry":
        from Security.hitl import sanitize_parameters
        return cls(
            session_id           = session_id,
            action_type          = ActionType.TOOL_CALL,
            tool_name            = tool_name,
            parameters_sanitized = sanitize_parameters(parameters),
            result_summary       = result_summary[:500],
            threat_level         = threat_level,
        )

    @classmethod
    def for_scan_result(
        cls,
        session_id:   str,
        threat_level: str,
        summary:      str,
        tool_name:    str | None = None,
    ) -> "AuditEntry":
        return cls(
            session_id     = session_id,
            action_type    = ActionType.SCAN_RESULT,
            tool_name      = tool_name,
            result_summary = summary[:500],
            threat_level   = threat_level,
        )

    @classmethod
    def for_approval(
        cls,
        session_id:   str,
        request_id:   str,
        action:       str,
        tool_name:    str,
        approved:     bool,
        decided_by:   str = "human",
    ) -> "AuditEntry":
        return cls(
            session_id     = session_id,
            action_type    = ActionType.APPROVAL,
            tool_name      = tool_name,
            result_summary = (
                f"request_id={request_id} action={action} "
                f"approved={approved} decided_by={decided_by}"
            )[:500],
            threat_level   = "safe",
        )

    @classmethod
    def for_circuit_breaker(
        cls,
        session_id: str,
        state:      str,
        reason:     str,
    ) -> "AuditEntry":
        return cls(
            session_id     = session_id,
            action_type    = ActionType.CIRCUIT_BREAKER,
            result_summary = f"state={state} reason={reason}"[:500],
            threat_level   = "safe",
        )

class _AuditSupabaseClient:

    TABLE = "audit_log"

    def __init__(self, url: str, service_key: str) -> None:
        self._rest_url = f"{url.rstrip('/')}/rest/v1"
        self._http     = httpx.AsyncClient(
            headers = {
                "apikey":        service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            timeout = 15.0,
        )

    async def insert(self, row: dict[str, Any]) -> bool:
        try:
            resp = await self._http.post(
                f"{self._rest_url}/{self.TABLE}",
                json = row,
            )
            if resp.status_code == 409:
                log.debug("audit_log_duplicate_entry", entry_hash=row.get("entry_hash", "?")[:16])
                return True
            if resp.status_code not in (200, 201, 204):
                log.error(
                    "audit_log_insert_failed",
                    status = resp.status_code,
                    body   = resp.text[:200],
                )
                return False
            return True
        except httpx.TimeoutException:
            log.warning("audit_log_insert_timeout")
            return False
        except Exception as exc:
            log.error("audit_log_insert_error", error=str(exc))
            return False

    async def select_all_ordered(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000

        while True:
            try:
                resp = await self._http.get(
                    f"{self._rest_url}/{self.TABLE}",
                    params = {
                        "select": "session_id,action_type,tool_name,parameters_sanitized,result_summary,threat_level,prev_hash,entry_hash,timestamp",
                        "order":  "timestamp.asc",
                        "limit":  page_size,
                        "offset": offset,
                    },
                    headers = {"Prefer": "count=none"},
                )
                resp.raise_for_status()
                page = resp.json()
                if not page:
                    break
                results.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
            except Exception as exc:
                log.error("audit_log_select_failed", error=str(exc))
                break

        return results

    async def select_latest_hash(self) -> str | None:
        try:
            resp = await self._http.get(
                f"{self._rest_url}/{self.TABLE}",
                params = {
                    "select": "entry_hash",
                    "order":  "timestamp.desc",
                    "limit":  1,
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            return rows[0]["entry_hash"] if rows else None
        except Exception as exc:
            log.warning("audit_log_select_latest_failed", error=str(exc))
            return None

    async def close(self) -> None:
        await self._http.aclose()

from dataclasses import dataclass

@dataclass
class VerifyResult:
    intact:          bool
    broken_at:       str | None = None
    entries_checked: int        = 0
    error:           str | None = None

class AuditLog:

    GENESIS_HASH = "GENESIS"

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ) -> None:
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

        if url and key:
            self._client      = _AuditSupabaseClient(url, key)
            self._use_supabase = True
            log.info("audit_log_initialized", backend="supabase", url=url[:40] + "...")
        else:
            self._client       = None
            self._use_supabase  = False
            self._memory_log: list[dict[str, Any]] = []
            log.warning(
                "audit_log_no_supabase",
                hint="Set SUPABASE_URL and SUPABASE_SERVICE_KEY for persistent audit log.",
            )

        self._last_hash = self.GENESIS_HASH
        self._chain_lock = asyncio.Lock()

        self._write_count = 0
        self._error_count = 0

    async def initialize(self) -> None:
        if not self._use_supabase:
            return

        latest = await self._client.select_latest_hash()
        if latest:
            self._last_hash = latest
            log.info(
                "audit_log_chain_resumed",
                last_hash = latest[:16] + "...",
            )
        else:
            log.info("audit_log_chain_started_fresh", genesis=self.GENESIS_HASH)

    async def record(self, entry: AuditEntry) -> str:
        try:
            async with self._chain_lock:
                return await self._record_locked(entry)
        except Exception as exc:
            log.error("audit_log_record_fatal", error=str(exc))
            return ""

    async def _record_locked(self, entry: AuditEntry) -> str:
        entry.timestamp  = datetime.now(tz=timezone.utc).isoformat()
        entry.prev_hash  = self._last_hash
        entry.result_summary = entry.result_summary[:500]

        hash_input = json.dumps(
            entry._fields_for_hash(),
            sort_keys  = True,
            ensure_ascii = False,
            default    = str,
        ).encode("utf-8")
        entry.entry_hash = hashlib.sha256(hash_input).hexdigest()

        row = entry.model_dump()

        success = await self._write_row(row)
        self._write_count += 1

        if success:
            self._last_hash = entry.entry_hash
            log.debug(
                "audit_log_entry_written",
                action_type = entry.action_type,
                entry_hash  = entry.entry_hash[:16] + "...",
                session_id  = entry.session_id,
            )
        else:
            self._error_count += 1
            log.error(
                "audit_log_write_failed",
                action_type = entry.action_type,
                entry_hash  = entry.entry_hash[:16] + "...",
            )

        return entry.entry_hash

    async def _write_row(self, row: dict[str, Any]) -> bool:
        if self._use_supabase:
            success = await self._client.insert(row)
            if not success:
                log.debug("audit_log_retrying_insert")
                success = await self._client.insert(row)
            return success
        else:
            self._memory_log.append(row)
            return True

    async def verify_chain(
        self,
        session_id: str | None = None,
    ) -> VerifyResult:
        try:
            entries = await self._fetch_all_entries()
        except Exception as exc:
            return VerifyResult(intact=False, error=str(exc))

        if session_id:
            entries = [e for e in entries if e.get("session_id") == session_id]

        if not entries:
            return VerifyResult(intact=True, entries_checked=0)

        expected_prev = self.GENESIS_HASH

        for i, entry in enumerate(entries):
            fields_for_hash = {
                k: v for k, v in entry.items()
                if k != "entry_hash"
            }
            fields_for_hash.pop("id", None)

            recomputed = hashlib.sha256(
                json.dumps(
                    fields_for_hash,
                    sort_keys    = True,
                    ensure_ascii = False,
                    default      = str,
                ).encode("utf-8")
            ).hexdigest()

            stored_hash = entry.get("entry_hash", "")
            stored_prev = entry.get("prev_hash", "")

            if stored_hash != recomputed:
                log.warning(
                    "audit_log_chain_broken_hash",
                    entry_index = i,
                    stored      = stored_hash[:16],
                    recomputed  = recomputed[:16],
                )
                return VerifyResult(
                    intact          = False,
                    broken_at       = stored_hash[:16] + "...",
                    entries_checked = i + 1,
                )

            if stored_prev != expected_prev:
                log.warning(
                    "audit_log_chain_broken_prev",
                    entry_index    = i,
                    stored_prev    = stored_prev[:16],
                    expected_prev  = expected_prev[:16],
                )
                return VerifyResult(
                    intact          = False,
                    broken_at       = stored_hash[:16] + "...",
                    entries_checked = i + 1,
                )

            expected_prev = stored_hash

        log.debug(
            "audit_log_chain_verified",
            entries = len(entries),
            last    = expected_prev[:16] + "...",
        )

        return VerifyResult(
            intact          = True,
            entries_checked = len(entries),
        )

    async def _fetch_all_entries(self) -> list[dict[str, Any]]:
        if self._use_supabase:
            return await self._client.select_all_ordered()
        else:
            return sorted(
                self._memory_log,
                key = lambda e: e.get("timestamp", ""),
            )

    async def record_tool_call(
        self,
        session_id:     str,
        tool_name:      str,
        parameters:     dict[str, Any],
        result_summary: str,
        threat_level:   str = "safe",
    ) -> str:
        return await self.record(
            AuditEntry.for_tool_call(
                session_id     = session_id,
                tool_name      = tool_name,
                parameters     = parameters,
                result_summary = result_summary,
                threat_level   = threat_level,
            )
        )

    async def record_scan(
        self,
        session_id:   str,
        threat_level: str,
        summary:      str,
        tool_name:    str | None = None,
    ) -> str:
        return await self.record(
            AuditEntry.for_scan_result(
                session_id   = session_id,
                threat_level = threat_level,
                summary      = summary,
                tool_name    = tool_name,
            )
        )

    async def record_approval(
        self,
        session_id: str,
        request_id: str,
        action:     str,
        tool_name:  str,
        approved:   bool,
        decided_by: str = "human",
    ) -> str:
        return await self.record(
            AuditEntry.for_approval(
                session_id = session_id,
                request_id = request_id,
                action     = action,
                tool_name  = tool_name,
                approved   = approved,
                decided_by = decided_by,
            )
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "write_count":   self._write_count,
            "error_count":   self._error_count,
            "last_hash":     self._last_hash[:16] + "..." if len(self._last_hash) > 16 else self._last_hash,
            "backend":       "supabase" if self._use_supabase else "memory",
            "memory_entries": len(self._memory_log) if not self._use_supabase else None,
        }

    async def close(self) -> None:
        if self._use_supabase and self._client:
            await self._client.close()
            log.debug("audit_log_closed")