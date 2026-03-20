from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

import httpx

from core.logging import get_logger

if TYPE_CHECKING:
    from Security.audit_log import AuditLog

log = get_logger(__name__)

class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

@dataclass
class Thresholds:
    tool_calls_per_minute:    float = 30.0
    failed_calls_per_minute:  float = 5.0
    cost_per_hour_usd:        float = 0.01
    memory_writes_per_minute: float = 20.0

    def as_dict(self) -> dict[str, float]:
        return {
            "tool_calls_per_minute":    self.tool_calls_per_minute,
            "failed_calls_per_minute":  self.failed_calls_per_minute,
            "cost_per_hour_usd":        self.cost_per_hour_usd,
            "memory_writes_per_minute": self.memory_writes_per_minute,
        }

METRIC_WINDOWS: dict[str, int] = {
    "tool_calls_per_minute":    60,
    "failed_calls_per_minute":  60,
    "cost_per_hour_usd":        3600,
    "memory_writes_per_minute": 60,
}

CIRCUIT_OPEN_ERROR = (
    "Circuit breaker is OPEN — agent execution halted to prevent "
    "quota exhaustion or runaway loop. "
    "The circuit will enter HALF_OPEN after the cooldown period."
)

@dataclass
class _Counter:
    """One in-process sliding window counter with TTL expiry."""
    value:       float = 0.0
    expiry_time: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return time.monotonic() > self.expiry_time

    def add(self, amount: float, window_s: int) -> float:
        """Increment counter. Resets to amount if window has expired."""
        if self.is_expired():
            self.value       = amount
            self.expiry_time = time.monotonic() + window_s
        else:
            self.value += amount
        return self.value

    def get(self) -> float:
        """Return current value, or 0.0 if expired."""
        if self.is_expired():
            return 0.0
        return self.value

class _UpstashCounterClient:

    def __init__(self, rest_url: str, rest_token: str) -> None:
        self._url  = rest_url.rstrip("/")
        self._http = httpx.AsyncClient(
            headers = {
                "Authorization": f"Bearer {rest_token}",
                "Content-Type":  "application/json",
            },
            timeout = 5.0,   
        )

    async def _command(self, *args: Any) -> Any:
        try:
            path     = "/".join(str(a) for a in args)
            response = await self._http.post(f"{self._url}/{path}")
            response.raise_for_status()
            return response.json().get("result")
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.debug("anomaly_upstash_command_failed", command=args[0], error=str(exc))
            return None

    async def incrbyfloat(self, key: str, value: float) -> float | None:
        result = await self._command("INCRBYFLOAT", key, value)
        if result is None:
            return None
        try:
            return float(result)
        except (TypeError, ValueError):
            return None

    async def expire(self, key: str, seconds: int) -> bool:
        result = await self._command("EXPIRE", key, seconds)
        return result == 1

    async def get(self, key: str) -> float:
        result = await self._command("GET", key)
        if result is None:
            return 0.0
        try:
            return float(result)
        except (TypeError, ValueError):
            return 0.0

    async def delete(self, key: str) -> int:
        result = await self._command("DEL", key)
        return int(result or 0)

    async def close(self) -> None:
        await self._http.aclose()

class AnomalyDetector:

    def __init__(
        self,
        upstash_url:   str | None = None,
        upstash_token: str | None = None,
        thresholds:    Thresholds | None = None,
        cooldown_s:    float = 60.0,
        audit_log:     "AuditLog | None" = None,
    ) -> None:
        url   = upstash_url   or os.getenv("UPSTASH_REDIS_REST_URL")
        token = upstash_token or os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if url and token:
            self._upstash      = _UpstashCounterClient(url, token)
            self._use_upstash  = True
            log.debug("anomaly_detector_upstash_mode", url=url[:40] + "...")
        else:
            self._upstash      = None
            self._use_upstash  = False
            self._local:  dict[str, _Counter] = {}
            log.debug(
                "anomaly_detector_local_fallback",
                hint="Set UPSTASH_REDIS_REST_URL + TOKEN for cross-process detection.",
            )

        self._thresholds  = thresholds or Thresholds()
        self._cooldown_s  = cooldown_s
        self._audit       = audit_log

        self._state:      CircuitState = CircuitState.CLOSED
        self._opened_at:  float | None = None   
        self._open_reason: str         = ""

        self._half_open_lock = asyncio.Lock()
        self._half_open_test_in_flight = False

    async def record(
        self,
        session_id: str,
        metric:     str,
        value:      float = 1.0,
    ) -> None:
        if metric not in METRIC_WINDOWS:
            log.debug("anomaly_unknown_metric", metric=metric)
            return

        window_s = METRIC_WINDOWS[metric]
        key      = _counter_key(session_id, metric, window_s)

        if self._use_upstash:
            new_val = await self._upstash.incrbyfloat(key, value)
            if new_val is not None:
                
                await self._upstash.expire(key, window_s)
        else:
            counter = self._local.setdefault(key, _Counter())
            counter.add(value, window_s)

    async def record_cost(
        self,
        session_id: str,
        cost_usd:   float,
    ) -> None:
        await self.record(session_id, "cost_per_hour_usd", cost_usd)

    async def check(
        self,
        session_id: str,
    ) -> tuple[bool, str]:
        
        if self._state == CircuitState.OPEN:
            
            if self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._cooldown_s:
                    self._transition_to_half_open()
                    return False, ""   
            return True, self._open_reason

        if self._state == CircuitState.HALF_OPEN:
            
            return False, ""
        
        threshold_dict = self._thresholds.as_dict()

        for metric, threshold in threshold_dict.items():
            window_s = METRIC_WINDOWS.get(metric, 60)
            key      = _counter_key(session_id, metric, window_s)
            current  = await self._get_counter(key)

            if current > threshold:
                reason = f"{metric} = {current:.1f} exceeds {threshold}"
                self._transition_to_open(reason)
                await self._record_state_change(session_id, "open", reason)
                return True, reason

        return False, ""

    

    def notify_success(self, session_id: str) -> None:
        """
        Notify the circuit breaker that a tool call succeeded.
        """
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_closed(session_id)

    def notify_failure(self, session_id: str) -> None:
        """
        Notify the circuit breaker that a tool call failed.
        """
        if self._state == CircuitState.HALF_OPEN:
            reason = "HALF_OPEN test call failed — re-opening circuit"
            self._transition_to_open(reason)
            log.info(
                "circuit_breaker_half_open_failed",
                session_id = session_id,
            )

    def reset_circuit(self) -> None:
        """
        Manually reset the circuit to CLOSED.
        """
        old_state   = self._state
        self._state = CircuitState.CLOSED
        self._opened_at   = None
        self._open_reason = ""
        self._half_open_test_in_flight = False
        log.info(
            "circuit_breaker_manual_reset",
            old_state = old_state.value,
        )

    async def reset_session(self, session_id: str) -> None:
        """
        Clear all counters for a session.
        """
        for metric, window_s in METRIC_WINDOWS.items():
            key = _counter_key(session_id, metric, window_s)
            if self._use_upstash:
                await self._upstash.delete(key)
            else:
                self._local.pop(key, None)
        log.info("anomaly_session_reset", session_id=session_id)

    @property
    def is_open(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return False
        if self._state == CircuitState.HALF_OPEN:
            return False
        
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._cooldown_s:
                self._transition_to_half_open()
                return False
        return True

    @property
    def circuit_state(self) -> CircuitState:
        return self._state

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    @property
    def open_reason(self) -> str:
        return self._open_reason

    @property
    def cooldown_remaining_s(self) -> float:
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return 0.0
        remaining = self._cooldown_s - (time.monotonic() - self._opened_at)
        return max(0.0, remaining)

    async def get_counters(
        self,
        session_id: str,
    ) -> dict[str, float]:
        result = {}
        for metric, window_s in METRIC_WINDOWS.items():
            key = _counter_key(session_id, metric, window_s)
            result[metric] = await self._get_counter(key)
        return result

    async def _get_counter(self, key: str) -> float:
        if self._use_upstash:
            return await self._upstash.get(key)
        counter = self._local.get(key)
        return counter.get() if counter else 0.0

    def _transition_to_open(self, reason: str) -> None:
        old_state         = self._state
        self._state       = CircuitState.OPEN
        self._opened_at   = time.monotonic()
        self._open_reason = reason
        self._half_open_test_in_flight = False
        log.warning(
            "circuit_breaker_opened",
            reason    = reason,
            old_state = old_state.value,
            cooldown_s = self._cooldown_s,
        )

    def _transition_to_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        log.info(
            "circuit_breaker_half_open",
            was_open_s = (
                round(time.monotonic() - self._opened_at, 1)
                if self._opened_at else 0
            ),
        )

    def _transition_to_closed(self, session_id: str) -> None:
        self._state                    = CircuitState.CLOSED
        self._opened_at                = None
        self._open_reason              = ""
        self._half_open_test_in_flight = False
        log.info(
            "circuit_breaker_closed",
            session_id = session_id,
        )

    async def _record_state_change(
        self,
        session_id: str,
        new_state:  str,
        reason:     str,
    ) -> None:
        if self._audit is None:
            return
        try:
            from Security.audit_log import AuditEntry, ActionType
            await self._audit.record(AuditEntry(
                session_id     = session_id,
                action_type    = ActionType.CIRCUIT_BREAKER,
                result_summary = f"state={new_state} reason={reason}"[:500],
                threat_level   = "safe",
            ))
        except Exception as exc:
            log.debug("anomaly_audit_failed", error=str(exc))

    async def close(self) -> None:
        if self._use_upstash and self._upstash:
            await self._upstash.close()

    def __repr__(self) -> str:
        return (
            f"<AnomalyDetector "
            f"state={self._state.value} "
            f"backend={'upstash' if self._use_upstash else 'local'}>"
        )

def _counter_key(session_id: str, metric: str, window_s: int) -> str:
    return f"anomaly:{session_id}:{metric}:{window_s}s"