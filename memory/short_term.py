from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger

log = get_logger(__name__)


# ─── Data Model ──────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single message in the conversation window."""

    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., description="Message text")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got: {v!r}")
        return v


# ─── Upstash Redis Client ─────────────────────────────────────────────────────

class UpstashRedisClient:

    def __init__(self, rest_url: str, rest_token: str) -> None:
        self._url = rest_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {rest_token}",
            "Content-Type": "application/json",
        }
        # httpx.AsyncClient is lightweight — one instance, reused across calls
        self._http = httpx.AsyncClient(
            headers=self._headers,
            timeout=10.0,  # 10-second timeout — Upstash is usually <100ms
        )

    async def _command(self, *args: Any) -> Any:
        try:
            # Build URL: base_url/COMMAND/arg1/arg2
            path = "/".join(str(a) for a in args)
            response = await self._http.post(f"{self._url}/{path}")
            response.raise_for_status()
            data = response.json()
            return data.get("result")
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            log.warning("upstash_command_failed", command=args[0], error=str(e))
            return None

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """SET key value EX seconds — store with expiry."""
        result = await self._command("SET", key, value, "EX", seconds)
        return result == "OK"

    async def get(self, key: str) -> str | None:
        """GET key — retrieve a value."""
        return await self._command("GET", key)

    async def delete(self, key: str) -> int:
        """DEL key — delete a key."""
        return await self._command("DEL", key) or 0

    async def exists(self, key: str) -> bool:
        """EXISTS key — check if a key exists."""
        return bool(await self._command("EXISTS", key))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()


# ─── Short-Term Memory ────────────────────────────────────────────────────────

class ShortTermMemory:
    def __init__(
        self,
        window_size: int = 20,
        session_id: str = "default",
        upstash_url: str | None = None,
        upstash_token: str | None = None,
    ) -> None:
        self._window_size = window_size
        self._session_id = session_id

        # In-process sliding window — always works, no network needed
        self._messages: deque[Message] = deque(maxlen=window_size)

        # Redis key for this session's data in Upstash
        self._redis_key = f"st:{session_id}"

        # Set up Upstash client if credentials are available
        # Priority: constructor args → environment variables → None (disabled)
        url = upstash_url or os.getenv("UPSTASH_REDIS_REST_URL")
        token = upstash_token or os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if url and token:
            self._upstash = UpstashRedisClient(url, token)
            log.info(
                "short_term_memory_upstash_enabled",
                session=session_id,
                window_size=window_size,
            )
        else:
            self._upstash = None
            log.info(
                "short_term_memory_local_only",
                session=session_id,
                window_size=window_size,
                reason="UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not set",
            )

    async def add(self, message: dict[str, Any]) -> None:
        msg = Message(**message)
        self._messages.append(msg)

        log.debug(
            "short_term_memory_add",
            session=self._session_id,
            role=msg.role,
            content_length=len(msg.content),
            window_fill=f"{len(self._messages)}/{self._window_size}",
        )

        # Persist to Upstash asynchronously — non-blocking, fire and forget
        # If this fails, it's logged but the in-process deque is unaffected
        if self._upstash:
            await self._persist()

    async def get_context(self) -> str:
        if not self._messages:
            return ""

        lines = []
        for msg in self._messages:
            role_label = msg.role.capitalize()
            # Truncate very long messages to keep context window usage sane
            content = (
                msg.content[:500] + "…"
                if len(msg.content) > 500
                else msg.content
            )
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    async def get_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    async def clear(self) -> None:
        self._messages.clear()
        if self._upstash:
            await self._upstash.delete(self._redis_key)
        log.info("short_term_memory_cleared", session=self._session_id)

    async def restore_from_cloud(self) -> bool:
        if not self._upstash:
            return False

        raw = await self._upstash.get(self._redis_key)
        if not raw:
            return False

        try:
            messages = json.loads(raw)
            # Only restore the last window_size messages
            for msg_dict in messages[-self._window_size:]:
                self._messages.append(Message(**msg_dict))
            log.info(
                "short_term_memory_restored",
                session=self._session_id,
                count=len(self._messages),
            )
            return True
        except (json.JSONDecodeError, KeyError) as e:
            log.error(
                "short_term_memory_restore_failed",
                session=self._session_id,
                error=str(e),
            )
            self._messages.clear()
            return False

    async def _persist(self) -> None:
        try:
            data = json.dumps([m.model_dump() for m in self._messages])
            await self._upstash.setex(self._redis_key, 86400, data)
        except Exception as e:
            log.warning(
                "short_term_memory_persist_failed",
                session=self._session_id,
                error=str(e),
            )

    async def close(self) -> None:
        """Close the Upstash HTTP client. Call at application shutdown."""
        if self._upstash:
            await self._upstash.close()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        backend = "upstash" if self._upstash else "local"
        return (
            f"ShortTermMemory("
            f"session={self._session_id!r}, "
            f"messages={len(self._messages)}/{self._window_size}, "
            f"backend={backend!r})"
        )