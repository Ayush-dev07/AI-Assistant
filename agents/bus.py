from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import Any, AsyncGenerator

import httpx
from pydantic import ValidationError

from agents.protocol import AgentMessage, Channels
from core.logging import get_logger

log = get_logger(__name__)

_SSE_TIMEOUT_S = 30.0

_QUEUE_MAX_SIZE = 512

_PUBLISH_TIMEOUT_S = 10.0


class MessageBus:

    def __init__(
        self,
        upstash_url:   str | None = None,
        upstash_token: str | None = None,
    ) -> None:
        url   = upstash_url   or os.getenv("UPSTASH_REDIS_REST_URL")
        token = upstash_token or os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if url and token:
            self._use_upstash = True
            self._base_url    = url.rstrip("/")
            self._http        = httpx.AsyncClient(
                base_url = url.rstrip("/"),
                headers  = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type":  "application/json",
                },
                timeout = httpx.Timeout(
                    connect = 5.0,
                    read    = _SSE_TIMEOUT_S,
                    write   = _PUBLISH_TIMEOUT_S,
                    pool    = 5.0,
                ),
            )
            log.debug("message_bus_upstash_mode", url=url[:40] + "...")
        else:
            self._use_upstash = False
            self._queues: dict[str, asyncio.Queue[str]] = defaultdict(
                lambda: asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
            )
            log.debug("message_bus_queue_fallback")

    async def publish(
        self,
        channel: str,
        message: AgentMessage,
    ) -> bool:
        payload = message.model_dump_json()

        if self._use_upstash:
            return await self._upstash_publish(channel, payload)
        else:
            return self._queue_publish(channel, payload)

    async def _upstash_publish(self, channel: str, payload: str) -> bool:
        try:
            resp = await self._http.post(
                f"/publish/{channel}",
                content = payload.encode("utf-8"),
                timeout = _PUBLISH_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            subscriber_count = data.get("result", 0)
            log.debug(
                "bus_published_upstash",
                channel     = channel,
                subscribers = subscriber_count,
                bytes       = len(payload),
            )
            return True
        except httpx.TimeoutException:
            log.warning("bus_publish_timeout", channel=channel)
            return False
        except httpx.HTTPStatusError as exc:
            log.warning(
                "bus_publish_http_error",
                channel = channel,
                status  = exc.response.status_code,
            )
            return False
        except Exception as exc:
            log.warning("bus_publish_error", channel=channel, error=str(exc))
            return False

    def _queue_publish(self, channel: str, payload: str) -> bool:
        q = self._queues[channel]
        if q.full():
            log.warning("bus_queue_full", channel=channel, maxsize=_QUEUE_MAX_SIZE)
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(payload)
            log.debug("bus_published_queue", channel=channel)
            return True
        except asyncio.QueueFull:
            log.warning("bus_queue_full_drop", channel=channel)
            return False

    async def subscribe(
        self,
        channel:    str,
        timeout_s:  float | None = None,
    ) -> AsyncGenerator[AgentMessage, None]:
        if self._use_upstash:
            async for msg in self._upstash_subscribe(channel, timeout_s):
                yield msg
        else:
            async for msg in self._queue_subscribe(channel, timeout_s):
                yield msg

    async def _upstash_subscribe(
        self,
        channel:   str,
        timeout_s: float | None,
    ) -> AsyncGenerator[AgentMessage, None]:
        """
        Stream SSE events from Upstash /subscribe/{channel}.

        Upstash SSE format (each event):
            data: {json_string}
            \\n
        """
        try:
            async with self._http.stream("GET", f"/subscribe/{channel}") as resp:
                resp.raise_for_status()
                log.debug("bus_subscribed_upstash", channel=channel)

                async for line in resp.aiter_lines():
                    if not line:
                        continue                  
                    if line.startswith(":"):
                        continue                    

                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if not raw:
                            continue

                        msg = _parse_message(raw, channel)
                        if msg is not None:
                            yield msg

        except httpx.TimeoutException:
            log.debug("bus_subscribe_sse_timeout", channel=channel)
            return
        except Exception as exc:
            log.warning("bus_subscribe_error", channel=channel, error=str(exc))
            return

    async def _queue_subscribe(
        self,
        channel:   str,
        timeout_s: float | None,
    ) -> AsyncGenerator[AgentMessage, None]:
        q = self._queues[channel]
        log.debug("bus_subscribed_queue", channel=channel)

        while True:
            try:
                if timeout_s is not None:
                    raw = await asyncio.wait_for(q.get(), timeout=timeout_s)
                else:
                    raw = await q.get()
            except asyncio.TimeoutError:
                log.debug("bus_subscribe_queue_timeout", channel=channel)
                return

            msg = _parse_message(raw, channel)
            if msg is not None:
                yield msg

    async def subscribe_one(
        self,
        channel:   str,
        timeout_s: float = 60.0,
    ) -> AgentMessage | None:
        async for msg in self.subscribe(channel, timeout_s=timeout_s):
            return msg
        return None

    async def close(self) -> None:
        if self._use_upstash and not self._http.is_closed:
            await self._http.aclose()
            log.debug("message_bus_closed")

    @property
    def mode(self) -> str:
        return "upstash" if self._use_upstash else "queue"

    def __repr__(self) -> str:
        return f"<MessageBus mode={self.mode!r}>"

_bus_instance: MessageBus | None = None


def get_bus(
    upstash_url:   str | None = None,
    upstash_token: str | None = None,
) -> MessageBus:
    """
    Return the process-level MessageBus singleton.
    """
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = MessageBus(
            upstash_url   = upstash_url,
            upstash_token = upstash_token,
        )
    return _bus_instance


def reset_bus() -> None:
    global _bus_instance
    _bus_instance = None

def _parse_message(raw: str, channel: str) -> AgentMessage | None:
    """
    Parse a raw JSON string into an AgentMessage.
    """
    try:
        return AgentMessage.model_validate_json(raw)
    except ValidationError:
        pass

    try:
        outer = json.loads(raw)
        if isinstance(outer, dict):
            inner = outer.get("data") or outer.get("message")
            if isinstance(inner, str):
                return AgentMessage.model_validate_json(inner)
            elif isinstance(inner, dict):
                return AgentMessage.model_validate(inner)
    except (json.JSONDecodeError, ValidationError, TypeError):
        pass

    log.warning(
        "bus_message_parse_failed",
        channel  = channel,
        raw_preview = raw[:120],
    )
    return None