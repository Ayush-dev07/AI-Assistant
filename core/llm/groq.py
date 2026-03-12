from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator

import httpx

from core.llm.base import (
    LLMAuthError,
    LLMContextWindowError,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from core.logging import get_logger

log = get_logger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Available free models and their context windows
GROQ_MODELS: dict[str, dict[str, Any]] = {
    "llama-3.1-8b-instant": {
        "context_window": 131072,
        "daily_tokens": 500_000,
        "supports_tools": True,
        "description": "Fast, lightweight. Good for dev/testing.",
    },
    "llama-3.3-70b-versatile": {
        "context_window": 131072,
        "daily_tokens": 100_000,
        "supports_tools": True,
        "description": "Powerful reasoning. Use for complex tasks.",
    },
    "mixtral-8x7b-32768": {
        "context_window": 32768,
        "daily_tokens": 100_000,
        "supports_tools": True,
        "description": "Strong instruction following. 32K context.",
    },
    "gemma2-9b-it": {
        "context_window": 8192,
        "daily_tokens": 250_000,
        "supports_tools": False,
        "description": "Google's Gemma 2. Good all-rounder.",
    },
}


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        max_retries: int = 3,
    ) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise LLMAuthError(
                "Groq API key is required. "
            )

        self._api_key = resolved_key
        self._model = model
        self._max_retries = max_retries
        self._model_info = GROQ_MODELS.get(model, {})

        self._http = httpx.AsyncClient(
            base_url=GROQ_API_BASE,
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        log.info(
            "groq_provider_initialized",
            model=model,
            context_window=self._model_info.get("context_window", "unknown"),
            daily_tokens=self._model_info.get("daily_tokens", "unknown"),
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tools(self) -> bool:
        return self._model_info.get("supports_tools", False)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion using Groq's inference API.
        """
        body = self._build_request(messages, tools, temperature, max_tokens)

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post("/chat/completions", json=body)
                return self._parse_response(response)

            except LLMRateLimitError:
                if attempt == self._max_retries:
                    raise
                wait = 2 ** attempt
                log.warning(
                    "groq_rate_limited",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens via Server-Sent Events.
        """
        body = self._build_request(messages, None, temperature, max_tokens)
        body["stream"] = True

        async with self._http.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def _build_request(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Build the OpenAI-compatible request body.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools and self.supports_tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters or {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"

        return body

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        """
        Parse OpenAI-format response into standard LLMResponse.
        """
        if response.status_code == 429:
            # Check how long until the rate limit resets
            reset_header = response.headers.get("x-ratelimit-reset-requests", "unknown")
            log.warning("groq_rate_limited", reset_in=reset_header)
            raise LLMRateLimitError(f"Groq rate limit. Resets in: {reset_header}")

        if response.status_code == 401:
            raise LLMAuthError("Invalid Groq API key")

        if response.status_code == 413:
            raise LLMContextWindowError("Input too long for Groq model context window")

        if response.status_code != 200:
            raise LLMProviderError(
                f"Groq API error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()

        if "error" in data:
            err = data["error"]
            code = err.get("code", "")
            msg = err.get("message", "")
            if "rate" in code.lower() or "rate" in msg.lower():
                raise LLMRateLimitError(msg)
            raise LLMProviderError(f"Groq error: {msg}")

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="", model=self._model)

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        stop_reason = choice.get("finish_reason", "")

        # Parse tool calls if present
        tool_calls = []
        raw_tool_calls = message.get("tool_calls", []) or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", f"groq-{func.get('name', 'unknown')}"),
                name=func.get("name", ""),
                arguments=args,
            ))

        # Token usage
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = 0.0

        log.debug(
            "groq_completion",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=len(tool_calls),
            stop_reason=stop_reason,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self._model,
            stop_reason=stop_reason,
        )

    async def close(self) -> None:
        await self._http.aclose()