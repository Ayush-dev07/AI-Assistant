from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator
from dotenv import load_dotenv
load_dotenv()

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

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        site_url: str = "http://localhost:8000",
        app_name: str = "SuperAgent",
        max_retries: int = 3,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_key:
            raise LLMAuthError(
                "OpenRouter API key is required. "
            )

        self._model = (
            model
            or os.getenv("OPENROUTER_MODEL")
            or "meta-llama/llama-3.1-8b-instruct:free"
        )
        self._max_retries = max_retries

        self._http = httpx.AsyncClient(
            base_url=OPENROUTER_API_BASE,
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
                # OpenRouter uses these for their usage dashboard and routing
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
            timeout=90.0,
        )

        is_free = self._model.endswith(":free")
        log.info(
            "openrouter_provider_initialized",
            model=self._model,
            free_tier=is_free,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tools(self) -> bool:
        return not self._model.endswith(":free")

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
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
                    "openrouter_rate_limited",
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
        if response.status_code == 429:
            raise LLMRateLimitError("OpenRouter rate limit exceeded")
        if response.status_code == 401:
            raise LLMAuthError("Invalid OpenRouter API key")
        if response.status_code == 400:
            body_text = response.text
            if "context" in body_text.lower():
                raise LLMContextWindowError("Input too long for model context window")
            raise LLMProviderError(f"OpenRouter bad request: {body_text[:300]}")
        if response.status_code != 200:
            raise LLMProviderError(
                f"OpenRouter error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()

        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err))
            if "rate" in msg.lower():
                raise LLMRateLimitError(msg)
            raise LLMProviderError(f"OpenRouter error: {msg}")

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="", model=self._model)

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Parse tool calls
        tool_calls = []
        raw_tool_calls = message.get("tool_calls", []) or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", f"or-{func.get('name', 'unknown')}"),
                name=func.get("name", ""),
                arguments=args,
            ))

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = float(data.get("usage", {}).get("cost", 0.0))

        log.debug(
            "openrouter_completion",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self._model,
            stop_reason=choice.get("finish_reason", ""),
        )

    async def close(self) -> None:
        await self._http.aclose()