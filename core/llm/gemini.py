from __future__ import annotations

import asyncio
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

# Gemini REST API base URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
class GeminiProvider(LLMProvider):

    def __init__( 
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise LLMAuthError(
                "Gemini API key is required. "
            )

        self._api_key = resolved_key
        self._model = model
        self._max_retries = max_retries

        self._endpoint = (
            f"{GEMINI_API_BASE}/models/{model}:generateContent"
            f"?key={resolved_key}"
        )
        self._stream_endpoint = (
            f"{GEMINI_API_BASE}/models/{model}:streamGenerateContent"
            f"?alt=sse&key={resolved_key}"
        )

        self._http = httpx.AsyncClient(timeout=120.0)
        log.info("gemini_provider_initialized", model=model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tools(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion using Gemini.
        """
        body = self._build_request(messages, tools, temperature, max_tokens)

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(self._endpoint, json=body)
                return self._parse_response(response)

            except LLMRateLimitError:
                if attempt == self._max_retries:
                    raise
                wait = 2 ** attempt  # 1s, 2s, 4s
                log.warning(
                    "gemini_rate_limited",
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
        Stream response tokens as they arrive.
        """
        body = self._build_request(messages, None, temperature, max_tokens)

        async with self._http.stream("POST", self._stream_endpoint, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                json_str = line[6:] 
                if json_str.strip() == "[DONE]":
                    break
                try:
                    import json
                    chunk = json.loads(json_str)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except Exception:
                    continue

    def _build_request(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """
        Convert standard LLMMessage list to Gemini API request format.
        """
        system_parts = []
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            else:
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": msg.content}],
                })

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "candidateCount": 1,
            },
        }

        if system_parts:
            body["system_instruction"] = {"parts": system_parts}

        if tools:
            body["tools"] = [{"function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {
                        "type": "object",
                        "properties": {},
                    },
                }
                for t in tools
            ]}]
            # AUTO mode: Gemini decides whether to call a tool or respond directly
            body["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

        return body

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        """
        Parse Gemini API response into a standard LLMResponse.
        """
        # Handle HTTP errors
        if response.status_code == 429:
            raise LLMRateLimitError("Gemini rate limit exceeded")
        if response.status_code == 401:
            raise LLMAuthError("Invalid Gemini API key")
        if response.status_code == 400:
            body = response.text
            if "context window" in body.lower() or "too long" in body.lower():
                raise LLMContextWindowError("Input exceeds Gemini context window")
            raise LLMProviderError(f"Gemini bad request: {body[:300]}")
        if response.status_code != 200:
            raise LLMProviderError(
                f"Gemini API error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()

        if "error" in data:
            err = data["error"]
            raise LLMProviderError(f"Gemini error {err.get('code')}: {err.get('message')}")

        candidates = data.get("candidates", [])
        if not candidates:
            finish_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
            log.warning("gemini_empty_candidates", block_reason=finish_reason)
            return LLMResponse(content="", stop_reason=finish_reason, model=self._model)

        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        text_parts = []
        tool_calls = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                # Gemini function call format
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=f"gemini-{fc['name']}-{id(fc)}",
                    name=fc["name"],
                    arguments=fc.get("args", {}),
                ))

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        cost = 0.0

        stop_reason = candidate.get("finishReason", "")

        log.debug(
            "gemini_completion",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 8),
            tool_calls=len(tool_calls),
            stop_reason=stop_reason,
        )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self._model,
            stop_reason=stop_reason,
        )

    async def close(self) -> None:
        await self._http.aclose()