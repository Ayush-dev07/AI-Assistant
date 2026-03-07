from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import anthropic

from core.llm.base import (
    LLMContextWindowError,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from core.logging import METRICS, get_logger

log = get_logger(__name__)

_CLAUDE_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001":   {"input": 0.80e-6,  "output": 4.00e-6},
    "claude-sonnet-4-20250514":     {"input": 3.00e-6,  "output": 15.00e-6},
    "claude-opus-4-20250514":       {"input": 15.00e-6, "output": 75.00e-6},
}
_DEFAULT_PRICING = {"input": 3.00e-6, "output": 15.00e-6}


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

        self._pricing = _CLAUDE_PRICING.get(model, _DEFAULT_PRICING)

        log.info(
            "claude_provider_initialized",
            model=model,
            input_price_per_token=self._pricing["input"],
            output_price_per_token=self._pricing["output"],
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"  # Anthropic takes system as a separate param
        ]

        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        # Retry loop 
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                log.warning(
                    "claude_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    delay_s=delay,
                    error=str(last_error),
                )
                await asyncio.sleep(delay)

            try:
                start_time = time.monotonic()

                # Make the actual API call
                kwargs: dict = {
                    "model": self._model,
                    "messages": anthropic_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if system:
                    kwargs["system"] = system
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                response = await self._client.messages.create(**kwargs)
                duration = time.monotonic() - start_time

                # Parse response 
                text_content = ""
                tool_calls: list[ToolCall] = []

                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            tool_name=block.name,
                            tool_call_id=block.id,
                            arguments=block.input,
                        ))

                # Cost calculation 
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (
                    input_tokens * self._pricing["input"]
                    + output_tokens * self._pricing["output"]
                )

                # Metrics recording 
                METRICS.llm_tokens_total.labels(
                    provider="anthropic", model=self._model, direction="input"
                ).inc(input_tokens)
                METRICS.llm_tokens_total.labels(
                    provider="anthropic", model=self._model, direction="output"
                ).inc(output_tokens)
                METRICS.llm_cost_usd_total.labels(
                    provider="anthropic", model=self._model
                ).inc(cost)
                METRICS.llm_call_duration.labels(
                    provider="anthropic", model=self._model
                ).observe(duration)

                log.info(
                    "llm_call_completed",
                    provider="anthropic",
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost, 6),
                    duration_s=round(duration, 3),
                    stop_reason=response.stop_reason,
                    has_tool_calls=len(tool_calls) > 0,
                )

                return LLMResponse(
                    content=text_content,
                    tool_calls=tool_calls,
                    model=self._model,
                    provider="anthropic",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    stop_reason=response.stop_reason or "end_turn",
                )

            # Error handling 
            except anthropic.RateLimitError as e:
                retry_after = float(e.response.headers.get("retry-after", 60))
                log.warning("claude_rate_limited", retry_after=retry_after)
                if attempt == self._max_retries:
                    raise LLMRateLimitError(provider="anthropic", retry_after=retry_after) from e
                await asyncio.sleep(retry_after)
                last_error = e

            except anthropic.BadRequestError as e:
                # Bad request = our fault. Don't retry.
                if "context_window" in str(e).lower():
                    raise LLMContextWindowError(str(e), provider="anthropic") from e
                raise LLMProviderError(str(e), provider="anthropic", status_code=400) from e

            except anthropic.AuthenticationError as e:
                # Wrong API key. Don't retry — it won't help.
                raise LLMProviderError(
                    "Anthropic API key is invalid. Check ANTHROPIC_API_KEY in .env",
                    provider="anthropic",
                    status_code=401,
                ) from e

            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:

                last_error = e
                if attempt == self._max_retries:
                    raise LLMProviderError(
                        f"Anthropic API failed after {self._max_retries} retries: {e}",
                        provider="anthropic",
                    ) from e

        raise LLMProviderError("Max retries exceeded", provider="anthropic")

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"
        ]

        async with self._client.messages.stream(
            model=self._model,
            messages=anthropic_messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text