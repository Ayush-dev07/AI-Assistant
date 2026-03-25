from __future__ import annotations

import os
import re

from core.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    ToolDefinition,
)
from core.logging import get_logger

log = get_logger(__name__)

# Patterns that indicate a task needs a more powerful model
_COMPLEX_SIGNALS = re.compile(
    r"\b(analyze|analysis|compare|research|explain|synthesize|evaluate|"
    r"argue|debate|critique|review|complex|detailed|comprehensive|thorough|"
    r"step.by.step|reason|reasoning|think through)\b",
    re.IGNORECASE,
)

# Patterns that indicate a coding task (usually needs stronger model)
_CODE_SIGNALS = re.compile(
    r"\b(code|implement|function|class|debug|refactor|optimize|"
    r"algorithm|program|script|api|database|sql|python|javascript)\b",
    re.IGNORECASE,
)

# Patterns that indicate a simple, fast task
_SIMPLE_SIGNALS = re.compile(
    r"\b(what is|who is|when|where|how many|define|list|translate|"
    r"summarize|spell|convert|calculate|simple|quick|brief)\b",
    re.IGNORECASE,
)


def create_provider_from_env(provider_name: str | None = None) -> LLMProvider:
    name = (provider_name or os.getenv("DEFAULT_PROVIDER", "gemini")).lower().strip()

    if name == "gemini":
        from core.llm.gemini import GeminiProvider
        model = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        return GeminiProvider(model=model)

    elif name == "groq":
        from core.llm.groq import GroqProvider
        model = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
        return GroqProvider(model=model)

    elif name == "openrouter":
        from core.llm.openrouter import OpenRouterProvider
        model = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        return OpenRouterProvider(model=model)

    elif name == "claude":
        # Keep Claude support — useful when you eventually get credits
        from core.llm.claude import ClaudeProvider
        model = os.getenv("DEFAULT_MODEL", "claude-haiku-4-5-20251001")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY not set. "
                "Either set the key or change DEFAULT_PROVIDER to gemini or groq."
            )
        return ClaudeProvider(api_key=api_key, model=model)

    else:
        raise LLMProviderError(
            f"Unknown provider: {name!r}. "
            f"Supported: gemini, groq, openrouter, claude. "
            f"Set DEFAULT_PROVIDER in your .env file."
        )


class RouterProvider(LLMProvider):
    def __init__(
        self,
        fast_provider: LLMProvider,
        powerful_provider: LLMProvider | None = None,
        auto_fallback: bool = True,
    ) -> None:
        self._fast = fast_provider
        # If no powerful provider, use fast for everything
        self._powerful = powerful_provider or fast_provider
        self._auto_fallback = auto_fallback

    @property
    def model_name(self) -> str:
        return f"router({self._fast.model_name}/{self._powerful.model_name})"

    @property
    def supports_tools(self) -> bool:
        return self._fast.supports_tools

    def _classify_task(self, messages: list[LLMMessage]) -> str:
        last_user_content = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_content = msg.content
                break

        if not last_user_content:
            return "simple"

        if _CODE_SIGNALS.search(last_user_content):
            return "code"
        if _COMPLEX_SIGNALS.search(last_user_content):
            return "complex"
        if _SIMPLE_SIGNALS.search(last_user_content):
            return "simple"
        
        word_count = len(last_user_content.split())
        return "complex" if word_count > 50 else "simple"

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        task_type = self._classify_task(messages)
        use_powerful = task_type in ("complex", "code")
        provider = self._powerful if use_powerful else self._fast

        log.debug(
            "router_selected_provider",
            task_type=task_type,
            provider=provider.model_name,
        )

        try:
            return await provider.complete(messages, tools, temperature, max_tokens)

        except LLMRateLimitError:
            if self._auto_fallback and use_powerful and provider is not self._fast:
                log.warning(
                    "router_falling_back",
                    reason="powerful_provider_rate_limited",
                    fallback=self._fast.model_name,
                )
                return await self._fast.complete(messages, tools, temperature, max_tokens)
            raise

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Route streaming to the fast provider (streaming is for real-time display)."""
        async for chunk in self._fast.stream(messages, temperature, max_tokens):
            yield chunk

    def routing_summary(self) -> str:
        """Return a human-readable summary of the routing configuration."""
        same = self._fast is self._powerful
        if same:
            return f"Single provider: {self._fast.model_name}"
        return (
            f"Fast (simple tasks):   {self._fast.model_name}\n"
            f"Powerful (complex):    {self._powerful.model_name}\n"
            f"Auto-fallback:         {self._auto_fallback}"
        )