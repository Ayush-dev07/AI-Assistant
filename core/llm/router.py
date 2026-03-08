from __future__ import annotations

import re

from core.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolDefinition
from core.logging import get_logger

log = get_logger(__name__)

# Keywords that signal a task needs more capable models
_COMPLEX_SIGNALS = re.compile(
    r"\b(analyze|critique|compare|evaluate|research|synthesize|"
    r"implement|architect|design|complex|thorough|comprehensive|"
    r"step.by.step|reason|carefully|in.depth)\b",
    re.IGNORECASE,
)

_CODE_SIGNALS = re.compile(
    r"\b(code|implement|debug|function|class|algorithm|script|"
    r"program|refactor|test|unit.test)\b",
    re.IGNORECASE,
)

_SIMPLE_SIGNALS = re.compile(
    r"\b(what.is|define|list|summarize|translate|convert|format|"
    r"spell.check|grammar|simple|quick|brief)\b",
    re.IGNORECASE,
)


class RouterProvider(LLMProvider):
    def __init__(
        self,
        providers: dict[str, LLMProvider],
        cost_limit_usd: float | None = None,
    ) -> None:
        if "balanced" not in providers:
            raise ValueError("RouterProvider requires at least a 'balanced' provider")

        self._providers = providers
        self._cost_limit = cost_limit_usd
        self._session_cost = 0.0

        # Track routing decisions for analysis
        self._routing_log: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "router"

    @property
    def model_name(self) -> str:
        return "auto"

    def _select_tier(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None,
    ) -> str:
        # Combine all message content for analysis
        all_text = " ".join(m.content for m in messages)
        total_chars = len(all_text)

        # Feature extraction
        has_tools = bool(tools)
        is_complex = bool(_COMPLEX_SIGNALS.search(all_text))
        is_code = bool(_CODE_SIGNALS.search(all_text))
        is_simple = bool(_SIMPLE_SIGNALS.search(all_text))
        is_long = total_chars > 3000   # Rough proxy for token count (1 token ≈ 4 chars)
        is_very_long = total_chars > 8000

        # ── Decision tree ─────────────────────────────────────────────────────
        # Upgrade to powerful
        if is_very_long and is_complex:
            tier = "powerful"
        elif is_complex and is_code:
            tier = "powerful"
        # Balanced
        elif has_tools or is_code or is_complex or is_long:
            tier = "balanced"
        # Fast (simple, short, no tools)
        elif is_simple and not is_long:
            tier = "fast"
        # Default
        else:
            tier = "balanced"

        # Fall back if the selected tier isn't configured
        if tier not in self._providers:
            log.debug("router_tier_fallback", requested=tier, using="balanced")
            tier = "balanced"

        # Downgrade if we're over cost limit
        if self._cost_limit and self._session_cost >= self._cost_limit:
            tier = "fast" if "fast" in self._providers else "balanced"
            log.warning(
                "router_cost_limit_reached",
                session_cost=self._session_cost,
                limit=self._cost_limit,
                downgraded_to=tier,
            )

        log.debug(
            "router_decision",
            tier=tier,
            has_tools=has_tools,
            is_complex=is_complex,
            is_code=is_code,
            is_simple=is_simple,
            input_chars=total_chars,
        )
        return tier

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Route to the best provider and run the completion."""
        tier = self._select_tier(messages, tools)
        provider = self._providers[tier]

        response = await provider.complete(
            messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Track cumulative cost for budget management
        self._session_cost += response.cost_usd
        self._routing_log.append({
            "tier": tier,
            "model": response.model,
            "cost_usd": response.cost_usd,
            "tokens": response.total_tokens,
        })

        return response

    async def stream(self, messages, *, system="", max_tokens=4096, temperature=0.7):  # type: ignore[override]
        """Route streaming to the balanced provider (fast enough for real-time output)."""
        provider = self._providers.get("balanced") or next(iter(self._providers.values()))
        async for chunk in provider.stream(
            messages, system=system, max_tokens=max_tokens, temperature=temperature
        ):
            yield chunk

    def routing_summary(self) -> dict:
        if not self._routing_log:
            return {"calls": 0, "total_cost_usd": 0.0, "by_tier": {}}

        by_tier: dict[str, dict] = {}
        for entry in self._routing_log:
            t = entry["tier"]
            if t not in by_tier:
                by_tier[t] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            by_tier[t]["calls"] += 1
            by_tier[t]["cost_usd"] += entry["cost_usd"]
            by_tier[t]["tokens"] += entry["tokens"]

        return {
            "calls": len(self._routing_log),
            "total_cost_usd": round(self._session_cost, 6),
            "by_tier": by_tier,
        }