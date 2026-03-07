"""
core/llm/base.py
================
Task 1.3 — LLM Provider Abstraction Layer (Base Interface)

The Problem This Solves:
------------------------
If your agent calls `anthropic.messages.create(...)` directly, you've painted
yourself into a corner. Swapping to GPT-4o means rewriting every single LLM
call in the codebase. Adding cost tracking means editing every call site.
Adding retry logic means copy-pasting it everywhere.

The Solution — Provider Pattern:
---------------------------------
Define one abstract interface (LLMProvider) that describes what any LLM must
be able to do. Then write concrete implementations (ClaudeProvider, OpenAIProvider,
OllamaProvider) that satisfy that interface. The rest of the codebase only
imports the abstract base — it never knows or cares which concrete provider
it's talking to.

This is a classic application of the Liskov Substitution Principle:
any LLMProvider subclass should be substitutable for any other.

Data Models:
------------
All inputs and outputs are Pydantic models, not raw dicts. This means:
- Autocomplete works in your IDE
- Type errors are caught at dev time (not at 2am in production)
- Serialization to/from JSON is automatic
- Every response carries cost metadata for budgeting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel, Field, field_validator


# ─── Data Models ─────────────────────────────────────────────────────────────

class LLMMessage(BaseModel):
    """
    A single message in a conversation with an LLM.

    role must be one of: "system" | "user" | "assistant"
    content is the text of the message.

    In a multi-turn conversation, you build a list of these and pass them
    to LLMProvider.complete(). The LLM uses the history to maintain context.
    """

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., min_length=1, description="The message text")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got: {v!r}")
        return v


class ToolDefinition(BaseModel):
    """
    Describes a tool that the LLM can call.

    This maps to Anthropic's tool_use format and OpenAI's function_call format.
    The agent's ToolRegistry converts BaseTool instances into these definitions
    before passing them to the LLM.
    """

    name: str = Field(..., description="Unique tool identifier. Snake_case by convention.")
    description: str = Field(
        ...,
        description=(
            "What this tool does and WHEN to use it. "
            "The LLM reads this to decide whether to call the tool. "
            "Be specific — vague descriptions lead to wrong tool selection."
        ),
    )
    input_schema: dict = Field(
        ...,
        description=(
            "JSON Schema for the tool's input parameters. "
            "Generated from a Pydantic model via .model_json_schema(). "
            "The LLM uses this to construct valid tool call arguments."
        ),
    )


class ToolCall(BaseModel):
    """
    A tool call requested by the LLM in its response.

    When the LLM decides to use a tool, it returns a ToolCall instead of
    (or in addition to) text. The agent's executor receives this, runs
    the tool, and feeds the result back as a user message.
    """

    tool_name: str = Field(..., description="Which tool to call (matches ToolDefinition.name)")
    tool_call_id: str = Field(..., description="Unique ID for this call, used to match results")
    arguments: dict = Field(default_factory=dict, description="Tool arguments as a dict")


class LLMResponse(BaseModel):
    """
    The complete response from an LLM call.

    Every response carries:
    - content: the text the LLM generated (empty string if it only made tool calls)
    - tool_calls: any tool invocations the LLM requested
    - cost metadata: tokens used and estimated USD cost (for budget tracking)
    - stop_reason: why generation ended (helps with retry logic)
    """

    # The actual generated text
    content: str = Field(default="", description="Text content of the response")

    # Tool calls the LLM wants to make (can be empty list)
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the LLM. Process these before continuing.",
    )

    # Provider metadata
    model: str = Field(..., description="Exact model string that generated this response")
    provider: str = Field(..., description="Provider name: 'anthropic' | 'openai' | 'ollama'")

    # Token counts — critical for cost tracking and context window management
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    # Calculated cost in USD (provider-specific rates applied by each adapter)
    cost_usd: float = Field(default=0.0, ge=0.0)

    # Why generation stopped
    stop_reason: str = Field(
        default="end_turn",
        description=(
            "Why the LLM stopped generating. "
            "'end_turn': natural completion. "
            "'tool_use': LLM wants to call a tool. "
            "'max_tokens': hit token limit (may need to increase max_tokens). "
            "'stop_sequence': hit a configured stop string."
        ),
    )

    @property
    def has_tool_calls(self) -> bool:
        """Convenience: did the LLM request any tool calls?"""
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this call."""
        return self.input_tokens + self.output_tokens


# ─── Abstract Provider Interface ──────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    To add a new provider:
      1. Create a new file: core/llm/myprovider.py
      2. Subclass LLMProvider
      3. Implement `complete()` and `stream()`
      4. Register in core/llm/router.py

    The rest of the codebase never changes.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider. Used in metrics labels."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The exact model string being used. Used in metrics labels."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a conversation to the LLM and return its complete response.

        This is the core method all agents use. It's async because LLM APIs
        are network calls that can take several seconds.

        Args:
            messages: Full conversation history. User and assistant turns alternate.
            system: System prompt — sets the agent's persona, constraints, and
                    output format requirements. Not included in messages list.
            tools: Tool definitions available to the LLM. If provided, the LLM
                   may choose to call one instead of (or alongside) generating text.
            max_tokens: Upper bound on response length. Hitting this limit means
                        the response was cut off — increase if you see truncated outputs.
            temperature: Sampling temperature 0.0–1.0.
                         0.0 = deterministic (good for structured outputs, tool calls)
                         0.7 = balanced (good for reasoning and planning)
                         1.0 = creative (good for brainstorming, variety)

        Returns:
            LLMResponse with content, any tool calls, and token/cost metadata.

        Raises:
            LLMProviderError: If the API call fails after retries.
            LLMRateLimitError: If rate limited (caller should back off and retry).
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream the LLM response token by token.

        Yields text chunks as they arrive from the API. Use this for the CLI
        and WebSocket endpoints where you want to show output progressively
        rather than waiting for the full response.

        Note: Streaming doesn't support tool calls — use complete() for those.
        """
        ...


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class LLMProviderError(Exception):
    """Base class for LLM provider errors."""
    def __init__(self, message: str, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM API returns a rate limit response (HTTP 429)."""
    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(f"Rate limited by {provider}", provider=provider, status_code=429)
        self.retry_after = retry_after


class LLMContextWindowError(LLMProviderError):
    """Raised when the conversation is too long for the model's context window."""
    pass