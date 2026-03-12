from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

class LLMMessage(BaseModel):
    role: str = Field(..., description="'system' | 'user' | 'assistant'")
    content: str


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    stop_reason: str = ""


class LLMProviderError(Exception):
    """Base class for all LLM provider errors."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider rate-limits your requests."""


class LLMContextWindowError(LLMProviderError):
    """Raised when the input exceeds the model's context window."""


class LLMAuthError(LLMProviderError):
    """Raised when the API key is invalid or missing."""



class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion for the given message history.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Stream the response token by token.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports function/tool calling."""
        return False