from __future__ import annotations

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # ANTHROPIC_API_KEY = anthropic_api_key
        extra="ignore",        # Don't fail on extra env vars (CI often has many)
    )

    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key. Required. Get one at console.anthropic.com",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key. Optional. Only needed if using GPT models.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server. Used for local model inference.",
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model identifier. Must be a valid model string.",
    )

    # Memory / Databases
    chroma_path: str = Field(
        default="./data/chroma",
        description="Filesystem path where ChromaDB persists vector embeddings.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL. Used for short-term memory and task queue.",
    )
    postgres_url: str = Field(
        default="postgresql://sa:changeme@localhost:5432/superagent",
        description="PostgreSQL connection URL. Used for episodic memory (task logs).",
    )

    # Agent Behaviour 
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max ReAct loop iterations per task. Higher = more capable but costlier.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Max tokens per LLM call. Larger allows longer reasoning chains.",
    )
    short_term_window: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Number of recent messages kept in short-term memory context.",
    )

    # Observability 
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity. Use DEBUG locally, INFO/WARNING in production.",
    )
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry distributed tracing. Requires a collector.",
    )
    prometheus_port: int = Field(
        default=9090,
        description="Port to expose Prometheus /metrics endpoint.",
    )

    # Security 
    vault_master_key: SecretStr = Field(
        ...,
        min_length=32,
        description=(
            "Master key for the local encrypted secrets vault. "
            "Must be at least 32 characters. Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        ),
    )
    secret_key: SecretStr = Field(
        ...,
        min_length=16,
        description="Secret key for JWT signing and CSRF protection.",
    )

    # API Server 
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    # Validators 

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return upper

    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        known_prefixes = ("claude-", "gpt-", "o1-", "llama", "mistral", "gemma")
        if not any(v.lower().startswith(p) for p in known_prefixes):
            # Don't hard-fail — custom/new models should still work
            import warnings
            warnings.warn(
                f"default_model {v!r} doesn't match any known provider prefix. "
                "Double-check the model string.",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def validate_at_least_one_provider(self) -> "Settings":
        if not self.anthropic_api_key.get_secret_value():
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Module-level singleton
settings = Settings()  # type: ignore[call-arg]