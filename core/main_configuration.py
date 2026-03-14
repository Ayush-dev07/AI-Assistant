from __future__ import annotations
import warnings

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
        extra="ignore",        
    )

    gemini_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Google Gemini API key. Primary free LLM provider. "
        ),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Groq API key. Backup free LLM provider (ultra-fast inference). "
        ),
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "OpenRouter API key."
        ),
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct:free",
        description=(
            "OpenRouter model identifier."
        ),
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Anthropic API key."
        ),
    )

    default_provider: str = Field(
        default="gemini",
        description=(
            "Which LLM provider to use by default. "
        ),
    )

    default_model: str = Field(
        default="gemini-1.5-flash",
        description=(
            "Default LLM model identifier. Must be a valid model string for "
        ),
    )

    upstash_redis_rest_url: str | None = Field(
        default=None,
        description=(
            "Upstash Redis REST API URL for short-term memory. "
        ),
    )
    upstash_redis_rest_token: SecretStr | None = Field(
        default=None,
        description=(
            "Upstash Redis REST API token. Found next to the REST URL. "
        ),
    )

    pinecone_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Pinecone API key for long-term vector memory. "
        ),
    )
    pinecone_index_name: str = Field(
        default="agent-memory",
        description=(
            "Name of the Pinecone index to use for long-term memory. "
        ),
    )
    pinecone_index_host: str | None = Field(
        default=None,
        description=(
            "Pinecone index host URL. Found in the Pinecone console after index creation. "
        ),
    )

    supabase_url: str | None = Field(
        default=None,
        description=(
            "Supabase project URL for episodic memory (task history). "
        ),
    )
    supabase_service_key: SecretStr | None = Field(
        default=None,
        description=(
            "Supabase service role key (not the anon key). "
        ),
    )

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max ReAct loop iterations per task. Higher = more capable but uses more tokens.",
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

    # ── Security
    vault_master_key: SecretStr = Field(
        ...,
        min_length=32,
        description=(
            "Master key for the local encrypted secrets vault. "
        ),
    )
    secret_key: SecretStr = Field(
        ...,
        min_length=16,
        description="Secret key for JWT signing and CSRF protection.",
    )

    # ── API Server 
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return upper

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str) -> str:
        allowed = {"gemini", "groq", "openrouter", "anthropic"}
        lower = v.lower().strip()
        if lower not in allowed:
            raise ValueError(
                f"default_provider must be one of {allowed}, got: {v!r}. "
                "Check core/llm/router.py for the full list of supported providers."
            )
        return lower

    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        known_prefixes = (
            "gemini-",       # Google Gemini
            "claude-",       # Anthropic Claude
            "gpt-",          # OpenAI GPT
            "o1-",           # OpenAI o1
            "llama",         # Meta Llama (Groq / OpenRouter)
            "mistral",       # Mistral AI (Groq / OpenRouter)
            "gemma",         # Google Gemma (Groq / OpenRouter)
            "mixtral",       # Mistral MoE (Groq)
            "meta-llama/",   # OpenRouter full path format
            "google/",       # OpenRouter full path format
            "mistralai/",    # OpenRouter full path format
            "microsoft/",    # OpenRouter full path format
            "qwen/",         # OpenRouter full path format
        )
        if not any(v.lower().startswith(p) for p in known_prefixes):
            warnings.warn(
                f"default_model {v!r} doesn't match any known provider prefix. "
                "Double-check the model string against your chosen provider's docs.",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def validate_at_least_one_llm_provider(self) -> "Settings":
        has_gemini = bool(
            self.gemini_api_key and self.gemini_api_key.get_secret_value()
        )
        has_groq = bool(
            self.groq_api_key and self.groq_api_key.get_secret_value()
        )
        has_openrouter = bool(
            self.openrouter_api_key and self.openrouter_api_key.get_secret_value()
        )
        has_anthropic = bool(
            self.anthropic_api_key and self.anthropic_api_key.get_secret_value()
        )

        if not any([has_gemini, has_groq, has_openrouter, has_anthropic]):
            raise ValueError(
                "No LLM provider API key found. Set at least one of:\n"
            )

        # Warn if the configured default_provider doesn't have a key
        provider_key_map = {
            "gemini": has_gemini,
            "groq": has_groq,
            "openrouter": has_openrouter,
            "anthropic": has_anthropic,
        }
        if not provider_key_map.get(self.default_provider, False):
            warnings.warn(
                f"default_provider is set to {self.default_provider!r} "
                f"but no API key was found for it. "
                f"Set the matching key or change DEFAULT_PROVIDER in your .env.",
                stacklevel=2,
            )

        return self

    @model_validator(mode="after")
    def validate_upstash_pair(self) -> "Settings":
        has_url = bool(self.upstash_redis_rest_url)
        has_token = bool(
            self.upstash_redis_rest_token
            and self.upstash_redis_rest_token.get_secret_value()
        )
        if has_url and not has_token:
            raise ValueError(
                "UPSTASH_REDIS_REST_URL is set but UPSTASH_REDIS_REST_TOKEN is missing. "
                "Both are required."
            )
        if has_token and not has_url:
            raise ValueError(
                "UPSTASH_REDIS_REST_TOKEN is set but UPSTASH_REDIS_REST_URL is missing. "
                "Both are required."
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def has_short_term_memory(self) -> bool:
        return bool(
            self.upstash_redis_rest_url
            and self.upstash_redis_rest_token
            and self.upstash_redis_rest_token.get_secret_value()
        )

    @property
    def has_long_term_memory(self) -> bool:
        return bool(
            self.pinecone_api_key
            and self.pinecone_api_key.get_secret_value()
            and self.pinecone_index_host
        )

    @property
    def has_episodic_memory(self) -> bool:
        """True if Supabase is fully configured (URL + service key both set)."""
        return bool(
            self.supabase_url
            and self.supabase_service_key
            and self.supabase_service_key.get_secret_value()
        )

settings = Settings()  # type: ignore[call-arg]