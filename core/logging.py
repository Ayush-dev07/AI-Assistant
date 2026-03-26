from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import structlog
from prometheus_client import Counter, Histogram, start_http_server

@dataclass
class AgentMetrics:

    # Task-level counters
    tasks_total: Counter = field(default_factory=lambda: Counter(
        "agent_tasks_total",
        "Total number of agent tasks submitted",
        ["status"],  # success | failed | timeout
    ))

    # LLM usage
    llm_tokens_total: Counter = field(default_factory=lambda: Counter(
        "llm_tokens_used_total",
        "Total LLM tokens consumed",
        ["provider", "model", "direction"],  # direction: input | output
    ))
    llm_cost_usd_total: Counter = field(default_factory=lambda: Counter(
        "llm_cost_usd_total",
        "Total cost of LLM calls in USD",
        ["provider", "model"],
    ))
    llm_call_duration: Histogram = field(default_factory=lambda: Histogram(
        "llm_call_duration_seconds",
        "Time taken for LLM API calls",
        ["provider", "model"],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    ))

    # Tool calls
    tool_calls_total: Counter = field(default_factory=lambda: Counter(
        "agent_tool_calls_total",
        "Total tool calls made by the agent",
        ["tool_name", "status"],  # status: success | failed | timeout
    ))
    tool_call_duration: Histogram = field(default_factory=lambda: Histogram(
        "agent_tool_call_duration_seconds",
        "Time taken for tool calls",
        ["tool_name"],
        buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 60.0),
    ))

    # Memory operations
    memory_writes_total: Counter = field(default_factory=lambda: Counter(
        "agent_memory_writes_total",
        "Total writes to long-term memory",
        ["tier"],  # short_term | long_term | episodic
    ))
    memory_reads_total: Counter = field(default_factory=lambda: Counter(
        "agent_memory_reads_total",
        "Total reads from memory",
        ["tier"],
    ))

    # Agent loop
    loop_iterations_total: Counter = field(default_factory=lambda: Counter(
        "agent_loop_iterations_total",
        "Total ReAct loop iterations across all tasks",
    ))
    task_duration: Histogram = field(default_factory=lambda: Histogram(
        "agent_task_duration_seconds",
        "End-to-end duration of agent tasks",
        buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
    ))

METRICS = AgentMetrics()

def configure_logging(log_level: str = "INFO", *, pretty: bool | None = None) -> None:

    if pretty is None:
        pretty = sys.stdout.isatty()

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
        stream=sys.stdout,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,          # Thread-local context (session_id, etc.)
        structlog.processors.add_log_level,               # Adds "level" key
        structlog.processors.TimeStamper(fmt="iso"),      # ISO-8601 timestamp
        structlog.stdlib.add_logger_name,                 # Adds "logger" key with module name
        structlog.processors.StackInfoRenderer(),         # Stack info for exceptions
        structlog.processors.format_exc_info,             # Format exception tracebacks
    ]

    if pretty:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)

def start_metrics_server(port: int = 8000) -> None:
    log = get_logger(__name__)
    start_http_server(port)
    log.info("prometheus_metrics_server_started", port=port, path="/metrics")

def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()