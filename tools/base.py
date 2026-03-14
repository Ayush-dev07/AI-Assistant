from __future__ import annotations

import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from core.logging import METRICS, get_logger

log = get_logger(__name__)


class PermissionManifest(BaseModel):

    network_domains:     list[str] = Field(default_factory=list)
    filesystem_read:     list[str] = Field(default_factory=list)
    filesystem_write:    list[str] = Field(default_factory=list)
    env_vars:            list[str] = Field(default_factory=list)
    can_spawn_processes: bool      = False
    max_response_bytes:  int       = 10 * 1024 * 1024  # 10 MB

    def allows_domain(self, hostname: str) -> bool:
        if not hostname:
            return False
        for pattern in self.network_domains:
            if pattern == "*":
                return True
            if pattern.startswith("*."):
                suffix = pattern[2:]  # strip "*."
                if hostname == suffix or hostname.endswith("." + suffix):
                    return True
            else:
                if hostname == pattern:
                    return True
        return False

    def allows_path(self, path: str, write: bool = False) -> bool:
        from pathlib import Path
        resolved = str(Path(path).resolve())
        allowed_list = self.filesystem_write if write else self.filesystem_read
        return any(
            resolved == str(Path(a).resolve()) or
            resolved.startswith(str(Path(a).resolve()) + "/")
            for a in allowed_list
        )

class ToolInput(BaseModel):

    tool_name:  str            = Field(..., description="Tool name (matches BaseTool.name)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool-specific arguments")
    session_id: str            = Field(default="default",    description="Caller session ID")
    call_id:    str            = Field(default_factory=lambda: str(uuid.uuid4())[:8],
                                       description="Unique call ID for tracing")


class ToolResult(BaseModel):

    success:     bool           = True
    output:      Any            = None
    error:       str | None     = None
    metadata:    dict[str, Any] = Field(default_factory=dict)
    duration_ms: int            = 0
    cached:      bool           = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        if self.cached:
            d["cached"] = True
        if self.metadata:
            d["metadata"] = self.metadata
        return d

class BaseTool(ABC):

    name:         str
    description:  str
    manifest:     PermissionManifest
    input_schema: type[BaseModel]

    @abstractmethod
    async def execute(self, input: ToolInput) -> ToolResult:
        """
        Run the tool and return a ToolResult.
        """

    def to_llm_schema(self) -> dict[str, Any]:
        """
        Convert this tool to the JSON Schema format used by all LLM providers.
        """
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  self.input_schema.model_json_schema(),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

class _ToolRegistry:

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            log.debug("tool_registry_overwrite", tool_name=tool.name)
        self._tools[tool.name] = tool
        log.debug("tool_registered", tool_name=tool.name, class_name=tool.__class__.__name__)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} is not registered. "
                f"Available tools: {sorted(self._tools)}"
            )
        return self._tools[name]

    def get_callable(self, name: str) -> Callable:
        tool = self.get(name)

        async def _callable(arguments: dict[str, Any]) -> dict[str, Any]:
            call_id = str(uuid.uuid4())[:8]
            start   = time.monotonic()

            log.debug(
                "tool_dispatch",
                tool_name=name,
                call_id=call_id,
                arg_keys=list(arguments.keys()),
            )

            try:
                try:
                    tool.input_schema(**arguments)
                except Exception as exc:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    METRICS.tool_calls_total.labels(
                        tool_name=name, status="validation_error"
                    ).inc()
                    log.warning(
                        "tool_input_validation_failed",
                        tool_name=name, call_id=call_id, error=str(exc),
                    )
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Invalid arguments for tool '{name}': {exc}. "
                              f"Check the tool schema and fix the arguments.",
                        duration_ms=duration_ms,
                    ).to_dict()

                manifest_ok, manifest_err = _check_manifest(tool.manifest, arguments)
                if not manifest_ok:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    METRICS.tool_calls_total.labels(
                        tool_name=name, status="permission_denied"
                    ).inc()
                    log.warning(
                        "tool_manifest_check_failed",
                        tool_name=name, call_id=call_id, reason=manifest_err,
                    )
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Permission denied for tool '{name}': {manifest_err}",
                        duration_ms=duration_ms,
                    ).to_dict()

                tool_input = ToolInput(
                    tool_name=name,
                    parameters=arguments,
                    call_id=call_id,
                )
                result = await tool.execute(tool_input)

            except Exception as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                METRICS.tool_calls_total.labels(
                    tool_name=name, status="exception"
                ).inc()
                log.error(
                    "tool_execute_exception",
                    tool_name=name,
                    call_id=call_id,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Tool '{name}' raised an unexpected error: {exc}",
                    duration_ms=duration_ms,
                ).to_dict()

            duration_ms = int((time.monotonic() - start) * 1000)
            result.duration_ms = duration_ms

            status = "success" if result.success else "failure"
            METRICS.tool_calls_total.labels(tool_name=name, status=status).inc()
            METRICS.tool_call_duration.labels(tool_name=name).observe(
                (time.monotonic() - start)
            )

            log.info(
                "tool_call_complete",
                tool_name=name,
                call_id=call_id,
                success=result.success,
                duration_ms=duration_ms,
                cached=result.cached,
                output_type=type(result.output).__name__,
            )

            return result.to_dict()

        _callable.__name__ = f"tool_{name}"
        return _callable

    def get_all_callables(self) -> dict[str, Callable]:
        return {name: self.get_callable(name) for name in self._tools}

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def list_all(self) -> list[str]:
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.list_all()}>"

ToolRegistry = _ToolRegistry()


def _check_manifest(
    manifest: PermissionManifest,
    arguments: dict[str, Any],
) -> tuple[bool, str]:
    import urllib.parse

    url = arguments.get("url", "")
    if url:
        try:
            hostname = urllib.parse.urlparse(url).hostname or ""
        except Exception:
            hostname = ""
        if hostname and not manifest.allows_domain(hostname):
            return False, (
                f"Domain '{hostname}' is not in this tool's manifest.network_domains "
                f"{manifest.network_domains!r}. "
                f"Use a tool that allows this domain, or expand the manifest."
            )

    path = arguments.get("path", "")
    if path:
        action = arguments.get("action", "read")
        is_write = action in ("write", "delete", "move", "append")
        if not manifest.allows_path(path, write=is_write):
            allowed = manifest.filesystem_write if is_write else manifest.filesystem_read
            return False, (
                f"Path '{path}' (resolved: action={action}) is not within "
                f"manifest.filesystem_{'write' if is_write else 'read'} = {allowed!r}."
            )

    return True, ""

def build_tool_definitions_for_llm(
    tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    names = tool_names if tool_names is not None else ToolRegistry.list_all()
    schemas = []
    for name in names:
        try:
            schemas.append(ToolRegistry.get(name).to_llm_schema())
        except KeyError:
            log.warning("build_tool_definitions_unknown_tool", tool_name=name)
    return schemas