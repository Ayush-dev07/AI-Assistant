from __future__ import annotations

import ast
import re
import resource
import subprocess
import sys
import textwrap
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult

log = get_logger(__name__)

BANNED_IMPORTS: frozenset[str] = frozenset({
    # OS / process access
    "os", "subprocess", "shutil", "pathlib", "glob", "fnmatch",
    "tempfile", "pty", "tty", "termios", "fcntl", "grp", "pwd",
    "nis", "syslog", "resource",
    # Network
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib",
    "imaplib", "poplib", "nntplib", "telnetlib", "xmlrpc",
    "socketserver", "wsgiref",
    "requests", "httpx", "aiohttp", "tornado", "flask", "fastapi",
    "django", "bottle", "starlette",
    # Concurrency / process control
    "threading", "multiprocessing", "concurrent", "asyncio",
    "signal", "ctypes", "_ctypes",
    # Introspection and code generation (bypass vectors)
    "gc", "inspect", "dis", "traceback", "linecache", "tokenize",
    "token", "symtable", "ast", "compile", "code", "codeop",
    "importlib", "pkgutil", "zipimport", "runpy",
    # Serialization (arbitrary code execution via pickle)
    "pickle", "pickletools", "shelve", "dbm", "marshal",
    # System / platform
    "sys", "builtins", "platform", "sysconfig", "_sysconfig_vars",
    "site", "sitecustomize", "usercustomize",
    "winreg", "winsound", "msvcrt", "_winapi",
    # Cryptography and low-level
    "hashlib", "hmac", "secrets", "ssl",
    # File formats that shell out or have RCE history
    "xml", "xmlrpc", "plistlib",
})

BANNED_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "__import__", "open", "compile",
    "breakpoint", "input",
    # Memory views can expose internal bytes
    "memoryview",
    # vars/dir/globals/locals allow introspection of live state
    "vars", "dir", "globals", "locals",
    # delattr/setattr can modify class internals
    "delattr", "setattr",
})

ALLOWED_IMPORTS: tuple[str, ...] = (
    "math", "cmath", "decimal", "fractions", "statistics",
    "random",
    "json", "csv", "configparser",
    "datetime", "calendar", "time",
    "re", "string", "textwrap", "difflib", "unicodedata",
    "struct", "array", "bisect", "heapq",
    "collections", "itertools", "functools", "operator",
    "copy", "pprint", "reprlib",
    "enum", "dataclasses", "abc", "typing",
    "io", "base64", "binascii", "codecs",
    "zlib", "gzip", "bz2", "lzma",
   
)

SAFE_IMPORTS: tuple[str, ...] = tuple(
    m for m in ALLOWED_IMPORTS if m not in BANNED_IMPORTS
)

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

DEFAULT_MEMORY_MB   = 256
DEFAULT_CPU_SECONDS = 30
DEFAULT_TIMEOUT_S   = 35    
DEFAULT_MAX_OUTPUT  = 8_000 
DEFAULT_MAX_FSIZE   = 1     


class CodeInput(BaseModel):
    code:            str  = Field(...,          description="Python source code to execute.")
    language:        str  = Field(default="python", description="Must be 'python'.")
    timeout_seconds: int  = Field(default=30,   ge=1, le=120,
                                  description="Wall-clock timeout (1–120s).")
    dry_run:         bool = Field(default=False,
                                  description="Scan and plan without executing.")
    stdin_data:      str  = Field(default="",
                                  description="Optional stdin for the subprocess.")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v.lower() != "python":
            raise ValueError(
                f"Only 'python' is supported. Got: {v!r}. "
                "JavaScript, Bash, and other languages will be added in future tasks."
            )
        return "python"

    @field_validator("code")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code cannot be empty.")
        if len(v) > 50_000:
            raise ValueError(
                f"Code is {len(v):,} chars. Maximum is 50,000 chars to prevent "
                "context window overflow in the result."
            )
        return v

class _HistoryEntry:
    __slots__ = ("code_hash", "code_preview", "success", "stdout", "stderr",
                 "returncode", "exec_time_ms", "timestamp")

    def __init__(
        self,
        code: str,
        success: bool,
        stdout: str,
        stderr: str,
        returncode: int,
        exec_time_ms: int,
    ) -> None:
        import hashlib
        self.code_hash    = hashlib.sha256(code.encode()).hexdigest()[:12]
        self.code_preview = code.strip()[:120]
        self.success      = success
        self.stdout       = stdout
        self.stderr       = stderr
        self.returncode   = returncode
        self.exec_time_ms = exec_time_ms
        self.timestamp    = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_hash":    self.code_hash,
            "code_preview": self.code_preview,
            "success":      self.success,
            "stdout":       self.stdout,
            "stderr":       self.stderr,
            "returncode":   self.returncode,
            "exec_time_ms": self.exec_time_ms,
        }


_execution_history: dict[str, deque[_HistoryEntry]] = {}
_HISTORY_MAXLEN = 10


class CodeExecutorTool(BaseTool):

    name = "code_executor"
    description = (
        "Execute Python code in a secure sandbox. "
    )
    manifest     = PermissionManifest(can_spawn_processes=True)
    input_schema = CodeInput

    def __init__(
        self,
        memory_limit_mb:   int = DEFAULT_MEMORY_MB,
        cpu_limit_seconds: int = DEFAULT_CPU_SECONDS,
        max_output_chars:  int = DEFAULT_MAX_OUTPUT,
        max_fsize_mb:      int = DEFAULT_MAX_FSIZE,
    ) -> None:
        self._memory_bytes  = memory_limit_mb   * 1024 * 1024
        self._cpu_seconds   = cpu_limit_seconds
        self._max_output    = max_output_chars
        self._max_fsize     = max_fsize_mb       * 1024 * 1024


    async def execute(self, input: ToolInput) -> ToolResult:
        params = self.input_schema(**input.parameters)

        scan_ok, scan_reason, scan_report = self._ast_scan(params.code)
        if not scan_ok:
            log.warning(
                "code_executor_ast_blocked",
                session_id=input.session_id,
                reason=scan_reason,
            )
            return ToolResult(
                success=False,
                error=(
                    f"Code blocked by security scanner: {scan_reason}\n\n"
                    f"Tip: Use only the allowed imports listed in the tool description. "
                    f"Do not use os, subprocess, socket, or any network libraries."
                ),
                output={
                    "blocked":       True,
                    "block_reason":  scan_reason,
                    "scan_report":   scan_report,
                    "allowed_imports": list(SAFE_IMPORTS),
                },
            )

        if params.dry_run:
            return self._dry_run_result(params.code, scan_report)

        return await self._run_subprocess(params, input.session_id)


    def _ast_scan(self, code: str) -> tuple[bool, str, dict[str, Any]]:
        report: dict[str, Any] = {
            "imports_found":  [],
            "builtins_used":  [],
            "banned_imports": [],
            "banned_calls":   [],
            "ast_nodes":      0,
        }

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}", report

        node_count = 0
        for node in ast.walk(tree):
            node_count += 1

            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    top = mod.split(".")[0]
                    report["imports_found"].append(mod)
                    if top in BANNED_IMPORTS:
                        report["banned_imports"].append(mod)
                        return False, f"Banned import: {mod!r}", report

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                report["imports_found"].append(mod)
                if top in BANNED_IMPORTS:
                    report["banned_imports"].append(mod)
                    return False, f"Banned import from: {mod!r}", report

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in BANNED_BUILTINS:
                        report["banned_calls"].append(name)
                        return False, f"Banned builtin call: {name!r}", report
                    report["builtins_used"].append(name)

                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr in BANNED_BUILTINS:
                        report["banned_calls"].append(f"*.{attr}")
                        return False, (
                            f"Banned attribute call: {attr!r}. "
                            "Attribute-style calls to restricted builtins are not allowed."
                        ), report

            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in (
                    "__builtins__", "builtins", "globals", "locals"
                ):
                    return False, (
                        "Subscript access to __builtins__ or globals is not allowed. "
                        "This pattern is commonly used to bypass sandbox restrictions."
                    ), report

        report["ast_nodes"] = node_count
        return True, "", report


    async def _run_subprocess(
        self,
        params: CodeInput,
        session_id: str,
    ) -> ToolResult:
        import asyncio

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self._blocking_run,
                params,
                session_id,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Subprocess launcher failed unexpectedly: {exc}",
            )
        return result

    def _blocking_run(self, params: CodeInput, session_id: str) -> ToolResult:
        memory_bytes  = self._memory_bytes
        cpu_seconds   = self._cpu_seconds
        max_fsize     = self._max_fsize

        def _preexec() -> None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_bytes, memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (cpu_seconds, cpu_seconds),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (max_fsize, max_fsize),
            )
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            except (ValueError, resource.error):
                pass

        wall_timeout = min(params.timeout_seconds + 5, DEFAULT_TIMEOUT_S)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", params.code],
                capture_output=True,
                text=True,
                timeout=wall_timeout,
                preexec_fn=_preexec,
                input=params.stdin_data or None,
                env={
                    "PATH":      "/usr/bin:/bin",
                    "PYTHONPATH": "",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONHASHSEED": "0",
                },
            )
            exec_time_ms = int((time.monotonic() - start) * 1000)

        except subprocess.TimeoutExpired:
            exec_time_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "code_executor_timeout",
                session_id=session_id,
                timeout_s=wall_timeout,
                exec_time_ms=exec_time_ms,
            )
            entry = _HistoryEntry(
                params.code, False, "", "Killed: wall-clock timeout", -1, exec_time_ms
            )
            self._record_history(session_id, entry)
            return ToolResult(
                success=False,
                error=(
                    f"Execution killed: exceeded {wall_timeout}s wall-clock timeout. "
                    f"Simplify the code or increase timeout_seconds (max 120)."
                ),
                output={
                    "stdout":       "",
                    "stderr":       "Process killed (timeout)",
                    "returncode":   -9,
                    "exec_time_ms": exec_time_ms,
                    "timed_out":    True,
                },
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to launch subprocess: {exc}",
            )

        raw_stdout = self._clean_output(proc.stdout or "")
        raw_stderr = self._clean_output(proc.stderr or "")

        stdout_truncated = len(raw_stdout) > self._max_output
        stderr_truncated = len(raw_stderr) > self._max_output

        stdout = raw_stdout[: self._max_output]
        stderr = raw_stderr[: self._max_output]

        success = proc.returncode == 0

        log.info(
            "code_executor_complete",
            session_id=session_id,
            returncode=proc.returncode,
            exec_time_ms=exec_time_ms,
            stdout_len=len(stdout),
            stderr_len=len(stderr),
            success=success,
        )

        entry = _HistoryEntry(
            params.code, success, stdout, stderr, proc.returncode, exec_time_ms
        )
        self._record_history(session_id, entry)

        output: dict[str, Any] = {
            "stdout":            stdout,
            "stderr":            stderr,
            "returncode":        proc.returncode,
            "exec_time_ms":      exec_time_ms,
            "stdout_truncated":  stdout_truncated,
            "stderr_truncated":  stderr_truncated,
            "memory_limit_mb":   self._memory_bytes // (1024 * 1024),
            "cpu_limit_s":       self._cpu_seconds,
        }

        if not success:
            last_error = _extract_last_exception(stderr)
            return ToolResult(
                success=False,
                output=output,
                error=(
                    f"Process exited with code {proc.returncode}. "
                    + (f"Error: {last_error}" if last_error else "See stderr for details.")
                ),
            )

        return ToolResult(success=True, output=output)

    def _dry_run_result(
        self, code: str, scan_report: dict[str, Any]
    ) -> ToolResult:
        lines      = code.count("\n") + 1
        char_count = len(code)

        loop_count = code.count("for ") + code.count("while ")
        func_count = code.count("def ")
        cls_count  = code.count("class ")

        preview = textwrap.shorten(code.strip(), width=200, placeholder=" ...")

        return ToolResult(
            success=True,
            output={
                "dry_run":          True,
                "would_execute":    True,
                "code_preview":     preview,
                "lines":            lines,
                "chars":            char_count,
                "imports_found":    scan_report.get("imports_found", []),
                "builtins_used":    scan_report.get("builtins_used", []),
                "loops":            loop_count,
                "functions":        func_count,
                "classes":          cls_count,
                "ast_nodes":        scan_report.get("ast_nodes", 0),
                "limits": {
                    "memory_mb":    self._memory_bytes // (1024 * 1024),
                    "cpu_seconds":  self._cpu_seconds,
                    "max_output":   self._max_output,
                },
                "allowed_imports":  list(SAFE_IMPORTS),
                "message": (
                    "AST scan passed. Code would execute in a subprocess with "
                    f"{self._memory_bytes // (1024*1024)}MB memory limit and "
                    f"{self._cpu_seconds}s CPU limit. "
                    "Remove dry_run=True to execute."
                ),
            },
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        text = _ANSI_ESCAPE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    @staticmethod
    def _record_history(session_id: str, entry: _HistoryEntry) -> None:
        if session_id not in _execution_history:
            _execution_history[session_id] = deque(maxlen=_HISTORY_MAXLEN)
        _execution_history[session_id].append(entry)

    @staticmethod
    def get_history(session_id: str) -> list[dict[str, Any]]:
        buf = _execution_history.get(session_id)
        if not buf:
            return []
        return [e.to_dict() for e in buf]

    @staticmethod
    def clear_history(session_id: str) -> None:
        _execution_history.pop(session_id, None)

def _extract_last_exception(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z][\w.]*(?:Error|Exception|Warning|Interrupt)[:\s]", line):
            return line[:300]
        if re.match(r"^[A-Za-z][\w.]*Error$", line):
            return line[:300]
    return lines[-1][:300] if lines else ""