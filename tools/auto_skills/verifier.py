from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)

SKILL_BANNED_IMPORTS: frozenset[str] = frozenset({
    # OS / process access
    "os", "subprocess", "shutil", "pathlib", "glob", "fnmatch",
    "tempfile", "pty", "tty", "termios", "fcntl", "grp", "pwd",
    "nis", "syslog", "resource",
    # Network 
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib",
    "imaplib", "poplib", "nntplib", "telnetlib", "xmlrpc",
    "socketserver", "wsgiref",
    "requests", "aiohttp", "tornado", "flask", "fastapi",
    "django", "bottle", "starlette",
    # Concurrency
    "threading", "multiprocessing", "concurrent",
    "signal", "ctypes", "_ctypes",
    # Introspection / code generation (bypass vectors)
    "gc", "inspect", "dis", "traceback", "linecache", "tokenize",
    "token", "symtable", "ast", "compile", "code", "codeop",
    "importlib", "pkgutil", "zipimport", "runpy",
    # Serialization with arbitrary-code-execution risk
    "pickle", "pickletools", "shelve", "dbm", "marshal",
    # System / platform
    "sys", "builtins", "platform", "sysconfig",
    "site", "sitecustomize", "usercustomize",
    "winreg", "winsound", "msvcrt", "_winapi",
    # Skills should not do their own crypto — use the vault
    "hashlib", "hmac", "secrets",
    # XML / plist (historical RCE surface)
    "xml", "xmlrpc", "plistlib",
})

SKILL_BANNED_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "__import__", "open", "compile",
    "breakpoint", "input", "memoryview",
    "vars", "dir", "globals", "locals",
    "delattr", "setattr",
})

REQUIRED_ATTRIBUTES: frozenset[str] = frozenset({
    "name", "description", "manifest", "input_schema",
})

BLOCKING_SEVERITIES: frozenset[str] = frozenset({"HIGH", "CRITICAL"})

PYTEST_TIMEOUT_S = 30

BANDIT_TIMEOUT_S = 30

@dataclass
class VerificationResult:
    passed:      bool
    stage:       str       = "unknown"
    reason:      str       = ""
    warnings:    list[str] = field(default_factory=list)
    skill_class: str       = ""
    report:      Any       = None
    duration_ms: int       = 0
    file_hash:   str       = ""

    def __bool__(self) -> bool:
        return self.passed

class SkillVerifier:

    def __init__(
        self,
        run_bandit:     bool = True,
        run_pytest:     bool = True,
        pytest_timeout: int  = PYTEST_TIMEOUT_S,
        bandit_timeout: int  = BANDIT_TIMEOUT_S,
    ) -> None:
        self._run_bandit     = run_bandit
        self._run_pytest     = run_pytest
        self._pytest_timeout = pytest_timeout
        self._bandit_timeout = bandit_timeout


    def verify(self, skill_path: Path) -> VerificationResult:
        start    = time.monotonic()
        warnings: list[str] = []

        try:
            source = skill_path.read_text(encoding="utf-8")
        except Exception as exc:
            return VerificationResult(
                passed=False, stage="read",
                reason=f"Cannot read skill file: {exc}",
            )

        file_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        log.info(
            "skill_verifier_start",
            skill_path=str(skill_path),
            file_hash=file_hash,
            size_bytes=len(source),
        )

        # Stage 1 — AST
        ast_result = self.ast_scan(skill_path, _source=source)
        warnings.extend(ast_result.warnings)
        if not ast_result.passed:
            return VerificationResult(
                passed=False, stage="ast",
                reason=ast_result.reason,
                warnings=warnings,
                skill_class=ast_result.skill_class,
                report=ast_result.report,
                duration_ms=int((time.monotonic() - start) * 1000),
                file_hash=file_hash,
            )

        skill_class = ast_result.skill_class

        # Stage 2 — Bandit
        if self._run_bandit:
            bandit_result = self.bandit_scan(skill_path)
            warnings.extend(bandit_result.warnings)
            if not bandit_result.passed:
                return VerificationResult(
                    passed=False, stage="bandit",
                    reason=bandit_result.reason,
                    warnings=warnings,
                    skill_class=skill_class,
                    report=bandit_result.report,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    file_hash=file_hash,
                )

        # Stage 3 — pytest
        if self._run_pytest:
            pytest_result = self.pytest_run(skill_path, _source=source)
            warnings.extend(pytest_result.warnings)
            if not pytest_result.passed:
                return VerificationResult(
                    passed=False, stage="pytest",
                    reason=pytest_result.reason,
                    warnings=warnings,
                    skill_class=skill_class,
                    report=pytest_result.report,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    file_hash=file_hash,
                )

        duration_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "skill_verifier_passed",
            skill_path=str(skill_path),
            skill_class=skill_class,
            file_hash=file_hash,
            duration_ms=duration_ms,
            warnings=len(warnings),
        )

        return VerificationResult(
            passed=True, stage="all",
            reason="",
            warnings=warnings,
            skill_class=skill_class,
            duration_ms=duration_ms,
            file_hash=file_hash,
        )

    # Stage 1: AST Scan

    def ast_scan(
        self,
        skill_path: Path,
        _source: str | None = None,
    ) -> VerificationResult:
        source = _source or skill_path.read_text(encoding="utf-8")
        warnings: list[str] = []

        # Parse
        try:
            tree = ast.parse(source, filename=str(skill_path))
        except SyntaxError as exc:
            return VerificationResult(
                passed=False, stage="ast",
                reason=f"SyntaxError in skill file: {exc}",
            )

        for node in ast.walk(tree):

            # Import checks
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in SKILL_BANNED_IMPORTS:
                        return VerificationResult(
                            passed=False, stage="ast",
                            reason=(
                                f"Banned import {alias.name!r} in skill. "
                                f"Skills must not import {top!r}. "
                                "Use httpx for network access and pydantic for models."
                            ),
                        )

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                if top in SKILL_BANNED_IMPORTS:
                    return VerificationResult(
                        passed=False, stage="ast",
                        reason=(
                            f"Banned import from {mod!r} in skill. "
                            f"Module {top!r} is not permitted in skills."
                        ),
                    )

            # Builtin call checks
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in SKILL_BANNED_BUILTINS:
                        return VerificationResult(
                            passed=False, stage="ast",
                            reason=f"Banned builtin call {node.func.id!r} in skill.",
                        )
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in SKILL_BANNED_BUILTINS:
                        return VerificationResult(
                            passed=False, stage="ast",
                            reason=(
                                f"Banned attribute call {node.func.attr!r} in skill. "
                                "Attribute-style access to restricted builtins is not allowed."
                            ),
                        )

            elif isinstance(node, ast.Subscript):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in ("__builtins__", "builtins", "globals", "locals")
                ):
                    return VerificationResult(
                        passed=False, stage="ast",
                        reason=(
                            "Subscript access to __builtins__ / globals / locals "
                            "is not permitted in skills."
                        ),
                    )

        base_tool_classes: list[tuple[str, set[str], bool]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)

            if "BaseTool" not in base_names:
                continue

            declared_attrs: set[str] = set()
            has_async_execute        = False

            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            declared_attrs.add(target.id)
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        declared_attrs.add(item.target.id)
                elif isinstance(item, ast.AsyncFunctionDef) and item.name == "execute":
                    has_async_execute = True
                elif isinstance(item, ast.FunctionDef) and item.name == "execute":

                    return VerificationResult(
                        passed=False, stage="ast",
                        reason=(
                            f"execute() in {node.name!r} must be async "
                            "(use 'async def execute(...)')."
                        ),
                    )

            base_tool_classes.append((node.name, declared_attrs, has_async_execute))

        if not base_tool_classes:
            return VerificationResult(
                passed=False, stage="ast",
                reason=(
                    "No BaseTool subclass found in the skill file. "
                    "The file must contain exactly one class that inherits from BaseTool. "
                    "Example: class MyTool(BaseTool): ..."
                ),
            )

        if len(base_tool_classes) > 1:
            names = [c[0] for c in base_tool_classes]
            return VerificationResult(
                passed=False, stage="ast",
                reason=(
                    f"Multiple BaseTool subclasses found: {names}. "
                    "Each skill file must contain exactly one BaseTool subclass."
                ),
            )

        class_name, declared_attrs, has_async_execute = base_tool_classes[0]

        missing = REQUIRED_ATTRIBUTES - declared_attrs
        if missing:
            return VerificationResult(
                passed=False, stage="ast",
                reason=(
                    f"BaseTool subclass {class_name!r} is missing required "
                    f"class attributes: {sorted(missing)}. "
                    "Every BaseTool subclass must declare: "
                    "name, description, manifest, input_schema."
                ),
            )

        if not has_async_execute:
            return VerificationResult(
                passed=False, stage="ast",
                reason=(
                    f"BaseTool subclass {class_name!r} has no async execute() method. "
                    "Add: async def execute(self, input: ToolInput) -> ToolResult: ..."
                ),
            )

        # Check for test functions
        test_fns = [
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")
        ]
        if not test_fns:
            warnings.append(
                f"Skill {class_name!r} has no test functions (def test_*). "
                "Tests are strongly recommended. Stage 3 (pytest) will be skipped."
            )

        log.debug(
            "skill_ast_scan_passed",
            skill_class=class_name,
            attrs=sorted(declared_attrs),
            test_fns=test_fns,
        )

        return VerificationResult(
            passed=True, stage="ast",
            warnings=warnings,
            skill_class=class_name,
            report={
                "declared_attrs": sorted(declared_attrs),
                "test_functions": test_fns,
            },
        )

    # Stage 2: Bandit Scan

    def bandit_scan(self, skill_path: Path) -> VerificationResult:
        # Check bandit is available
        bandit_bin = _find_bandit()
        if bandit_bin is None:
            log.warning("skill_bandit_not_installed")
            return VerificationResult(
                passed=True, stage="bandit",
                warnings=[
                    "bandit is not installed — security scan was skipped. "
                    "Install with: pip install bandit"
                ],
            )

        try:
            proc = subprocess.run(
                [bandit_bin, "-r", str(skill_path), "-f", "json", "-ll"],
                capture_output=True,
                text=True,
                timeout=self._bandit_timeout,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False, stage="bandit",
                reason=f"Bandit scan timed out after {self._bandit_timeout}s.",
            )
        except Exception as exc:
            return VerificationResult(
                passed=False, stage="bandit",
                reason=f"Bandit execution failed: {exc}",
            )

        raw_output = proc.stdout or proc.stderr or ""
        try:
            report = json.loads(raw_output)
        except json.JSONDecodeError:
            if proc.returncode not in (0, 1):
                return VerificationResult(
                    passed=False, stage="bandit",
                    reason=f"Bandit produced no JSON output. stderr: {proc.stderr[:300]}",
                )
            return VerificationResult(passed=True, stage="bandit")

        results  = report.get("results", [])
        blocking = [r for r in results if r.get("issue_severity") in BLOCKING_SEVERITIES]
        medium   = [r for r in results if r.get("issue_severity") == "MEDIUM"]

        warnings: list[str] = []
        for item in medium:
            warnings.append(
                f"[MEDIUM] {item.get('issue_text', '')} "
                f"(line {item.get('line_number', '?')}, test {item.get('test_id', '?')})"
            )

        if blocking:
            reasons = "; ".join(
                f"{r.get('issue_text', '?')} (line {r.get('line_number', '?')}, "
                f"{r.get('issue_severity', '?')})"
                for r in blocking[:5]
            )
            return VerificationResult(
                passed=False, stage="bandit",
                reason=f"Bandit found {len(blocking)} blocking issue(s): {reasons}",
                warnings=warnings,
                report=report,
            )

        log.debug(
            "skill_bandit_scan_passed",
            skill_path=str(skill_path),
            findings=len(results),
            medium=len(medium),
        )

        return VerificationResult(
            passed=True, stage="bandit",
            warnings=warnings,
            report=report,
        )

    # Stage 3: pytest

    def pytest_run(
        self,
        skill_path: Path,
        _source: str | None = None,
    ) -> VerificationResult:
        source = _source or skill_path.read_text(encoding="utf-8")

        try:
            tree     = ast.parse(source)
            test_fns = [
                n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("test_")
            ]
        except SyntaxError:
            test_fns = []

        if not test_fns:
            return VerificationResult(
                passed=True, stage="pytest",
                warnings=[
                    "Skill has no test functions — pytest skipped. "
                    "Add def test_*() functions to improve confidence."
                ],
            )

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(skill_path),
                    "-x",           # stop on first failure
                    "-q",           # quiet output
                    "--tb=short",   # short traceback on failures
                    "--no-header",  # cleaner output
                ],
                capture_output=True,
                text=True,
                timeout=self._pytest_timeout,
            )
        except FileNotFoundError:
            # pytest not installed
            return VerificationResult(
                passed=True, stage="pytest",
                warnings=[
                    "pytest is not installed — test run was skipped. "
                    "Install with: pip install pytest"
                ],
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False, stage="pytest",
                reason=(
                    f"pytest timed out after {self._pytest_timeout}s. "
                    "Ensure tests don't contain slow operations or infinite loops."
                ),
            )
        except Exception as exc:
            return VerificationResult(
                passed=False, stage="pytest",
                reason=f"pytest execution failed: {exc}",
            )

        combined_output = (proc.stdout + proc.stderr).strip()

        if proc.returncode != 0:
            # pytest failed — return last 800 chars of output so the agent
            # knows exactly which assertions failed
            tail = combined_output[-800:] if len(combined_output) > 800 else combined_output
            return VerificationResult(
                passed=False, stage="pytest",
                reason=(
                    f"pytest failed ({len(test_fns)} test(s) found). "
                    f"Output:\n{tail}"
                ),
                report=combined_output,
            )

        log.debug(
            "skill_pytest_passed",
            skill_path=str(skill_path),
            test_count=len(test_fns),
        )

        return VerificationResult(
            passed=True, stage="pytest",
            report=combined_output,
        )


# Helpers 

def _find_bandit() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "--version"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return sys.executable + " -m bandit"   
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["bandit", "--version"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return "bandit"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _build_bandit_cmd(bandit_bin: str, skill_path: Path) -> list[str]:
    """Build the bandit command list."""
    if bandit_bin.endswith("-m bandit"):
        return [sys.executable, "-m", "bandit", "-r", str(skill_path), "-f", "json", "-ll"]
    return [bandit_bin, "-r", str(skill_path), "-f", "json", "-ll"]