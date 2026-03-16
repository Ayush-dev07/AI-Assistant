from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult, ToolRegistry
from tools.auto_skills.installer import SkillInstaller
from tools.auto_skills.verifier import SkillVerifier

log = get_logger(__name__)

_GUIDANCE: dict[str, str] = {
    "ast": (
        "Fix the AST scan failure by removing banned imports and calls. "
        "Allowed imports: httpx (for network), pydantic (for models), "
        "asyncio, json, re, math, datetime, collections, itertools, "
        "functools, typing, dataclasses, enum, copy, io, base64. "
        "Do not import: os, subprocess, socket, sys, requests, gc, "
        "inspect, threading, multiprocessing, importlib, pickle, ctypes. "
        "Do not call: exec(), eval(), open(), __import__(), compile()."
    ),
    "bandit": (
        "Fix the Bandit security finding. Common issues: "
        "hardcoded passwords/tokens (use ToolInput parameters instead), "
        "use of assert for security checks (use if/raise instead), "
        "subprocess calls (not permitted in skills), "
        "use of MD5 or SHA1 for security purposes (use SHA256). "
        "Remove or refactor the flagged code and resubmit."
    ),
    "pytest": (
        "Fix the failing test(s). The test output above shows exactly "
        "which assertions failed and why. Update either the skill code "
        "or the test expectations to make all tests pass. "
        "Then resubmit the corrected skill."
    ),
    "structure": (
        "The skill file must contain exactly one class that inherits from BaseTool "
        "and declares: name (str), description (str), manifest (PermissionManifest), "
        "input_schema (Pydantic BaseModel subclass), and "
        "async def execute(self, input: ToolInput) -> ToolResult. "
        "Example structure:\n"
        "    from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult\n"
        "    from pydantic import BaseModel\n"
        "    class MyInput(BaseModel):\n"
        "        query: str\n"
        "    class MyTool(BaseTool):\n"
        "        name = 'my_tool'\n"
        "        description = 'Does X.'\n"
        "        manifest = PermissionManifest()\n"
        "        input_schema = MyInput\n"
        "        async def execute(self, input: ToolInput) -> ToolResult:\n"
        "            params = self.input_schema(**input.parameters)\n"
        "            return ToolResult(success=True, output={'result': params.query})"
    ),
}

class SkillInput(BaseModel):

    code:       str  = Field(
        ...,
        min_length=50,
        description="Python source code for the new skill.",
    )
    skill_name: str  = Field(
        default="",
        description="Expected skill name (optional, for logging).",
    )
    dry_run:    bool = Field(
        default=False,
        description="If True, verify but do not install.",
    )

    @field_validator("code")
    @classmethod
    def validate_code_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code cannot be empty or whitespace only.")
        return v

class SkillInstallTool(BaseTool):

    name = "install_skill"
    description = (
        "Install a new Python tool (skill) into the agent. "
        "Pass the complete Python source code for a BaseTool subclass. "
        "The code is verified with AST scan, Bandit security scan, and pytest "
        "before installation. "
        "On success, the new tool is immediately available to use. "
        "Use dry_run=True to check if the code will pass verification "
        "without actually installing it. "
        "The skill code must: import from tools.base, define one class "
        "inheriting BaseTool, declare name/description/manifest/input_schema, "
        "and implement async execute(self, input: ToolInput) -> ToolResult."
    )
    manifest     = PermissionManifest(
        can_spawn_processes=True,
        filesystem_write=["tools/skills/verified/"],
    )
    input_schema = SkillInput

    def __init__(
        self,
        verifier:  SkillVerifier  | None = None,
        installer: SkillInstaller | None = None,
    ) -> None:
        self._verifier  = verifier  or SkillVerifier()
        self._installer = installer or SkillInstaller()

    async def execute(self, input: ToolInput) -> ToolResult:
        params = self.input_schema(**input.parameters)

        code_hash = hashlib.sha256(params.code.encode()).hexdigest()[:12]
        hint_name = params.skill_name or f"skill_{code_hash}"

        log.info(
            "install_skill_start",
            session_id=input.session_id,
            hint_name=hint_name,
            dry_run=params.dry_run,
            code_length=len(params.code),
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"{hint_name}_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(params.code)
            tmp_path = Path(tmp.name)

        try:
            result = self._verifier.verify(tmp_path)
        finally:
            # Always clean up the temp file
            tmp_path.unlink(missing_ok=True)

        if not result.passed:
            log.info(
                "install_skill_rejected",
                session_id=input.session_id,
                stage=result.stage,
                reason=result.reason[:200],
            )
            guidance_key = result.stage if result.stage in _GUIDANCE else "structure"
            return ToolResult(
                success=False,
                output={
                    "installed":    False,
                    "stage":        result.stage,
                    "reason":       result.reason,
                    "warnings":     result.warnings,
                    "guidance":     _GUIDANCE[guidance_key],
                    "duration_ms":  result.duration_ms,
                },
                error=f"Skill rejected at stage '{result.stage}': {result.reason}",
            )

        if params.dry_run:
            return ToolResult(
                success=True,
                output={
                    "installed":   False,
                    "dry_run":     True,
                    "passed":      True,
                    "skill_class": result.skill_class,
                    "file_hash":   result.file_hash,
                    "warnings":    result.warnings,
                    "duration_ms": result.duration_ms,
                    "message": (
                        f"Skill {result.skill_class!r} passed all verification stages. "
                        "Remove dry_run=True to install it."
                    ),
                },
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"{result.skill_class}_",
            delete=False,
            encoding="utf-8",
        ) as install_tmp:
            install_tmp.write(params.code)
            install_path = Path(install_tmp.name)

        try:
            record = self._installer.install(
                skill_path  = install_path,
                skill_class = result.skill_class,
                file_hash   = result.file_hash,
                warnings    = result.warnings,
            )
        except RuntimeError as exc:
            log.error(
                "install_skill_install_failed",
                session_id=input.session_id,
                error=str(exc),
            )
            return ToolResult(
                success=False,
                output={
                    "installed": False,
                    "reason":    str(exc),
                },
                error=f"Installation failed: {exc}",
            )
        finally:
            install_path.unlink(missing_ok=True)

        log.info(
            "install_skill_complete",
            session_id=input.session_id,
            skill_name=record.skill_name,
            skill_class=record.class_name,
            file_hash=record.file_hash,
        )

        return ToolResult(
            success=True,
            output={
                "installed":       True,
                "skill_name":      record.skill_name,
                "class_name":      record.class_name,
                "file_hash":       record.file_hash,
                "installed_at":    record.installed_at,
                "installed_path":  record.installed_path,
                "warnings":        record.warnings,
                "verification_ms": result.duration_ms,
                "available_tools": ToolRegistry.list_all(),
                "message": (
                    f"Skill {record.skill_name!r} is now installed and ready to use. "
                    f"Call it with tool_name='{record.skill_name}'."
                ),
            },
        )