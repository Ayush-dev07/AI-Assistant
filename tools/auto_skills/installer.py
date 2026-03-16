from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logging import get_logger
from tools.base import BaseTool, ToolRegistry

log = get_logger(__name__)

_DEFAULT_VERIFIED_DIR = Path(__file__).parent.parent / "skills" / "verified"

_MANIFEST_FILENAME = "manifest.json"

@dataclass
class InstallRecord:
    skill_name:     str
    class_name:     str
    file_hash:      str
    installed_at:   str
    source_path:    str
    installed_path: str
    warnings:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name":     self.skill_name,
            "class_name":     self.class_name,
            "file_hash":      self.file_hash,
            "installed_at":   self.installed_at,
            "source_path":    self.source_path,
            "installed_path": self.installed_path,
            "warnings":       self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstallRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

class SkillInstaller:

    def __init__(
        self,
        verified_dir: Path | None = None,
    ) -> None:
        self._verified_dir = Path(verified_dir) if verified_dir else _DEFAULT_VERIFIED_DIR
        self._verified_dir.mkdir(parents=True, exist_ok=True)

        init_file = self._verified_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                '"""Auto-generated package for verified agent skills."""\n'
            )

    def install(
        self,
        skill_path:  Path,
        skill_class: str,
        file_hash:   str = "",
        warnings:    list[str] | None = None,
    ) -> InstallRecord:
        if not skill_path.exists():
            raise RuntimeError(f"Skill file not found: {skill_path}")

        source = skill_path.read_text(encoding="utf-8")
        if not file_hash:
            file_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        dest_path = self._verified_dir / skill_path.name

        log.info(
            "skill_installer_start",
            skill_path=str(skill_path),
            skill_class=skill_class,
            file_hash=file_hash,
            dest=str(dest_path),
        )

        # 1. Copy file 
        try:
            shutil.copy2(skill_path, dest_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to copy skill to {dest_path}: {exc}"
            ) from exc

        # 2. Dynamic import 
        module_name = f"tools.skills.verified.{skill_path.stem}"
        try:
            module = _load_module(module_name, dest_path)
        except Exception as exc:
            # Roll back the copy on import failure
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to import skill module {module_name!r}: {exc}"
            ) from exc

        # 3. Find and instantiate the BaseTool subclass 
        skill_cls = getattr(module, skill_class, None)
        if skill_cls is None:
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Class {skill_class!r} not found in imported module {module_name!r}. "
                "The class name must match the name found by SkillVerifier.ast_scan()."
            )

        if not (isinstance(skill_cls, type) and issubclass(skill_cls, BaseTool)):
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{skill_class!r} is not a BaseTool subclass. "
                "The skill must inherit from tools.base.BaseTool."
            )

        try:
            skill_instance = skill_cls()
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to instantiate {skill_class!r}: {exc}. "
                "Ensure the class has no required __init__ arguments."
            ) from exc

        skill_name = skill_instance.name

        # 4. Register into ToolRegistry 
        ToolRegistry.register(skill_instance)

        log.info(
            "skill_installed",
            skill_name=skill_name,
            skill_class=skill_class,
            file_hash=file_hash,
            dest=str(dest_path),
        )

        # 5. Write manifest
        record = InstallRecord(
            skill_name     = skill_name,
            class_name     = skill_class,
            file_hash      = file_hash,
            installed_at   = datetime.now(tz=timezone.utc).isoformat(),
            source_path    = str(skill_path),
            installed_path = str(dest_path),
            warnings       = warnings or [],
        )
        self._write_manifest(record)

        return record

    # Uninstall 
    def uninstall(
        self,
        skill_name:  str,
        delete_file: bool = False,
    ) -> bool:
        if skill_name not in ToolRegistry:
            log.warning("skill_uninstall_not_found", skill_name=skill_name)
            return False

        ToolRegistry.unregister(skill_name)

        if delete_file:
            record = self._read_manifest().get(skill_name)
            if record:
                installed_path = Path(record.installed_path)
                installed_path.unlink(missing_ok=True)
                self._remove_from_manifest(skill_name)
                log.info("skill_file_deleted", path=str(installed_path))

        log.info("skill_uninstalled", skill_name=skill_name)
        return True

    def list_installed(self) -> list[InstallRecord]:
        return list(self._read_manifest().values())

    def load_all_verified(self) -> list[str]:
        loaded: list[str] = []

        for py_file in sorted(self._verified_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            try:
                module_name = f"tools.skills.verified.{py_file.stem}"
                module      = _load_module(module_name, py_file)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name, None)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseTool)
                        and attr is not BaseTool
                    ):
                        try:
                            instance = attr()
                            ToolRegistry.register(instance)
                            loaded.append(instance.name)
                            log.debug(
                                "skill_loaded_at_startup",
                                skill_name=instance.name,
                                file=py_file.name,
                            )
                        except Exception as exc:
                            log.warning(
                                "skill_load_failed_at_startup",
                                file=py_file.name,
                                class_name=attr_name,
                                error=str(exc),
                            )

            except Exception as exc:
                log.warning(
                    "skill_import_failed_at_startup",
                    file=py_file.name,
                    error=str(exc),
                )

        if loaded:
            log.info("skills_loaded_at_startup", count=len(loaded), names=loaded)

        return loaded

    def _manifest_path(self) -> Path:
        return self._verified_dir / _MANIFEST_FILENAME

    def _read_manifest(self) -> dict[str, InstallRecord]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: InstallRecord.from_dict(v) for k, v in data.items()}
        except Exception as exc:
            log.warning("skill_manifest_read_error", error=str(exc))
            return {}

    def _write_manifest(self, record: InstallRecord) -> None:
        manifest = self._read_manifest()
        manifest[record.skill_name] = record

        path = self._manifest_path()
        tmp  = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {k: v.to_dict() for k, v in manifest.items()},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log.warning("skill_manifest_write_error", error=str(exc))

    def _remove_from_manifest(self, skill_name: str) -> None:
        manifest = self._read_manifest()
        manifest.pop(skill_name, None)
        path = self._manifest_path()
        tmp  = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {k: v.to_dict() for k, v in manifest.items()},
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)

def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module