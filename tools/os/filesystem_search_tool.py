from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult

log = get_logger(__name__)

MAX_READ_CHARS    = 8_000     
MAX_SEARCH_RESULTS = 20       
MAX_DIR_ENTRIES   = 100       
MAX_CSV_ROWS      = 50        

_COMMON_DIRS: list[Path] = [
    Path.home(),
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Pictures",
    Path.home() / "Videos",
    Path.home() / "Music",
    Path.home() / ".config",
    Path("/tmp"),
]

_TEXT_SIZE_LIMIT = 32 * 1024 * 1024

_PROTECTED_PATHS: set[str] = {
    "/", "/etc", "/boot", "/sys", "/proc", "/dev",
    "/usr", "/bin", "/sbin", "/lib", "/lib64",
    str(Path.home() / ".ssh"),
    str(Path.home() / ".gnupg"),
}

HITL_ACTIONS: set[str] = {"delete_file"}

class FilesystemSearchInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Action to perform. One of: search_files, open_file, read_file, "
            "list_directory, get_file_info, delete_file (HITL)."
        ),
    )
    query: str = Field(
        default="",
        description=(
            "File name or pattern to search for. "
            "Supports wildcards: '*.pdf', 'resume*', '*.py'. "
            "Case-insensitive."
        ),
    )
    search_type: str = Field(
        default="any",
        description=(
            "Filter by type: any | file | directory | "
            "pdf | image | video | audio | code | document | archive"
        ),
    )
    directory: str = Field(
        default="",
        description=(
            "Root directory to search within. "
            "Defaults to the user's home directory. "
            "Use '/' for full-system search (slower)."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Maximum number of results to return.",
    )
    path: str = Field(
        default="",
        description="Absolute or relative path for open_file, read_file, list_directory, get_file_info, delete_file.",
    )
    max_chars: int = Field(
        default=MAX_READ_CHARS,
        ge=100,
        le=MAX_READ_CHARS,
        description="Maximum characters to read from a text file.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding to try when reading (utf-8, latin-1, etc.)",
    )
    show_hidden: bool = Field(
        default=False,
        description="Include hidden files (starting with '.') in directory listing.",
    )
    permanent: bool = Field(
        default=False,
        description=(
            "If True: delete permanently with rm (UNRECOVERABLE). "
            "If False (default): move to Trash with gio trash (recoverable). "
            "HITL approval always required."
        ),
    )

class FilesystemSearchTool(BaseTool):
    name         = "filesystem_search"
    description  = (
        "Search for files anywhere on the machine, open them with the default "
        "application, read their contents, or list directories. "
        "Actions: search_files (query across full filesystem), open_file (xdg-open), "
        "read_file (returns text content up to 8000 chars), "
        "list_directory (files + subdirs with sizes and dates), "
        "get_file_info (size, type, permissions, dates), "
        "delete_file (HITL — moves to Trash by default, or permanent with rm)."
    )
    manifest     = PermissionManifest(
        filesystem_read  = ["/"],               
        filesystem_write = [str(Path.home())],  
        can_spawn_processes = True,
    )
    input_schema = FilesystemSearchInput

    async def execute(self, inp: ToolInput) -> ToolResult:
        try:
            params = FilesystemSearchInput(**inp.parameters)
        except Exception as exc:
            return ToolResult(success=False, error=f"Invalid parameters: {exc}")

        action = params.action.lower().strip()
        log.debug("filesystem_search_execute", action=action, session=inp.session_id)

        dispatch = {
            "search_files":   self._search_files,
            "open_file":      self._open_file,
            "read_file":      self._read_file,
            "list_directory": self._list_directory,
            "get_file_info":  self._get_file_info,
            "delete_file":    self._delete_file,
        }

        handler = dispatch.get(action)
        if handler is None:
            return ToolResult(
                success = False,
                error   = (
                    f"Unknown action: {action!r}. "
                    f"Valid actions: {', '.join(sorted(dispatch.keys()))}"
                ),
            )
        try:
            return await handler(params)
        except Exception as exc:
            log.error("filesystem_search_error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"{action} failed: {exc}")

    async def _search_files(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.query:
            return ToolResult(
                success = False,
                error   = "query is required for search_files.",
            )
        query        = p.query.strip()
        search_root  = p.directory.strip() or str(Path.home())
        type_filter  = p.search_type.lower()
        max_results  = p.max_results

        if "*" in query or "?" in query:
            pattern = query            
        else:
            pattern = f"*{query}*"    

        results: list[dict[str, Any]] = []

        if shutil.which("locate") and results == []:
            results = _locate_search(pattern, search_root, max_results * 2)
            log.debug("filesystem_search_locate", query=query, found=len(results))

        if not results:
            results = _find_search(pattern, search_root, max_results * 2)
            log.debug("filesystem_search_find", query=query, found=len(results))

        if not results:
            results = _glob_search(pattern, max_results * 2)
            log.debug("filesystem_search_glob", query=query, found=len(results))

        if type_filter != "any":
            results = [r for r in results if _matches_type_filter(r["path"], type_filter)]

        results.sort(key=lambda r: r.get("mtime", 0), reverse=True)
        results = results[:max_results]

        for r in results:
            r["mime_type"]    = _detect_mime(r["path"])
            r["modified_ago"] = _human_time_ago(r.get("mtime", 0))
            r.pop("mtime", None)  

        if not results:
            return ToolResult(
                success = True,
                output  = {
                    "files":   [],
                    "query":   query,
                    "message": (
                        f"No files found matching '{query}'. "
                        "Try a broader query or check the search directory."
                    ),
                },
            )
        return ToolResult(
            success = True,
            output  = {
                "files":        results,
                "total_found":  len(results),
                "query":        query,
                "search_root":  search_root,
                "type_filter":  type_filter,
                "tip":          "Use open_file(path=...) to open a result, or read_file(path=...) to read its contents.",
            },
        )

    async def _open_file(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.path:
            return ToolResult(success=False, error="path is required for open_file.")
        target = Path(p.path).expanduser().resolve()
        if not target.exists():
            return ToolResult(
                success = False,
                error   = f"Path does not exist: {target}",
            )
        if not shutil.which("xdg-open"):
            return ToolResult(
                success = False,
                error   = (
                    "xdg-open not found. "
                    "Install: sudo apt install xdg-utils"
                ),
            )
        mime = _detect_mime(str(target))

        try:
            subprocess.Popen(
                ["xdg-open", str(target)],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                start_new_session = True,   
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"xdg-open failed: {exc}")

        log.info("filesystem_search_opened", path=str(target), mime=mime)
        return ToolResult(
            success = True,
            output  = {
                "path":      str(target),
                "mime_type": mime,
                "is_dir":    target.is_dir(),
                "message":   f"Opened '{target.name}' with the default application.",
            },
        )

    async def _read_file(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.path:
            return ToolResult(success=False, error="path is required for read_file.")

        target = Path(p.path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, error=f"Path does not exist: {target}")

        if not target.is_file():
            return ToolResult(
                success = False,
                error   = f"{target} is a directory. Use list_directory instead.",
            )
        stat      = target.stat()
        size_bytes = stat.st_size
        mime_type  = _detect_mime(str(target))

        if _is_binary_mime(mime_type) and size_bytes > 0:
            return ToolResult(
                success = True,
                output  = {
                    "path":      str(target),
                    "mime_type": mime_type,
                    "size_kb":   round(size_bytes / 1024, 1),
                    "readable":  False,
                    "message":   (
                        f"'{target.name}' is a binary file ({mime_type}). "
                        "Cannot display contents. "
                        "Use open_file to open it with the default application."
                    ),
                },
            )
        if size_bytes > _TEXT_SIZE_LIMIT:
            return ToolResult(
                success = False,
                error   = (
                    f"File too large to read ({size_bytes / 1e6:.1f} MB). "
                    "Max readable size is 32 MB."
                ),
            )
        content, used_encoding = _read_text_file(target, p.encoding)
        if content is None:
            return ToolResult(
                success = False,
                error   = (
                    f"Could not decode '{target.name}' as text. "
                    "It may be a binary file with a text extension."
                ),
            )
        ext = target.suffix.lower()
        parsed_output: dict[str, Any] = {
            "path":          str(target),
            "name":          target.name,
            "mime_type":     mime_type,
            "size_bytes":    size_bytes,
            "encoding_used": used_encoding,
            "truncated":     len(content) > p.max_chars,
        }

        if ext == ".json":
            try:
                data = json.loads(content)
                parsed_output["format"]  = "json"
                parsed_output["content"] = json.dumps(data, indent=2, ensure_ascii=False)[:p.max_chars]
                parsed_output["keys"]    = list(data.keys()) if isinstance(data, dict) else f"list[{len(data)}]"
            except json.JSONDecodeError:
                parsed_output["format"]  = "text"
                parsed_output["content"] = content[:p.max_chars]

        elif ext == ".csv":
            try:
                reader = csv.DictReader(io.StringIO(content))
                rows   = list(reader)[:MAX_CSV_ROWS]
                parsed_output["format"]   = "csv"
                parsed_output["columns"]  = reader.fieldnames or []
                parsed_output["rows"]     = rows
                parsed_output["row_count_shown"]  = len(rows)
                parsed_output["content"]  = content[:2000]  
            except Exception:
                parsed_output["format"]  = "text"
                parsed_output["content"] = content[:p.max_chars]

        else:
            parsed_output["format"]  = "text"
            parsed_output["content"] = content[:p.max_chars]

        log.info("filesystem_search_read", path=str(target), size=size_bytes)
        return ToolResult(success=True, output=parsed_output)

    async def _list_directory(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.path:
            return ToolResult(success=False, error="path is required for list_directory.")
        target = Path(p.path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, error=f"Path does not exist: {target}")
        if not target.is_dir():
            
            target = target.parent
        entries: list[dict[str, Any]] = []
        dirs:    list[dict[str, Any]] = []
        total_size = 0
        try:
            items = sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return ToolResult(
                success = False,
                error   = f"Permission denied reading directory: {target}",
            )

        for item in items:
            if not p.show_hidden and item.name.startswith("."):
                continue
            try:
                stat      = item.stat()
                size_bytes = stat.st_size
                total_size += size_bytes

                entry = {
                    "name":        item.name,
                    "path":        str(item),
                    "is_dir":      item.is_dir(),
                    "size_kb":     round(size_bytes / 1024, 1) if not item.is_dir() else None,
                    "modified":    time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    "mime_type":   _detect_mime(str(item)) if item.is_file() else "directory",
                }
                if item.is_dir():
                    dirs.append(entry)
                else:
                    entries.append(entry)
            except (PermissionError, OSError):
                continue
        all_entries = (dirs + entries)[:MAX_DIR_ENTRIES]

        return ToolResult(
            success = True,
            output  = {
                "path":          str(target),
                "entries":       all_entries,
                "file_count":    len(entries),
                "dir_count":     len(dirs),
                "total_shown":   len(all_entries),
                "total_size_mb": round(total_size / 1e6, 2),
                "hidden_shown":  p.show_hidden,
                "tip":           "Use open_file(path=...) to open any entry.",
            },
        )

    async def _get_file_info(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.path:
            return ToolResult(success=False, error="path is required for get_file_info.")

        target = Path(p.path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, error=f"Path does not exist: {target}")

        try:
            stat      = target.stat()
            lstat     = target.lstat()       
            is_link   = target.is_symlink()
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))

        import pwd, grp
        try:
            owner = pwd.getpwuid(stat.st_uid).pw_name
        except (KeyError, ImportError):
            owner = str(stat.st_uid)
        try:
            group = grp.getgrgid(stat.st_gid).gr_name
        except (KeyError, ImportError):
            group = str(stat.st_gid)

        perms_octal  = oct(stat.st_mode)[-3:]
        perms_human  = _format_permissions(stat.st_mode)

        child_count = None
        if target.is_dir():
            try:
                child_count = sum(1 for _ in target.iterdir())
            except PermissionError:
                child_count = "?"
        mime_type = _detect_mime(str(target))

        output: dict[str, Any] = {
            "path":           str(target),
            "name":           target.name,
            "parent":         str(target.parent),
            "type":           "directory" if target.is_dir() else ("symlink" if is_link else "file"),
            "mime_type":      mime_type,
            "size_bytes":     stat.st_size,
            "size_human":     _human_size(stat.st_size),
            "created":        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)),
            "modified":       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            "accessed":       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_atime)),
            "owner":          owner,
            "group":          group,
            "permissions":    perms_human,
            "permissions_octal": perms_octal,
            "readable":       os.access(target, os.R_OK),
            "writable":       os.access(target, os.W_OK),
            "executable":     os.access(target, os.X_OK),
            "is_symlink":     is_link,
            "symlink_target": str(target.readlink()) if is_link else None,
            "child_count":    child_count,
        }
        return ToolResult(success=True, output=output)

    async def _delete_file(self, p: FilesystemSearchInput) -> ToolResult:
        if not p.path:
            return ToolResult(success=False, error="path is required for delete_file.")

        target = Path(p.path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, error=f"Path does not exist: {target}")

        target_str = str(target)
        for protected in _PROTECTED_PATHS:
            if target_str == protected or target_str.startswith(protected + "/"):
                return ToolResult(
                    success = False,
                    error   = (
                        f"Refusing to delete protected path: {target}. "
                        "This path is essential for system stability."
                    ),
                )
        try:
            stat = target.stat()
            size_bytes = stat.st_size
            if target.is_dir():
                
                try:
                    size_bytes = sum(
                        f.stat().st_size
                        for f in target.rglob("*")
                        if f.is_file()
                    )
                except Exception:
                    pass
        except OSError:
            size_bytes = 0

        modified = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(target.stat().st_mtime),
        ) if target.exists() else ""

        log.warning(
            "filesystem_search_delete",
            path      = str(target),
            permanent = p.permanent,
            size      = size_bytes,
        )

        if not p.permanent:
            
            if shutil.which("gio"):
                result = subprocess.run(
                    ["gio", "trash", str(target)],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    return ToolResult(
                        success = True,
                        output  = {
                            "path":       str(target),
                            "method":     "trash",
                            "size_human": _human_size(size_bytes),
                            "modified":   modified,
                            "message":    (
                                f"'{target.name}' moved to Trash. "
                                "You can restore it from the Trash folder."
                            ),
                        },
                    )
                return ToolResult(
                    success = False,
                    error   = f"gio trash failed: {result.stderr.strip()}",
                )

            if shutil.which("trash-put"):
                result = subprocess.run(
                    ["trash-put", str(target)],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    return ToolResult(
                        success = True,
                        output  = {
                            "path":       str(target),
                            "method":     "trash-put",
                            "size_human": _human_size(size_bytes),
                            "message":    f"'{target.name}' moved to Trash.",
                        },
                    )

            return ToolResult(
                success = False,
                error   = (
                    "Neither gio nor trash-put available. "
                    "Install: sudo apt install glib2.0-bin  or  sudo apt install trash-cli"
                ),
            )
        else:
            try:
                if target.is_dir():
                    shutil.rmtree(str(target))
                else:
                    target.unlink()
                return ToolResult(
                    success = True,
                    output  = {
                        "path":       str(target),
                        "method":     "permanent",
                        "size_human": _human_size(size_bytes),
                        "modified":   modified,
                        "message":    (
                            f"'{target.name}' permanently deleted. "
                            f"Freed {_human_size(size_bytes)}. "
                            "This cannot be undone."
                        ),
                    },
                )
            except PermissionError:
                return ToolResult(
                    success = False,
                    error   = (
                        f"Permission denied deleting: {target}. "
                        "The file may be owned by another user."
                    ),
                )
            except OSError as exc:
                return ToolResult(success=False, error=str(exc))

def _locate_search(
    pattern:    str,
    search_root: str,
    limit:      int,
) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["locate", "-i", "--limit", str(limit), pattern],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0:
            return []

        results = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if search_root and search_root != "/" and not line.startswith(search_root):
                continue
            p = Path(line)
            if p.exists():
                try:
                    st = p.stat()
                    results.append({
                        "path":     line,
                        "name":     p.name,
                        "size_kb":  round(st.st_size / 1024, 1) if p.is_file() else None,
                        "is_dir":   p.is_dir(),
                        "mtime":    st.st_mtime,
                    })
                except OSError:
                    continue
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

def _find_search(
    pattern:    str,
    search_root: str,
    limit:      int,
) -> list[dict[str, Any]]:
    root = search_root if search_root else str(Path.home())
    exclude_paths = ["/proc", "/sys", "/dev"]
    cmd = ["find", root]
    for ep in exclude_paths:
        if root == "/" or root.startswith(ep):
            pass  
        else:
            cmd += ["-not", "-path", f"{ep}/*"]
    cmd += ["-iname", pattern, "-not", "-name", ".*"]  

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30,
        )
        results = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.exists():
                try:
                    st = p.stat()
                    results.append({
                        "path":    line,
                        "name":    p.name,
                        "size_kb": round(st.st_size / 1024, 1) if p.is_file() else None,
                        "is_dir":  p.is_dir(),
                        "mtime":   st.st_mtime,
                    })
                    if len(results) >= limit:
                        break
                except OSError:
                    continue
        return results
    except subprocess.TimeoutExpired:
        log.warning("filesystem_search_find_timeout", pattern=pattern, root=root)
        return []
    except FileNotFoundError:
        return []

def _glob_search(
    pattern: str,
    limit:   int,
) -> list[dict[str, Any]]:
    results = []
    seen:    set[str] = set()
    for base_dir in _COMMON_DIRS:
        if not base_dir.exists():
            continue
        try:
            for match in base_dir.rglob(pattern):
                path_str = str(match)
                if path_str in seen:
                    continue
                seen.add(path_str)
                try:
                    st = match.stat()
                    results.append({
                        "path":    path_str,
                        "name":    match.name,
                        "size_kb": round(st.st_size / 1024, 1) if match.is_file() else None,
                        "is_dir":  match.is_dir(),
                        "mtime":   st.st_mtime,
                    })
                    if len(results) >= limit:
                        return results
                except OSError:
                    continue
        except (PermissionError, OSError):
            continue

    return results

_TYPE_EXTENSIONS: dict[str, set[str]] = {
    "pdf":      {".pdf"},
    "image":    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".tiff", ".ico"},
    "video":    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"},
    "audio":    {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"},
    "code":     {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
                 ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh", ".sql",
                 ".html", ".css", ".scss", ".yaml", ".yml", ".toml", ".json",
                 ".xml", ".md", ".txt", ".csv", ".ini", ".cfg", ".env"},
    "document": {".pdf", ".doc", ".docx", ".odt", ".ppt", ".pptx", ".xls", ".xlsx",
                 ".ods", ".odp", ".rtf", ".epub", ".md", ".tex"},
    "archive":  {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".deb", ".rpm"},
}

def _matches_type_filter(path_str: str, filter_type: str) -> bool:
    if filter_type == "any":
        return True
    if filter_type == "file":
        return Path(path_str).is_file()
    if filter_type == "directory":
        return Path(path_str).is_dir()
    extensions = _TYPE_EXTENSIONS.get(filter_type, set())
    return Path(path_str).suffix.lower() in extensions

def _detect_mime(path_str: str) -> str:
    try:
        import magic
        return magic.from_file(path_str, mime=True)
    except (ImportError, Exception):
        pass

    mime, _ = mimetypes.guess_type(path_str)
    return mime or "application/octet-stream"

def _is_binary_mime(mime: str) -> bool:
    if mime.startswith("text/"):
        return False
    text_like = {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-sh",
        "application/x-python",
        "application/toml",
        "application/x-yaml",
    }
    return mime not in text_like

def _read_text_file(path: Path, preferred_encoding: str) -> tuple[str | None, str]:
    encodings = [preferred_encoding, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    seen: set[str] = set()

    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            content = path.read_text(encoding=enc, errors="strict")
            return content, enc
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError as exc:
            log.warning("read_text_file_error", path=str(path), error=str(exc))
            return None, ""

    return None, ""

def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"

def _human_time_ago(timestamp: float) -> str:
    if not timestamp:
        return "unknown"
    diff = time.time() - timestamp
    if diff < 60:
        return "just now"
    elif diff < 3600:
        return f"{int(diff / 60)} min ago"
    elif diff < 86400:
        return f"{int(diff / 3600)} hours ago"
    elif diff < 86400 * 7:
        return f"{int(diff / 86400)} days ago"
    elif diff < 86400 * 30:
        return f"{int(diff / 86400 / 7)} weeks ago"
    else:
        return time.strftime("%Y-%m-%d", time.localtime(timestamp))

def _format_permissions(mode: int) -> str:
    import stat
    flags = [
        (stat.S_IRUSR, "r"), (stat.S_IWUSR, "w"), (stat.S_IXUSR, "x"),
        (stat.S_IRGRP, "r"), (stat.S_IWGRP, "w"), (stat.S_IXGRP, "x"),
        (stat.S_IROTH, "r"), (stat.S_IWOTH, "w"), (stat.S_IXOTH, "x"),
    ]
    result = ""
    for flag, char in flags:
        result += char if (mode & flag) else "-"
    return result