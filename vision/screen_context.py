from __future__ import annotations

import asyncio
import base64
import io
import logging
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PIL_Image

logger = logging.getLogger(__name__)

try:
    import pyautogui  
    pyautogui.FAILSAFE = False  
    _PYAUTOGUI_OK = True
except ImportError:
    pyautogui = None  
    _PYAUTOGUI_OK = False
    logger.warning(
        "pyautogui not installed — screenshots unavailable. "
        "Run: pip install pyautogui pillow"
    )
try:
    from PIL import Image  
    _PIL_OK = True
except ImportError:
    Image = None  
    _PIL_OK = False
    logger.warning("Pillow not installed. Run: pip install pillow")

AMBIENT_INTERVAL_S: float = 8.0        
BUFFER_SIZE: int = 3                   
AMBIENT_RESIZE: tuple[int, int] = (1280, 800)   
CURSOR_REGION_SIZE: int = 400          

@dataclass
class ScreenContext:

    full_screenshot_b64: str | None = None
    cursor_region_b64:   str | None = None
    cursor_pos:          tuple[int, int] = field(default_factory=lambda: (0, 0))
    active_window:       str = ""
    window_title:        str = ""
    screen_width:        int = 0
    screen_height:       int = 0
    timestamp:           str = field(default_factory=lambda: _iso_now())
    ambient_age_s:       float = 0.0

    def has_screenshot(self) -> bool:
        return self.full_screenshot_b64 is not None or self.cursor_region_b64 is not None

    def summary(self) -> str:
        parts = []
        if self.active_window:
            parts.append(self.active_window)
        if self.window_title:
            parts.append(f'"{self.window_title[:60]}"')
        if self.cursor_pos != (0, 0):
            parts.append(f"cursor@{self.cursor_pos}")
        has = []
        if self.full_screenshot_b64:
            has.append(f"screenshot(age={self.ambient_age_s:.1f}s)")
        if self.cursor_region_b64:
            has.append(f"cursor_region({CURSOR_REGION_SIZE}×{CURSOR_REGION_SIZE})")
        if has:
            parts.append(" ".join(has))
        return " | ".join(parts) or "empty context"

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _run(cmd: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def _pil_to_b64_png(img: "PIL_Image.Image") -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()

def _capture_screenshot_b64(
    resize: tuple[int, int] | None = AMBIENT_RESIZE,
    region: tuple[int, int, int, int] | None = None,
) -> str | None:
    if not _PYAUTOGUI_OK or not _PIL_OK:
        return None
    try:
        pil_img = pyautogui.screenshot(region=region)
        if resize and region is None:
            pil_img = pil_img.resize(resize, Image.LANCZOS)
        return _pil_to_b64_png(pil_img)
    except Exception as exc:
        logger.debug("Screenshot capture failed: %s", exc)
        return None

def _get_screen_size() -> tuple[int, int]:
    if not _PYAUTOGUI_OK:
        return (0, 0)
    try:
        size = pyautogui.size()
        return (size.width, size.height)
    except Exception:
        return (0, 0)

def _get_cursor_pos() -> tuple[int, int]:
    if not _PYAUTOGUI_OK:
        return (0, 0)
    try:
        pos = pyautogui.position()
        return (pos.x, pos.y)
    except Exception:
        return (0, 0)

def _cursor_region_box(
    cx: int,
    cy: int,
    size: int,
    screen_w: int,
    screen_h: int,
) -> tuple[int, int, int, int]:
    half = size // 2
    left = cx - half
    top  = cy - half

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if screen_w > 0 and left + size > screen_w:
        left = max(0, screen_w - size)
    if screen_h > 0 and top + size > screen_h:
        top = max(0, screen_h - size)

    return (left, top, size, size)

def _capture_cursor_region_b64(
    cursor_pos: tuple[int, int],
    screen_size: tuple[int, int],
) -> str | None:
    if not _PYAUTOGUI_OK or not _PIL_OK:
        return None
    cx, cy = cursor_pos
    sw, sh = screen_size
    region = _cursor_region_box(cx, cy, CURSOR_REGION_SIZE, sw, sh)
    return _capture_screenshot_b64(resize=None, region=region)

def _xdotool_available() -> bool:
    return shutil.which("xdotool") is not None

def get_active_window() -> tuple[str, str]:
    if not _xdotool_available():
        logger.debug("xdotool not found — active window detection unavailable")
        return ("", "")
    try:
        wid_res = _run(["xdotool", "getactivewindow"])
        if wid_res.returncode != 0 or not wid_res.stdout.strip():
            return ("", "")
        wid = wid_res.stdout.strip()

        title_res = _run(["xdotool", "getwindowname", wid])
        window_title = title_res.stdout.strip() if title_res.returncode == 0 else ""

        pid_res = _run(["xdotool", "getwindowpid", wid])
        app_name = ""
        if pid_res.returncode == 0 and pid_res.stdout.strip():
            pid = pid_res.stdout.strip()
            try:
                with open(f"/proc/{pid}/comm") as fh:
                    app_name = fh.read().strip()
            except (OSError, IOError):
                pass

        if not app_name and window_title:
            app_name = _infer_app_name(window_title)

        return (app_name.lower(), window_title)
    except Exception as exc:
        logger.debug("get_active_window failed: %s", exc)
        return ("", "")

def _infer_app_name(window_title: str) -> str:
    title_lower = window_title.lower()
    known: list[tuple[str, str]] = [
        ("firefox", "firefox"),
        ("chrome", "chrome"),
        ("chromium", "chromium"),
        ("visual studio code", "code"),
        ("vs code", "code"),
        ("code", "code"),
        ("terminal", "terminal"),
        ("spotify", "spotify"),
        ("vlc", "vlc"),
        ("sublime", "sublime"),
        ("gedit", "gedit"),
        ("nautilus", "nautilus"),
        ("thunar", "thunar"),
        ("gimp", "gimp"),
        ("inkscape", "inkscape"),
        ("libreoffice", "libreoffice"),
        ("slack", "slack"),
        ("discord", "discord"),
        ("zoom", "zoom"),
        ("telegram", "telegram"),
    ]
    for keyword, name in known:
        if keyword in title_lower:
            return name
    first_word = window_title.split()[0] if window_title.split() else ""
    return first_word.lower()

@dataclass
class _AmbientFrame:
    b64_png:    str
    captured_at: float  

class ScreenContextCapture:

    def __init__(
        self,
        interval_s: float = AMBIENT_INTERVAL_S,
        buffer_size: int = BUFFER_SIZE,
    ) -> None:
        self._interval     = interval_s
        self._buffer: Deque[_AmbientFrame] = deque(maxlen=buffer_size)
        self._task: asyncio.Task | None    = None
        self._running      = False
        self._screen_size  = _get_screen_size()

    async def start(self) -> None:
        if self._running:
            return
        self._running    = True
        self._screen_size = _get_screen_size()
        self._do_ambient_capture()
        self._task = asyncio.create_task(self._capture_loop(), name="ambient_capture")
        logger.info(
            "ScreenContextCapture started (interval=%.1fs, buffer=%d)",
            self._interval, self._buffer.maxlen,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ScreenContextCapture stopped.")

    async def _capture_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._do_ambient_capture)
            except Exception as exc:
                logger.debug("Ambient capture error: %s", exc)

    def _do_ambient_capture(self) -> None:
        b64 = _capture_screenshot_b64(resize=AMBIENT_RESIZE, region=None)
        if b64:
            self._buffer.append(
                _AmbientFrame(b64_png=b64, captured_at=time.monotonic())
            )
            logger.debug("Ambient buffer updated (%d frames)", len(self._buffer))

    async def get_context(
        self,
        include_cursor_region: bool = True,
        include_full_screenshot: bool = True,
    ) -> ScreenContext:
        now_mono = time.monotonic()
        active_window, window_title = get_active_window()
        cursor_pos  = _get_cursor_pos()
        screen_size = self._screen_size or _get_screen_size()
        sw, sh      = screen_size
        full_b64:   str | None = None
        ambient_age = 0.0
        if include_full_screenshot and self._buffer:
            latest    = self._buffer[-1]
            full_b64  = latest.b64_png
            ambient_age = round(now_mono - latest.captured_at, 1)

        cursor_b64: str | None = None
        if include_cursor_region and _PYAUTOGUI_OK and _PIL_OK:
            try:
                loop = asyncio.get_running_loop()
                cursor_b64 = await loop.run_in_executor(
                    None,
                    _capture_cursor_region_b64,
                    cursor_pos,
                    screen_size,
                )
            except Exception as exc:
                logger.debug("Cursor region capture failed: %s", exc)

        ctx = ScreenContext(
            full_screenshot_b64 = full_b64,
            cursor_region_b64   = cursor_b64,
            cursor_pos          = cursor_pos,
            active_window       = active_window,
            window_title        = window_title,
            screen_width        = sw,
            screen_height       = sh,
            timestamp           = _iso_now(),
            ambient_age_s       = ambient_age,
        )
        logger.debug("ScreenContext assembled: %s", ctx.summary())
        return ctx

    def buffer_size(self) -> int:
        return len(self._buffer)

    def latest_screenshot_age_s(self) -> float | None:
        if not self._buffer:
            return None
        return round(time.monotonic() - self._buffer[-1].captured_at, 1)

    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return (
            f"ScreenContextCapture("
            f"running={self._running}, "
            f"buffer={len(self._buffer)}/{self._buffer.maxlen}, "
            f"interval={self._interval}s)"
        )

def capture_context_sync() -> ScreenContext:
    active_window, window_title = get_active_window()
    cursor_pos  = _get_cursor_pos()
    screen_size = _get_screen_size()
    sw, sh      = screen_size

    full_b64   = _capture_screenshot_b64(resize=AMBIENT_RESIZE, region=None)
    cursor_b64 = _capture_cursor_region_b64(cursor_pos, screen_size)

    return ScreenContext(
        full_screenshot_b64 = full_b64,
        cursor_region_b64   = cursor_b64,
        cursor_pos          = cursor_pos,
        active_window       = active_window,
        window_title        = window_title,
        screen_width        = sw,
        screen_height       = sh,
        timestamp           = _iso_now(),
        ambient_age_s       = 0.0,
    )