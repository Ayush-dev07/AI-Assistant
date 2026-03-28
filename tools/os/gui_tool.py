from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = True          
    pyautogui.PAUSE    = 0.05          
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None                   
    _PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not installed — GUI automation unavailable.")

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    Image = None                       
    _PIL_AVAILABLE = False

_SCREENSHOT_DIR = Path(os.environ.get("SUPERAGENT_TMP", "/tmp/superagent")) / "screenshots"
_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ToolResult:
    success: bool
    output: dict[str, Any]
    error: str | None = None

def _run(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

def _require_pyautogui() -> None:
    if not _PYAUTOGUI_AVAILABLE:
        raise RuntimeError(
            "pyautogui is not installed. Run: pip install pyautogui pillow"
        )

def _require_display() -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No DISPLAY environment variable set. "
            "GUI automation requires an active X11 session."
        )

class GUITool:
    
    _ACTION_MAP: dict[str, str] = {
        "screenshot":                "_screenshot",
        "mouse_click":               "_mouse_click",
        "mouse_move":                "_mouse_move",
        "scroll":                    "_scroll",
        "type_text":                 "_type_text",
        "hotkey":                    "_hotkey",
        "press_key":                 "_press_key",
        "type_password":             "_type_password",
        "list_windows":              "_list_windows",
        "focus_window":              "_focus_window",
        "resize_window":             "_resize_window",
        "minimize_window":           "_minimize_window",
        "maximize_window":           "_maximize_window",
        "verify_with_vision":        "_verify_with_vision",
        "get_screen_size":           "_get_screen_size",
        "get_mouse_position":        "_get_mouse_position",
    }
    HITL_ACTIONS: frozenset[str] = frozenset({
        "type_password",
        "confirm_dialog",
        "click_destructive_button",
    })

    def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate method."""
        method_name = self._ACTION_MAP.get(action)
        if method_name is None:
            return ToolResult(
                success=False,
                output={},
                error=(
                    f"Unknown action '{action}'. "
                    f"Available: {sorted(self._ACTION_MAP)}"
                ),
            )
        try:
            _require_display()
            method = getattr(self, method_name)
            return method(**kwargs)
        except Exception as exc:
            logger.exception("GUITool.execute(%s) failed", action)
            return ToolResult(success=False, output={}, error=str(exc))

    def _screenshot(
        self,
        region: tuple[int, int, int, int] | None = None,
        save: bool = True,
    ) -> ToolResult:
        _require_pyautogui()
        img = pyautogui.screenshot(region=region)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode()

        path = None
        if save:
            ts = int(time.time() * 1000)
            path = str(_SCREENSHOT_DIR / f"screenshot_{ts}.png")
            with open(path, "wb") as fh:
                fh.write(png_bytes)

        return ToolResult(
            success=True,
            output={
                "path":       path,
                "base64_png": b64,
                "width":      img.width,
                "height":     img.height,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def _mouse_click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> ToolResult:
        _require_pyautogui()
        if button not in ("left", "right", "middle"):
            return ToolResult(
                success=False,
                output={},
                error=f"Invalid button '{button}'. Use 'left', 'right', or 'middle'.",
            )
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
        return ToolResult(
            success=True,
            output={
                "action":  "mouse_click",
                "x":       x,
                "y":       y,
                "button":  button,
                "clicks":  clicks,
            },
        )

    def _mouse_move(self, x: int, y: int, duration: float = 0.2) -> ToolResult:
        _require_pyautogui()
        pyautogui.moveTo(x, y, duration=duration)
        return ToolResult(
            success=True,
            output={"action": "mouse_move", "x": x, "y": y},
        )

    def _scroll(self, x: int, y: int, amount: int = 3) -> ToolResult:
        _require_pyautogui()
        pyautogui.moveTo(x, y)
        pyautogui.scroll(amount)
        return ToolResult(
            success=True,
            output={"action": "scroll", "x": x, "y": y, "amount": amount},
        )

    def _get_mouse_position(self) -> ToolResult:
        _require_pyautogui()
        pos = pyautogui.position()
        return ToolResult(
            success=True,
            output={"x": pos.x, "y": pos.y},
        )

    def _get_screen_size(self) -> ToolResult:
        _require_pyautogui()
        size = pyautogui.size()
        return ToolResult(
            success=True,
            output={"width": size.width, "height": size.height},
        )

    def _type_text(self, text: str, interval: float = 0.05) -> ToolResult:
        _require_pyautogui()
        pyautogui.typewrite(text, interval=interval)
        return ToolResult(
            success=True,
            output={"action": "type_text", "length": len(text)},
        )

    def _hotkey(self, keys: list[str]) -> ToolResult:
        _require_pyautogui()
        if not keys:
            return ToolResult(
                success=False, output={}, error="'keys' list must not be empty."
            )
        pyautogui.hotkey(*keys)
        return ToolResult(
            success=True,
            output={"action": "hotkey", "keys": keys},
        )

    def _press_key(self, key: str) -> ToolResult:
        _require_pyautogui()
        pyautogui.press(key)
        return ToolResult(
            success=True,
            output={"action": "press_key", "key": key},
        )

    def _type_password(
        self,
        password: str,
        interval: float = 0.05,
    ) -> ToolResult:
        _require_pyautogui()
        pyautogui.typewrite(password, interval=interval)
        
        return ToolResult(
            success=True,
            output={"action": "type_password", "result": "[PASSWORD TYPED]"},
        )

    @staticmethod
    def _xdotool_available() -> bool:
        return _run(["which", "xdotool"]).returncode == 0

    def _list_windows(self, name_filter: str = "") -> ToolResult:
        if not self._xdotool_available():
            return ToolResult(
                success=False,
                output={},
                error="xdotool not found. Install with: sudo apt install xdotool",
            )

        result = _run(["xdotool", "search", "--onlyvisible", "--name", ""])
        if result.returncode != 0:
            return ToolResult(
                success=False,
                output={},
                error=f"xdotool search failed: {result.stderr.strip()}",
            )

        window_ids = [wid for wid in result.stdout.strip().splitlines() if wid]
        windows: list[dict[str, Any]] = []

        for wid in window_ids:
            title_res = _run(["xdotool", "getwindowname", wid])
            title = title_res.stdout.strip() if title_res.returncode == 0 else ""
            if name_filter and name_filter.lower() not in title.lower():
                continue

            geom_res = _run(["xdotool", "getwindowgeometry", "--shell", wid])
            geom: dict[str, int] = {}
            if geom_res.returncode == 0:
                for line in geom_res.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        try:
                            geom[k.strip()] = int(v.strip())
                        except ValueError:
                            pass

            windows.append(
                {
                    "id":     wid,
                    "title":  title,
                    "x":      geom.get("X", 0),
                    "y":      geom.get("Y", 0),
                    "width":  geom.get("WIDTH", 0),
                    "height": geom.get("HEIGHT", 0),
                }
            )

        return ToolResult(
            success=True,
            output={"windows": windows, "count": len(windows)},
        )

    def _focus_window(
        self,
        window_id: str | None = None,
        window_name: str | None = None,
    ) -> ToolResult:
        if not self._xdotool_available():
            return ToolResult(
                success=False,
                output={},
                error="xdotool not installed.",
            )

        if window_id is None and window_name is not None:
            res = _run(
                ["xdotool", "search", "--onlyvisible", "--name", window_name]
            )
            ids = res.stdout.strip().splitlines()
            if not ids:
                return ToolResult(
                    success=False,
                    output={},
                    error=f"No visible window found with name containing '{window_name}'.",
                )
            window_id = ids[0]

        if window_id is None:
            return ToolResult(
                success=False,
                output={},
                error="Provide window_id or window_name.",
            )

        res = _run(["xdotool", "windowfocus", "--sync", window_id])
        if res.returncode != 0:
            return ToolResult(
                success=False,
                output={},
                error=f"windowfocus failed: {res.stderr.strip()}",
            )

        _run(["xdotool", "windowraise", window_id])

        return ToolResult(
            success=True,
            output={"action": "focus_window", "window_id": window_id},
        )

    def _resize_window(
        self,
        window_id: str,
        width: int,
        height: int,
    ) -> ToolResult:
        if not self._xdotool_available():
            return ToolResult(success=False, output={}, error="xdotool not installed.")

        res = _run(["xdotool", "windowsize", window_id, str(width), str(height)])
        if res.returncode != 0:
            return ToolResult(
                success=False,
                output={},
                error=f"windowsize failed: {res.stderr.strip()}",
            )
        return ToolResult(
            success=True,
            output={"action": "resize_window", "window_id": window_id,
                    "width": width, "height": height},
        )

    def _minimize_window(self, window_id: str) -> ToolResult:
        if not self._xdotool_available():
            return ToolResult(success=False, output={}, error="xdotool not installed.")

        res = _run(["xdotool", "windowminimize", window_id])
        if res.returncode != 0:
            return ToolResult(
                success=False,
                output={},
                error=f"windowminimize failed: {res.stderr.strip()}",
            )
        return ToolResult(
            success=True,
            output={"action": "minimize_window", "window_id": window_id},
        )

    def _maximize_window(self, window_id: str) -> ToolResult:
        if not self._xdotool_available():
            return ToolResult(success=False, output={}, error="xdotool not installed.")
        
        res = _run(["xdotool", "windowactivate", "--sync", window_id])
        if res.returncode != 0:
            return ToolResult(
                success=False,
                output={},
                error=f"windowactivate failed: {res.stderr.strip()}",
            )
        _run(["xdotool", "key", "--window", window_id, "super+Up"])
        return ToolResult(
            success=True,
            output={"action": "maximize_window", "window_id": window_id},
        )

    def _verify_with_vision(
        self,
        question: str,
        region: tuple[int, int, int, int] | None = None,
        model: str | None = None,
    ) -> ToolResult:
        
        ss_result = self._screenshot(region=region, save=True)
        if not ss_result.success:
            return ss_result

        b64_png = ss_result.output["base64_png"]
        screenshot_path = ss_result.output["path"]
        answer = self._call_vision_api(b64_png, question, model=model)

        return ToolResult(
            success=True,
            output={
                "question":   question,
                "answer":     answer,
                "screenshot": screenshot_path,
            },
        )

    @staticmethod
    def _call_vision_api(
        b64_png: str,
        question: str,
        model: str | None = None,
    ) -> str:
        try:
            import anthropic  
        except ImportError:
            return (
                "[Vision API unavailable — install 'anthropic' SDK: "
                "pip install anthropic]"
            )

        _model = model or "claude-opus-4-6"
        client = anthropic.Anthropic()

        message = client.messages.create(
            model=_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/png",
                                "data":       b64_png,
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
        )
        return message.content[0].text

def create_gui_tool() -> GUITool:
    return GUITool()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    tool = GUITool()

    tests = [
        ("get_screen_size",     {}),
        ("get_mouse_position",  {}),
        ("list_windows",        {"name_filter": ""}),
        ("screenshot",          {"save": True}),
    ]

    all_passed = True
    for action, params in tests:
        result = tool.execute(action, **params)
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{status}  {action}")
        if not result.success:
            print(f"       error: {result.error}")
            all_passed = False
        else:
            
            safe = {k: v for k, v in result.output.items() if k != "base64_png"}
            print(f"       {safe}")

    sys.exit(0 if all_passed else 1)