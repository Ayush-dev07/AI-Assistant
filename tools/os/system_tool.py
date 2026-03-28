from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field

from core.logging import get_logger
from tools.base import BaseTool, PermissionManifest, ToolInput, ToolResult

log = get_logger(__name__)

HITL_ACTIONS: set[str] = {
    "kill_process",
    "systemctl_stop",
    "systemctl_disable",
}

_SYSTEMCTL_HITL = {"stop", "disable"}

_PROTECTED_PROCESSES: set[str] = {
    "systemd", "init", "kernel", "kthreadd",
    "python3", "python",  
}

_OWN_PID = os.getpid()

class SystemInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Action to perform. One of: get_system_info, get_processes, "
            "get_volume, set_volume, get_brightness, set_brightness, "
            "send_notification, get_network_info, "
            "kill_process (HITL), systemctl_action (HITL for stop/disable)."
        ),
    )
    pid: int | None = Field(
        default=None,
        description="Process ID for kill_process.",
    )
    name_filter: str = Field(
        default="",
        description="Filter processes by name substring (case-insensitive).",
    )
    sort_by: str = Field(
        default="cpu",
        description="Sort processes by: cpu | memory | name | pid",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max number of processes to return.",
    )

    level: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Volume or brightness level as percentage 0–100.",
    )

    title: str = Field(default="Assistant", description="Notification title.")
    body: str  = Field(default="",           description="Notification body text.")
    urgency: str = Field(
        default="normal",
        description="Notification urgency: low | normal | critical",
    )

    service: str = Field(
        default="",
        description="systemd service name for systemctl_action (e.g. 'nginx', 'docker').",
    )
    systemctl_cmd: str = Field(
        default="status",
        description="systemctl subcommand: status | start | stop | restart | enable | disable",
    )

class SystemTool(BaseTool):
    name         = "system_tool"
    description  = (
        "Query and control the local system. "
        "Safe actions: get_system_info (CPU/RAM/disk/battery/OS), get_processes, "
        "get_volume, set_volume, get_brightness, set_brightness, send_notification, "
        "get_network_info. "
        "HITL-guarded: kill_process (send SIGTERM/SIGKILL to a PID), "
        "systemctl_action with stop/disable (can break running services). "
        "systemctl_action with status/start/restart is always safe."
    )
    manifest     = PermissionManifest(
        can_spawn_processes = True,
        filesystem_read     = ["/sys/class/backlight", "/proc", "/sys"],
    )
    input_schema = SystemInput

    async def execute(self, inp: ToolInput) -> ToolResult:
        try:
            params = SystemInput(**inp.parameters)
        except Exception as exc:
            return ToolResult(success=False, error=f"Invalid parameters: {exc}")

        action = params.action.lower().strip()
        log.debug("system_tool_execute", action=action, session=inp.session_id)

        dispatch = {
            "get_system_info":    self._get_system_info,
            "get_processes":      self._get_processes,
            "kill_process":       self._kill_process,
            "get_volume":         self._get_volume,
            "set_volume":         self._set_volume,
            "get_brightness":     self._get_brightness,
            "set_brightness":     self._set_brightness,
            "send_notification":  self._send_notification,
            "get_network_info":   self._get_network_info,
            "systemctl_action":   self._systemctl_action,
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
            log.error("system_tool_error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"{action} failed: {exc}")

    async def _get_system_info(self, p: SystemInput) -> ToolResult:
        cpu_pct     = psutil.cpu_percent(interval=0.5)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_count   = psutil.cpu_count(logical=True)
        cpu_freq    = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "mountpoint":  part.mountpoint,
                    "device":      part.device,
                    "fstype":      part.fstype,
                    "total_gb":    round(usage.total  / 1e9, 2),
                    "used_gb":     round(usage.used   / 1e9, 2),
                    "free_gb":     round(usage.free   / 1e9, 2),
                    "percent":     usage.percent,
                })
            except (PermissionError, OSError):
                continue

        bat = psutil.sensors_battery()
        battery = None
        if bat is not None:
            battery = {
                "percent":   round(bat.percent, 1),
                "plugged_in": bat.power_plugged,
                "time_left_min": (
                    round(bat.secsleft / 60, 1)
                    if bat.secsleft not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN)
                    else None
                ),
            }
        boot_time = psutil.boot_time()
        uptime_s  = time.time() - boot_time
        uptime_h  = round(uptime_s / 3600, 2)
        uname = platform.uname()
        resolution = _get_screen_resolution()
        output = {
            "hostname":        uname.node,
            "os":              f"{uname.system} {uname.release}",
            "os_version":      uname.version[:80],
            "architecture":    uname.machine,
            "python_version":  platform.python_version(),
            "cpu": {
                "percent":        cpu_pct,
                "per_core":       cpu_per_core,
                "count_logical":  cpu_count,
                "count_physical": psutil.cpu_count(logical=False),
                "freq_mhz":       round(cpu_freq.current, 0) if cpu_freq else None,
                "model":          _get_cpu_model(),
            },
            "memory": {
                "total_gb":     round(mem.total     / 1e9, 2),
                "used_gb":      round(mem.used      / 1e9, 2),
                "available_gb": round(mem.available / 1e9, 2),
                "percent":      mem.percent,
            },
            "swap": {
                "total_gb": round(swap.total / 1e9, 2),
                "used_gb":  round(swap.used  / 1e9, 2),
                "percent":  swap.percent,
            },
            "disks":         disks,
            "battery":       battery,
            "uptime_hours":  uptime_h,
            "boot_time":     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(boot_time)),
            "screen":        resolution,
        }
        log.info("system_tool_info_retrieved", cpu=cpu_pct, ram_pct=mem.percent)
        return ToolResult(success=True, output=output)

    async def _get_processes(self, p: SystemInput) -> ToolResult:
        procs = []
        name_filter = p.name_filter.lower()
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info",
             "status", "username", "cmdline", "create_time"]
        ):
            try:
                info = proc.info
                name = (info.get("name") or "").lower()

                if name_filter and name_filter not in name:
                    continue
                mem_mb = 0.0
                if info.get("memory_info"):
                    mem_mb = round(info["memory_info"].rss / 1e6, 1)
                cmd = " ".join(info.get("cmdline") or [])[:120]
                procs.append({
                    "pid":       info["pid"],
                    "name":      info.get("name", ""),
                    "cpu_pct":   round(info.get("cpu_percent", 0.0) or 0.0, 1),
                    "mem_mb":    mem_mb,
                    "status":    info.get("status", ""),
                    "user":      info.get("username", ""),
                    "command":   cmd or info.get("name", ""),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        sort_key = {
            "cpu":    lambda x: -x["cpu_pct"],
            "memory": lambda x: -x["mem_mb"],
            "name":   lambda x: x["name"].lower(),
            "pid":    lambda x: x["pid"],
        }.get(p.sort_by, lambda x: -x["cpu_pct"])
        procs.sort(key=sort_key)
        procs = procs[: p.limit]
        return ToolResult(
            success = True,
            output  = {
                "processes":    procs,
                "total_shown":  len(procs),
                "filter_used":  p.name_filter or None,
                "sorted_by":    p.sort_by,
                "tip": "Use kill_process(pid=...) to terminate a process. HITL approval required.",
            },
        )

    async def _kill_process(self, p: SystemInput) -> ToolResult:
        if p.pid is None:
            return ToolResult(success=False, error="pid is required for kill_process.")
        pid = p.pid
        if pid == 1:
            return ToolResult(
                success = False,
                error   = "Refusing to kill PID 1 (init/systemd). This would crash the system.",
            )
        if pid == _OWN_PID:
            return ToolResult(
                success = False,
                error   = "Refusing to kill own process (the agent itself).",
            )
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return ToolResult(
                success = False,
                error   = f"Process {pid} does not exist.",
            )
        proc_name = proc.name()

        if proc_name.lower() in _PROTECTED_PROCESSES:
            return ToolResult(
                success = False,
                error   = (
                    f"Refusing to kill protected process '{proc_name}' (PID {pid}). "
                    "This process is essential for system stability."
                ),
            )
        try:
            mem_mb     = round(proc.memory_info().rss / 1e6, 1)
            cpu_pct    = proc.cpu_percent(interval=0.1)
            proc_user  = proc.username()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            mem_mb, cpu_pct, proc_user = 0.0, 0.0, "unknown"

        log.warning(
            "system_tool_kill_process",
            pid       = pid,
            name      = proc_name,
            user      = proc_user,
            mem_mb    = mem_mb,
        )   
        try:
            proc.terminate()
        except psutil.AccessDenied:
            return ToolResult(
                success = False,
                error   = f"Permission denied killing process {pid} ({proc_name}). Try with sudo.",
            )
        except psutil.NoSuchProcess:
            return ToolResult(
                success = True,
                output  = {"pid": pid, "name": proc_name, "method": "already_gone"},
            )

        import asyncio
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: proc.wait(timeout=3)),
                timeout=4.0,
            )
            method = "SIGTERM"
        except (psutil.TimeoutExpired, asyncio.TimeoutError):
            
            try:
                proc.kill()
                method = "SIGKILL"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                method = "SIGTERM_then_gone"

        return ToolResult(
            success = True,
            output  = {
                "pid":       pid,
                "name":      proc_name,
                "user":      proc_user,
                "mem_mb":    mem_mb,
                "cpu_pct":   cpu_pct,
                "method":    method,
                "message":   f"Process '{proc_name}' (PID {pid}) terminated via {method}.",
            },
        )

    async def _get_volume(self, p: SystemInput) -> ToolResult:
        volume = _read_volume_pactl()
        if volume is None:
            volume = _read_volume_amixer()

        if volume is None:
            return ToolResult(
                success = False,
                error   = (
                    "Could not read volume. "
                    "Ensure pactl (PulseAudio) or amixer (ALSA) is installed."
                ),
            )
        return ToolResult(
            success = True,
            output  = {
                "volume_percent": volume,
                "message":        f"Current volume: {volume}%",
            },
        )

    async def _set_volume(self, p: SystemInput) -> ToolResult:
        if p.level is None:
            return ToolResult(success=False, error="level is required for set_volume.")
        level = max(0, min(100, p.level))
        previous = _read_volume_pactl() or _read_volume_amixer()

        result = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("system_tool_volume_set", level=level, method="pactl")
            return ToolResult(
                success = True,
                output  = {
                    "previous_percent": previous,
                    "current_percent":  level,
                    "method":           "pactl",
                    "message":          f"Volume set to {level}%.",
                },
            )

        result = subprocess.run(
            ["amixer", "sset", "Master", f"{level}%"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("system_tool_volume_set", level=level, method="amixer")
            return ToolResult(
                success = True,
                output  = {
                    "previous_percent": previous,
                    "current_percent":  level,
                    "method":           "amixer",
                    "message":          f"Volume set to {level}% (via ALSA).",
                },
            )

        return ToolResult(
            success = False,
            error   = (
                "Could not set volume — neither pactl nor amixer succeeded. "
                "Install pulseaudio-utils or alsa-utils: "
                "sudo apt install pulseaudio-utils"
            ),
        )

    async def _get_brightness(self, p: SystemInput) -> ToolResult:
        level = _read_brightness_xrandr()
        if level is None:
            level = _read_brightness_sysfs()

        if level is None:
            return ToolResult(
                success = False,
                error   = (
                    "Could not read brightness. "
                    "Ensure xrandr is installed: sudo apt install x11-xserver-utils"
                ),
            )

        return ToolResult(
            success = True,
            output  = {
                "brightness_percent": level,
                "message":            f"Current brightness: {level}%",
            },
        )

    async def _set_brightness(self, p: SystemInput) -> ToolResult:
        if p.level is None:
            return ToolResult(success=False, error="level is required for set_brightness.")
        level = max(0, min(100, p.level))
        xrandr_val = round(level / 100.0, 2)  

        display = _get_primary_display()

        if display:
            result = subprocess.run(
                ["xrandr", "--output", display, "--brightness", str(xrandr_val)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                log.info("system_tool_brightness_set", level=level, display=display)
                return ToolResult(
                    success = True,
                    output  = {
                        "current_percent": level,
                        "display":         display,
                        "method":          "xrandr",
                        "message":         f"Brightness set to {level}% on {display}.",
                    },
                )
        backlight_set = _set_brightness_sysfs(level)
        if backlight_set:
            return ToolResult(
                success = True,
                output  = {
                    "current_percent": level,
                    "method":          "sysfs",
                    "message":         f"Brightness set to {level}% via /sys/class/backlight.",
                },
            )

        return ToolResult(
            success = False,
            error   = (
                "Could not set brightness. "
                "Ensure xrandr is installed: sudo apt install x11-xserver-utils"
            ),
        )

    async def _send_notification(self, p: SystemInput) -> ToolResult:
        if not shutil.which("notify-send"):
            log.info(
                "system_tool_notification_fallback",
                title  = p.title,
                body   = p.body,
                reason = "notify-send not installed",
            )
            return ToolResult(
                success  = True,
                output   = {
                    "delivered": False,
                    "method":    "log_only",
                    "message":   (
                        f"Notification logged (notify-send not installed): "
                        f"{p.title} — {p.body}. "
                        "Install: sudo apt install libnotify-bin"
                    ),
                },
            )
        urgency = p.urgency if p.urgency in ("low", "normal", "critical") else "normal"
        result = subprocess.run(
            [
                "notify-send",
                "--urgency", urgency,
                "--app-name", "SuperAgent",
                p.title,
                p.body,
            ],
            capture_output=True, text=True,
        )

        if result.returncode == 0:
            log.info("system_tool_notification_sent", title=p.title, urgency=urgency)
            return ToolResult(
                success = True,
                output  = {
                    "delivered": True,
                    "title":     p.title,
                    "body":      p.body,
                    "urgency":   urgency,
                },
            )

        return ToolResult(
            success = False,
            error   = f"notify-send failed: {result.stderr.strip()}",
        )

    async def _get_network_info(self, p: SystemInput) -> ToolResult:
        interfaces = []
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        for name, addr_list in addrs.items():
            iface_stats = stats.get(name)
            ips: list[str] = []
            mac = ""
            for addr in addr_list:
                if addr.family.name == "AF_INET":
                    ips.append(addr.address)
                elif addr.family.name == "AF_PACKET":
                    mac = addr.address

            interfaces.append({
                "name":     name,
                "ips":      ips,
                "mac":      mac,
                "is_up":    iface_stats.isup if iface_stats else False,
                "speed_mb": iface_stats.speed if iface_stats else 0,
            })
 
        io = psutil.net_io_counters()
        return ToolResult(
            success = True,
            output  = {
                "interfaces":      interfaces,
                "sent_mb":         round(io.bytes_sent  / 1e6, 1),
                "received_mb":     round(io.bytes_recv  / 1e6, 1),
                "packets_sent":    io.packets_sent,
                "packets_recv":    io.packets_recv,
            },
        )

    async def _systemctl_action(self, p: SystemInput) -> ToolResult:
        if not p.service:
            return ToolResult(
                success = False,
                error   = "service name is required for systemctl_action.",
            )
        cmd = p.systemctl_cmd.lower().strip()
        valid_cmds = {"status", "start", "stop", "restart", "enable", "disable", "is-active"}
        if cmd not in valid_cmds:
            return ToolResult(
                success = False,
                error   = (
                    f"Invalid systemctl command: {cmd!r}. "
                    f"Valid: {', '.join(sorted(valid_cmds))}"
                ),
            )
        service = p.service.strip()
        result = subprocess.run(
            ["systemctl", cmd, service],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout  = result.stdout.strip()
        stderr  = result.stderr.strip()
        success = result.returncode == 0
        if cmd == "status":
            active   = "active (running)" in stdout
            enabled  = "enabled" in stdout
            return ToolResult(
                success = True,
                output  = {
                    "service":    service,
                    "command":    cmd,
                    "active":     active,
                    "enabled":    enabled,
                    "raw_status": stdout[:800],
                },
            )

        return ToolResult(
            success = success,
            output  = {
                "service":  service,
                "command":  cmd,
                "stdout":   stdout[:500],
                "message":  f"systemctl {cmd} {service} {'succeeded' if success else 'failed'}.",
            },
            error = stderr[:300] if not success else None,
        )

def _read_volume_pactl() -> int | None:
    if not shutil.which("pactl"):
        return None
    try:
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        
        for part in result.stdout.split("/"):
            stripped = part.strip()
            if stripped.endswith("%"):
                return int(stripped[:-1])
    except Exception:
        return None
    return None

def _read_volume_amixer() -> int | None:
    if not shutil.which("amixer"):
        return None
    try:
        result = subprocess.run(
            ["amixer", "sget", "Master"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        
        import re
        match = re.search(r"\[(\d+)%\]", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        return None
    return None

def _get_primary_display() -> str | None:
    if not shutil.which("xrandr"):
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if " connected" in line and ("primary" in line or line.split()[1] == "connected"):
                return line.split()[0]
    except Exception:
        return None
    return None

def _read_brightness_xrandr() -> int | None:
    if not shutil.which("xrandr"):
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--verbose"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        import re
        match = re.search(r"Brightness:\s*([\d.]+)", result.stdout)
        if match:
            return round(float(match.group(1)) * 100)
    except Exception:
        return None
    return None

def _read_brightness_sysfs() -> int | None:
    backlight_dir = Path("/sys/class/backlight")
    if not backlight_dir.exists():
        return None
    try:
        devices = list(backlight_dir.iterdir())
        if not devices:
            return None
        device = devices[0]
        actual  = int((device / "brightness").read_text().strip())
        maximum = int((device / "max_brightness").read_text().strip())
        if maximum > 0:
            return round((actual / maximum) * 100)
    except Exception:
        return None
    return None

def _set_brightness_sysfs(level: int) -> bool:
    backlight_dir = Path("/sys/class/backlight")
    if not backlight_dir.exists():
        return False
    try:
        devices = list(backlight_dir.iterdir())
        if not devices:
            return False
        device  = devices[0]
        maximum = int((device / "max_brightness").read_text().strip())
        target  = round((level / 100.0) * maximum)
        (device / "brightness").write_text(str(target))
        return True
    except Exception:
        return False

def _get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"

def _get_screen_resolution() -> dict | None:
    if not shutil.which("xrandr"):
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        import re
        for line in result.stdout.splitlines():
            if "*" in line:  
                match = re.search(r"(\d+)x(\d+)", line)
                if match:
                    return {
                        "width":  int(match.group(1)),
                        "height": int(match.group(2)),
                    }
    except Exception:
        pass
    return None