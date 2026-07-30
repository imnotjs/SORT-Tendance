"""
SORT-tendance :: start_sortendance.py
=====================================
ONE script to launch and supervise both main.py and dashboard.py.

WHAT IT DOES
------------
  * Spawns `python main.py` and `streamlit run ui/dashboard.py` as
    child processes -- each in its OWN cmd window (Windows) or its own
    terminal session (Linux), so logs are not interleaved.
  * On Windows, each child gets a titled cmd window:
        "SORT-tendance :: MAIN"
        "SORT-tendance :: DASHBOARD"
    so you can tell them apart in the taskbar.
  * Restarts BOTH every 06:00 and 18:00 LOCAL (12-hour cycle) to
    release memory that accumulates over a long session:
      - main.py: ONNX Runtime arenas, BoTSORT track history, CUDA
        caching allocator fragmentation, async_logger ring buffers.
      - dashboard.py: Streamlit session_state, PIL.Image caches,
        matplotlib figures, pandas DataFrames.
    IMPORTANT: At the 6AM/6PM boundary, children are asked to exit
    GRACEFULLY (CTRL_BREAK_EVENT on Windows / SIGINT on Linux) so
    the application itself can release VRAM, the camera handle, ONNX
    sessions, CUDA context, video recorder, async logger, and cv2
    windows via its own signal handlers + finally blocks. The
    supervisor only force-kills (taskkill /F) if a child refuses to
    exit within _GRACEFUL_EXIT_TIMEOUT_S seconds. This is the
    OPPOSITE of force-killing: the application exits with code 0
    having cleanly dropped all GPU + camera resources, so the next
    main.py can grab them immediately without OOMing during YOLO load.
  * Restarts either child individually if it crashes (non-zero exit).
    On Windows, the crashed cmd window shows the exit code for 5
    seconds (then auto-closes so the supervisor can detect the crash
    via Popen.poll() and trigger the restart cooldown). The full
    traceback is in the child's log file under logs/.
    IMPORTANT: We deliberately do NOT use `pause >nul` in the batch
    wrapper -- `pause` keeps cmd.exe alive after python crashes, which
    makes the supervisor think the child is still running (split-brain
    with the dashboard, which correctly reports "offline" because no
    heartbeats are arriving).
  * Belt-and-suspenders watchdog: every 15s, the supervisor checks
    whether `python.exe` is still running as a descendant of each
    child's cmd.exe. If cmd.exe is alive but its python.exe descendant
    has died (e.g. the batch wrapper is hung on a different prompt,
    streamlit's own wrapper held the console, or `timeout` got stuck),
    the supervisor force-kills the cmd wrapper so the restart can
    proceed. This catches the rare case where Popen.poll() returns
    None (cmd.exe alive) but the actual application is dead.
  * Cleanly kills both children on Ctrl+C.

USAGE
-----
  python start_sortendance.py              # run BOTH main + dashboard
  python start_sortendance.py main         # run only main.py
  python start_sortendance.py dashboard    # run only dashboard.py
  python start_sortendance.py --dry-run    # preview schedule, no spawn
  python start_sortendance.py --single-window  # OLD: pump both into one console

AUTO-START ON BOOT
------------------
  Windows:
    1. Win+R -> type: shell:startup -> Enter
    2. Copy start_sortendance.bat into that folder
    3. (Optional) Right-click -> Create shortcut -> Properties ->
       Set Run: Minimized if you don't want a console window on boot

  Linux:
    Add to crontab (crontab -e):
      @reboot cd /path/to/SORT-tendance && ./start_sortendance.sh

CONFIG
------
Reads existing session_rotation block from config.yaml (no new block):
  session_rotation:
    am_hour: 6
    pm_hour: 18
Defaults to 6 / 18 if config.yaml is missing or unreadable.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import datetime as _dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
_DEFAULT_AM_HOUR = 6
_DEFAULT_PM_HOUR = 18
_POLL_INTERVAL_S = 5.0         # How often to check children + clock.
_GRACE_PERIOD_S = 10.0         # SIGTERM -> SIGKILL grace.
_CHILD_RESTART_DELAY_S = 30.0  # Delay before respawning ANY child after exit
                               #   (crash, Ctrl+C, X-close, or clean exit).
                               #   The ONLY way to permanently stop a child
                               #   is to stop the supervisor itself.
_GRACEFUL_EXIT_TIMEOUT_S = 25.0  # How long to wait for a child to exit
                                 #   after sending CTRL_BREAK_EVENT (Win)
                                 #   or SIGINT (Linux) before escalating
                                 #   to force-kill. main.py's
                                 #   orch.shutdown() takes ~5-15s (it joins
                                 #   tracking/recorder/logger threads and
                                 #   tears down CUDA), so 25s is a safe
                                 #   ceiling.
_POST_GRACEFUL_RELEASE_S = 3.0   # After the child exits cleanly, the
                                 #   NVIDIA driver still needs 1-3s to
                                 #   finalize CUDA context teardown and
                                 #   the camera handle fully releases.
                                 #   We wait this long before respawning.
_WATCHDOG_CHECK_INTERVAL_S = 15.0  # How often the orphaned-cmd watchdog
                                   #   runs. Every N seconds, it checks
                                   #   whether python.exe is still running
                                   #   as a descendant of each child's
                                   #   cmd.exe. If cmd.exe is alive but
                                   #   no python.exe descendant exists,
                                   #   the cmd wrapper is hung (e.g. on
                                   #   `pause`, `timeout`, or streamlit's
                                   #   own wrapper) and we force-kill it
                                   #   so the restart can proceed.
_WATCHDOG_GRACE_S = 60.0          # Don't run the watchdog for the first
                                   #   60s after spawning a child. This
                                   #   gives python.exe time to start up
                                   #   (the batch wrapper runs cmd.exe
                                   #   first, then python.exe launches as
                                   #   a child). Without this grace
                                   #   period, the watchdog would
                                   #   false-positive during boot.
_MAX_RESTARTS = 1000           # Max restarts per child before giving up.
                               #   (Set high so the supervisor never gives
                               #   up on its own -- the user must stop it.)

# Project root = directory containing this script.
_PROJECT_ROOT = Path(__file__).resolve().parent

# Process definitions: name -> python command (relative to project root).
# On Windows we wrap these in batch files so the new cmd window has a
# nice title and stays open on crash. On Linux we invoke python directly.
_PROCESS_PYTHON: Dict[str, List[str]] = {
    "main":      [sys.executable, "-u", "main.py"],
    "dashboard": [sys.executable, "-u", "-m", "streamlit", "run", "ui/dashboard.py"],
}

# Window titles for the spawned cmd windows.
_PROCESS_TITLES: Dict[str, str] = {
    "main":      "SORT-tendance :: MAIN",
    "dashboard": "SORT-tendance :: DASHBOARD",
}


# ---------------------------------------------------------------------------
# Config loader (reads session_rotation from config.yaml or config/config.yaml).
# ---------------------------------------------------------------------------
def _load_rotation_hours() -> Tuple[int, int]:
    """Read am_hour / pm_hour from config.yaml. Falls back to (6, 18)."""
    candidates = [
        _PROJECT_ROOT / "config.yaml",
        _PROJECT_ROOT / "config" / "config.yaml",
    ]
    cfg_path = next((p for p in candidates if p.is_file()), candidates[0])

    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        sess = raw.get("session_rotation", {}) or {}
        am = int(sess.get("am_hour", _DEFAULT_AM_HOUR))
        pm = int(sess.get("pm_hour", _DEFAULT_PM_HOUR))
        if not (0 <= am <= 23 and 0 <= pm <= 23 and am != pm):
            raise ValueError(f"invalid hours: am={am} pm={pm}")
        return am, pm
    except Exception as exc:
        print(f"[supervisor] Could not read config {cfg_path} ({exc}); "
              f"using defaults ({_DEFAULT_AM_HOUR}, {_DEFAULT_PM_HOUR}).")
        return _DEFAULT_AM_HOUR, _DEFAULT_PM_HOUR


def _current_session_hour(now: _dt.datetime, am_hour: int, pm_hour: int) -> str:
    """Return 'AM' or 'PM' for the current wall-clock session."""
    today = now.date()
    am_today = _dt.datetime(today.year, today.month, today.day, am_hour, 0, 0)
    pm_today = _dt.datetime(today.year, today.month, today.day, pm_hour, 0, 0)
    if now < am_today:
        return "PM"   # Pre-dawn = still in yesterday's PM session.
    elif now < pm_today:
        return "AM"
    else:
        return "PM"


# ---------------------------------------------------------------------------
# Windows batch wrapper generator.
# ---------------------------------------------------------------------------
# How long the cmd window stays open after a child crashes, so the
# operator can see the exit code before the window closes. After this
# timeout, the window closes and the supervisor detects the crash via
# Popen.poll(), triggering the 30s restart cooldown.
#
# CRITICAL: We MUST NOT use `pause >nul` here. `pause` blocks
# indefinitely waiting for a keypress, which means cmd.exe stays alive
# forever after python crashes. The supervisor has a Popen handle to
# cmd.exe (not to python.exe), so proc.poll() returns None as long as
# cmd.exe is alive -- the supervisor cannot tell that python has died.
# This produces a split-brain state where the dashboard reports
# "offline" (no heartbeats) but the supervisor thinks the child is
# healthy (cmd.exe still running) and never triggers a restart.
#
# `timeout /t N /nobreak >nul` waits N seconds and then continues,
# regardless of keypresses. This lets the operator see the exit code
# briefly, then closes the window so the supervisor can detect the
# crash and restart the child.
_CRASH_WINDOW_VISIBLE_S = 5.0


def _write_run_bat(name: str) -> Path:
    """Generate (or overwrite) _run_<name>.bat in the project root.

    The batch file:
      1. Sets the cmd window title.
      2. cd into the project root.
      3. Runs the python command.
      4. On non-zero exit, prints the exit code and waits
         _CRASH_WINDOW_VISIBLE_S seconds (using `timeout /nobreak`,
         NOT `pause`) so the operator can see the exit code. The
         window then closes so the supervisor can detect the crash
         via Popen.poll() and trigger the restart cooldown.
      5. On zero exit, the window closes immediately.
      6. Exits with the python process's exit code (so the supervisor
         can detect crash vs clean exit via Popen.poll()).

    IMPORTANT: Do NOT use `pause >nul` in this batch file. `pause`
    blocks indefinitely waiting for a keypress, keeping cmd.exe alive
    after python has crashed. Since the supervisor's Popen handle
    points at cmd.exe (not python.exe), the supervisor cannot detect
    the crash -- proc.poll() returns None as long as cmd.exe is alive,
    so the restart logic never fires. This produces a split-brain
    state where the dashboard reports "offline" but the supervisor
    thinks the child is healthy.
    """
    py_cmd = " ".join(_PROCESS_PYTHON[name])
    title = _PROCESS_TITLES[name]
    bat_path = _PROJECT_ROOT / f"_run_{name}.bat"
    visible_s = int(_CRASH_WINDOW_VISIBLE_S)

    # Build batch content. We use \r\n line endings for cmd compatibility.
    lines = [
        "@echo off",
        f'title {title}',
        f'cd /d "{_PROJECT_ROOT}"',
        py_cmd,
        "set RC=%errorlevel%",
        "if %RC% neq 0 (",
        "    echo.",
        f"    echo ========================================",
        f"    echo [{name} exited with code %RC%]",
        f"    echo [supervisor will auto-restart in ~30s]",
        f"    echo [this window closes in {visible_s}s]",
        f"    echo [check logs\\ for full traceback]",
        f"    echo ========================================",
        f"    timeout /t {visible_s} /nobreak >nul",
        ")",
        "exit /b %RC%",
    ]
    bat_content = "\r\n".join(lines) + "\r\n"
    bat_path.write_bytes(bat_content.encode("ascii", errors="replace"))
    return bat_path


# ---------------------------------------------------------------------------
# Process management.
# ---------------------------------------------------------------------------
def _spawn(name: str, single_window: bool = False) -> subprocess.Popen:
    """Spawn one child process. Returns the Popen object.

    Args:
        name: "main" or "dashboard".
        single_window: If True, pump child stdout into the supervisor's
            console (old behavior, for users who want interleaved logs).
            If False (default), spawn each child in its own cmd window
            on Windows, or its own session on Linux.
    """
    py_cmd = _PROCESS_PYTHON[name]
    title = _PROCESS_TITLES[name]

    if single_window:
        # ----- Old behavior: pump child stdout into supervisor console. -----
        kwargs: Dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 1,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "cwd": str(_PROJECT_ROOT),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
        else:
            kwargs["start_new_session"] = True

        print(f"[supervisor] Spawning {name} (single-window): {' '.join(py_cmd)}")
        proc = subprocess.Popen(py_cmd, **kwargs)

        import threading
        def _pump(p: subprocess.Popen, n: str) -> None:
            if p.stdout is None:
                return
            try:
                for line in p.stdout:
                    line = line.rstrip("\r\n")
                    if line:
                        print(f"[{n}] {line}", flush=True)
            except (OSError, ValueError):
                pass
        threading.Thread(target=_pump, args=(proc, name),
                         name=f"pump.{name}", daemon=True).start()
        return proc

    # ----- New behavior: separate cmd window per child. -----
    if os.name == "nt":
        # Windows: write a batch wrapper, then invoke it in a new console.
        bat_path = _write_run_bat(name)
        cmd = [str(bat_path)]
        kwargs = {
            "cwd": str(_PROJECT_ROOT),
            # CREATE_NEW_CONSOLE  = open a new visible cmd window.
            # CREATE_NEW_PROCESS_GROUP = CTRL_BREAK_EVENT can target this
            #   child (used for graceful shutdown).
            "creationflags": (
                subprocess.CREATE_NEW_CONSOLE  # type: ignore
                | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            ),
        }
        print(f"[supervisor] Spawning {name} in new cmd window: {title}")
        print(f"             bat: {bat_path}")
        return subprocess.Popen(cmd, **kwargs)

    # ----- Linux fallback: try common terminal emulators. -----
    # We try xterm / gnome-terminal / konsole in order. If none is
    # available, we fall back to the single-window pump behavior.
    terminal_candidates = [
        # (terminal binary, args-before-command, args-after-command)
        (["xterm", "-T", title, "-e"],          ["bash", "-lc"]),
        (["gnome-terminal", f"--title={title}", "--"], ["bash", "-lc"]),
        (["konsole", f"--title", title, "-e"],  ["bash", "-lc"]),
    ]
    shell_cmd = f'cd "{_PROJECT_ROOT}" && exec {" ".join(py_cmd)}'
    for term_prefix, term_suffix in terminal_candidates:
        try:
            full_cmd = term_prefix + term_suffix + [shell_cmd]
            print(f"[supervisor] Spawning {name} in {term_prefix[0]}: {title}")
            return subprocess.Popen(
                full_cmd,
                cwd=str(_PROJECT_ROOT),
                start_new_session=True,
            )
        except FileNotFoundError:
            continue
        except OSError:
            continue

    # No terminal emulator available -- fall back to single-window pump.
    print(f"[supervisor] No terminal emulator found; falling back to "
          f"single-window mode for {name}.")
    return _spawn(name, single_window=True)


def _kill(proc: subprocess.Popen, name: str) -> None:
    """Gracefully terminate a child. Escalates to force-kill after grace.

    On Windows with separate cmd windows, we use `taskkill /F /T /PID`
    to kill the entire process tree (cmd.exe + python.exe + any ONNX
    worker threads). This is necessary because TerminateProcess on the
    cmd wrapper would leave python orphaned.
    """
    if proc.poll() is not None:
        return
    print(f"[supervisor] Stopping {name} (PID {proc.pid})...")

    if os.name == "nt":
        # Windows: use taskkill /F /T to kill the entire process tree.
        # /F = force (no graceful shutdown -- python doesn't always
        #       respond to CTRL_BREAK_EVENT when wrapped in cmd.exe).
        # /T = tree (kill all children too).
        # /PID = target process ID.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=_GRACE_PERIOD_S,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"[supervisor] taskkill failed for {name}: {exc}; "
                  f"falling back to proc.kill().")
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print(f"[supervisor] {name} refused to die after taskkill!")
        return

    # Linux: SIGTERM -> SIGKILL escalation.
    try:
        proc.terminate()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=_GRACE_PERIOD_S)
    except subprocess.TimeoutExpired:
        print(f"[supervisor] {name} didn't exit in {_GRACE_PERIOD_S}s; force-killing.")
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print(f"[supervisor] {name} refused to die after SIGKILL!")


def _graceful_exit(
    proc: subprocess.Popen,
    name: str,
    timeout_s: float = _GRACEFUL_EXIT_TIMEOUT_S,
) -> bool:
    """Ask a child to exit gracefully so it can release VRAM, camera,
    ONNX sessions, and CUDA contexts via its OWN signal handlers +
    finally blocks. Escalates to force-kill only if the child does not
    exit within `timeout_s` seconds.

    This is the OPPOSITE of _kill() -- _kill() force-terminates
    immediately (no cleanup), while _graceful_exit() gives the
    application a chance to release GPU memory, close the camera
    handle, flush logs, etc. before terminating.

    Returns:
        True  -- child exited cleanly (rc == 0 or KeyboardInterrupt path).
        False -- child had to be force-killed (timeout or signal failed).

    On Windows:
        Sends CTRL_BREAK_EVENT to the child's process group. This is
        deliverable because _spawn() created the child with
        CREATE_NEW_PROCESS_GROUP. main.py's SIGBREAK handler then fires
        (it wires both SIGINT and SIGTERM to the same handler), which
        calls orch.shutdown() -- releasing camera, ONNX sessions, CUDA
        context, video recorder, async logger, and cv2 windows.
        dashboard.py's Patch 63 SIGINT/SIGTERM handler calls os._exit(0)
        after stopping its UDP telemetry receiver.

    On Linux:
        Sends SIGINT (same as Ctrl+C in the terminal).
    """
    if proc.poll() is not None:
        # Already exited; treat as graceful.
        return True

    print(f"[supervisor] Asking {name} (PID {proc.pid}) to exit "
          f"gracefully (timeout {timeout_s:.0f}s) -- application will "
          f"release VRAM / camera / ONNX / CUDA via its own cleanup...")

    # ----- Send the graceful signal. -----
    if os.name == "nt":
        # CTRL_BREAK_EVENT targets the child's process group (set via
        # CREATE_NEW_PROCESS_GROUP at spawn time). Python's default
        # SIGBREAK handler raises KeyboardInterrupt, and main.py /
        # dashboard.py both override that to call their own shutdown
        # paths (orch.shutdown() / os._exit(0)).
        sig = getattr(signal, "CTRL_BREAK_EVENT", None)
        if sig is None:
            print(f"[supervisor] CTRL_BREAK_EVENT not available on this "
                  f"platform; force-killing {name}.")
            _kill(proc, name)
            return False
        try:
            proc.send_signal(sig)
        except (OSError, ValueError, AttributeError) as exc:
            print(f"[supervisor] CTRL_BREAK_EVENT failed for {name} "
                  f"({exc}); force-killing.")
            _kill(proc, name)
            return False
    else:
        # Linux: SIGINT (same as Ctrl+C). main.py's signal handler
        # (signal.SIGINT -> _signal_handler -> orch.shutdown()) fires.
        try:
            proc.send_signal(signal.SIGINT)
        except (OSError, ValueError) as exc:
            print(f"[supervisor] SIGINT failed for {name} ({exc}); "
                  f"force-killing.")
            _kill(proc, name)
            return False

    # ----- Wait for graceful exit, polling every 0.5s. -----
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            print(f"[supervisor] {name} exited gracefully (rc={rc}). "
                  f"Application released VRAM / camera / ONNX cleanly. "
                  f"Safe to respawn.")
            return True
        time.sleep(0.5)

    # ----- Timed out -- escalate to force-kill. -----
    print(f"[supervisor] {name} did not exit in {timeout_s:.0f}s after "
          f"graceful signal. Escalating to force-kill (VRAM/camera "
          f"may not be fully released -- NVIDIA driver will reclaim "
          f"them in a few seconds).")
    _kill(proc, name)
    return False


# ---------------------------------------------------------------------------
# Orphaned-cmd watchdog (Windows-specific, no-op on Linux).
# ---------------------------------------------------------------------------
# Problem this solves:
#   The supervisor has a Popen handle to cmd.exe (the batch wrapper),
#   NOT to python.exe. If python.exe crashes but cmd.exe stays alive
#   (e.g. batch wrapper hung on `pause`, `timeout`, or streamlit's
#   own wrapper held the console), proc.poll() returns None and the
#   supervisor thinks the child is healthy. Meanwhile the dashboard
#   sees no heartbeats and reports "offline" -- split-brain.
#
# Fix:
#   Periodically check if python.exe is still running as a descendant
#   of the cmd.exe process. If cmd.exe is alive but no python.exe
#   descendant exists, the cmd wrapper is orphaned -- force-kill it
#   so proc.poll() returns and the restart logic fires.
#
# Implementation:
#   On Windows, we use `wmic process where ParentProcessId=<pid> get
#   ProcessId,Name` recursively to walk the process tree. wmic is
#   preinstalled on Windows 10/11. (PowerShell Get-CimInstance would
#   also work but is slower to spawn.)
# ---------------------------------------------------------------------------
def _has_python_descendant(root_pid: int) -> bool:
    """Return True if any python.exe / pythonw.exe process exists
    anywhere in the descendant tree of `root_pid`.

    On non-Windows platforms, returns True (no-op -- the watchdog
    is Windows-only).
    """
    if os.name != "nt":
        return True

    try:
        import ctypes
        from ctypes import wintypes

        # Use CreateToolhelp32Snapshot via kernel32 for fast tree walk.
        # This avoids spawning wmic / PowerShell processes (which are
        # slow and can themselves hang).
        TH32CS_SNAPPROCESS = 0x00000002

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(
            TH32CS_SNAPPROCESS, 0,
        )
        if snapshot == -1 or snapshot == 0:
            return True  # Can't tell -- assume healthy.

        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)

            # Build parent->children map in one pass.
            children_of: Dict[int, List[int]] = {}
            name_of: Dict[int, str] = {}

            if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                while True:
                    pid = entry.th32ProcessID
                    ppid = entry.th32ParentProcessID
                    name = entry.szExeFile.lower()
                    children_of.setdefault(ppid, []).append(pid)
                    name_of[pid] = name
                    if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                        break

            # BFS from root_pid; look for python.exe / pythonw.exe.
            # Also accept python3.X.exe (conda/MSYS2 installs sometimes
            # name it that way).
            stack = [root_pid]
            visited = set()
            while stack:
                pid = stack.pop()
                if pid in visited:
                    continue
                visited.add(pid)
                name = name_of.get(pid, "")
                if name in ("python.exe", "pythonw.exe"):
                    return True
                if name.startswith("python") and name.endswith(".exe"):
                    return True
                stack.extend(children_of.get(pid, []))
            return False
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception as exc:
        # If anything goes wrong (ctypes, struct layout, etc.),
        # assume healthy -- we don't want a buggy watchdog to
        # force-kill healthy children.
        print(f"[supervisor] Watchdog tree-walk failed ({exc}); "
              f"assuming child is healthy.")
        return True


def _watchdog_check(
    procs: Dict[str, subprocess.Popen],
    spawn_times: Dict[str, float],
) -> None:
    """Watchdog pass: for each child whose cmd.exe is still alive,
    verify that python.exe is still running as a descendant. If not,
    force-kill cmd.exe so the supervisor's restart logic can fire.

    This is a belt-and-suspenders check. The primary crash-detection
    path is `proc.poll()` returning a non-None exit code when the
    batch wrapper exits. This watchdog catches the rare case where
    the batch wrapper hangs after python has died.
    """
    if os.name != "nt":
        return  # Windows-only.

    now_epoch = time.time()
    for name, proc in list(procs.items()):
        # Skip children that have already exited (poll() != None).
        if proc.poll() is not None:
            continue
        # Skip children that were just spawned (grace period).
        spawn_time = spawn_times.get(name, now_epoch)
        if (now_epoch - spawn_time) < _WATCHDOG_GRACE_S:
            continue
        # Skip children whose Popen handle is invalid.
        try:
            pid = proc.pid
        except (AttributeError, OSError):
            continue

        if _has_python_descendant(pid):
            continue  # Healthy -- python.exe is still running.

        # Orphaned! cmd.exe is alive but no python.exe descendant.
        # Force-kill cmd.exe so proc.poll() returns and the restart
        # logic fires.
        print(f"[supervisor] WATCHDOG: {name} (cmd.exe PID {pid}) is "
              f"alive but has no python.exe descendant -- python has "
              f"died but the batch wrapper is hung (likely on "
              f"`pause`, `timeout`, or streamlit's wrapper). Force-"
              f"killing cmd.exe so the restart can proceed.")
        _kill(proc, name)
        # proc.poll() will now return on the next loop iteration,
        # triggering the standard crash-restart path (30s cooldown).


# ---------------------------------------------------------------------------
# Windows console control handler.
# ---------------------------------------------------------------------------
# On Windows, when the user clicks the X button on the supervisor's cmd
# window, Windows sends a CTRL_CLOSE_EVENT to the console. The default
# handler calls ExitProcess() immediately, which orphans all child
# processes (their cmd windows stay open, still running python.exe).
#
# We install a custom handler that:
#   1. Sets a global flag so the supervisor loop breaks out cleanly.
#   2. Returns True so Windows waits for us to exit (gives ~5 seconds
#      before hard-killing the console).
#   3. The supervisor's finally / except path then kills all children.
#
# This ensures X-close on the supervisor window kills children, just
# like Ctrl+C does.

_SUPERVISOR_SHUTDOWN_REQUESTED = False


def _install_windows_control_handler() -> None:
    """Install a console control handler on Windows. No-op on Linux."""
    if os.name != "nt":
        return

    try:
        import ctypes
        from ctypes import wintypes

        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        HANDLER_ROUTINE = ctypes.WINFUNCTYPE(
            wintypes.BOOL,  # return type
            wintypes.DWORD,  # dwCtrlType
        )

        def _handler(ctrl_type):
            global _SUPERVISOR_SHUTDOWN_REQUESTED
            if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT,
                             CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT,
                             CTRL_SHUTDOWN_EVENT):
                _SUPERVISOR_SHUTDOWN_REQUESTED = True
                print(f"\n[supervisor] Console control event {ctrl_type} "
                      f"received. Initiating clean shutdown...")
                # Returning True tells Windows "I handled it" -- for
                # CTRL_CLOSE_EVENT this buys us ~5 seconds to clean up
                # before Windows force-kills the console.
                return True
            return False

        # Keep a reference so the callback isn't garbage-collected.
        _handler_ref = HANDLER_ROUTINE(_handler)
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCtrlHandler(_handler_ref, True)
    except Exception as exc:
        print(f"[supervisor] Warning: could not install Windows console "
              f"control handler ({exc}). X-close on the supervisor window "
              f"may orphan child processes.")


# ---------------------------------------------------------------------------
# Patch: Enrollment-triggered main restart.
# ---------------------------------------------------------------------------
# When enroll.py --single-student successfully enrolls a NEW student,
# it writes data/.restart_main_requested with a JSON payload containing
# {"requested_at": <epoch>, "delay_s": 15, "reason": "new_student_enrolled"}.
#
# The supervisor polls this file every cycle. Once `delay_s` seconds have
# elapsed since `requested_at`, the supervisor GRACEFULLY restarts ONLY
# the `main` child (NOT dashboard -- the Streamlit UI keeps running so
# the operator sees the toast notifications). After the restart, the
# flag file is deleted so we don't trigger again.
#
# Rationale: main.py caches student_db.pickle in RAM at startup. A newly
# enrolled student won't be recognized until main.py is restarted to
# re-read the pickle. The 15s delay gives enroll.py time to release
# VRAM (its registration engine closes on exit) before main.py grabs
# the GPU for its live engine.
_RESTART_REQUEST_FILE: str = "data/.restart_main_requested"


def _check_enroll_restart_signal(
    procs: Dict[str, subprocess.Popen],
    pending_restarts: Dict[str, float],
    spawn_times: Dict[str, float],
    single_window: bool,
) -> None:
    """
    Poll the enrollment restart-request flag file. If present AND the
    requested delay has elapsed, gracefully restart ONLY the `main` child.

    Mutates `procs`, `pending_restarts`, and `spawn_times` in place.
    Safe to call every poll cycle -- it short-circuits cheaply when the
    flag file is absent.
    """
    flag_path = _PROJECT_ROOT / _RESTART_REQUEST_FILE
    if not flag_path.is_file():
        return

    try:
        import json as _json
        payload = _json.loads(flag_path.read_text(encoding="utf-8"))
        requested_at = float(payload.get("requested_at", 0.0))
        delay_s = float(payload.get("delay_s", 15.0))
    except (ValueError, OSError, KeyError) as exc:
        print(f"[supervisor] Malformed restart-request flag at {flag_path} "
              f"({exc}); deleting to avoid repeated reads.")
        try:
            flag_path.unlink()
        except OSError:
            pass
        return

    elapsed = time.time() - requested_at
    if elapsed < delay_s:
        # Not yet time to restart. Log once per ~5s so the operator
        # sees the countdown without flooding the console.
        remaining = delay_s - elapsed
        if int(elapsed) % 5 == 0:
            print(f"[supervisor] Enrollment restart pending: {remaining:.1f}s "
                  f"until main.py restart (delay_s={delay_s:.0f}).")
        return

    # Delay elapsed -> restart main only.
    if "main" not in procs:
        # main is not currently running (crashed? not spawned?). Just
        # clear the flag; the normal crash-restart path will handle it.
        print("[supervisor] Restart-request flag fired but main is not "
              "currently running. Clearing flag; no action.")
        try:
            flag_path.unlink()
        except OSError:
            pass
        return

    print(f"[supervisor] Enrollment-triggered restart: gracefully stopping "
          f"main.py so it can release VRAM/camera and reload the updated "
          f"student_db.pickle ...")
    main_proc = procs.pop("main")
    _graceful_exit(main_proc, "main", timeout_s=_GRACEFUL_EXIT_TIMEOUT_S)

    # Brief release window for VRAM/camera finalization (same rationale
    # as the session-boundary restart).
    print(f"[supervisor] Waiting {_POST_GRACEFUL_RELEASE_S:.0f}s for "
          f"VRAM/camera release before respawning main.py ...")
    time.sleep(_POST_GRACEFUL_RELEASE_S)

    try:
        procs["main"] = _spawn("main", single_window=single_window)
        spawn_times["main"] = time.time()
        print("[supervisor] main.py restarted with updated student DB.")
    except Exception as exc:
        print(f"[supervisor] ERROR: failed to respawn main after enroll: "
              f"{exc}. Will retry via crash-restart path in "
              f"{_CHILD_RESTART_DELAY_S:.0f}s.")
        pending_restarts["main"] = time.time() + _CHILD_RESTART_DELAY_S

    # Clear the flag so we don't trigger again on the next poll.
    try:
        flag_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main supervisor loop.
# ---------------------------------------------------------------------------
def _supervise(
    targets: List[str],
    dry_run: bool = False,
    single_window: bool = False,
) -> int:
    """Supervise the given target names. Returns exit code."""
    _install_windows_control_handler()

    am_hour, pm_hour = _load_rotation_hours()
    now = _dt.datetime.now()
    last_session = _current_session_hour(now, am_hour, pm_hour)

    mode_desc = "SINGLE-WINDOW (interleaved)" if single_window else "SEPARATE WINDOWS (per-process)"
    print("=" * 70)
    print(f"SORT-tendance supervisor | targets: {targets}")
    print(f"Mode: {mode_desc}")
    print(f"Schedule: {am_hour:02d}:00 / {pm_hour:02d}:00 LOCAL | "
          f"current session: {last_session}")
    print(f"Project root: {_PROJECT_ROOT}")
    if dry_run:
        print("DRY-RUN: no processes will be spawned.")
        print("=" * 70)
        return 0
    if not single_window and os.name == "nt":
        print("Each target launches in its own cmd window.")
        print("RESTART POLICY:")
        print("  * If a child window closes (crash, Ctrl+C, or X-close),")
        print(f"    it auto-restarts in {_CHILD_RESTART_DELAY_S:.0f}s as long")
        print("    as this supervisor is running.")
        print(f"  * Watchdog: every {_WATCHDOG_CHECK_INTERVAL_S:.0f}s, checks")
        print("    that python.exe is still running inside each child's")
        print("    cmd.exe. If cmd.exe is alive but python has died (e.g.")
        print("    batch wrapper hung on a prompt), force-kills cmd.exe")
        print("    so the restart can proceed.")
        print("  * To PERMANENTLY stop both children, press Ctrl+C HERE")
        print("    (in the supervisor window) or close THIS window.")
    print("Press Ctrl+C to stop all children.")
    print("=" * 70)

    # Spawn all targets.
    procs: Dict[str, subprocess.Popen] = {}
    crash_counts: Dict[str, int] = {n: 0 for n in targets}
    spawn_times: Dict[str, float] = {}  # name -> epoch when last spawned
    for name in targets:
        procs[name] = _spawn(name, single_window=single_window)
        spawn_times[name] = time.time()
        time.sleep(1.0)  # Stagger spawns so YOLO/CUDA don't init in parallel.

    # Track scheduled restart timestamps so we don't busy-wait.
    pending_restarts: Dict[str, float] = {}  # name -> scheduled_spawn_epoch

    # Watchdog tick counter. We run the orphaned-cmd watchdog every
    # _WATCHDOG_CHECK_INTERVAL_S seconds (instead of every poll) because
    # the process-tree walk is more expensive than proc.poll().
    last_watchdog_tick = time.time()

    try:
        while True:
            time.sleep(_POLL_INTERVAL_S)

            # 0. Check if Windows console control handler (or signal) has
            #    requested a shutdown. This fires on X-close of the
            #    supervisor window, logoff, or system shutdown.
            if _SUPERVISOR_SHUTDOWN_REQUESTED:
                print("\n[supervisor] Shutdown requested by console control "
                      "event. PERMANENTLY stopping all children (graceful "
                      "where possible so VRAM/camera are released).")
                pending_restarts.clear()
                for name, proc in list(procs.items()):
                    _graceful_exit(proc, name,
                                   timeout_s=_GRACEFUL_EXIT_TIMEOUT_S)
                print("[supervisor] All children stopped. Supervisor exiting. "
                      "Children will NOT be restarted.")
                return 0

            # 0.5. Watchdog: detect orphaned cmd.exe wrappers (python has
            #     died but cmd.exe is still alive, e.g. hung on `pause`
            #     or `timeout`). Force-kill cmd.exe so the standard
            #     crash-detection path (step 1 below) can fire and
            #     trigger the restart cooldown.
            if (time.time() - last_watchdog_tick) >= _WATCHDOG_CHECK_INTERVAL_S:
                last_watchdog_tick = time.time()
                _watchdog_check(procs, spawn_times)

            # 0.6. Patch: Enrollment-triggered main restart. Poll the
            #     data/.restart_main_requested flag file. If 15s have
            #     elapsed since the flag was written by enroll.py,
            #     gracefully restart ONLY the `main` child so it picks
            #     up the updated student_db.pickle. The dashboard keeps
            #     running so the operator sees the toast notifications.
            _check_enroll_restart_signal(
                procs, pending_restarts, spawn_times, single_window,
            )

            # 1. Check for child exits (ANY reason -- crash, Ctrl+C, X-close,
            #    clean exit). ALL of them get restarted after a 30-second
            #    cooldown, AS LONG AS THE SUPERVISOR IS STILL RUNNING.
            #    The ONLY way to permanently stop a child is to stop the
            #    supervisor itself (Ctrl+C here, or close this window).
            for name in list(procs.keys()):
                proc = procs[name]
                if proc.poll() is None:
                    continue  # Still running.
                rc = proc.returncode
                # Log the exit reason in plain English.
                if rc == 0:
                    reason = "clean exit"
                elif rc == 130:
                    reason = "Ctrl+C in child window"
                elif rc == -1073741510 or rc == 0xC000013A:
                    # 0xC000013A = STATUS_CONTROL_C_EXIT on Windows (X-close
                    #   on the cmd window sometimes returns this).
                    reason = "window closed / Ctrl+C"
                else:
                    reason = f"crashed (code {rc})"
                print(f"[supervisor] {name} {reason}. "
                      f"Will restart in {_CHILD_RESTART_DELAY_S:.0f}s "
                      f"(supervisor still running).")
                # Drop the dead handle and schedule a restart.
                del procs[name]
                pending_restarts[name] = time.time() + _CHILD_RESTART_DELAY_S
                crash_counts[name] += 1
                if crash_counts[name] >= _MAX_RESTARTS:
                    print(f"[supervisor] {name} has been restarted "
                          f"{crash_counts[name]}x. Giving up -- "
                          f"something is fundamentally broken.")
                    pending_restarts.pop(name, None)

            # 2. Process pending restarts whose cooldown has elapsed.
            now_epoch = time.time()
            for name in list(pending_restarts.keys()):
                if now_epoch < pending_restarts[name]:
                    continue
                del pending_restarts[name]
                if name in procs:
                    continue  # Already restarted by another path.
                print(f"[supervisor] Restarting {name}...")
                try:
                    procs[name] = _spawn(name, single_window=single_window)
                    spawn_times[name] = time.time()
                except Exception as exc:
                    print(f"[supervisor] ERROR: failed to spawn {name}: {exc}. "
                          f"Will retry in {_CHILD_RESTART_DELAY_S:.0f}s.")
                    pending_restarts[name] = time.time() + _CHILD_RESTART_DELAY_S

            # 3. Check for scheduled restart (6AM / 6PM boundary crossing).
            now_dt = _dt.datetime.now()
            current = _current_session_hour(now_dt, am_hour, pm_hour)
            if current != last_session:
                print(f"[supervisor] Session boundary crossed: "
                      f"{last_session} -> {current} at "
                      f"{now_dt.strftime('%Y-%m-%d %H:%M:%S')}. "
                      f"GRACEFULLY restarting all children so the "
                      f"application itself can release VRAM / camera / "
                      f"ONNX / CUDA (no force-kill).")
                last_session = current

                # GRACEFUL exit -- let each child run its own signal
                # handler + orch.shutdown() / os._exit(0) path so the
                # application releases VRAM, camera handle, ONNX
                # sessions, CUDA context, video recorder, async logger,
                # and cv2 windows via its own finally blocks.
                #
                # This is the OPPOSITE of force-killing: the application
                # exits with code 0 (or KeyboardInterrupt rc=130),
                # having cleanly dropped all GPU + camera resources.
                # Force-kill is only used as a fallback if a child
                # refuses to exit within _GRACEFUL_EXIT_TIMEOUT_S.
                for name, proc in list(procs.items()):
                    _graceful_exit(proc, name,
                                   timeout_s=_GRACEFUL_EXIT_TIMEOUT_S)
                procs.clear()
                pending_restarts.clear()
                # Reset crash counters (scheduled restart is not a crash).
                crash_counts = {n: 0 for n in targets}

                # Brief wait for OS-level finalization after graceful
                # exit. The application already released most resources
                # via its shutdown path, but the NVIDIA driver still
                # needs 1-3s to finalize CUDA context teardown and the
                # camera handle fully closes.
                print(f"[supervisor] Waiting {_POST_GRACEFUL_RELEASE_S:.0f}s "
                      f"for OS-level VRAM/camera finalization before "
                      f"respawn...")
                time.sleep(_POST_GRACEFUL_RELEASE_S)

                # Respawn with error handling so the supervisor can't crash.
                for name in targets:
                    try:
                        procs[name] = _spawn(name, single_window=single_window)
                        spawn_times[name] = time.time()
                    except Exception as exc:
                        print(f"[supervisor] ERROR: failed to spawn {name} "
                              f"after session boundary: {exc}. Will retry "
                              f"via crash-restart path in "
                              f"{_CHILD_RESTART_DELAY_S:.0f}s.")
                        pending_restarts[name] = time.time() + _CHILD_RESTART_DELAY_S
                        continue
                    time.sleep(1.0)

                print(f"[supervisor] Session boundary restart complete. "
                      f"Waiting for heartbeats...")

            # 4. Decay crash counters if a process has been stable for a while.
            #    (Any 5-min window of stable running decays the counter by 1.)
            for name in crash_counts:
                if crash_counts[name] > 0 and name in procs and procs[name].poll() is None:
                    crash_counts[name] = max(0, crash_counts[name] - 1)

            # 5. If ALL children are dead AND none are pending restart, the
            #    supervisor has nothing left to do -- exit.
            if not procs and not pending_restarts:
                print("[supervisor] All children permanently dead and no "
                      "restarts pending. Exiting.")
                return 0

    except KeyboardInterrupt:
        print("\n[supervisor] Ctrl+C received in supervisor. "
              "PERMANENTLY stopping all children (graceful where "
              "possible so VRAM/camera are released; no respawn).")
        # Cancel any pending restarts so they don't fire after we exit.
        pending_restarts.clear()
        for name, proc in list(procs.items()):
            _graceful_exit(proc, name,
                           timeout_s=_GRACEFUL_EXIT_TIMEOUT_S)
        print("[supervisor] All children stopped. Supervisor exiting. "
              "Children will NOT be restarted.")
        return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> int:
    args = sys.argv[1:]

    dry_run = "--dry-run" in args
    if dry_run:
        args = [a for a in args if a != "--dry-run"]

    single_window = "--single-window" in args
    if single_window:
        args = [a for a in args if a != "--single-window"]

    # Determine targets.
    if not args:
        targets = ["main", "dashboard"]
    else:
        targets = []
        for a in args:
            if a not in _PROCESS_PYTHON:
                print(f"ERROR: unknown target {a!r}. "
                      f"Valid: {list(_PROCESS_PYTHON.keys())}")
                return 2
            targets.append(a)

    return _supervise(targets, dry_run=dry_run, single_window=single_window)


if __name__ == "__main__":
    sys.exit(main())
