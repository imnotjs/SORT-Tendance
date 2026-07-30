"""
SORT-tendance :: ui/dashboard.py

Comprehensive 3-Column Enterprise Dashboard (Streamlit visualizer engine).

This module is intended to be launched as an INDEPENDENT OPERATING SYSTEM
PROCESS via `streamlit run ui/dashboard.py` (or `python -m streamlit run`).
Running the UI in a separate process ensures that Streamlit's re-render
loop, JIT widget recomputation, and script-rerun semantics NEVER introduce
cycle blockages into the core capture / AI / recorder threads of the
SORT-tendance orchestrator (`main.py`).

Layout (strict 3-column enterprise dashboard):

  * Column 1 (Performance Monitor):
      Opens a non-blocking UDP socket server loop that ingests, decodes,
      and displays live system telemetry packages broadcast by the
      orchestrator's PerformanceBroadcaster. Renders:
        - Capture FPS (rolling 1s window)
        - Inference FPS (rolling 1s window)
        - Processing / tracking latency (ms)
        - Thread CPU affinity scopes (per-thread core mask)
        - GPU VRAM occupancy
        - Active / locked / pending track counts

  * Column 2 (Live Attendance Journal):
      Asynchronously polls the active daily CSV log file written by
      `src/utils/async_logger.py`. Presents a clean, auto-refreshing
      grid showing student records registered during the current
      session (NRP, Student Name, Status, State, Hardware Capture Index,
      Precision Timestamp Microseconds, Match Similarity).

  * Column 3 (Stranger Alert Gallery):
      Scans the `storage/cache_strangers/` directory and dynamically
      renders cropped images of unverified targets alongside their
      sequential tracking labels ("Stranger_01", "Stranger_02", ...)
      and computed spatial anomaly scores (centroid displacement
      variance over the cached snapshot set).

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import atexit
import gc
import socket
import struct
import logging
import threading
import traceback
import datetime as _dt
import tempfile
import uuid
import subprocess
import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Patch 36 :: Optional charting + dataframe deps for the performance
# time-series graphs. pandas is used to build the time-indexed
# DataFrame that feeds the chart renderer. When pandas is absent,
# _render_performance_charts() falls back to a plain text table of
# the latest values (no crash).
#
# Patch 56 :: Plotly has been REMOVED ENTIRELY from dashboard.py.
# Previous patches (52/54/55) only made Plotly *conditional* on
# Altair being absent -- but on the user's Windows machine Altair
# is NOT installed, so the conditional still evaluated True and
# Plotly was still being imported. Every render still called
# st.plotly_chart(fig, ...), which invokes the recursive
# convert_to_base64() walker, which access-violates (0xC0000005)
# during garbage collection on long-running Windows processes,
# killing the Streamlit process AFTER a successful render.
#
# The companion error 'tuple' object has no attribute 'pop' comes
# from Streamlit's _HashStack (thread-local) being corrupted by the
# same recursive walker when it tries to hash the Plotly figure for
# caching -- the stack is left in an inconsistent state after a
# previous failed hash.
#
# Three-level fallback chain (NO Plotly anywhere):
#   Path A : Altair  (preferred -- pure JSON spec, no recursion)
#   Path B : matplotlib PNG  (fallback -- pure bytes, no native crash)
#   Path C : text-only  (final safety net)
#
# matplotlib is a pure-Python (with a non-corrupting C-extension)
# renderer that outputs PNG bytes. st.image(png_bytes) has no
# recursive walker and no GC access violation.
try:
    import altair as alt
    _ALTAIR_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _ALTAIR_AVAILABLE = False
    alt = None  # type: ignore

# Patch 56 :: matplotlib is the safe fallback when Altair is absent.
# We force the Agg backend (no GUI dependency, thread-safe) before
# importing pyplot. matplotlib's C-extension does not corrupt memory
# during GC on Windows, unlike Plotly's.
import matplotlib
matplotlib.use("Agg", force=True)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _MPL_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _MPL_AVAILABLE = False
    plt = None  # type: ignore
    mdates = None  # type: ignore

# Patch 56 :: io.BytesIO is used to serialize matplotlib figures to
# PNG bytes for st.image(). Pure Python, no native crash.
import io as _io

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _PIL_AVAILABLE = False
    Image = None  # type: ignore

# Streamlit is the UI runtime; we import it lazily inside the entry point
# so that this module can be imported for its helper classes without
# requiring Streamlit at import time (useful for unit tests).
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

# Local config registry import (absolute path resolution).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))

# Patch :: Class Scheduling & Attendance UI pages.
# Lives in ui/scheduling_pages.py to keep dashboard.py from bloating.
try:
    from ui.scheduling_pages import (
        render_schedule_page,
        render_students_page,
        render_live_attendance_page,
        render_reports_page,
    )
    _SCHED_PAGES_AVAILABLE = True
    _SCHED_PAGES_ERROR = None
except Exception as _sched_imp_exc:  # pragma: no cover
    _SCHED_PAGES_AVAILABLE = False
    _SCHED_PAGES_ERROR = str(_sched_imp_exc)
    render_schedule_page = None  # type: ignore
    render_students_page = None  # type: ignore
    render_live_attendance_page = None  # type: ignore
    render_reports_page = None  # type: ignore
try:
    from utils.database_manager import ConfigRegistry
except ImportError:                         # pragma: no cover
    ConfigRegistry = None  # type: ignore

# Patch 48 :: Enable faulthandler for native-crash diagnostics.
# This installs a SIGSEGV/SIGABRT handler that prints the Python
# stack trace to stderr when a C extension access-violates. Without
# this, native crashes (0xC0000005) die silently with no stack trace.
import faulthandler
faulthandler.enable()

# Patch 18 :: Shared 12-hour session helpers from async_logger. These
# guarantee that the dashboard, the CSV writer, and snap_strangers all
# agree on the LOCAL 06:00 / 18:00 session boundary and the
# "YYYY-MM-DD" / "6AM Session" / "6PM Session" folder naming.
try:
    from utils.async_logger import (
        compute_session_key,
        session_label_to_dir,
        session_has_started,
        current_active_session_label,
        SESSION_AM_START_HOUR,
        SESSION_PM_START_HOUR,
        SESSION_LABEL_AM,
        SESSION_LABEL_PM,
        SESSION_DIR_AM,
        SESSION_DIR_PM,
    )
    _SESSION_HELPERS_AVAILABLE = True
except ImportError:                         # pragma: no cover
    # Fallback: re-implement the helpers locally so the dashboard can
    # still render even if async_logger is unavailable (e.g. in a
    # stripped-down monitoring-only deployment on the laptop that does
    # not have the full src/utils package).
    _SESSION_HELPERS_AVAILABLE = False
    SESSION_AM_START_HOUR = 6
    SESSION_PM_START_HOUR = 18
    SESSION_LABEL_AM = "06AM"
    SESSION_LABEL_PM = "06PM"
    SESSION_DIR_AM = "6AM Session"
    SESSION_DIR_PM = "6PM Session"

    def compute_session_key(ts_us):
        local_dt = _dt.datetime.fromtimestamp(ts_us / 1_000_000.0).astimezone()
        hour = local_dt.hour
        if hour < SESSION_AM_START_HOUR:
            session_start_date = (local_dt - _dt.timedelta(days=1)).date()
            session_label = SESSION_LABEL_PM
        elif hour < SESSION_PM_START_HOUR:
            session_start_date = local_dt.date()
            session_label = SESSION_LABEL_AM
        else:
            session_start_date = local_dt.date()
            session_label = SESSION_LABEL_PM
        return (session_start_date.strftime("%Y-%m-%d"), session_label)

    def session_label_to_dir(session_label):
        if session_label == SESSION_LABEL_AM:
            return SESSION_DIR_AM
        if session_label == SESSION_LABEL_PM:
            return SESSION_DIR_PM
        return session_label

    def current_active_session_label():
        return compute_session_key(int(time.time() * 1_000_000))[1]

    def session_has_started(date_str, session_label):
        try:
            d = _dt.date.fromisoformat(date_str)
        except ValueError:
            return False
        if session_label == SESSION_LABEL_AM:
            start_dt = _dt.datetime.combine(d, _dt.time(hour=SESSION_AM_START_HOUR)).astimezone()
        elif session_label == SESSION_LABEL_PM:
            start_dt = _dt.datetime.combine(d, _dt.time(hour=SESSION_PM_START_HOUR)).astimezone()
        else:
            return False
        return _dt.datetime.now().astimezone() >= start_dt


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.dashboard")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class TelemetryPacket:
    """
    Wire-format telemetry packet schema (broadcast by main.py's
    PerformanceBroadcaster on each AI cycle).

    The UDP payload is a UTF-8 JSON document with this exact schema.
    """
    capture_fps: float
    inference_fps: float
    processing_latency_ms: float
    tracking_latency_ms: float
    gui_latency_ms: float
    ai_queue_depth: int
    ai_queue_maxsize: int
    active_track_count: int
    pending_track_count: int
    verified_track_count: int
    stranger_track_count: int
    anomaly_count: int
    gpu_vram_used_bytes: int
    gpu_vram_total_bytes: int
    throttle_mode: str                       # IDLE | BURST
    encoder_kind: str                        # h264_nvenc | libx264
    thread_affinity: Dict[str, List[int]]    # thread_name -> core list
    frame_index: int
    broadcast_us: int
    # Patch 20 :: Session-boundary telemetry. The orchestrator broadcasts
    # the session it is currently writing into (e.g. "06AM" / "2025-01-15"),
    # so the dashboard can surface "Orchestrator active session" in the
    # Performance column without recomputing the boundary locally and
    # risking a clock-skew mismatch. Defaults to "" so older broadcasts
    # (which do not include these fields) still parse cleanly.
    current_session_label: str = ""
    current_session_date: str = ""
    # Patch 36 :: Process-scoped CPU% + RSS bytes for the performance
    # time-series graphs. Defaults to 0.0 / 0 so older broadcasts from
    # pre-Patch-35 main.py still parse cleanly (the charts will render
    # flat zero lines for those packets, not crash).
    cpu_percent: float = 0.0
    rss_bytes: int = 0
    # Patch 63 (hotfix B) :: Heartbeat fields. Let the dashboard show
    # main.py's status even when no frames are being processed.
    #   STARTUP : AI thread just entered run(), before first frame
    #   IDLE    : camera disconnected / not producing frames
    #   LIVE    : frames being processed normally
    #   ERROR   : _process_frame crashing every cycle
    heartbeat: str = "LIVE"
    ai_errors: int = 0
    frames_processed: int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_fps": self.capture_fps,
            "inference_fps": self.inference_fps,
            "processing_latency_ms": self.processing_latency_ms,
            "tracking_latency_ms": self.tracking_latency_ms,
            "gui_latency_ms": self.gui_latency_ms,
            "ai_queue_depth": self.ai_queue_depth,
            "ai_queue_maxsize": self.ai_queue_maxsize,
            "active_track_count": self.active_track_count,
            "pending_track_count": self.pending_track_count,
            "verified_track_count": self.verified_track_count,
            "stranger_track_count": self.stranger_track_count,
            "anomaly_count": self.anomaly_count,
            "gpu_vram_used_bytes": self.gpu_vram_used_bytes,
            "gpu_vram_total_bytes": self.gpu_vram_total_bytes,
            "throttle_mode": self.throttle_mode,
            "encoder_kind": self.encoder_kind,
            "thread_affinity": dict(self.thread_affinity),
            "frame_index": self.frame_index,
            "broadcast_us": self.broadcast_us,
            "current_session_label": self.current_session_label,
            "current_session_date": self.current_session_date,
            "cpu_percent": self.cpu_percent,
            "rss_bytes": self.rss_bytes,
            "heartbeat": self.heartbeat,
            "ai_errors": self.ai_errors,
            "frames_processed": self.frames_processed,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, payload: str) -> "TelemetryPacket":
        d = json.loads(payload)
        return cls(
            capture_fps=float(d.get("capture_fps", 0.0)),
            inference_fps=float(d.get("inference_fps", 0.0)),
            processing_latency_ms=float(d.get("processing_latency_ms", 0.0)),
            tracking_latency_ms=float(d.get("tracking_latency_ms", 0.0)),
            gui_latency_ms=float(d.get("gui_latency_ms", 0.0)),
            ai_queue_depth=int(d.get("ai_queue_depth", 0)),
            ai_queue_maxsize=int(d.get("ai_queue_maxsize", 0)),
            active_track_count=int(d.get("active_track_count", 0)),
            pending_track_count=int(d.get("pending_track_count", 0)),
            verified_track_count=int(d.get("verified_track_count", 0)),
            stranger_track_count=int(d.get("stranger_track_count", 0)),
            anomaly_count=int(d.get("anomaly_count", 0)),
            gpu_vram_used_bytes=int(d.get("gpu_vram_used_bytes", 0)),
            gpu_vram_total_bytes=int(d.get("gpu_vram_total_bytes", 0)),
            throttle_mode=str(d.get("throttle_mode", "IDLE")),
            encoder_kind=str(d.get("encoder_kind", "libx264")),
            thread_affinity=dict(d.get("thread_affinity", {})),
            frame_index=int(d.get("frame_index", 0)),
            broadcast_us=int(d.get("broadcast_us", 0)),
            current_session_label=str(d.get("current_session_label", "")),
            current_session_date=str(d.get("current_session_date", "")),
            cpu_percent=float(d.get("cpu_percent", 0.0)),
            rss_bytes=int(d.get("rss_bytes", 0)),
            heartbeat=str(d.get("heartbeat", "LIVE")),
            ai_errors=int(d.get("ai_errors", 0)),
            frames_processed=int(d.get("frames_processed", 0)),
        )


@dataclass
class AttendanceRow:
    """A single row from the daily CSV log file."""
    timestamp_us: int
    frame_index: int
    track_id: int
    nrp: str
    student_name: str
    resolved_label: str
    state: str
    similarity_score: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int


# ============================================================================
# UDP Telemetry Receiver (Column 1)
# ============================================================================
class UDPTelemetryReceiver:
    """
    Non-blocking UDP socket server that ingests telemetry packets from
    the orchestrator's PerformanceBroadcaster.

    The receiver runs a background daemon thread that recvfrom()'s on
    the configured port and pushes decoded TelemetryPacket instances
    onto a deque of bounded capacity. The Streamlit loop polls the
    deque on each re-render cycle.

    The socket is bound in non-blocking mode with a short timeout so
    that the receiver thread can be cleanly shut down via the stop
    event without hanging on recvfrom.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        buffer_bytes: int = 65536,
        # Patch 36 :: history_capacity 60 -> 43200 to support the 3h
        # performance time-series window (3h @ 250 ms = 43200 packets).
        # Memory cost is ~8.6 MB at ~200 bytes per packet -- negligible.
        history_capacity: int = 43200,
    ) -> None:
        self._host: str = str(host)
        self._port: int = int(port)
        self._buffer_bytes: int = int(buffer_bytes)
        self._history_capacity: int = int(history_capacity)
        self._history: Deque[TelemetryPacket] = deque(maxlen=self._history_capacity)
        self._lock: threading.RLock = threading.RLock()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._running: bool = False
        # Patch 63 (hotfix E) :: One-way shutdown flag. Once stop()
        # has been called, _shutdown is set True and start() will
        # refuse to re-start the receiver. This prevents the rapid
        # start/stop cycle during Streamlit's shutdown sequence:
        # Streamlit re-runs the script multiple times while shutting
        # down, and each re-run used to call receiver.start() (because
        # _running was False), creating dozens of "loop exited" +
        # "thread joined cleanly" log pairs.
        self._shutdown: bool = False
        self._packets_received: int = 0
        self._packets_dropped: int = 0
        self._decode_errors: int = 0
        self._last_packet_at_us: int = 0

    # ------------------------------------------------------------------
    def start(self) -> bool:
        # Patch 63 (hotfix E) :: Refuse to start after shutdown.
        # Once stop() has been called (either by atexit or by the
        # __main__ block), the receiver is permanently retired.
        # This prevents Streamlit's shutdown reruns from respawning
        # the receiver thread dozens of times.
        if self._shutdown:
            logger.info(
                "UDPTelemetryReceiver: refusing to start (already "
                "shut down). This is expected during process exit."
            )
            return False
        if self._running:
            logger.warning("UDPTelemetryReceiver already running.")
            return True

        try:
            self._socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP,
            )
            # Patch 63 (hotfix H) :: REMOVED SO_REUSEADDR.
            #
            # On Linux, SO_REUSEADDR for UDP allows rebinding during
            # TIME_WAIT (harmless). But on WINDOWS, SO_REUSEADDR allows
            # MULTIPLE sockets to bind to the same UDP port SIMULTANEOUSLY.
            # The OS then distributes incoming datagrams unpredictably
            # among all bound sockets. If a zombie process (from a
            # previous crashed dashboard) is still bound to port 9999,
            # it steals ALL the packets from main.py, and the new
            # receiver gets zero -- exactly the symptom we saw:
            #   main.py: broadcaster_total=7915 (packets sent OK)
            #   dashboard: packets_received=0 (none arrived)
            #
            # UDP has no TIME_WAIT (that's TCP-only), so once a UDP
            # socket is closed, the port is immediately available.
            # SO_REUSEADDR is NOT needed for UDP restart-safety.
            #
            # Without SO_REUSEADDR, if a zombie holds the port, bind()
            # will FAIL with WSAEADDRINUSE (errno 10048 on Windows),
            # which we handle below with a clear error message telling
            # the operator to find and kill the zombie process.
            #
            # Increase the OS receive buffer to absorb bursts.
            try:
                self._socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, self._buffer_bytes * 4,
                )
            except OSError as exc:
                logger.warning(
                    "UDPTelemetryReceiver: SO_RCVBUF set failed (continuing): %s",
                    exc,
                )
            self._socket.settimeout(0.5)
            self._socket.bind((self._host, self._port))
        except OSError as exc:
            # Patch 63 (hotfix H) :: Enhanced bind-failure error with
            # zombie-process diagnosis. On Windows, errno 10048
            # (WSAEADDRINUSE) means another process is holding the port.
            # Tell the operator exactly how to find and kill it.
            _errno = getattr(exc, "errno", None) or getattr(exc, "winerror", None)
            _is_addr_in_use = (
                _errno in (10048, 98)  # 10048=WSAEADDRINUSE, 98=EADDRINUSE
                or "10048" in str(exc)
                or "Addr already in use" in str(exc)
                or "Address already in use" in str(exc)
            )
            if _is_addr_in_use:
                logger.error(
                    "======================================================\n"
                    "UDPTelemetryReceiver: PORT %d IS ALREADY IN USE!\n"
                    "Another process (likely a zombie dashboard from a\n"
                    "previous run) is holding UDP %s:%d and stealing\n"
                    "all telemetry packets from main.py.\n\n"
                    "TO FIX:\n"
                    "  1. Open Task Manager (Ctrl+Shift+Esc)\n"
                    "  2. Go to 'Details' tab\n"
                    "  3. Find ALL python.exe and streamlit.exe processes\n"
                    "  4. Right-click -> End Task for EACH one\n"
                    "  5. OR run in cmd:  taskkill /F /IM python.exe\n"
                    "                      taskkill /F /IM streamlit.exe\n"
                    "  6. OR check what holds the port:\n"
                    "       netstat -ano | findstr :%d\n"
                    "     Then taskkill /F /PID <pid>\n"
                    "  7. Restart run_dashboard.bat\n"
                    "======================================================",
                    self._port, self._host, self._port, self._port,
                )
            else:
                logger.error(
                    "UDPTelemetryReceiver: failed to bind %s:%d -- %s "
                    "(errno=%s, winerror=%s)",
                    self._host, self._port, exc,
                    getattr(exc, "errno", None),
                    getattr(exc, "winerror", None),
                )
            self._socket = None
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sortendance.dashboard.udp_rx",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info(
            "UDPTelemetryReceiver listening on %s:%d (buf=%d bytes)",
            self._host, self._port, self._buffer_bytes,
        )
        return True

    # ------------------------------------------------------------------
    def stop(self, timeout_s: float = 3.0) -> None:
        # Patch 63 (hotfix E) :: stop() is now truly one-way. Once
        # called, _shutdown is set True and start() will refuse to
        # restart the receiver. This prevents the rapid start/stop
        # cycle during Streamlit's shutdown sequence.
        if self._shutdown:
            # Already fully stopped -- silent return (not even a log,
            # since this may be called dozens of times by atexit +
            # __main__ + Streamlit reruns).
            return
        if not self._running:
            # Not running yet, but mark as shut down so start() refuses.
            self._shutdown = True
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.error(
                    "UDPTelemetryReceiver thread did not exit within %.2fs",
                    timeout_s,
                )
            else:
                logger.info("UDPTelemetryReceiver thread joined cleanly.")
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self._running = False
        self._shutdown = True

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        logger.info("UDPTelemetryReceiver loop entered.")
        # Patch 63 (hotfix G) :: Periodic alive log. If no packets
        # arrive, log every 5 seconds so the operator can confirm
        # the receiver thread is still running (not crashed). This
        # is critical for diagnosing "no telemetry" issues: if the
        # alive log appears but no packets arrive, the problem is
        # network/firewall/stale-process; if the alive log STOPS,
        # the receiver thread crashed.
        #
        # NOTE: This log MUST be at the TOP of the while loop, BEFORE
        # the try/except. The socket.timeout handler uses `continue`,
        # which skips anything after the try block. If the alive log
        # were at the bottom, it would never fire when no packets
        # arrive (the exact scenario we need to diagnose).
        _last_alive_log_ts: float = time.time()
        _loop_start_ts: float = time.time()  # hotfix H: for uptime tracking
        _zombie_warned: bool = False  # hotfix H: warn once about zombie
        while not self._stop_event.is_set():
            # Patch 63 (hotfix G) :: Alive log at top of loop.
            _now = time.time()
            if _now - _last_alive_log_ts >= 5.0:
                _last_alive_log_ts = _now
                _uptime_s = _now - _loop_start_ts
                logger.info(
                    "UDPTelemetryReceiver: alive (uptime=%.0fs) | "
                    "total_received=%d | decode_errors=%d | "
                    "history=%d/%d",
                    _uptime_s,
                    self._packets_received, self._decode_errors,
                    len(self._history), self._history_capacity,
                )
                # Patch 63 (hotfix H) :: Zombie detection. If we've been
                # alive for >10 seconds with 0 packets, warn about a
                # possible zombie process stealing packets (Windows
                # SO_REUSEADDR allows multiple binds -- but we removed
                # it in hotfix H, so this should no longer happen.
                # If it does, the bind() should have failed. This log
                # is a safety net for edge cases.)
                if (
                    not _zombie_warned
                    and _uptime_s > 10.0
                    and self._packets_received == 0
                ):
                    _zombie_warned = True
                    logger.warning(
                        "================================================\n"
                        "UDPTelemetryReceiver: 0 packets after %.0fs!\n"
                        "main.py may not be running, OR a zombie process\n"
                        "is stealing packets on port %d.\n\n"
                        "Check:\n"
                        "  1. Is main.py running? (run_main.bat)\n"
                        "  2. netstat -ano | findstr :%d\n"
                        "  3. Task Manager -> kill ALL python.exe\n"
                        "================================================",
                        _uptime_s, self._port, self._port,
                    )
            try:
                if self._socket is None:
                    break
                try:
                    data, _addr = self._socket.recvfrom(self._buffer_bytes)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stop_event.is_set():
                        break
                    logger.warning(
                        "UDPTelemetryReceiver: recvfrom error: %s", exc,
                    )
                    continue

                self._packets_received += 1
                # Patch 63 (hotfix G) :: Log the FIRST packet received
                # so the operator can confirm the receiver is actually
                # getting data from main.py. Without this, the receiver
                # only logs every 100 packets, so the first ~100 seconds
                # of operation produce no receiver-side feedback.
                if self._packets_received == 1:
                    logger.info(
                        "UDPTelemetryReceiver: FIRST packet received! "
                        "(len=%d bytes) -- telemetry stream is live.",
                        len(data),
                    )
                try:
                    payload = data.decode("utf-8", errors="replace")
                    packet = TelemetryPacket.from_json(payload)
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    self._decode_errors += 1
                    logger.warning(
                        "UDPTelemetryReceiver: decode error (%s): %s",
                        exc, payload[:200],
                    )
                    continue

                self._last_packet_at_us = int(time.time() * 1_000_000)
                with self._lock:
                    self._history.append(packet)
                # Patch 44 :: Heartbeat log every 100 packets so the
                # operator can see the receiver is still alive.
                if self._packets_received % 100 == 0:
                    logger.info(
                        "UDPTelemetryReceiver heartbeat | packets=%d | "
                        "history=%d/%d",
                        self._packets_received, len(self._history),
                        self._history_capacity,
                    )
            except Exception as exc:
                logger.error(
                    "UDPTelemetryReceiver loop exception: %s\n%s",
                    exc, traceback.format_exc(),
                )
                # Brief sleep to prevent tight error loops.
                time.sleep(0.05)

        logger.info("UDPTelemetryReceiver loop exited.")

    # ------------------------------------------------------------------
    def latest(self) -> Optional[TelemetryPacket]:
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    # ------------------------------------------------------------------
    def history(self) -> List[TelemetryPacket]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Patch 41 :: history_last_n() -- fetch only the tail of the deque
    # without materializing the entire history. Used by the performance
    # charts for small windows (e.g. 15m needs only ~3600 packets, not
    # the full 43200-packet deque).
    # ------------------------------------------------------------------
    def history_last_n(self, n: int) -> List[TelemetryPacket]:
        with self._lock:
            if n >= len(self._history):
                return list(self._history)
            # itertools.islice on a deque tail.
            from collections import deque as _deque
            start = len(self._history) - n
            return [
                self._history[i]
                for i in range(start, len(self._history))
            ]

    # ------------------------------------------------------------------
    # Patch 57 :: clear_history() -- drop the 43200-packet deque at the
    # 6AM/6PM session boundary. Without this, the deque retains packets
    # from the PREVIOUS session indefinitely, mixing old-session
    # telemetry with new-session telemetry in the performance charts
    # (data correctness issue + ~8.6 MB of stale baseline memory).
    #
    # Safe to call from any thread -- acquires the same RLock used by
    # history() / history_last_n() / latest(). The daemon receive loop
    # will continue appending fresh packets after clear() returns.
    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        with self._lock:
            n = len(self._history)
            self._history.clear()
        logger.info(
            "UDPTelemetryReceiver: history deque cleared (%d packets dropped).",
            n,
        )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "host": self._host,
                "port": self._port,
                "running": self._running,
                "packets_received": self._packets_received,
                "packets_dropped": self._packets_dropped,
                "decode_errors": self._decode_errors,
                "history_size": len(self._history),
                "history_capacity": self._history_capacity,
                "last_packet_age_us": (
                    int(time.time() * 1_000_000) - self._last_packet_at_us
                    if self._last_packet_at_us > 0 else -1
                ),
            }


# ============================================================================
# CSV Attendance Poller (Column 2)
# ============================================================================
class CSVAttendancePoller:
    """
    Asynchronously polls the active daily CSV log file produced by
    `AsyncLoggingEngine.DailyCSVWriter`.

    The poller:
      * Resolves the active daily file path (UTC YYYY-MM-DD).
      * Reads + parses the file on each `poll()` invocation.
      * Caches the parsed rows keyed by (timestamp_us, track_id) to
        avoid re-emitting stale rows.
      * Returns the rows sorted by timestamp (descending) so the
        dashboard can render the most-recent-first.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        log_dir: str,
        prefix: str,
        columns: List[str],
        max_rows: int = 500,
    ) -> None:
        self._log_dir: str = os.path.abspath(log_dir)
        self._prefix: str = str(prefix)
        self._columns: List[str] = list(columns)
        self._max_rows: int = int(max_rows)
        self._lock: threading.RLock = threading.RLock()
        self._cache: Dict[Tuple[int, int], AttendanceRow] = {}
        self._last_file_path: Optional[str] = None
        self._last_file_size: int = -1
        self._last_poll_at_us: int = 0
        self._poll_count: int = 0
        self._parse_errors: int = 0

        try:
            os.makedirs(self._log_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "CSVAttendancePoller: failed to create log_dir %s: %s",
                self._log_dir, exc,
            )

        logger.info(
            "CSVAttendancePoller initialized | dir=%s | prefix=%s | cols=%d",
            self._log_dir, self._prefix, len(self._columns),
        )

    # ------------------------------------------------------------------
    def _resolve_active_path(self, ts_us: int) -> str:
        """
        Patch 18 :: Resolve the active 12h-session CSV file path.

        Uses the same `compute_session_key()` helper as the writer so
        the dashboard poller and the writer agree on which file is
        "active" at any given moment. The path takes the form:
            {log_dir}/{prefix}_{YYYY-MM-DD}_{06AM|06PM}.csv
        """
        date_str, session_label = compute_session_key(ts_us)
        return os.path.join(
            self._log_dir, f"{self._prefix}_{date_str}_{session_label}.csv",
        )

    # ------------------------------------------------------------------
    def _resolve_path_for_session(
        self,
        date_str: str,
        session_label: str,
    ) -> str:
        """
        Patch 20 :: Resolve the CSV path for an explicit (date, session)
        pair, bypassing the active-session computation. Used by
        ``poll_session`` for historical review.
        """
        return os.path.join(
            self._log_dir,
            f"{self._prefix}_{date_str}_{session_label}.csv",
        )

    # ------------------------------------------------------------------
    def list_available_sessions(self) -> List[Tuple[str, str]]:
        """
        Patch 20 :: Enumerate all sessions whose CSV file exists in
        ``log_dir`` and that have already started.

        Returns:
            List of (date_str, session_label) tuples, sorted newest-first.

        A session qualifies if:
          1. Its CSV file exists in log_dir (any size, including the
             header-only case the writer creates on rotation).
          2. Its start time has passed (session_has_started). Future
             sessions are NEVER listed -- e.g. at 08:00 local on
             2025-01-15, only the 2025-01-15/06AM session is listed.

        The currently-active session is always included even if its
        CSV file does not yet exist (e.g. the session just rolled over
        and no rows have been written yet).
        """
        sessions: List[Tuple[str, str]] = []
        if os.path.isdir(self._log_dir):
            try:
                for entry in os.scandir(self._log_dir):
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.endswith(".csv"):
                        continue
                    if not name.startswith(self._prefix):
                        continue
                    # Strip prefix and .csv to recover "{date}_{session}".
                    stem = name[len(self._prefix):-len(".csv")]
                    # stem looks like "2025-01-15_06AM".
                    parts = stem.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    date_str, session_label = parts
                    if session_label not in (SESSION_LABEL_AM, SESSION_LABEL_PM):
                        continue
                    try:
                        _dt.date.fromisoformat(date_str)
                    except ValueError:
                        continue
                    sessions.append((date_str, session_label))
            except OSError as exc:
                logger.warning(
                    "CSVAttendancePoller.list_available_sessions: "
                    "scan failed at %s: %s",
                    self._log_dir, exc,
                )

        # Always include the currently-active session even if its file
        # does not exist yet (just-rolled-over edge case).
        active_date, active_label = compute_session_key(int(time.time() * 1_000_000))
        if (active_date, active_label) not in sessions:
            sessions.append((active_date, active_label))

        # Filter out not-yet-started sessions (future-session filter).
        visible = [
            (d, s) for (d, s) in sessions
            if session_has_started(d, s)
        ]

        # Sort newest-first by session start wall-clock.
        def _key(item: Tuple[str, str]) -> float:
            d_str, s_label = item
            try:
                d = _dt.date.fromisoformat(d_str)
            except ValueError:
                return 0.0
            hour = (
                SESSION_AM_START_HOUR
                if s_label == SESSION_LABEL_AM
                else SESSION_PM_START_HOUR
            )
            return _dt.datetime.combine(
                d, _dt.time(hour=hour),
            ).astimezone().timestamp()

        visible.sort(key=_key, reverse=True)
        return visible

    # ------------------------------------------------------------------
    def poll_session(
        self,
        date_str: str,
        session_label: str,
    ) -> List[AttendanceRow]:
        """
        Patch 20 :: Read a specific (date, session) CSV file and return
        all rows it contains, sorted by timestamp descending.

        Unlike ``poll()``, this method does NOT touch ``self._cache`` --
        it builds a fresh local list on every call. This keeps the live
        cache (used by the active-session poll loop) pristine when the
        operator switches the attendance column to a historical session
        for review.

        On I/O failure or missing file, returns an empty list.
        """
        path = self._resolve_path_for_session(date_str, session_label)
        rows: List[AttendanceRow] = []
        if not os.path.exists(path):
            return rows

        try:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    return rows
                for row in reader:
                    try:
                        ts_us = int(row.get("timestamp_us", 0))
                        frame_index = int(row.get("frame_index", 0))
                        track_id = int(row.get("track_id", -1))
                        rows.append(AttendanceRow(
                            timestamp_us=ts_us,
                            frame_index=frame_index,
                            track_id=track_id,
                            nrp=row.get("nrp", "") or "",
                            student_name=row.get("student_name", "") or "",
                            resolved_label=row.get("resolved_label", "") or "",
                            state=row.get("state", "") or "",
                            similarity_score=float(row.get("similarity_score", 0.0) or 0.0),
                            bbox_x1=int(row.get("bbox_x1", 0) or 0),
                            bbox_y1=int(row.get("bbox_y1", 0) or 0),
                            bbox_x2=int(row.get("bbox_x2", 0) or 0),
                            bbox_y2=int(row.get("bbox_y2", 0) or 0),
                        ))
                    except (ValueError, KeyError) as exc:
                        self._parse_errors += 1
                        logger.debug(
                            "CSVAttendancePoller.poll_session: row parse "
                            "error (%s): %s",
                            exc, row,
                        )
                        continue
        except OSError as exc:
            logger.warning(
                "CSVAttendancePoller.poll_session: failed to read %s: %s",
                path, exc,
            )
            return rows

        rows.sort(key=lambda r: r.timestamp_us, reverse=True)
        return rows

    # ------------------------------------------------------------------
    def poll(self) -> List[AttendanceRow]:
        """
        Re-read the active daily CSV and return all rows (newly cached
        + previously cached), sorted by timestamp descending.

        On I/O failure, returns the last successful cache.
        """
        now_us = int(time.time() * 1_000_000)
        self._last_poll_at_us = now_us
        self._poll_count += 1

        path = self._resolve_active_path(now_us)
        if not os.path.exists(path):
            # No file yet for the current UTC day; return cached rows.
            with self._lock:
                return self._sorted_cache_locked()

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = -1

        # If the file path + size haven't changed since the last poll,
        # skip the re-read (the writer only appends; no need to re-parse).
        if (
            path == self._last_file_path
            and file_size == self._last_file_size
            and file_size > 0
        ):
            with self._lock:
                return self._sorted_cache_locked()

        # Patch 46 :: Session rotation detection.
        # When the active file path changes (12h session rotated at
        # 06:00 / 18:00 local), CLEAR the cache before reading the new
        # file. Without this, verified_students() / strangers() /
        # anomalies() would return rows from ALL sessions ever polled,
        # mixed together -- causing the dashboard to show stale data
        # from old sessions indefinitely.
        if (
            self._last_file_path is not None
            and path != self._last_file_path
        ):
            old_name = os.path.basename(self._last_file_path)
            new_name = os.path.basename(path)
            with self._lock:
                old_count = len(self._cache)
                self._cache.clear()
            logger.info(
                "CSVAttendancePoller: session rotation detected -- "
                "cache cleared (%d rows dropped). Old file: %s | "
                "New file: %s",
                old_count, old_name, new_name,
            )

        self._last_file_path = path
        self._last_file_size = file_size

        # Parse the file.
        try:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    # Empty file (header not yet written).
                    with self._lock:
                        return self._sorted_cache_locked()

                # Validate the header against our configured columns.
                # We do NOT fail on column mismatch; we just log it and
                # use whatever columns the file declares.
                if set(reader.fieldnames) != set(self._columns):
                    logger.debug(
                        "CSVAttendancePoller: header mismatch | file=%s | "
                        "configured=%s | actual=%s",
                        path, self._columns, reader.fieldnames,
                    )

                with self._lock:
                    for row in reader:
                        try:
                            self._ingest_row_locked(row)
                        except (ValueError, KeyError) as exc:
                            self._parse_errors += 1
                            logger.debug(
                                "CSVAttendancePoller: row parse error (%s): %s",
                                exc, row,
                            )
                            continue
        except OSError as exc:
            logger.warning(
                "CSVAttendancePoller: failed to read %s: %s",
                path, exc,
            )
            with self._lock:
                return self._sorted_cache_locked()

        with self._lock:
            return self._sorted_cache_locked()

    # ------------------------------------------------------------------
    def _ingest_row_locked(self, row: Dict[str, str]) -> None:
        """Ingest a single CSV row into the cache (caller holds the lock)."""
        ts_us = int(row.get("timestamp_us", 0))
        frame_index = int(row.get("frame_index", 0))
        track_id = int(row.get("track_id", -1))
        key = (ts_us, track_id)
        ar = AttendanceRow(
            timestamp_us=ts_us,
            frame_index=frame_index,
            track_id=track_id,
            nrp=row.get("nrp", "") or "",
            student_name=row.get("student_name", "") or "",
            resolved_label=row.get("resolved_label", "") or "",
            state=row.get("state", "") or "",
            similarity_score=float(row.get("similarity_score", 0.0) or 0.0),
            bbox_x1=int(row.get("bbox_x1", 0) or 0),
            bbox_y1=int(row.get("bbox_y1", 0) or 0),
            bbox_x2=int(row.get("bbox_x2", 0) or 0),
            bbox_y2=int(row.get("bbox_y2", 0) or 0),
        )
        self._cache[key] = ar

        # Bound the cache size if it grows unbounded (defensive).
        if len(self._cache) > self._max_rows * 4:
            # Evict the oldest entries by timestamp.
            sorted_keys = sorted(self._cache.keys(), key=lambda k: k[0])
            for k in sorted_keys[: len(self._cache) - self._max_rows * 2]:
                self._cache.pop(k, None)

    # ------------------------------------------------------------------
    def _sorted_cache_locked(self) -> List[AttendanceRow]:
        """Return the cache sorted by timestamp descending (caller holds lock)."""
        return sorted(
            self._cache.values(),
            key=lambda r: r.timestamp_us,
            reverse=True,
        )

    # ------------------------------------------------------------------
    def verified_students(self) -> List[AttendanceRow]:
        """Return only the VERIFIED_STUDENT rows (deduplicated by NRP)."""
        with self._lock:
            seen_nrps: Dict[str, AttendanceRow] = {}
            for row in self._cache.values():
                if row.state != "VERIFIED_STUDENT":
                    continue
                if not row.nrp:
                    continue
                # Keep the most recent registration per NRP.
                if (
                    row.nrp not in seen_nrps
                    or row.timestamp_us > seen_nrps[row.nrp].timestamp_us
                ):
                    seen_nrps[row.nrp] = row
            return sorted(
                seen_nrps.values(),
                key=lambda r: r.timestamp_us,
                reverse=True,
            )

    # ------------------------------------------------------------------
    def strangers(self) -> List[AttendanceRow]:
        """Return only the STRANGER rows (deduplicated by resolved_label)."""
        with self._lock:
            seen_labels: Dict[str, AttendanceRow] = {}
            for row in self._cache.values():
                if row.state != "STRANGER":
                    continue
                if (
                    row.resolved_label not in seen_labels
                    or row.timestamp_us > seen_labels[row.resolved_label].timestamp_us
                ):
                    seen_labels[row.resolved_label] = row
            return sorted(
                seen_labels.values(),
                key=lambda r: r.timestamp_us,
                reverse=True,
            )

    # ------------------------------------------------------------------
    def anomalies(self) -> List[AttendanceRow]:
        """Return only the ANOMALY rows."""
        with self._lock:
            return sorted(
                [r for r in self._cache.values() if r.state == "ANOMALY"],
                key=lambda r: r.timestamp_us,
                reverse=True,
            )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "log_dir": self._log_dir,
                "prefix": self._prefix,
                "last_file_path": self._last_file_path,
                "last_file_size": self._last_file_size,
                "cache_size": len(self._cache),
                "max_rows": self._max_rows,
                "poll_count": self._poll_count,
                "parse_errors": self._parse_errors,
                "last_poll_age_us": (
                    int(time.time() * 1_000_000) - self._last_poll_at_us
                    if self._last_poll_at_us > 0 else -1
                ),
            }


# ============================================================================
# Stranger Gallery Scanner (Column 3) -- Patch 18 :: session-aware
# ============================================================================
class StrangerGalleryScanner:
    """
    Patch 18 :: Session-aware stranger gallery scanner.

    Walks the per-date+session hierarchy written by snap_strangers.py:

        {cache_dir}/{YYYY-MM-DD}/{6AM Session | 6PM Session}/...png

    Each PNG filename follows the snap_strangers.py convention:
        {ts_ms}_track{tid}_BIRTH.png
        {ts_ms}_track{tid}_STRANGER_{label}.png
        {ts_ms}_ANOMALY.png

    The scanner exposes:
      * `list_available_sessions()` -- enumerate sessions that have
        already started (today's 6PM Session is NOT listed before
        18:00 local; tomorrow's sessions are never listed).
      * `current_active_session()` -- the (date, label) of the session
        that should be selected by default in the UI.
      * `scan_session(date_str, session_label)` -- return gallery
        entries for a specific session.
      * `scan()` -- convenience wrapper that scans the active session.

    Gallery entries are grouped by stranger label (extracted from the
    filename). For each stranger the scanner selects the most recent
    snapshot as the thumbnail and computes a per-stranger temporal
    spread (variance of capture timestamps) as a proxy for movement.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        cache_dir: str,
        max_strangers: int = 24,
    ) -> None:
        self._cache_dir: str = os.path.abspath(cache_dir)
        self._max_strangers: int = int(max_strangers)
        self._lock: threading.RLock = threading.RLock()
        self._scan_count: int = 0
        self._last_scan_at_us: int = 0
        self._last_file_count: int = 0

        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "StrangerGalleryScanner: failed to create cache_dir %s: %s",
                self._cache_dir, exc,
            )

        logger.info(
            "StrangerGalleryScanner initialized (Patch 18 :: session-aware) | "
            "dir=%s | max=%d",
            self._cache_dir, self._max_strangers,
        )

    # ------------------------------------------------------------------
    def _session_dir(self, date_str: str, session_label: str) -> str:
        """Return the absolute path of the per-session subfolder."""
        return os.path.join(
            self._cache_dir, date_str, session_label_to_dir(session_label),
        )

    # ------------------------------------------------------------------
    def current_active_session(self) -> Tuple[str, str]:
        """Return the (date_str, session_label) of the currently-active session."""
        return compute_session_key(int(time.time() * 1_000_000))

    # ------------------------------------------------------------------
    def list_available_sessions(self) -> List[Tuple[str, str]]:
        """
        Enumerate all sessions visible on disk that have ALREADY STARTED.

        Returns:
            List of (date_str, session_label) tuples, sorted newest-first.

        A session qualifies for the list if:
          1. Its start time has passed (session_has_started returns True).
             Future sessions are NEVER listed -- e.g. at 08:00 local on
             2025-01-15, only the 2025-01-15/06AM session is listed (the
             2025-01-15/06PM session is hidden until 18:00 local).
          2. Either its folder exists on disk with at least one PNG, OR
             it is the currently-active session (so the operator can see
             the empty folder placeholder even before any strangers have
             been captured in the new session).

        Past sessions are kept in the list indefinitely for forensic
        review.
        """
        now_us = int(time.time() * 1_000_000)
        active_date, active_session = compute_session_key(now_us)

        # Discover all sessions that exist on disk.
        sessions_on_disk: List[Tuple[str, str]] = []
        if os.path.isdir(self._cache_dir):
            try:
                for date_entry in os.scandir(self._cache_dir):
                    if not date_entry.is_dir():
                        continue
                    # Validate the directory name looks like YYYY-MM-DD.
                    try:
                        _dt.date.fromisoformat(date_entry.name)
                    except ValueError:
                        continue
                    for sess_entry in os.scandir(date_entry.path):
                        if not sess_entry.is_dir():
                            continue
                        # Map the human-readable folder name back to a
                        # session label token.
                        if sess_entry.name == SESSION_DIR_AM:
                            label = SESSION_LABEL_AM
                        elif sess_entry.name == SESSION_DIR_PM:
                            label = SESSION_LABEL_PM
                        else:
                            continue
                        sessions_on_disk.append((date_entry.name, label))
            except OSError as exc:
                logger.warning(
                    "StrangerGalleryScanner: list_available_sessions scan "
                    "failed at %s: %s",
                    self._cache_dir, exc,
                )

        # Always include the currently-active session (even if its folder
        # does not yet exist on disk -- e.g. the session just rolled over
        # and no strangers have been captured yet).
        if (active_date, active_session) not in sessions_on_disk:
            sessions_on_disk.append((active_date, active_session))

        # Filter to sessions that have already started (hide future sessions).
        # Also implicitly hide sessions that have invalid date formats.
        visible = [
            (d, s) for (d, s) in sessions_on_disk
            if session_has_started(d, s)
        ]

        # Sort newest-first. A session is "newer" if its (date, session)
        # tuple represents a later wall-clock start time. We compute the
        # session start datetime for comparison.
        def _session_sort_key(item: Tuple[str, str]) -> float:
            d_str, s_label = item
            try:
                d = _dt.date.fromisoformat(d_str)
            except ValueError:
                return 0.0
            hour = SESSION_AM_START_HOUR if s_label == SESSION_LABEL_AM else SESSION_PM_START_HOUR
            start_dt = _dt.datetime.combine(d, _dt.time(hour=hour)).astimezone()
            return start_dt.timestamp()

        visible.sort(key=_session_sort_key, reverse=True)
        return visible

    # ------------------------------------------------------------------
    def scan_session(
        self,
        date_str: str,
        session_label: str,
    ) -> List[Dict[str, Any]]:
        """
        Scan a specific session folder and return gallery entries.

        Args:
            date_str: "YYYY-MM-DD" (the day the session started on).
            session_label: "06AM" or "06PM".

        Returns:
            List of gallery entry dicts (see class docstring) sorted by
            stranger_id ascending, capped at self._max_strangers.
        """
        self._last_scan_at_us = int(time.time() * 1_000_000)
        self._scan_count += 1

        session_dir = self._session_dir(date_str, session_label)
        if not os.path.isdir(session_dir):
            self._last_file_count = 0
            return []

        # Group files by stranger label (extracted from filename).
        # The snap_strangers.py filename conventions are:
        #   {ts_ms}_track{tid}_BIRTH.png
        #   {ts_ms}_track{tid}_STRANGER_{label}.png
        #   {ts_ms}_ANOMALY.png
        groups: Dict[str, List[Dict[str, Any]]] = {}
        total_files = 0
        try:
            for entry in os.scandir(session_dir):
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png"):
                    continue
                total_files += 1
                parsed = self._parse_filename(entry.name)
                if parsed is None:
                    continue
                label = parsed["label"]
                if label not in groups:
                    groups[label] = []
                parsed["abs_path"] = entry.path
                try:
                    parsed["file_size"] = entry.stat().st_size
                except OSError:
                    parsed["file_size"] = 0
                groups[label].append(parsed)
        except OSError as exc:
            logger.warning(
                "StrangerGalleryScanner: scan_session failed at %s: %s",
                session_dir, exc,
            )
            return []

        self._last_file_count = total_files

        # Build the gallery entries.
        entries: List[Dict[str, Any]] = []
        for label, snapshots in groups.items():
            if not snapshots:
                continue
            # Sort by timestamp (ascending).
            snapshots.sort(key=lambda s: s["ts_us"])
            first = snapshots[0]
            last = snapshots[-1]
            # Per-stranger temporal-spread score = variance of capture
            # timestamps in seconds^2. A higher score means the stranger
            # was visible across a longer time window (not just a one-off
            # frame glitch).
            ts_seconds = [s["ts_us"] / 1_000_000.0 for s in snapshots]
            if len(ts_seconds) >= 2:
                mean_ts = sum(ts_seconds) / len(ts_seconds)
                variance = sum((t - mean_ts) ** 2 for t in ts_seconds) / len(ts_seconds)
                temporal_spread = float(variance)
            else:
                temporal_spread = 0.0

            # Extract numeric stranger ID (0 for unresolved birth snapshots).
            sid = self._extract_stranger_id(label)

            # Patch 63 :: Build hourly buckets for the Extend Log feature.
            hourly: Dict[int, List[Dict[str, Any]]] = {}
            for snap in snapshots:
                hour = int(time.strftime("%H", time.localtime(snap["ts_us"] / 1_000_000.0)))
                hourly.setdefault(hour, []).append(snap)
            entries.append({
                "stranger_label": label,
                "stranger_id": sid,
                "thumbnail_path": last["abs_path"],
                "snapshot_count": len(snapshots),
                "first_seen_us": first["ts_us"],
                "last_seen_us": last["ts_us"],
                "temporal_spread_score": temporal_spread,
                "last_track_id": last.get("track_id", -1),
                "is_stranger": last.get("is_stranger", False),
                # Patch 63 :: Full snapshot list + hourly buckets for Extend Log.
                "snapshots": list(snapshots),
                "hourly_buckets": hourly,
            })

        # Sort by stranger_id ascending and cap.
        # Patch 63 (hotfix K) :: max_strangers == 0 means UNLIMITED
        # (config knob for environments that want zero truncation).
        entries.sort(key=lambda e: e["stranger_id"])
        if self._max_strangers > 0:
            return entries[: self._max_strangers]
        return entries

    # ------------------------------------------------------------------
    def scan_session_verified(self, date_str: str, session_label: str) -> List[Dict[str, Any]]:
        """Patch 63 :: Scan the identified/ subfolder for verified-person snapshots.

        Returns a list of gallery entries (same shape as scan_session),
        but for verified students. Each entry includes the full per-snapshot
        list (not just the thumbnail) so the Event Log page can render
        hourly history when the user clicks "Extend Log".
        """
        session_dir = self._session_dir(date_str, session_label)
        identified_dir = os.path.join(session_dir, "identified")
        if not os.path.isdir(identified_dir):
            return []

        try:
            files = sorted(os.listdir(identified_dir))
        except OSError as exc:
            logger.warning(
                "StrangerGalleryScanner: scan_session_verified failed at %s: %s",
                identified_dir, exc,
            )
            return []

        # Parse all files and group by student label.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for filename in files:
            if not filename.lower().endswith(".png"):
                continue
            parsed = self._parse_filename(filename)
            if parsed is None:
                continue
            if not parsed.get("is_verified"):
                continue
            abs_path = os.path.join(identified_dir, filename)
            entry = {**parsed, "abs_path": abs_path}
            label = parsed["label"]
            groups.setdefault(label, []).append(entry)

        # Build gallery entries with full snapshot lists + hourly buckets.
        entries: List[Dict[str, Any]] = []
        for label, snapshots in groups.items():
            if not snapshots:
                continue
            snapshots.sort(key=lambda s: s["ts_us"])
            first = snapshots[0]
            last = snapshots[-1]
            # Hourly buckets: {hour_int: [snapshot, ...]}
            hourly: Dict[int, List[Dict[str, Any]]] = {}
            for snap in snapshots:
                hour = int(time.strftime("%H", time.localtime(snap["ts_us"] / 1_000_000.0)))
                hourly.setdefault(hour, []).append(snap)
            entries.append({
                "student_label": label,
                "thumbnail_path": last["abs_path"],
                "snapshot_count": len(snapshots),
                "first_seen_us": first["ts_us"],
                "last_seen_us": last["ts_us"],
                "last_track_id": last.get("track_id", -1),
                "snapshots": snapshots,
                "hourly_buckets": hourly,
                "is_verified": True,
            })

        # Sort by first_seen ascending.
        entries.sort(key=lambda e: e["first_seen_us"])
        return entries

    # ------------------------------------------------------------------
    def scan(self) -> List[Dict[str, Any]]:
        """
        Convenience: scan the currently-active session.

        Equivalent to:
            scan_session(*self.current_active_session())
        """
        date_str, session_label = self.current_active_session()
        return self.scan_session(date_str, session_label)

    # ------------------------------------------------------------------
    def _parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a snap_strangers.py snapshot filename into its components.

        Supported formats (Patch 18 + Patch 63 + Patch 65):
            {ts_ms}_track{tid}_BIRTH.png
            {ts_ms}_track{tid}_STRANGER_{label}.png
            {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY}.png   (Patch 65)
            {ts_ms}_track{tid}_VERIFIED_{label}.png                   (Patch 63)
            {ts_ms}_ANOMALY.png

        Returns None if the filename does not match any of these.
        """
        import re

        # Patch 65 :: Format 0 (check FIRST -- non-greedy label capture):
        #   {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY}.png
        #
        # Clearshots are additional periodic snapshots captured for
        # STRANGER-locked tracks when YOLO confidence is high. They
        # serve as OSNet "memory recall" reference frames. The YY
        # counter is 1-indexed per stranger (01, 02, 03, ...) so the
        # operator can browse the sequence.
        #
        # MUST be matched BEFORE Format 1 because Format 1's greedy
        # (.+) would otherwise swallow the _CLEARSHOT_YY suffix into
        # the label group. Using non-greedy (.+?) here ensures the
        # label stops at the first _CLEARSHOT_ occurrence.
        m = re.match(
            r"^(\d+)_track(\d+)_STRANGER_(.+?)_CLEARSHOT_(\d+)\.png$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            ts_ms = int(m.group(1))
            track_id = int(m.group(2))
            raw_label = m.group(3)
            clearshot_idx = int(m.group(4))
            # The label stored in the filename is the sanitized stranger
            # label (e.g. "Stranger_07"). Recover the stranger_id.
            sid_match = re.search(r"Stranger_(\d+)", raw_label)
            sid = int(sid_match.group(1)) if sid_match else 0
            return {
                "label": f"{raw_label}_CLEARSHOT_{clearshot_idx:02d}",
                "stranger_id": sid,
                "track_id": track_id,
                "ts_us": ts_ms * 1000,   # convert ms -> us
                "is_stranger": True,
                "is_clearshot": True,
                "clearshot_idx": clearshot_idx,
                # Display label = the parent stranger (so clearshots can
                # be grouped under their stranger in the gallery).
                "parent_label": raw_label,
            }

        # Format 1: {ts_ms}_track{tid}_STRANGER_{label}.png
        m = re.match(
            r"^(\d+)_track(\d+)_STRANGER_(.+)\.png$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            ts_ms = int(m.group(1))
            track_id = int(m.group(2))
            raw_label = m.group(3)
            # The label stored in the filename is the sanitized stranger
            # label (e.g. "Stranger_07"). Recover the stranger_id.
            sid_match = re.search(r"Stranger_(\d+)", raw_label)
            sid = int(sid_match.group(1)) if sid_match else 0
            return {
                "label": raw_label,
                "stranger_id": sid,
                "track_id": track_id,
                "ts_us": ts_ms * 1000,   # convert ms -> us
                "is_stranger": True,
            }

        # Patch 63 :: Format 1b: {ts_ms}_track{tid}_VERIFIED_{label}.png
        # (moved into identified/ subfolder by SnapStrangersEngine).
        m = re.match(
            r"^(\d+)_track(\d+)_VERIFIED_(.+)\.png$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            ts_ms = int(m.group(1))
            track_id = int(m.group(2))
            raw_label = m.group(3)
            return {
                "label": raw_label,
                "stranger_id": 0,
                "track_id": track_id,
                "ts_us": ts_ms * 1000,
                "is_stranger": False,
                "is_verified": True,
            }

        # Format 2: {ts_ms}_track{tid}_BIRTH.png
        m = re.match(
            r"^(\d+)_track(\d+)_BIRTH\.png$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            ts_ms = int(m.group(1))
            track_id = int(m.group(2))
            return {
                "label": f"Pending_Track_{track_id}",
                "stranger_id": 0,
                "track_id": track_id,
                "ts_us": ts_ms * 1000,
                "is_stranger": False,
            }

        # Format 3: {ts_ms}_ANOMALY.png (and any suffix variant)
        m = re.match(
            r"^(\d+)_ANOMALY.*\.png$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            ts_ms = int(m.group(1))
            return {
                "label": "ANOMALY",
                "stranger_id": -1,
                "track_id": -1,
                "ts_us": ts_ms * 1000,
                "is_stranger": False,
            }

        # Legacy format (pre-Patch 18): Stranger_{NN}_frame_{FFFFFFF}_ts_{TTTTTTTT}.jpg
        # Kept for backward compat with snapshots captured before the
        # snap_strangers migration.
        m = re.match(
            r"^(Stranger_(\d+))_frame_(\d+)_ts_(\d+)\.(?:jpg|jpeg|png)$",
            filename,
            re.IGNORECASE,
        )
        if m is not None:
            return {
                "label": m.group(1),
                "stranger_id": int(m.group(2)),
                "track_id": -1,
                "ts_us": int(m.group(4)),
                "is_stranger": True,
            }

        return None

    # ------------------------------------------------------------------
    def _extract_stranger_id(self, label: str) -> int:
        """Extract the numeric ID from 'Stranger_NN' (or 0 for pending)."""
        import re
        m = re.search(r"Stranger_(\d+)", label)
        if m is not None:
            return int(m.group(1))
        # Pending_Track_{N} -> use a high offset so they sort after
        # finalized strangers in the gallery.
        m = re.search(r"Track_(\d+)", label)
        if m is not None:
            return 10000 + int(m.group(1))
        return 0

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "cache_dir": self._cache_dir,
            "max_strangers": self._max_strangers,
            "scan_count": self._scan_count,
            "last_file_count": self._last_file_count,
            "last_scan_age_us": (
                int(time.time() * 1_000_000) - self._last_scan_at_us
                if self._last_scan_at_us > 0 else -1
            ),
        }


# ============================================================================
# Streamlit Application (Entry Point)
# ============================================================================
def _render_metric_card(
    container: Any,
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """Render a single metric card inside a Streamlit container."""
    if not _STREAMLIT_AVAILABLE:
        return
    try:
        container.metric(label=label, value=value, delta=delta, help=help_text)
    except Exception as exc:                    # pragma: no cover
        logger.warning("render_metric_card failed for %s: %s", label, exc)


def _render_performance_column(
    container: Any,
    receiver: UDPTelemetryReceiver,
) -> None:
    """Render Column 1: Performance Monitor."""
    if not _STREAMLIT_AVAILABLE:
        return

    container.subheader("Performance Monitor")

    latest = receiver.latest()
    rx_telemetry = receiver.telemetry()
    packets_received = int(rx_telemetry.get("packets_received", 0))
    last_age_us = int(rx_telemetry.get("last_packet_age_us", -1))
    last_age_s = last_age_us / 1_000_000.0 if last_age_us > 0 else -1.0

    if latest is None:
        # Patch 22 :: Distinguish "never received" from "went stale".
        if packets_received == 0:
            # Patch 63 (hotfix G) :: Enhanced with receiver diagnostics.
            _rx_running = receiver._running
            _rx_shutdown = receiver._shutdown
            _thread_alive = (
                receiver._thread is not None
                and receiver._thread.is_alive()
            )
            _decode_errs = int(rx_telemetry.get("decode_errors", 0))
            container.warning(
                "main.py has not sent any telemetry yet.\n\n"
                f"Listening on UDP `{receiver._host}:{receiver._port}`.\n\n"
                f"**Receiver diagnostics:**\n"
                f"- Thread running: `{_rx_running}`\n"
                f"- Thread alive: `{_thread_alive}`\n"
                f"- Shutdown flag: `{_rx_shutdown}`\n"
                f"- Packets received: `{packets_received}`\n"
                f"- Decode errors: `{_decode_errs}`\n\n"
                "**Possible causes:**\n"
                "1. main.py is NOT running -- launch it with `python main.py`\n"
                "2. main.py just started -- wait 2-3 seconds for the heartbeat thread\n"
                "3. main.py crashed during startup -- check its console for errors\n"
                "4. Firewall blocking UDP 127.0.0.1:9999\n"
                "5. **Stale process holding port 9999** -- kill ALL python.exe "
                "and streamlit.exe in Task Manager, then restart both "
                "run_main.bat and run_dashboard.bat\n\n"
                "The heartbeat thread starts in the orchestrator's __init__, "
                "so you should see the first packet within 1-2 seconds of "
                "main.py launching."
            )
        else:
            container.error(
                f"main.py appears OFFLINE -- last packet received "
                f"{last_age_s:.1f}s ago (packets_received={packets_received}).\n\n"
                "The dashboard socket is still listening, but main.py "
                "has stopped broadcasting. Check main.py's console for "
                "errors or crashes."
            )
        container.caption(f"Receiver: {rx_telemetry}")
        return

    # Patch 63 (hotfix C) :: Heartbeat status banner.
    # The heartbeat field now carries the ORCHESTRATOR state (not just
    # the AI thread state). States:
    #   BOOTING                : orchestrator __init__ just started
    #   INITIALIZING           : initialize() entered
    #   INITIALIZING_LOGGER    : loading async logger
    #   INITIALIZING_SNAP_ENGINE: loading stranger snapshot engine
    #   INITIALIZING_RES_OPT   : loading resource optimizer
    #   INITIALIZING_TRACKING  : loading YOLO + BoTSORT (heaviest, 5-15s)
    #   INITIALIZING_IDENTITY  : loading identity matcher
    #   INITIALIZING_FACES     : loading InsightFace + ArcFace (5-10s)
    #   INITIALIZING_GATING    : loading gating engine
    #   INITIALIZED            : all engines loaded, ready to start
    #   STARTING_THREADS       : starting camera + AI threads
    #   RUNNING                : normal operation (no banner)
    #   IDLE                   : AI thread alive but no camera frames
    #   LIVE                   : AI thread processing frames normally
    #   ERROR                  : AI thread crashing every cycle
    #   CRASHED                : orchestrator crashed, auto-restarting
    #   SHUTTING_DOWN          : graceful shutdown in progress
    _hb = getattr(latest, "heartbeat", "LIVE")
    _ai_err = getattr(latest, "ai_errors", 0)
    _frames_proc = getattr(latest, "frames_processed", 0)

    # --- Orchestrator-level states (HeartbeatThread) ---
    if _hb in ("BOOTING", "INITIALIZING", "INITIALIZED", "STARTING_THREADS"):
        _phase_msg = {
            "BOOTING": "constructing orchestrator",
            "INITIALIZING": "entering initialize()",
            "INITIALIZED": "all engines loaded, starting threads",
            "STARTING_THREADS": "starting camera + AI threads",
        }.get(_hb, _hb)
        container.info(
            f"main.py is **BOOTING** -- {_phase_msg}. "
            f"This is normal during startup (5-30s). "
            f"(packets_received={packets_received})"
        )
    elif _hb.startswith("INITIALIZING_"):
        _engine_msg = {
            "INITIALIZING_LOGGER": "async logger",
            "INITIALIZING_SNAP_ENGINE": "stranger snapshot engine",
            "INITIALIZING_RES_OPT": "resource optimizer",
            "INITIALIZING_TRACKING": "YOLO + BoTSORT tracking (heaviest, 5-15s)",
            "INITIALIZING_IDENTITY": "identity matcher",
            "INITIALIZING_FACES": "InsightFace + ArcFace (5-10s)",
            "INITIALIZING_GATING": "gating engine",
        }.get(_hb, _hb)
        container.info(
            f"main.py is **LOADING {_engine_msg}**... "
            f"If this persists for > 30s, check main.py's console for "
            f"a hang or crash during model loading."
        )
    elif _hb == "CRASHED":
        container.error(
            f"main.py **CRASHED** -- the orchestrator caught an uncaught "
            f"exception and is auto-restarting. Check main.py's console "
            f"log for the traceback. The dashboard will receive new "
            f"heartbeats once the restart completes (~2s)."
        )
    elif _hb == "SHUTTING_DOWN":
        container.warning(
            f"main.py is **SHUTTING DOWN** -- graceful shutdown in progress."
        )
    # --- AI-thread-level states ---
    elif _hb == "STARTUP":
        container.info(
            f"main.py is **STARTING UP** -- AI thread is alive and "
            f"loading models. No frames processed yet "
            f"(packets_received={packets_received})."
        )
    elif _hb == "IDLE":
        container.warning(
            f"main.py is **IDLE** -- AI thread is running but the camera "
            f"is not producing frames. Check: is the camera connected? "
            f"Is the RTSP URL correct? (idle since frame {_frames_proc}, "
            f"packets_received={packets_received})"
        )
    elif _hb == "ERROR":
        container.error(
            f"main.py is **CRASHING** -- _process_frame has raised "
            f"{_ai_err} exception(s). The AI thread is still alive and "
            f"broadcasting heartbeats, but no frames are being processed. "
            f"Check main.py's console log for the traceback."
        )
    # LIVE / RUNNING: no banner, proceed to show normal metrics.

    # Patch 22 :: Stale-packet warning (main.py still alive but lagging).
    if last_age_s > 5.0:
        container.error(
            f"main.py appears OFFLINE -- last packet received "
            f"{last_age_s:.1f}s ago. Showing last known data below."
        )

    # Patch 22 :: FPS cards removed per operator request.
    # Top-row metric cards (AI Queue + Throttle only).
    c1, c2 = container.columns(2)
    _render_metric_card(
        c1, "AI Queue", f"{latest.ai_queue_depth}/{latest.ai_queue_maxsize}",
        delta=None, help_text="Strict maxsize=2 policy",
    )
    _render_metric_card(
        c2, "Throttle", latest.throttle_mode,
        delta=None, help_text="IDLE=25FPS cap / BURST=max",
    )

    container.divider()

    # Latency cards.
    l1, l2, l3 = container.columns(3)
    _render_metric_card(
        l1, "Processing Latency", f"{latest.processing_latency_ms:.1f} ms",
    )
    _render_metric_card(
        l2, "Tracking Latency", f"{latest.tracking_latency_ms:.1f} ms",
    )
    _render_metric_card(
        l3, "GUI Latency", f"{latest.gui_latency_ms:.1f} ms",
    )

    container.divider()

    # Track counts.
    t1, t2, t3, t4 = container.columns(4)
    _render_metric_card(t1, "Active", str(latest.active_track_count))
    _render_metric_card(t2, "Pending", str(latest.pending_track_count))
    _render_metric_card(t3, "Verified", str(latest.verified_track_count))
    _render_metric_card(t4, "Strangers", str(latest.stranger_track_count))

    # GPU VRAM.
    if latest.gpu_vram_total_bytes > 0:
        used_mb = latest.gpu_vram_used_bytes / (1024.0 * 1024.0)
        total_mb = latest.gpu_vram_total_bytes / (1024.0 * 1024.0)
        pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
        container.progress(
            value=min(1.0, max(0.0, pct / 100.0)),
            text=f"GPU VRAM: {used_mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)",
        )

    # Thread affinity scopes.
    container.caption("Thread CPU Affinity Scopes")
    if latest.thread_affinity:
        aff_data = [
            {"Thread": k, "Cores": ", ".join(str(c) for c in v)}
            for k, v in latest.thread_affinity.items()
        ]
        try:
            container.table(aff_data)
        except Exception as exc:                # pragma: no cover
            container.warning(f"Failed to render affinity table: {exc}")
    else:
        container.caption("(no affinity scopes broadcast)")

    # Anomaly count + encoder kind.
    cap_row = container.columns(2)
    cap_row[0].caption(f"Encoder: `{latest.encoder_kind}`")
    cap_row[1].caption(f"Anomalies: **{latest.anomaly_count}**")

    # Patch 20 :: Surface the orchestrator's broadcasted active session.
    # The orchestrator's SessionBoundaryWatcher stamps every telemetry
    # packet with the (label, date) it is currently writing into. We show
    # that here so the operator can confirm the orchestrator and the
    # dashboard agree on the active 12-hour session. If the field is
    # empty (older orchestrator build that does not broadcast it), we
    # fall back to computing the session locally.
    if latest.current_session_label and latest.current_session_date:
        orch_dir = session_label_to_dir(latest.current_session_label)
        cap_row2 = container.columns(2)
        cap_row2[0].caption(
            f"Orchestrator session: **{latest.current_session_date} / {orch_dir}**"
        )
        # Cross-check against the locally-computed active session.
        local_date, local_label = compute_session_key(int(time.time() * 1_000_000))
        if (
            local_date == latest.current_session_date
            and local_label == latest.current_session_label
        ):
            cap_row2[1].caption("Local clock: **in sync**")
        else:
            cap_row2[1].warning(
                f"Local clock mismatch (local: {local_date} {local_label}). "
                "Boundary may have just crossed; watcher will fire shortly."
            )
    else:
        # Fallback for older orchestrator builds without Patch 20 fields.
        local_date, local_label = compute_session_key(int(time.time() * 1_000_000))
        container.caption(
            f"Active session (local): **{local_date} / "
            f"{session_label_to_dir(local_label)}** "
            "(orchestrator did not broadcast session fields)"
        )

    # Receiver telemetry footer.
    container.caption(f"Receiver: {receiver.telemetry()}")


def _filter_verified_rows(rows: List[AttendanceRow]) -> List[AttendanceRow]:
    """Patch 20 :: Dedup-by-NRP filter for an arbitrary row list."""
    seen: Dict[str, AttendanceRow] = {}
    for row in rows:
        if row.state != "VERIFIED_STUDENT" or not row.nrp:
            continue
        if (
            row.nrp not in seen
            or row.timestamp_us > seen[row.nrp].timestamp_us
        ):
            seen[row.nrp] = row
    return sorted(seen.values(), key=lambda r: r.timestamp_us, reverse=True)


def _filter_stranger_rows(rows: List[AttendanceRow]) -> List[AttendanceRow]:
    """Patch 20 :: Dedup-by-label filter for an arbitrary row list."""
    seen: Dict[str, AttendanceRow] = {}
    for row in rows:
        if row.state != "STRANGER":
            continue
        if (
            row.resolved_label not in seen
            or row.timestamp_us > seen[row.resolved_label].timestamp_us
        ):
            seen[row.resolved_label] = row
    return sorted(seen.values(), key=lambda r: r.timestamp_us, reverse=True)


def _filter_anomaly_rows(rows: List[AttendanceRow]) -> List[AttendanceRow]:
    """Patch 20 :: Anomaly filter for an arbitrary row list."""
    return sorted(
        [r for r in rows if r.state == "ANOMALY"],
        key=lambda r: r.timestamp_us,
        reverse=True,
    )


def _render_attendance_column(
    container: Any,
    poller: CSVAttendancePoller,
) -> None:
    """Render Column 2: Live Attendance Journal.

    Patch 20 :: Adds a session selector dropdown at the top of the
    column. When the active session is selected, behavior is unchanged
    (poller.poll() + cached accessors). When a historical session is
    selected, the column calls poller.poll_session(date, label) and
    filters the returned rows via the module-level _filter_* helpers
    so the operator can review any past 12-hour session without
    disturbing the live poller cache.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    container.subheader("Live Attendance Journal")

    # Patch 18 :: Show the active 12h session label in the header.
    active_date, active_session = compute_session_key(int(time.time() * 1_000_000))
    active_session_dir = session_label_to_dir(active_session)
    container.caption(
        f"Active session: **{active_date} / {active_session_dir}** "
        f"(LOCAL time, rotates at 06:00 / 18:00)"
    )

    # ---- Patch 20 :: Session selector -------------------------------------
    # List only sessions whose CSV exists on disk AND that have already
    # started. Future sessions are hidden (same rule as the stranger
    # gallery). The active session is always selectable even if its CSV
    # does not yet exist (just-rolled-over edge case).
    available_sessions = poller.list_available_sessions()

    session_options: List[str] = []
    session_keys: List[Tuple[str, str]] = []
    default_index = 0
    for idx, (d, s) in enumerate(available_sessions):
        is_active = (d == active_date and s == active_session)
        sess_dir_name = session_label_to_dir(s)
        label_str = f"{d} — {sess_dir_name}" + (" (active)" if is_active else "")
        session_options.append(label_str)
        session_keys.append((d, s))
        if is_active:
            default_index = idx

    if not session_options:
        # Defensive -- list_available_sessions always returns at least
        # the active session.
        container.warning("No sessions available.")
        return

    # ----------------------------------------------------------------
    # Patch 46 :: Auto-advance session selector on rotation.
    # ----------------------------------------------------------------
    # The selectbox widget with key="attendance_session_select" stores
    # its selected VALUE in st.session_state[key]. Once set, the index=
    # parameter is IGNORED on subsequent renders. When the session
    # rotates, the old value becomes stale (the active session's label
    # changes -- "(active)" suffix moves to the new session). We detect
    # this by tracking the active session key separately; when it
    # changes, we DELETE the widget key from session_state so Streamlit
    # falls back to the index= parameter, and reset the selected index
    # to the new active session.
    # ----------------------------------------------------------------
    _active_session_key = f"{active_date}_{active_session}"
    _last_active_key = st.session_state.get(
        "attendance_last_active_session_key", None
    )
    if _last_active_key is not None and _last_active_key != _active_session_key:
        # Session rotated since the last render. Force the selector to
        # switch to the new active session.
        logger.info(
            "Dashboard: attendance session rotation detected -- "
            "auto-advancing selector. Old: %s | New: %s",
            _last_active_key, _active_session_key,
        )
        # Delete the widget key so Streamlit doesn't reuse the stale
        # value (which is no longer in the options list).
        if "attendance_session_select" in st.session_state:
            del st.session_state["attendance_session_select"]
        # Reset the selected index to the new active session.
        st.session_state.attendance_selected_session = default_index
    st.session_state.attendance_last_active_session_key = _active_session_key

    # Persist the selected session across reruns.
    if "attendance_selected_session" not in st.session_state:
        st.session_state.attendance_selected_session = default_index
    cached_idx = st.session_state.attendance_selected_session
    if cached_idx < 0 or cached_idx >= len(session_options):
        cached_idx = default_index
        st.session_state.attendance_selected_session = default_index

    selected_label = container.selectbox(
        "Session",
        options=session_options,
        index=cached_idx,
        key="attendance_session_select",
        help=(
            "Sessions rotate at LOCAL 06:00 and 18:00. The selector "
            "auto-advances to the new active session on rotation. "
            "Past sessions are kept for forensic review."
        ),
    )

    try:
        selected_idx = session_options.index(selected_label)
    except ValueError:
        selected_idx = default_index
    st.session_state.attendance_selected_session = selected_idx
    selected_date, selected_session = session_keys[selected_idx]

    is_viewing_active = (
        selected_date == active_date and selected_session == active_session
    )
    if is_viewing_active:
        container.caption(
            f"Viewing the **currently-active** session: "
            f"**{selected_date} / {session_label_to_dir(selected_session)}**"
        )
    else:
        container.caption(
            f"Viewing **historical** session: "
            f"**{selected_date} / {session_label_to_dir(selected_session)}** "
            f"(active session is {active_date} / "
            f"{session_label_to_dir(active_session)})"
        )

    # ---- Fetch rows for the selected session ------------------------------
    if is_viewing_active:
        # Live path -- uses the cached poller. verified_students() /
        # strangers() / anomalies() read from the poller's cache.
        rows = poller.poll()
        if not rows:
            container.info(
                f"No attendance records yet for the current 12h session "
                f"({active_date} {active_session}).\n\n"
                f"Watching: `{poller._log_dir}/{poller._prefix}_*.csv`",
            )
            container.caption(f"Poller: {poller.telemetry()}")
            return

        verified_rows = poller.verified_students()
        stranger_rows = poller.strangers()
        anomaly_rows = poller.anomalies()
    else:
        # Historical path -- poll_session() returns a fresh local list,
        # leaving the live cache untouched. Filter inline via the
        # module-level _filter_* helpers.
        rows = poller.poll_session(selected_date, selected_session)
        if not rows:
            container.info(
                f"No attendance records in this historical session "
                f"({selected_date} {selected_session}).\n\n"
                f"Path: `{poller._resolve_path_for_session(selected_date, selected_session)}`",
            )
            container.caption(f"Poller: {poller.telemetry()}")
            return

        verified_rows = _filter_verified_rows(rows)
        stranger_rows = _filter_stranger_rows(rows)
        anomaly_rows = _filter_anomaly_rows(rows)

    # Summary metrics.
    s1, s2, s3 = container.columns(3)
    _render_metric_card(s1, "Verified Students", str(len(verified_rows)))
    _render_metric_card(s2, "Strangers", str(len(stranger_rows)))
    _render_metric_card(s3, "Anomalies", str(len(anomaly_rows)))

    container.divider()

    # Verified students table.
    if verified_rows:
        container.markdown("#### Verified Students")
        table_data = [
            {
                "NRP": r.nrp,
                "Name": r.student_name,
                "Status": r.resolved_label,
                "Sim": f"{r.similarity_score:.3f}",
                # Patch 26 :: 'Frame #' column removed per operator
                # request ("we haven't removed the #frame as well
                # on the GUI").
                # Patch 18 :: Display LOCAL time (not UTC) -- the CSV
                # writer now rotates per 12h LOCAL session, so showing
                # UTC would be misleading.
                "Time (Local)": _dt.datetime.fromtimestamp(
                    r.timestamp_us / 1_000_000.0,
                ).astimezone().strftime("%H:%M:%S.%f")[:-3],
            }
            for r in verified_rows[:50]
        ]
        try:
            container.table(table_data)
        except Exception as exc:                # pragma: no cover
            container.warning(f"Failed to render verified table: {exc}")
    else:
        container.caption("_No verified students in this session._")

    container.divider()

    # Strangers + Anomalies summary.
    if stranger_rows:
        container.markdown("#### Strangers (most recent first)")
        stranger_data = [
            {
                "Label": r.resolved_label,
                "Sim": f"{r.similarity_score:.3f}",
                # Patch 26 :: 'Frame #' column removed.
                "Time (Local)": _dt.datetime.fromtimestamp(
                    r.timestamp_us / 1_000_000.0,
                ).astimezone().strftime("%H:%M:%S.%f")[:-3],
            }
            for r in stranger_rows[:20]
        ]
        try:
            container.table(stranger_data)
        except Exception as exc:                # pragma: no cover
            container.warning(f"Failed to render strangers table: {exc}")

    if anomaly_rows:
        container.markdown("#### Anomalies (face without body)")
        anomaly_data = [
            {
                "Label": r.resolved_label,
                # Patch 26 :: 'Frame #' column removed.
                "Time (Local)": _dt.datetime.fromtimestamp(
                    r.timestamp_us / 1_000_000.0,
                ).astimezone().strftime("%H:%M:%S.%f")[:-3],
                "BBox": f"({r.bbox_x1},{r.bbox_y1})-({r.bbox_x2},{r.bbox_y2})",
            }
            for r in anomaly_rows[:20]
        ]
        try:
            container.table(anomaly_data)
        except Exception as exc:                # pragma: no cover
            container.warning(f"Failed to render anomaly table: {exc}")

    container.caption(f"Poller: {poller.telemetry()}")


# ============================================================================
# Patch 63 (hotfix K) :: _render_snapshot_grid helper.
#
# Renders a list of snapshot dicts (as produced by
# StrangerGalleryScanner._parse_filename) as a grid of small thumbnails.
# Each thumbnail is shown with a timestamp caption. Corrupt/partial PNGs
# are skipped silently (a caption is shown in place of the image) so one
# bad file doesn't break the entire grid.
#
# Used by _render_stranger_gallery_column to show ALL snapshots per
# stranger (not just the latest). Also reused by the Event Log page.
# ============================================================================
def _render_snapshot_grid(
    snapshots: List[Dict[str, Any]],
    columns: int = 3,
    thumb_px: int = 160,
) -> None:
    """Render a grid of snapshot thumbnails.

    Args:
        snapshots: List of snapshot dicts from StrangerGalleryScanner.
            Each must have: abs_path, ts_us, track_id.
        columns: Number of thumbnails per row.
        thumb_px: Max dimension of each thumbnail in pixels (downscaled
            via LANCZOS for display; the full PNG stays on disk).
    """
    if not _STREAMLIT_AVAILABLE or not snapshots:
        return

    # Render in rows of `columns` thumbnails each.
    for row_start in range(0, len(snapshots), columns):
        row_snaps = snapshots[row_start:row_start + columns]
        cols = st.columns(columns)
        for col, snap in zip(cols, row_snaps):
            _path = snap.get("abs_path", "")
            _ts_us = snap.get("ts_us", 0)
            _tid = snap.get("track_id", -1)
            # Format timestamp as HH:MM:SS (local time).
            try:
                _time_str = _dt.datetime.fromtimestamp(
                    _ts_us / 1_000_000.0,
                ).astimezone().strftime('%H:%M:%S')
            except Exception:
                _time_str = "??"

            if not _PIL_AVAILABLE or not os.path.exists(_path):
                col.caption(f"(missing) {_time_str}")
                continue

            # Patch 47 defense layers (same as the latest-thumbnail path):
            #   1. File-size gate (< 1 KB = still being written)
            #   2. Image.verify() (catchable corruption check)
            #   3. img.load() (force decode inside try/except)
            try:
                _fsize = os.path.getsize(_path)
                if _fsize < 1024:
                    col.caption(f"(writing...) {_time_str}")
                    continue
                with open(_path, "rb") as _vf:
                    _verif = Image.open(_vf)
                    _verif.verify()
                _img = Image.open(_path)
                _img.load()
                # Downscale for grid display.
                if max(_img.size) > thumb_px:
                    _ratio = thumb_px / max(_img.size)
                    _new_size = (
                        int(_img.size[0] * _ratio),
                        int(_img.size[1] * _ratio),
                    )
                    _img = _img.resize(_new_size, Image.LANCZOS)
                col.image(
                    _img,
                    caption=f"{_time_str} (track #{_tid})",
                    use_container_width=True,
                )
            except Exception as exc:
                col.caption(f"(error: {exc}) {_time_str}")


def _render_stranger_gallery_column(
    container: Any,
    scanner: StrangerGalleryScanner,
) -> None:
    """
    Render Column 3: Stranger Alert Gallery (Patch 18 :: session-aware).

    Adds a session selector dropdown at the top of the column. The
    dropdown lists all sessions that have already started (today's
    6PM Session is NOT listed before 18:00 local; tomorrow's sessions
    are never listed). The currently-active session is selected by
    default and marked with an "(active)" tag.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    container.subheader("Stranger Alert Gallery")

    # ---- Patch 18 :: Session selector -------------------------------------
    # List only sessions that have already started. Future sessions are
    # hidden per the operator's request: "only make dashboard showing
    # 6AM Session if it not hits 6PM yet, and vice versa".
    available_sessions = scanner.list_available_sessions()
    active_date, active_session = scanner.current_active_session()

    # Build the dropdown options. Each option is a human-readable label
    # like "2025-01-15 — 6AM Session (active)" or "2025-01-14 — 6PM Session".
    # We keep a parallel list of (date, label) tuples so we can recover
    # the selection.
    session_options: List[str] = []
    session_keys: List[Tuple[str, str]] = []
    default_index = 0
    for idx, (d, s) in enumerate(available_sessions):
        is_active = (d == active_date and s == active_session)
        sess_dir_name = session_label_to_dir(s)
        label_str = f"{d} — {sess_dir_name}" + (" (active)" if is_active else "")
        session_options.append(label_str)
        session_keys.append((d, s))
        if is_active:
            default_index = idx

    if not session_options:
        # Defensive -- list_available_sessions always returns at least
        # the active session, but guard anyway.
        container.warning("No sessions available.")
        return

    # ----------------------------------------------------------------
    # Patch 46 :: Auto-advance session selector on rotation.
    # Same logic as the attendance column -- see the comment there.
    # ----------------------------------------------------------------
    _active_session_key = f"{active_date}_{active_session}"
    _last_active_key = st.session_state.get(
        "stranger_gallery_last_active_session_key", None
    )
    if _last_active_key is not None and _last_active_key != _active_session_key:
        logger.info(
            "Dashboard: stranger gallery session rotation detected -- "
            "auto-advancing selector. Old: %s | New: %s",
            _last_active_key, _active_session_key,
        )
        if "stranger_gallery_session_select" in st.session_state:
            del st.session_state["stranger_gallery_session_select"]
        st.session_state.stranger_gallery_selected_session = default_index
    st.session_state.stranger_gallery_last_active_session_key = _active_session_key

    # Persist the selected session across reruns. If the previously
    # selected session is no longer in the list (e.g. it was hidden by
    # the future-session filter on a future date -- shouldn't happen
    # but just in case), fall back to the active session.
    if "stranger_gallery_selected_session" not in st.session_state:
        st.session_state.stranger_gallery_selected_session = default_index
    # Clamp the cached index to the current list bounds.
    cached_idx = st.session_state.stranger_gallery_selected_session
    if cached_idx < 0 or cached_idx >= len(session_options):
        cached_idx = default_index
        st.session_state.stranger_gallery_selected_session = default_index

    selected_label = container.selectbox(
        "Session",
        options=session_options,
        index=cached_idx,
        key="stranger_gallery_session_select",
        help=(
            "Sessions rotate at LOCAL 06:00 and 18:00. The selector "
            "auto-advances to the new active session on rotation. "
            "Past sessions are kept for forensic review."
        ),
    )

    # Recover the (date, label) tuple for the selected option.
    try:
        selected_idx = session_options.index(selected_label)
    except ValueError:
        selected_idx = default_index
    st.session_state.stranger_gallery_selected_session = selected_idx
    selected_date, selected_session = session_keys[selected_idx]

    # Show a small active/historical banner.
    if selected_date == active_date and selected_session == active_session:
        container.caption(
            f"Viewing the **currently-active** session: "
            f"**{selected_date} / {session_label_to_dir(selected_session)}**"
        )
    else:
        container.caption(
            f"Viewing **historical** session: "
            f"**{selected_date} / {session_label_to_dir(selected_session)}** "
            f"(active session is {active_date} / {session_label_to_dir(active_session)})"
        )

    # ---- Scan the selected session ---------------------------------------
    entries = scanner.scan_session(selected_date, selected_session)
    if not entries:
        container.info(
            f"No stranger snapshots in this session yet.\n\n"
            f"Watching: `{scanner._session_dir(selected_date, selected_session)}`",
        )
        container.caption(f"Scanner: {scanner.telemetry()}")
        return

    container.caption(
        f"Showing {len(entries)} stranger(s) | "
        f"scanner: {scanner.telemetry()}",
    )

    # Render each stranger in a vertical scroll of cards.
    # Patch 63 (hotfix K) :: Show ALL snapshots per stranger (not just
    # the last thumbnail). Each card now renders a mini-grid of up to
    # 6 thumbnail images (2 rows x 3 cols). If the stranger has more
    # than 6 snapshots, an "Expand all N snapshots" expander reveals
    # the rest in a scrollable grid. The latest snapshot is always
    # shown first (largest, with zoom toggle) so operators can quickly
    # identify the stranger, then the history grid follows.
    for entry in entries:
        with container.container(border=True):
            hdr = st.columns([3, 2])
            hdr[0].markdown(f"##### {entry['stranger_label']}")
            anomaly = entry["temporal_spread_score"]
            hdr[1].metric(
                label="Temporal Spread",
                value=f"{anomaly:.1f}",
                help="Variance of snapshot capture timestamps (s^2) -- higher means the stranger was visible across a longer time window.",
            )

            # Metadata row (moved to top so it's always visible).
            # Patch 18 :: Display LOCAL time, not UTC.
            meta = st.columns(3)
            meta[0].caption(f"Snapshots: **{entry['snapshot_count']}**")
            # Patch :: Python 3.10 compat -- f-strings cannot contain
            # newlines inside {...} braces pre-3.12 (PEP 701). Extract
            # the datetime computation into a local variable first.
            first_local = _dt.datetime.fromtimestamp(
                entry['first_seen_us'] / 1_000_000.0,
            ).astimezone().strftime('%H:%M:%S')
            last_local = _dt.datetime.fromtimestamp(
                entry['last_seen_us'] / 1_000_000.0,
            ).astimezone().strftime('%H:%M:%S')
            meta[1].caption(f"First: {first_local}")
            meta[2].caption(f"Last: {last_local}")

            # ---- Latest snapshot (with zoom toggle) ----
            # This is the primary identification image -- shown at a
            # larger size than the history thumbnails below.
            # Patch 27 :: Per-card 'Zoom to full size' toggle.
            # When enabled, the snapshot PNG is shown at its native
            # resolution (e.g. 1280x720) instead of the 320px
            # downscale. The full-size PNG is already saved by
            # snap_strangers (ticket.frame is the full annotated
            # frame, not a crop), so no extra disk I/O is needed --
            # we just skip the LANCZOS downscale.
            _all_snaps = entry.get("snapshots", [])
            _latest_snap = _all_snaps[-1] if _all_snaps else None
            try:
                if _latest_snap is None:
                    thumb_path = entry.get("thumbnail_path", "")
                else:
                    thumb_path = _latest_snap.get("abs_path", entry.get("thumbnail_path", ""))

                if not _PIL_AVAILABLE or not os.path.exists(thumb_path):
                    st.caption(f"(thumbnail unavailable: {thumb_path})")
                else:
                    # Patch 47 :: Validate PNG is complete before PIL
                    # opens it. cv2.imwrite() in main.py's snap_strangers
                    # worker and PIL.Image.open() here race when the
                    # snapshot worker is mid-write on the same path.
                    # A partial PNG causes PIL's libpng C decoder to
                    # access-violate (0xC0000005), killing the dashboard
                    # with no Python stack trace.
                    #
                    # Defense layers (in order):
                    #   1. File-size gate: < 1 KB = still being written.
                    #      A complete 1280x720 PNG is 200 KB+; anything
                    #      < 1 KB is a partial write.
                    #   2. Image.verify(): full checksum pass without
                    #      decoding pixels. Raises a catchable Python
                    #      exception if the PNG is truncated or corrupt,
                    #      instead of access-violating in C.
                    #   3. img.load(): force full pixel decode now so
                    #      any residual corruption surfaces inside this
                    #      try/except block as a catchable exception.
                    try:
                        file_size = os.path.getsize(thumb_path)
                    except OSError as size_exc:
                        st.caption(f"(thumbnail stat failed: {size_exc})")
                        file_size = -1

                    if file_size < 1024:
                        st.caption(
                            f"(thumbnail still being written: "
                            f"{file_size} bytes; will retry next refresh)"
                        )
                    else:
                        # Step 2: verify() the PNG integrity without
                        # decoding pixels. This is the critical line --
                        # it raises a Python exception on corrupt files
                        # instead of letting libpng access-violate.
                        with open(thumb_path, "rb") as _vf:
                            _verif = Image.open(_vf)
                            _verif.verify()  # raises if corrupt; no decode

                        # Step 3: re-open for pixel decoding. verify()
                        # invalidates the image object, so we must
                        # re-open. Then load() forces the decode now
                        # so any corruption surfaces as a catchable
                        # Python exception inside this try block.
                        img = Image.open(thumb_path)
                        img.load()

                        # Per-card zoom toggle. The key includes the
                        # stranger_label so each card has its own state.
                        zoom_key = (
                            f"zoom_full_{entry['stranger_label']}"
                        )
                        zoom_full = st.toggle(
                            "Zoom to full size",
                            value=st.session_state.get(zoom_key, False),
                            key=zoom_key,
                            help=(
                                "Show this stranger's snapshot at its "
                                "native resolution (e.g. 1280x720) "
                                "instead of the 320px thumbnail."
                            ),
                        )

                        if not zoom_full:
                            # Thumbnail: downscale to 320px max.
                            max_dim = 320
                            if max(img.size) > max_dim:
                                ratio = max_dim / max(img.size)
                                new_size = (
                                    int(img.size[0] * ratio),
                                    int(img.size[1] * ratio),
                                )
                                img = img.resize(new_size, Image.LANCZOS)

                        if not zoom_full:
                            cap_text = (
                                f"Latest snapshot (track #{entry['last_track_id']})"
                            )
                        else:
                            cap_text = (
                                f"Full-size snapshot "
                                f"({img.size[0]}x{img.size[1]}) -- "
                                f"track #{entry['last_track_id']}"
                            )
                        st.image(
                            img,
                            caption=cap_text,
                            width='stretch',
                        )
            except Exception as exc:
                st.caption(f"(thumbnail load failed: {exc})")

            # ---- Patch 63 (hotfix K) :: History mini-grid ----
            # Show ALL snapshots for this stranger (not just the latest).
            # The snapshots are displayed in a grid of small thumbnails
            # (160px each, 3 per row). If there are more than 6, the
            # first 6 are shown inline and the rest are hidden behind
            # an "Expand all N snapshots" expander to avoid vertical
            # bloat on busy sessions.
            _snap_count = len(_all_snaps)
            if _snap_count > 1:
                st.markdown(
                    f"**History ({_snap_count} snapshots):**"
                )
                # Show up to 6 thumbnails inline (excluding the latest,
                # which is already shown above). Reverse order so the
                # most recent history appears first.
                _history_snaps = list(reversed(_all_snaps[:-1]))
                _inline_count = min(6, len(_history_snaps))
                _inline_snaps = _history_snaps[:_inline_count]
                _remaining_snaps = _history_snaps[_inline_count:]

                if _inline_snaps:
                    _render_snapshot_grid(
                        _inline_snaps, columns=3, thumb_px=160,
                    )

                # If there are more snapshots, put them in an expander.
                if _remaining_snaps:
                    with st.expander(
                        f"Show all {len(_all_snaps)} snapshots "
                        f"({len(_remaining_snaps)} more)"
                    ):
                        _render_snapshot_grid(
                            _remaining_snaps, columns=3, thumb_px=160,
                        )



# Patch 25 (original) :: @st.cache_resource was used here to make the
# receiver/poller/scanner singletons that persist across Streamlit
# re-renders. Without singleton behavior, every re-render constructs
# a fresh UDPTelemetryReceiver, binds a new UDP socket to 127.0.0.1:9999,
# and spawns a new daemon thread -- but the previous socket is never
# .stop()'d. On Windows UDP, even with SO_REUSEADDR, the OS may deliver
# incoming datagrams to the OLD (defunct) socket, causing the dashboard
# to see zero packets despite main.py broadcasting correctly.
#
# Patch 53 :: REPLACED @st.cache_resource with a module-level singleton.
# Reason: Streamlit's cache_resource decorator hashes all arguments to
# compute a cache key. On long-running dashboard processes (with the
# UDP receiver daemon thread active), the hasher's _HashStack thread-
# local context gets corrupted, producing:
#   AttributeError: 'tuple' object has no attribute 'pop'
#   AttributeError: '_HashStack' object has no attribute '_stack'
# at streamlit/runtime/caching/hashing.py lines 193/196/337. This
# cascades through 5+ levels of "During handling of the above
# exception" and ultimately crashes main() at the _build_components
# call site. A subsequent GC pass then triggers a native Windows
# access violation (0xC0000005) during script_runner teardown,
# killing the Streamlit process with exit code -1073741819.
#
# The fix: manage the singleton ourselves with a module-level dict
# keyed by (config_path, config_mtime). This achieves identical
# singleton semantics (one receiver/poller/scanner per process per
# config-file-version) WITHOUT invoking Streamlit's argument hasher.
#
# Patch 63 (hotfix J) :: CRITICAL FIX -- persistent state across
# Streamlit re-runs.
#
# PROBLEM: When dashboard.py is the Streamlit entry script (`streamlit
# run dashboard.py`), Streamlit re-executes the ENTIRE script top-to-
# bottom on every autorefresh / user interaction via exec(). This means
# EVERY module-level variable assignment (including the `_COMPONENTS_CACHE
# = {}` below) is RE-EXECUTED on every re-run, WIPING the cache.
#
# This caused THREE cascading bugs:
#   1. _COMPONENTS_CACHE wiped -> _build_components() cache-misses every
#      re-render -> creates a NEW UDPTelemetryReceiver -> .start() ->
#      .bind() FAILS with WSAEADDRINUSE because the PREVIOUS receiver's
#      daemon thread is still alive and holding port 9999. Result:
#      "PORT 9999 IS ALREADY IN USE!" error spam every 2s, and the
#      dashboard UI uses the broken (0-packet) receiver instead of the
#      working orphaned one. The user sees "no telemetry" despite
#      main.py successfully broadcasting 7000+ packets.
#   2. _ATEXIT_REGISTERED wiped -> atexit.register() re-fires every
#      re-render -> ~86,000-closure/day leak (per Patch 57 audit B1).
#      Each re-run registers a NEW lambda closing over the NEW (broken)
#      receiver object.
#   3. _DASHBOARD_SHUTTING_DOWN wiped -> the Ctrl+C early-return guard
#      in main() stops working after the first re-render.
#
# FIX: Stash all persistent state on the `sys` module, which is a
# builtin C module imported ONCE per process and NEVER re-executed by
# Streamlit's script runner. The hasattr() check ensures the state dict
# is created only on the first script execution; all subsequent re-runs
# reuse the existing dict. This achieves true process-global singleton
# semantics that survive Streamlit's exec() re-runs, are shared across
# browser tabs (unlike st.session_state), and are accessible from
# signal handlers and atexit callbacks (unlike st.session_state).
#
# The old module-level names (_COMPONENTS_CACHE, _LAST_CONFIG_KEY,
# _ATEXIT_REGISTERED, _DASHBOARD_SHUTTING_DOWN) are kept as REFERENCES
# to the dict entries for backward-compatible read access. For writes,
# code must assign into _DASH_STATE[...] directly (immutable types like
# bool/tuple cannot be updated through a reference).
if not hasattr(sys, "_sortendance_dashboard_state"):
    sys._sortendance_dashboard_state = {
        "components_cache": {},            # _build_components singleton
        "last_config_key": (("", 0.0)),    # last cache key used
        "atex_registered": False,          # Patch 57 idempotency flag
        "dashboard_shutting_down": False,  # hotfix F shutdown flag
    }
_DASH_STATE = sys._sortendance_dashboard_state

# Backward-compatible read-only references (mutable dict/list types
# can be read AND mutated through these references; only reassignment
# of the name itself would break the link, which is why all WRITE
# operations below use _DASH_STATE[...] directly).
_COMPONENTS_CACHE: Dict[Tuple[str, float], Tuple[Any, Any, Any]] = _DASH_STATE["components_cache"]
_LAST_CONFIG_KEY: Tuple[str, float] = _DASH_STATE["last_config_key"]

# Patch 57 :: Persistent flag to make atexit.register idempotent.
# Set to True the first time main() runs; never reset. Prevents the
# ~86,000-closure/day leak documented in the audit (B1).
# READ via _DASH_STATE["atex_registered"]; WRITE via _DASH_STATE["atex_registered"] = True.
_ATEXIT_REGISTERED: bool = _DASH_STATE["atex_registered"]

# Patch 63 (hotfix F) :: Shutdown flag. Set to True when the dashboard
# is shutting down (Ctrl+C / SIGINT handler). Once set, main() returns
# immediately on subsequent Streamlit reruns, preventing the start/stop
# cycle that produced 40+ seconds of "loop exited / thread joined
# cleanly" spam.
# READ via _DASH_STATE["dashboard_shutting_down"]; WRITE via
# _DASH_STATE["dashboard_shutting_down"] = True.
_DASHBOARD_SHUTTING_DOWN: bool = _DASH_STATE["dashboard_shutting_down"]



def _render_event_log_page(
    container: Any,
    scanner: "StrangerGalleryScanner",
) -> None:
    """Patch 63 :: Render the Event Log page.

    Shows latest snapshots of identified persons + strangers, with an
    "Extend Log" button per entry that expands the hourly history.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    container.subheader("Event Log")
    container.caption(
        "Hourly snapshot grid of identified + stranger catches. "
        "Click **Extend Log** on any card to expand its hourly history."
    )

    # ---- Session selector (reuses the stranger gallery pattern) ----
    available_sessions = scanner.list_available_sessions()
    active_date, active_session = scanner.current_active_session()

    session_options: List[str] = []
    session_keys: List[Tuple[str, str]] = []
    default_index = 0
    for idx, (d, s) in enumerate(available_sessions):
        is_active = (d == active_date and s == active_session)
        sess_dir_name = session_label_to_dir(s)
        label_str = f"{d} — {sess_dir_name}" + (" (active)" if is_active else "")
        session_options.append(label_str)
        session_keys.append((d, s))
        if is_active:
            default_index = idx

    if not session_options:
        container.warning("No sessions available.")
        return

    # Auto-advance on session rotation.
    _active_session_key = f"{active_date}_{active_session}"
    _last_active_key = st.session_state.get(
        "event_log_last_active_session_key", None
    )
    if _last_active_key is not None and _last_active_key != _active_session_key:
        if "event_log_session_select" in st.session_state:
            del st.session_state["event_log_session_select"]
        st.session_state.event_log_selected_session = default_index
    st.session_state.event_log_last_active_session_key = _active_session_key

    if "event_log_selected_session" not in st.session_state:
        st.session_state.event_log_selected_session = default_index
    cached_idx = st.session_state.event_log_selected_session
    if cached_idx < 0 or cached_idx >= len(session_options):
        cached_idx = default_index
        st.session_state.event_log_selected_session = default_index

    selected_label = container.selectbox(
        "Session",
        options=session_options,
        index=cached_idx,
        key="event_log_session_select",
    )
    selected_idx = session_options.index(selected_label) if selected_label in session_options else default_index
    st.session_state.event_log_selected_session = selected_idx
    selected_date, selected_session = session_keys[selected_idx]

    # ---- Scan both identified + strangers ----
    identified_entries = scanner.scan_session_verified(selected_date, selected_session)
    stranger_entries = scanner.scan_session(selected_date, selected_session)

    # ---- Section 1: Identified Persons ----
    container.markdown("### Identified Persons")
    if not identified_entries:
        container.info(
            "No identified-person snapshots yet for this session. "
            "Snapshots appear here when a track is verified as a known student."
        )
    else:
        _render_event_log_entries(container, identified_entries, is_verified=True)

    container.markdown("---")

    # ---- Section 2: Strangers ----
    container.markdown("### Strangers")
    if not stranger_entries:
        container.info("No stranger snapshots for this session yet.")
    else:
        _render_event_log_entries(container, stranger_entries, is_verified=False)


def _render_event_log_entries(
    container: Any,
    entries: List[Dict[str, Any]],
    is_verified: bool,
) -> None:
    """Render a grid of event-log entries with Extend Log toggles.

    Patch 63 (hotfix L) :: Replaced st.button + manual session_state
    toggle with st.toggle. The previous pattern:
        if st.button("Extend Log", key=extend_key):
            st.session_state[extend_key] = not st.session_state.get(extend_key, False)
    crashed with:
        StreamlitAPIException: st.session_state[extend_key] cannot be
        modified after the widget with key extend_key is instantiated.
    Because st.button binds session_state[key] to the button's pressed
    state, the next line tried to write to a widget-owned key AFTER
    instantiation -- which Streamlit forbids.

    The fix uses st.toggle, whose state IS the persistent toggle state.
    No manual session_state manipulation is needed.
    """
    for entry in entries:
        if is_verified:
            label = entry.get("student_label", "UNKNOWN")
            title = f"Identified: {label}"
        else:
            label = entry.get("stranger_label", "UNKNOWN")
            title = f"Stranger: {label}"
        # Unique key per (kind, label). Stays stable across reruns so
        # the toggle remembers its expanded/collapsed state.
        extend_key = f"event_log_extend_{'verified' if is_verified else 'stranger'}_{label}"

        with container.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{title}**")
                count = entry.get("snapshot_count", 0)
                first_us = entry.get("first_seen_us", 0)
                last_us = entry.get("last_seen_us", 0)
                if first_us > 0:
                    first_str = time.strftime("%H:%M:%S", time.localtime(first_us / 1_000_000.0))
                    last_str = time.strftime("%H:%M:%S", time.localtime(last_us / 1_000_000.0))
                    st.caption(f"{count} snapshots | {first_str} — {last_str}")
                else:
                    st.caption(f"{count} snapshots")
            with col2:
                # Patch 63 (hotfix L) :: st.toggle's state is bound
                # directly to session_state[extend_key]. No manual
                # mutation needed; reading the return value is enough.
                extended = st.toggle(
                    "Extend Log",
                    value=st.session_state.get(extend_key, False),
                    key=extend_key,
                    help="Show hourly history for this person.",
                )

            # Show the latest thumbnail.
            thumb_path = entry.get("thumbnail_path")
            if thumb_path and os.path.isfile(thumb_path):
                try:
                    from PIL import Image
                    img = Image.open(thumb_path)
                    img.verify()
                    img = Image.open(thumb_path)
                    st.image(img, caption="Latest snapshot", use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not load thumbnail: {exc}")

            # Extend Log: show hourly history.
            if extended:
                st.markdown("#### Hourly History")
                hourly = entry.get("hourly_buckets", {})
                if not hourly:
                    st.caption("No hourly data available.")
                else:
                    for hour in sorted(hourly.keys()):
                        snaps = hourly[hour]
                        st.markdown(f"**Hour {hour:02d}:00** — {len(snaps)} snapshot(s)")
                        # Show up to 4 thumbnails per hour in a row.
                        thumbs = snaps[:4]
                        if thumbs:
                            cols = st.columns(min(4, len(thumbs)))
                            for i, snap in enumerate(thumbs):
                                with cols[i]:
                                    snap_path = snap.get("abs_path", "")
                                    if snap_path and os.path.isfile(snap_path):
                                        try:
                                            from PIL import Image
                                            img = Image.open(snap_path)
                                            img.verify()
                                            img = Image.open(snap_path)
                                            ts_str = time.strftime(
                                                "%H:%M:%S",
                                                time.localtime(snap["ts_us"] / 1_000_000.0),
                                            )
                                            st.image(img, caption=ts_str)
                                        except Exception:
                                            st.caption("(corrupt)")
                                    if len(snaps) > 4:
                                        st.caption(f"+ {len(snaps) - 4} more")


def _build_components(
    config: Dict[str, Any],
    config_path: str = "",
) -> Tuple[UDPTelemetryReceiver, CSVAttendancePoller, StrangerGalleryScanner]:
    """Build the three dashboard column components from config.

    Patch 53 :: Module-level singleton. Returns the SAME
    receiver/poller/scanner tuple across Streamlit re-renders as long
    as the config file's path+mtime are unchanged. If the config file
    changes on disk, the next call rebuilds the components (after
    stopping the previous receiver to release the UDP socket).

    Patch 63 (hotfix J) :: The cache now lives in _DASH_STATE (stashed
    on the sys module) so it survives Streamlit's script re-execution.
    See the hotfix J comment block above for the full rationale.
    """
    # Read the current last-config key from persistent state.
    _last_config_key: Tuple[str, float] = _DASH_STATE["last_config_key"]

    # Compute a stable cache key from the config file's path + mtime.
    # If config_path is unknown (empty), fall back to a single
    # "default" slot so we still get singleton behavior.
    cache_key: Tuple[str, float] = ("__default__", 0.0)
    if config_path:
        try:
            cache_key = (
                os.path.abspath(config_path),
                os.path.getmtime(config_path),
            )
        except OSError:
            cache_key = (os.path.abspath(config_path), 0.0)

    # Cache hit: return the existing singleton tuple.
    if cache_key in _COMPONENTS_CACHE:
        _DASH_STATE["last_config_key"] = cache_key
        # Patch 63 (hotfix J) :: Debug log for cache hits. This should
        # appear on every Streamlit re-render AFTER the first. If you
        # see "built components" on every re-render instead of this
        # message, the persistent state fix is not working.
        logger.debug(
            "dashboard: components cache HIT (key=%s) -- reusing "
            "existing singleton receiver/poller/scanner.",
            cache_key,
        )
        return _COMPONENTS_CACHE[cache_key]

    # Cache miss. If a previous singleton exists under a different key
    # (e.g. config file was edited), stop the old receiver to release
    # its UDP socket before building a new one.
    if _last_config_key in _COMPONENTS_CACHE:
        try:
            _old = _COMPONENTS_CACHE.pop(_last_config_key)
            _old_receiver = _old[0]
            try:
                _old_receiver.stop(timeout_s=2.0)
            except Exception as exc:
                logger.warning(
                    "dashboard: failed to stop previous UDPTelemetryReceiver "
                    "during config-change rebuild: %s", exc,
                )
        except Exception:
            pass

    dash_cfg = config.get("dashboard", {})
    log_cfg = config.get("async_logger", {})

    # Patch 18 :: The stranger snapshot gallery now reads from
    # snap_strangers.output_dir (the new per-date+session hierarchy),
    # not the legacy video_recorder.stranger_crop_dir. Fall back to the
    # legacy section for graceful migration if snap_strangers is absent.
    snap_cfg = config.get("snap_strangers")
    if snap_cfg is None:
        snap_cfg = config.get("video_recorder", {})
        if snap_cfg:
            logger.warning(
                "dashboard: snap_strangers config section not found; "
                "falling back to legacy video_recorder section. Please "
                "rename 'video_recorder:' to 'snap_strangers:' in "
                "config.yaml.",
            )

    receiver = UDPTelemetryReceiver(
        host=dash_cfg.get("host", "127.0.0.1"),
        port=int(dash_cfg.get("udp_metrics_port", 9999)),
        buffer_bytes=int(dash_cfg.get("udp_buffer_bytes", 65536)),
        # Patch 36 :: 3h window @ 250 ms refresh = 43200 packets.
        history_capacity=43200,
    )

    poller = CSVAttendancePoller(
        log_dir=log_cfg.get("log_dir", "storage/logs"),
        prefix=log_cfg.get("daily_csv_prefix", "attendance_"),
        columns=list(log_cfg.get("csv_columns", [])),
        max_rows=500,
    )

    scanner = StrangerGalleryScanner(
        cache_dir=snap_cfg.get("output_dir", "storage/snap_strangers"),
        # Patch 63 (hotfix K) :: Default raised from 24 -> 200.
        max_strangers=int(dash_cfg.get("max_stranger_thumbnails", 200)),
    )

    # Patch 53 :: Cache the built components as a singleton.
    # Patch 63 (hotfix J) :: Write through _DASH_STATE so the cache
    # persists across Streamlit re-runs. _COMPONENTS_CACHE is the SAME
    # dict object (assigned at module load from _DASH_STATE), so
    # mutating it here also updates the persistent state.
    _COMPONENTS_CACHE[cache_key] = (receiver, poller, scanner)
    _DASH_STATE["last_config_key"] = cache_key
    logger.info(
        "dashboard: built components (cache_key=%s, port=%d) -- "
        "singleton will persist across re-renders until config "
        "file changes.",
        cache_key, int(dash_cfg.get("udp_metrics_port", 9999)),
    )
    return receiver, poller, scanner


# ============================================================================
# Patch 36 :: Performance time-series charts.
# ============================================================================

# Interval selector options (label, seconds). The 3h maximum matches
# the receiver's history_capacity (43200 packets @ 250 ms).
_PERF_CHART_INTERVALS: List[Tuple[str, int]] = [
    ("15m", 15 * 60),
    ("30m", 30 * 60),
    ("1h",  60 * 60),
    ("3h",  3 * 60 * 60),
]


def _render_performance_charts(
    container: Any,
    receiver: UDPTelemetryReceiver,
) -> None:
    """Render Patch 36 :: Performance time-series graphs.

    Draws 6 time-series charts in a 2-column grid:
      - Processing latency (ms)
      - Tracking latency  (ms)
      - GUI latency       (ms)
      - VRAM usage        (MB) with total as a horizontal reference line
      - CPU usage         (%) -- process-scoped via psutil
      - Process RSS       (MB)

    Three-level fallback chain (Patch 56, NO Plotly):
      Path A: Altair  -- preferred, pure JSON spec
      Path B: matplotlib PNG  -- fallback, pure bytes
      Path C: text-only  -- final safety net

    An interval selector (15m / 30m / 1h / 3h) slices the receiver's
    history deque to the requested window.

    If the receiver has no history yet (main.py not broadcasting),
    renders a friendly "waiting for telemetry" notice.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    container.subheader("Performance Time-Series")
    container.caption(
        "Latency / VRAM / CPU trends from the orchestrator's UDP "
        "telemetry stream. Window selectable below."
    )

    # --------------------------------------------------------------
    # Interval selector + latest-packet snapshot.
    # --------------------------------------------------------------
    history = receiver.history()
    if not history:
        # Patch 63 (hotfix G) :: Enhanced "no telemetry" message with
        # receiver diagnostics. Show the receiver's internal state so
        # the operator can distinguish between:
        #   - receiver thread not running (start() failed or never called)
        #   - receiver running but 0 packets received (network/firewall/
        #     stale-process issue)
        #   - receiver running, packets received, but decode errors
        #     (schema mismatch between main.py and dashboard)
        try:
            _diag = receiver.telemetry()
            _rx_running = receiver._running
            _rx_shutdown = receiver._shutdown
            _pkts_rx = _diag.get("packets_received", 0)
            _decode_errs = _diag.get("decode_errors", 0)
            _last_pkt_us = _diag.get("last_packet_at_us", 0)
            _thread_alive = (
                receiver._thread is not None
                and receiver._thread.is_alive()
            )
        except Exception:
            _rx_running = "?"
            _rx_shutdown = "?"
            _pkts_rx = -1
            _decode_errs = -1
            _last_pkt_us = 0
            _thread_alive = "?"

        _last_pkt_str = "never"
        if _last_pkt_us > 0:
            _last_pkt_ago = (int(time.time() * 1_000_000) - _last_pkt_us) / 1_000_000
            _last_pkt_str = f"{_last_pkt_ago:.1f}s ago"

        container.info(
            f"Waiting for telemetry -- main.py has not sent any packets "
            f"yet. Charts will populate once the orchestrator starts "
            f"broadcasting.\n\n"
            f"**Receiver diagnostics:**\n"
            f"- Thread running: `{_rx_running}`\n"
            f"- Thread alive: `{_thread_alive}`\n"
            f"- Shutdown flag: `{_rx_shutdown}`\n"
            f"- Packets received: `{_pkts_rx}`\n"
            f"- Decode errors: `{_decode_errs}`\n"
            f"- Last packet: `{_last_pkt_str}`\n\n"
            f"**If packets_received=0:** main.py is either not running, "
            f"crashed during startup, or a stale process is holding "
            f"port 9999. Kill ALL python.exe/streamlit processes in "
            f"Task Manager, then restart both run_main.bat and "
            f"run_dashboard.bat.\n\n"
            f"**If packets_received>0 but decode_errors>0:** schema "
            f"mismatch between main.py and dashboard.py. Check the "
            f"dashboard console for 'decode error' logs."
        )
        return

    ctl = container.columns([1, 1, 1, 1, 2])
    # Track selection in session_state so it persists across reruns.
    # Default to 15m on the very first render.
    if "perf_chart_interval_label" not in st.session_state:
        st.session_state["perf_chart_interval_label"] = "15m"
    for i, (lbl, secs) in enumerate(_PERF_CHART_INTERVALS):
        # Highlight the active selection via the button label.
        is_active = st.session_state["perf_chart_interval_label"] == lbl
        btn_label = f"\u25cf {lbl}" if is_active else lbl
        if ctl[i].button(
            btn_label,
            key=f"perf_interval_{lbl}",
            width='stretch',
            help=f"Show last {lbl} of telemetry",
        ):
            st.session_state["perf_chart_interval_label"] = lbl
            st.rerun()
    selected_label = st.session_state.get("perf_chart_interval_label", "15m")
    selected_seconds = dict(_PERF_CHART_INTERVALS).get(selected_label, 15 * 60)

    ctl[4].caption(
        f"Window: **{selected_label}** | "
        f"History: {len(history)} packets | "
        f"Receiver: {receiver.telemetry().get('packets_received', 0)} total"
    )

    # --------------------------------------------------------------
    # Slice history to the selected window.
    # Each packet has broadcast_us (epoch microseconds). Compute the
    # cutoff and keep only packets newer than (now - selected_seconds).
    # --------------------------------------------------------------
    # Patch 41 :: Bounded fetch. For 15m (3600 packets @ 250ms) we use
    # history_last_n() to avoid materializing the full 43200-packet
    # deque. For larger windows we fall back to the full history() but
    # then downsample below.
    # --------------------------------------------------------------
    now_us = int(time.time() * 1_000_000)
    cutoff_us = now_us - (selected_seconds * 1_000_000)

    # Heuristic: if the window needs <= 5000 packets, fetch the tail.
    # 15m = 3600 packets, 30m = 7200 packets. So 15m uses the fast path,
    # 30m / 1h / 3h use the full history path (then downsample).
    _window_packet_budget = int(selected_seconds * 1000 / 250) + 60
    if _window_packet_budget <= 5000:
        # Fast path: fetch only the tail.
        _tail = receiver.history_last_n(_window_packet_budget + 200)
        window = [p for p in _tail if p.broadcast_us >= cutoff_us]
    else:
        window = [p for p in history if p.broadcast_us >= cutoff_us]
    if not window:
        # Fall back to the most recent N packets if the window is empty
        # (e.g. clock skew or main.py just started).
        window = history[-min(60, len(history)):]

    # --------------------------------------------------------------
    # Patch 41 :: Downsample to a bounded number of points per chart.
    # Plotly + Streamlit's WebSocket transport struggles to render
    # 43200 points x 5 charts every 250ms. Stride sampling to 600 max
    # points preserves the time-series shape while bounding the render
    # cost. Always keeps the first and last sample.
    # --------------------------------------------------------------
    # Patch 50 :: Reduced from 600 to 300 to lower GC pressure.
    # Fewer points = fewer Plotly trace objects = less cyclic GC
    # work = lower probability of GC firing during batch_update()
    # commit phase (the root cause of the 0xC0000005 crash).
    MAX_POINTS_PER_CHART: int = 300
    if len(window) > MAX_POINTS_PER_CHART:
        stride = max(1, len(window) // MAX_POINTS_PER_CHART)
        window = window[::stride]
        # Always keep the last sample so the chart shows up-to-date data.
        if window[-1] is not history[-1]:
            window.append(history[-1])

    # --------------------------------------------------------------
    # Patch 56 :: pandas is the only hard dependency for the DataFrame.
    # If pandas is absent, bail out with a text summary before trying
    # to build df. (Altair / matplotlib availability is checked in the
    # decision block below; if both are absent, Path C renders text.)
    # --------------------------------------------------------------
    if not _PANDAS_AVAILABLE:
        latest = history[-1]
        container.warning(
            "pandas is not installed -- rendering text summary "
            "instead of time-series charts. Install with: "
            "`pip install pandas matplotlib` (or `pip install pandas altair`)."
        )
        container.code(
            f"Latest packet (frame #{latest.frame_index}):\n"
            f"  processing_latency_ms = {latest.processing_latency_ms:.2f}\n"
            f"  tracking_latency_ms   = {latest.tracking_latency_ms:.2f}\n"
            f"  gui_latency_ms        = {latest.gui_latency_ms:.2f}\n"
            f"  gpu_vram_used         = {latest.gpu_vram_used_bytes / 1048576:.1f} MB\n"
            f"  gpu_vram_total        = {latest.gpu_vram_total_bytes / 1048576:.1f} MB\n"
            f"  cpu_percent           = {latest.cpu_percent:.2f}%\n"
            f"  rss_bytes             = {latest.rss_bytes / 1048576:.1f} MB\n"
            f"  window_packets        = {len(window)}",
            language="text",
        )
        return

    # ------------------------------------------------------------------
    # Patch 56 :: Three-level fallback chain. Plotly is GONE.
    #   Path A: Altair  -- preferred, pure JSON spec
    #   Path B: matplotlib PNG  -- fallback, pure bytes
    #   Path C: text-only  -- final safety net
    # ------------------------------------------------------------------
    # Previous Plotly-based path (Patches 36/41/50/52/54/55) is fully
    # removed. Plotly's C-extension caused native 0xC0000005 access
    # violations during GC on long-running Windows processes, killing
    # the Streamlit process AFTER successful renders. matplotlib's PNG
    # serialization is pure bytes and has no recursive walker, so it
    # does not have this issue.
    # ------------------------------------------------------------------
    use_altair = _ALTAIR_AVAILABLE and _PANDAS_AVAILABLE
    use_mpl = (not use_altair) and _MPL_AVAILABLE and _PANDAS_AVAILABLE

    # One-time log of which path is active.
    if use_altair and not getattr(main, "_infoed_altair_active", False):
        main._infoed_altair_active = True  # type: ignore[attr-defined]
        logger.info(
            "dashboard: Altair is available -- using Altair for "
            "performance charts (pure JSON spec, no native crash)."
        )
    elif use_mpl and not getattr(main, "_infoed_mpl_fallback", False):
        main._infoed_mpl_fallback = True  # type: ignore[attr-defined]
        logger.info(
            "dashboard: Altair is not installed -- using matplotlib "
            "PNG fallback for performance charts. This is safe on "
            "long-running Windows processes (no recursive walker, "
            "no GC access violation). Install Altair for interactive "
            "tooltips: pip install altair pandas"
        )
    elif (not use_altair) and (not use_mpl):
        if not getattr(main, "_warned_textonly_charts", False):
            main._warned_textonly_charts = True  # type: ignore[attr-defined]
            logger.warning(
                "dashboard: Neither Altair nor matplotlib is available "
                "-- falling back to text-only performance summary. "
                "Install one of them for charts: "
                "pip install altair pandas  (or)  pip install matplotlib pandas"
            )

    # Build a pandas DataFrame indexed by datetime (shared by both paths).
    rows = []
    for p in window:
        rows.append({
            "time": _dt.datetime.fromtimestamp(p.broadcast_us / 1_000_000.0),
            "Processing (ms)": float(p.processing_latency_ms),
            "Tracking (ms)":   float(p.tracking_latency_ms),
            "GUI (ms)":        float(p.gui_latency_ms),
            "VRAM Used (MB)":  float(p.gpu_vram_used_bytes) / (1024.0 * 1024.0),
            "VRAM Total (MB)": float(p.gpu_vram_total_bytes) / (1024.0 * 1024.0),
            "CPU (%)":         float(p.cpu_percent),
            "RSS (MB)":        float(p.rss_bytes) / (1024.0 * 1024.0),
        })
    df = pd.DataFrame(rows)

    # ==================================================================
    # Path A: Altair (preferred -- stable on long-running Windows)
    # ==================================================================
    if use_altair:
        try:
            # Altair charts are pure JSON specs. No recursive Python
            # serialization, no `convert_to_base64`, no access violation.
            # The chart spec is a declarative Vega-Lite dict that
            # Streamlit serializes via standard json.dumps().
            def _alt_line_chart(
                col_name: str,
                title: str,
                y_title: str,
                color: str = "#1f77b4",
                total_line: Optional[float] = None,
            ):
                # Main line trace.
                chart = (
                    alt.Chart(df, title=title)
                    .mark_line(color=color, strokeWidth=1.5)
                    .encode(
                        x=alt.X("time:T", title=None),
                        y=alt.Y(f"{col_name}:Q", title=y_title, scale=alt.Scale(zero=True)),
                        tooltip=[
                            alt.Tooltip("time:T", title="Time"),
                            alt.Tooltip(f"{col_name}:Q", title=title, format=".2f"),
                        ],
                    )
                    .properties(height=200)
                )

                # Optional total reference line (for VRAM chart).
                if total_line is not None and total_line > 0:
                    rule_df = pd.DataFrame({
                        "time": [df["time"].iloc[0], df["time"].iloc[-1]],
                        "total": [total_line, total_line],
                    })
                    rule = (
                        alt.Chart(rule_df)
                        .mark_line(color="#888", strokeWidth=1, strokeDash=[4, 4])
                        .encode(
                            x="time:T",
                            y="total:Q",
                        )
                    )
                    chart = chart + rule

                # Axis theming: light grid, no chart-junk.
                chart = chart.configure_view(strokeWidth=0)
                chart = chart.configure_axis(
                    grid=True,
                    gridColor="#eee",
                    gridWidth=1,
                    labelFontSize=10,
                    titleFontSize=11,
                )
                chart = chart.configure_title(fontSize=13, anchor="start")
                return chart

            def _safe_altair(chart, key_hint: str) -> None:
                try:
                    st.altair_chart(chart, use_container_width=True)
                except Exception as exc:
                    container.warning(
                        f"Altair render failed for {key_hint}: {exc}"
                    )

            c_left, c_right = container.columns(2)

            with c_left:
                _safe_altair(
                    _alt_line_chart("Processing (ms)", "Processing Latency", "ms", "#1f77b4"),
                    "Processing Latency",
                )
                _safe_altair(
                    _alt_line_chart("GUI (ms)", "GUI Latency", "ms", "#2ca02c"),
                    "GUI Latency",
                )
                _safe_altair(
                    _alt_line_chart("CPU (%)", "CPU Usage (this process)", "%", "#9467bd"),
                    "CPU Usage",
                )

            with c_right:
                _safe_altair(
                    _alt_line_chart("Tracking (ms)", "Tracking Latency", "ms", "#ff7f0e"),
                    "Tracking Latency",
                )
                vram_total = float(df["VRAM Total (MB)"].iloc[-1]) if len(df) else 0.0
                _safe_altair(
                    _alt_line_chart(
                        "VRAM Used (MB)", "GPU VRAM Usage", "MB", "#d62728",
                        total_line=vram_total,
                    ),
                    "GPU VRAM Usage",
                )
                _safe_altair(
                    _alt_line_chart("RSS (MB)", "Process RSS Memory", "MB", "#8c564b"),
                    "Process RSS Memory",
                )

        except Exception as exc:
            container.warning(
                f"Altair performance charts render failed (showing text fallback): {exc}"
            )
            try:
                latest = history[-1]
                container.code(
                    f"Latest packet (frame #{latest.frame_index}):\n"
                    f"  processing_latency_ms = {latest.processing_latency_ms:.2f}\n"
                    f"  tracking_latency_ms   = {latest.tracking_latency_ms:.2f}\n"
                    f"  gui_latency_ms        = {latest.gui_latency_ms:.2f}\n"
                    f"  gpu_vram_used         = {latest.gpu_vram_used_bytes / 1048576:.1f} MB\n"
                    f"  gpu_vram_total        = {latest.gpu_vram_total_bytes / 1048576:.1f} MB\n"
                    f"  cpu_percent           = {latest.cpu_percent:.2f}%\n"
                    f"  rss_bytes             = {latest.rss_bytes / 1048576:.1f} MB\n"
                    f"  window_packets        = {len(window)}",
                    language="text",
                )
            except Exception:
                pass

    # ==================================================================
    # Path B: matplotlib PNG fallback (Patch 56)
    # ------------------------------------------------------------------
    # Used when Altair is not installed. matplotlib renders the chart
    # to a PNG byte buffer, which is passed to st.image(). Pure bytes,
    # no recursive walker, no native 0xC0000005 access violation.
    # ==================================================================
    elif use_mpl:
        try:
            # Helper: build a single-line matplotlib chart as PNG bytes.
            # IMPORTANT: constrained_layout=True manages spacing. We do
            # NOT also call tight_layout() / subplots_adjust() / pass
            # bbox_inches='tight' to savefig -- these conflict with
            # constrained_layout and silently break its margins.
            def _mpl_line_chart_png(
                col_name: str,
                title: str,
                y_title: str,
                color: str = "#1f77b4",
                total_line: Optional[float] = None,
            ) -> bytes:
                # Patch 57 :: wrap body in try/finally so plt.close(fig)
                # ALWAYS runs, even if savefig or plot raises. Without
                # this, an exception leaves the figure in matplotlib's
                # internal _pylab_helpers.Gcf.figs dict (~1 MB/figure)
                # and it never gets freed for the process lifetime.
                fig, ax = plt.subplots(
                    figsize=(5.0, 1.8),
                    constrained_layout=True,
                )
                try:
                    # X-axis as datetime.
                    ax.plot(
                        df["time"],
                        df[col_name],
                        color=color,
                        linewidth=1.5,
                        label=title,
                    )
                    # Optional total reference line (VRAM chart).
                    if total_line is not None and total_line > 0:
                        ax.plot(
                            [df["time"].iloc[0], df["time"].iloc[-1]],
                            [total_line, total_line],
                            color="#888888",
                            linewidth=1.0,
                            linestyle="--",
                            label=f"Total ({total_line:.0f} {y_title})",
                        )
                    # Format x-axis as HH:MM:SS.
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                    for label in ax.get_xticklabels():
                        label.set_rotation(0)
                        label.set_fontsize(8)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(8)
                    ax.set_title(title, fontsize=11, loc="left")
                    ax.set_ylabel(y_title, fontsize=9)
                    ax.grid(True, color="#eeeeee", linewidth=0.8)
                    ax.set_facecolor("#fafafa")
                    ax.set_ylim(bottom=0)
                    # Serialize to PNG bytes.
                    buf = _io.BytesIO()
                    try:
                        fig.savefig(buf, format="png", dpi=80)
                    finally:
                        # buf.close() is safe even if savefig raised.
                        # We read bytes BEFORE close in the success path.
                        pass
                    buf.seek(0)
                    png_bytes = buf.getvalue()
                    buf.close()
                    return png_bytes
                finally:
                    # CRITICAL: always close the figure, even on
                    # exception. Prevents the matplotlib figure leak.
                    plt.close(fig)

            def _safe_mpl(png_bytes: bytes, key_hint: str) -> None:
                try:
                    st.image(png_bytes, use_container_width=True)
                except Exception as exc:
                    container.warning(
                        f"matplotlib render failed for {key_hint}: {exc}"
                    )

            c_left, c_right = container.columns(2)

            with c_left:
                _safe_mpl(
                    _mpl_line_chart_png("Processing (ms)", "Processing Latency", "ms", "#1f77b4"),
                    "Processing Latency",
                )
                _safe_mpl(
                    _mpl_line_chart_png("GUI (ms)", "GUI Latency", "ms", "#2ca02c"),
                    "GUI Latency",
                )
                _safe_mpl(
                    _mpl_line_chart_png("CPU (%)", "CPU Usage (this process)", "%", "#9467bd"),
                    "CPU Usage",
                )

            with c_right:
                _safe_mpl(
                    _mpl_line_chart_png("Tracking (ms)", "Tracking Latency", "ms", "#ff7f0e"),
                    "Tracking Latency",
                )
                vram_total = float(df["VRAM Total (MB)"].iloc[-1]) if len(df) else 0.0
                _safe_mpl(
                    _mpl_line_chart_png(
                        "VRAM Used (MB)", "GPU VRAM Usage", "MB", "#d62728",
                        total_line=vram_total,
                    ),
                    "GPU VRAM Usage",
                )
                _safe_mpl(
                    _mpl_line_chart_png("RSS (MB)", "Process RSS Memory", "MB", "#8c564b"),
                    "Process RSS Memory",
                )
        except Exception as exc:
            container.warning(
                f"matplotlib performance charts render failed (showing text fallback): {exc}"
            )
            try:
                latest = history[-1]
                container.code(
                    f"Latest packet (frame #{latest.frame_index}):\n"
                    f"  processing_latency_ms = {latest.processing_latency_ms:.2f}\n"
                    f"  tracking_latency_ms   = {latest.tracking_latency_ms:.2f}\n"
                    f"  gui_latency_ms        = {latest.gui_latency_ms:.2f}\n"
                    f"  gpu_vram_used         = {latest.gpu_vram_used_bytes / 1048576:.1f} MB\n"
                    f"  gpu_vram_total        = {latest.gpu_vram_total_bytes / 1048576:.1f} MB\n"
                    f"  cpu_percent           = {latest.cpu_percent:.2f}%\n"
                    f"  rss_bytes             = {latest.rss_bytes / 1048576:.1f} MB\n"
                    f"  window_packets        = {len(window)}",
                    language="text",
                )
            except Exception:
                pass

    # ==================================================================
    # Path C: text-only fallback (Patch 56)
    # ------------------------------------------------------------------
    # Used when neither Altair nor matplotlib is available. Renders a
    # plain-text summary of the latest telemetry packet and the
    # window's min/max/avg for each metric.
    # ==================================================================
    else:
        try:
            latest = history[-1]
            # Compute min/max/avg across the window for each metric.
            def _stats(col_name: str) -> str:
                try:
                    vals = df[col_name]
                    return (
                        f"min={vals.min():.2f}  "
                        f"max={vals.max():.2f}  "
                        f"avg={vals.mean():.2f}"
                    )
                except Exception:
                    return "n/a"

            container.code(
                f"Latest packet (frame #{latest.frame_index}):\n"
                f"  processing_latency_ms = {latest.processing_latency_ms:.2f}  "
                f"[{_stats('Processing (ms)')}]\n"
                f"  tracking_latency_ms   = {latest.tracking_latency_ms:.2f}  "
                f"[{_stats('Tracking (ms)')}]\n"
                f"  gui_latency_ms        = {latest.gui_latency_ms:.2f}  "
                f"[{_stats('GUI (ms)')}]\n"
                f"  gpu_vram_used         = {latest.gpu_vram_used_bytes / 1048576:.1f} MB\n"
                f"  gpu_vram_total        = {latest.gpu_vram_total_bytes / 1048576:.1f} MB\n"
                f"  cpu_percent           = {latest.cpu_percent:.2f}%  "
                f"[{_stats('CPU (%)')}]\n"
                f"  rss_bytes             = {latest.rss_bytes / 1048576:.1f} MB  "
                f"[{_stats('RSS (MB)')}]\n"
                f"  window_packets        = {len(window)}",
                language="text",
            )
        except Exception as exc:
            container.warning(
                f"Text-only performance summary failed: {exc}"
            )


def main() -> None:
    """
    Streamlit application entry point.

    Run as an independent process via:
        streamlit run ui/dashboard.py
    or:
        python -m streamlit run ui/dashboard.py -- --config config/config.yaml
    """
    if not _STREAMLIT_AVAILABLE:
        logger.error(
            "Streamlit is not available -- install via `pip install streamlit`.",
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # Patch 63 (hotfix F) :: Early-return guard for Streamlit shutdown
    # reruns. When Streamlit receives Ctrl+C, it re-executes the script
    # multiple times during its shutdown sequence. Each re-run calls
    # main(). If the receiver has already been shut down (by a previous
    # rerun or by the SIGINT handler), return immediately so we don't
    # attempt any Streamlit calls that would trigger another start/stop
    # cycle on the receiver.
    # Patch 63 (hotfix J) :: Read from _DASH_STATE (persistent on sys)
    # instead of the module-level bool, which gets wiped on every
    # Streamlit re-run.
    # ----------------------------------------------------------------
    if _DASH_STATE["dashboard_shutting_down"]:
        logger.debug(
            "Dashboard main(): shutdown flag is set -- returning early "
            "(Streamlit shutdown rerun)."
        )
        return

    # Load config.
    config: Dict[str, Any] = {}
    # Patch 53 :: Initialize config_path outside the try block so it
    # is always defined (needed for _build_components cache key).
    config_path: str = "config/config.yaml"
    if ConfigRegistry is not None:
        try:
            # Allow --config path override from streamlit CLI args.
            argv = sys.argv[1:]
            if "--config" in argv:
                idx = argv.index("--config")
                if idx + 1 < len(argv):
                    config_path = argv[idx + 1]
            config = ConfigRegistry.load(config_path)
        except Exception as exc:
            logger.error(
                "Failed to load config: %s -- falling back to defaults.", exc,
            )

    # Page config.
    st.set_page_config(
        page_title="SORT-tendance Dashboard",
        page_icon=":material/security:",
        layout="wide",
        # Patch 60 :: Flip from "collapsed" -> "expanded" so the
        # sidebar radio page selector is visible on first load.
        initial_sidebar_state="expanded",
    )

    st.title("SORT-tendance :: Enterprise Attendance & Spatial Security")
    st.caption(
        f"Build: {_dt.datetime.now(tz=_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    # Build the column components.
    # Patch 53 :: _build_components uses a module-level singleton
    # (_COMPONENTS_CACHE) keyed by (config_path, config_mtime).
    # Returns the SAME receiver/poller/scanner across re-renders
    # as long as the config file is unchanged. This replaces the
    # previous @st.cache_resource decorator, which crashed with
    # 'tuple' object has no attribute 'pop' in Streamlit's hasher.
    receiver, poller, scanner = _build_components(config, config_path)

    # ----------------------------------------------------------------
    # Patch 57 :: atexit registration MUST be idempotent. The previous
    # code called _atexit.register(lambda: receiver.stop(...)) inside
    # main(), which re-runs on every Streamlit rerun. Python's
    # atexit._exithandlers list accumulates these closures -- ~86,000
    # leaked closures/day at 1 rerun/sec. Each closure captures the
    # `receiver` local (one reference; same singleton, so no extra
    # receiver objects -- but the closure objects themselves grow
    # ~17 MB/day).
    #
    # Fix: guard with a persistent flag so we register exactly once.
    # Patch 63 (hotfix J) :: The flag MUST live in _DASH_STATE (stashed
    # on sys), NOT as a module-level bool. A module-level bool gets
    # re-initialized to False on every Streamlit re-run (because
    # dashboard.py is the entry script and Streamlit re-executes it
    # top-to-bottom via exec()). This caused the atexit lambda to be
    # re-registered on EVERY re-render -- the "idempotent" log message
    # below was appearing every 2s in production, leaking ~86,000
    # closures/day.
    # ----------------------------------------------------------------
    if not _DASH_STATE["atex_registered"]:
        import atexit as _atexit
        _atexit.register(lambda: receiver.stop(timeout_s=2.0))
        _DASH_STATE["atex_registered"] = True
        logger.info(
            "Patch 57: atexit cleanup registered (idempotent -- will "
            "not re-register on subsequent Streamlit reruns)."
        )

    # ----------------------------------------------------------------
    # Patch 57 :: 6AM/6PM session-boundary memory drop.
    # Detects when the local wall clock has just crossed the 06:00 or
    # 18:00 boundary by comparing the current session_key against the
    # one stored in st.session_state on the previous render. On
    # crossing, drops the receiver's 43200-packet history deque and
    # runs gc.collect() to free transient chart-render objects.
    #
    # This mirrors main.py's SessionBoundaryWatcher._fire_rotation()
    # which fires the OSNet dynamic-memory reset at the same boundary.
    # Together they ensure BOTH processes drop memory at 6AM/6PM.
    # ----------------------------------------------------------------
    try:
        _now_us = int(time.time() * 1_000_000)
        _active_session_key = "_".join(compute_session_key(_now_us))
        _prev_session_key = st.session_state.get(
            "_patch57_last_session_key", None
        )
        if _prev_session_key is not None and            _prev_session_key != _active_session_key:
            logger.info(
                "Patch 57: dashboard session boundary crossed "
                "(%s -> %s) -- dropping memory",
                _prev_session_key, _active_session_key,
            )
            # D1 :: clear the telemetry receiver's 43200-packet deque.
            try:
                receiver.clear_history()
            except Exception as exc:
                logger.warning(
                    "Patch 57: receiver.clear_history() failed: %s", exc,
                )
            # D5 :: force a cyclic GC pass to free transient chart-
            # render objects (history list materialization, pandas df,
            # matplotlib figure closures).
            try:
                gc.collect()
            except Exception:
                pass
        st.session_state["_patch57_last_session_key"] = _active_session_key
    except Exception as exc:
        logger.debug("Patch 57: session-boundary check failed: %s", exc)

    # Start the UDP receiver (idempotent -- Patch 25).
    # receiver is now a Patch 53 module-level singleton, so on the first
    # re-render this binds the socket and starts the daemon thread; on
    # every subsequent re-render it returns the SAME already-running
    # receiver. The single check below is sufficient.
    # Patch 63 (hotfix E) :: Also check _shutdown -- if the receiver
    # was permanently stopped (during process exit), do NOT try to
    # restart it. This prevents the rapid start/stop cycle during
    # Streamlit's shutdown sequence.
    if not receiver._running and not receiver._shutdown:
        receiver.start()

    # Patch 34/51/60 :: Auto-refresh interval.
    # Was hardcoded 250ms (Patch 34), then 1000ms (Patch 51).
    # Patch 60 :: Now read from config.yaml dashboard.refresh_interval_ms
    # (default 2000ms = 2s). The 2s cadence halves the GC pressure
    # from per-rerun transient objects. Operator requested "lazier
    # loading, over 2 seconds instead for the update for stability".
    _dash_cfg = config.get("dashboard", {}) if isinstance(config, dict) else {}
    DASHBOARD_REFRESH_INTERVAL_MS: int = int(
        _dash_cfg.get("refresh_interval_ms", 2000)
    )

    # ----------------------------------------------------------------
    # Patch 44 :: Top-level try/except around the entire render block.
    # If ANY exception escapes, log the full traceback BEFORE
    # re-raising so the operator can see what crashed in the log file
    # (Streamlit's own exception handler only shows it in the browser,
    # which is useless if the server process dies).
    # ----------------------------------------------------------------
    try:
        _dashboard_render_loop(
            receiver=receiver,
            poller=poller,
            scanner=scanner,
            refresh_interval_ms=DASHBOARD_REFRESH_INTERVAL_MS,
            config=config,
            config_path=config_path,
        )
    except SystemExit:
        # Don't catch intentional exits (st.rerun raises RerunException
        # which Streamlit catches; sys.exit raises SystemExit which we
        # should let propagate).
        raise
    except Exception as exc:
        logger.error(
            "Dashboard main() CRASHED: %s\n%s",
            exc, traceback.format_exc(),
        )
        # Try to show the error in the browser too.
        try:
            st.error(f"Dashboard crashed: {exc}\n\n```\n{traceback.format_exc()}\n```")
        except Exception:
            pass
        # Re-raise so Streamlit knows the script failed.
        raise


# Patch 51 :: One-time-per-process flag for the autorefresh warning.
_AUTOREFRESH_WARNING_PRINTED: bool = False


# ============================================================================
# Patch: Student Enrollment Page
# ----------------------------------------------------------------------------
# Implements interactive single-student enrollment from the Streamlit UI:
#   * Student number (NRP) text input (digits only).
#   * Optional student name (defaults to the ID if blank).
#   * Two face photos:
#       - Photo 1: flat / neutral expression (canonical anchor).
#       - Photo 2: subtle expression (smile / squint -- robustness).
#     Each photo can be acquired via Gallery upload (st.file_uploader)
#     OR live camera capture (st.camera_input).
#   * Submit -> spawns `python enroll.py --single-student ...` as a
#     NON-blocking subprocess so the Streamlit UI stays responsive.
#   * Status panel + dynamic st.toast() notifications (top-right,
#     auto-dismiss in ~3-4s).
#   * On success, enroll.py writes data/.restart_main_requested; the
#     supervisor (start_sortendance.py) waits 15s then restarts only
#     `main` so it reloads the updated student_db.pickle.
# ============================================================================

# Session-state keys for the enrollment flow.
_ENROLL_SESSION_KEY: str = "_enroll_subprocess_state"
_ENROLL_TMPDIR_KEY: str = "_enroll_tmpdir"


def _save_uploaded_photo(uploaded, dest_dir: str, label: str) -> Optional[str]:
    """
    Persist a Streamlit UploadedFile (from st.file_uploader OR
    st.camera_input) to dest_dir/label.<ext>. Returns the absolute
    path, or None if `uploaded` is None.
    """
    if uploaded is None:
        return None
    # Infer extension from the uploaded filename; default to .jpg for
    # camera captures (which arrive as PNG via streamlit but we save
    # as .jpg to match the rest of the pipeline's expectations).
    name = getattr(uploaded, "name", "") or ""
    ext = ".jpg"
    lower = name.lower()
    if lower.endswith(".png"):
        ext = ".png"
    elif lower.endswith(".jpeg"):
        ext = ".jpeg"
    dest = os.path.join(dest_dir, f"{label}{ext}")
    try:
        # Streamlit >= 1.27 exposes .getvalue() on UploadedFile.
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        with open(dest, "wb") as fh:
            fh.write(data)
        return dest
    except (OSError, AttributeError) as exc:
        logger.error("Failed to save uploaded photo %s: %s", label, exc)
        return None


def _spawn_enroll_subprocess(
    student_id: str,
    student_name: str,
    photo1_path: str,
    photo2_path: str,
    config_path: str,
) -> Tuple[subprocess.Popen, str]:
    """
    Spawn `python enroll.py --single-student ...` as a NON-blocking
    subprocess. Returns (Popen, tmpdir_path). The tmpdir holds the
    uploaded photos; the caller is responsible for cleaning it up
    after the subprocess completes.

    Captures stdout/stderr to a single pipe so we can read the
    machine-readable status line ("ENROLLED ...", "DUPLICATE_ID ...",
    etc.) that enroll.py emits as its LAST output line.
    """
    tmpdir = tempfile.mkdtemp(prefix="sortendance_enroll_")
    # Copy the photos into the tmpdir so we have stable paths even if
    # the original uploaded files are garbage-collected by Streamlit
    # between reruns.
    p1 = os.path.join(tmpdir, "photo1" + os.path.splitext(photo1_path)[1])
    p2 = os.path.join(tmpdir, "photo2" + os.path.splitext(photo2_path)[1])
    shutil.copy2(photo1_path, p1)
    shutil.copy2(photo2_path, p2)

    py_cmd = [
        sys.executable, "-u",
        os.path.join("scripts", "enroll.py"),
        "--single-student",
        "--student-id", student_id,
        "--student-name", student_name,
        "--photo1", p1,
        "--photo2", p2,
        "--config", config_path,
    ]
    logger.info("Enroll page: spawning subprocess: %s", " ".join(py_cmd))
    proc = subprocess.Popen(
        py_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, tmpdir


def _read_enroll_status(proc: subprocess.Popen) -> Tuple[Optional[int], str]:
    """
    Block until the subprocess exits, then return (returncode, last_line).
    `last_line` is the last non-empty stdout line (the machine-readable
    status that enroll.py prints). Falls back to "" if stdout is empty.
    """
    try:
        stdout, _ = proc.communicate(timeout=180)  # 3min ceiling
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        return -1, "TIMEOUT after 180s"
    last_line = ""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line:
            last_line = line
    return proc.returncode, last_line


def _render_enroll_page(
    container,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config/config.yaml",
) -> None:
    """
    Render the Student Enrollment page. See module docstring above for
    the full flow.
    """
    container.markdown("## Student Enrollment")
    container.caption(
        "Onboard a new student into the recognition database. "
        "Two photos are required (flat + subtle expression); each can be "
        "uploaded from the gallery OR captured live via the camera. "
        "Duplicate detection runs on student ID first, then on face "
        "cosine similarity (threshold 0.60)."
    )

    # ------------------------------------------------------------------
    # 1. Check whether an enrollment subprocess is already running.
    # ------------------------------------------------------------------
    state: Dict[str, Any] = st.session_state.get(_ENROLL_SESSION_KEY, None)
    if state is not None and isinstance(state, dict):
        proc: subprocess.Popen = state.get("proc")
        # If the proc is still running, poll + show a spinner.
        if proc.poll() is None:
            with container.spinner(
                f"Enrolling student {state.get('student_id', '?')}... "
                f"this can take ~10-30s (engine warmup + 2 embeds)."
            ):
                # Block until done. The spinner keeps the UI alive.
                rc, last_line = _read_enroll_status(proc)
        else:
            rc, last_line = _read_enroll_status(proc) if proc.stdout else (proc.returncode, "")

        # Process the result.
        _display_enroll_result(container, rc, last_line, state)

        # Clean up: kill the proc handle and remove the tmpdir.
        try:
            tmpdir = st.session_state.get(_ENROLL_TMPDIR_KEY)
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as exc:
            logger.warning("Enroll tmpdir cleanup failed: %s", exc)
        st.session_state.pop(_ENROLL_SESSION_KEY, None)
        st.session_state.pop(_ENROLL_TMPDIR_KEY, None)

        container.divider()
        # Allow the operator to start a new enrollment.
        if st.button("Enroll Another Student", key="_enroll_another_btn",
                     use_container_width=False):
            st.rerun()
        return

    # ------------------------------------------------------------------
    # 2. Render the input form (no subprocess running yet).
    # ------------------------------------------------------------------
    with container.form("enroll_form", clear_on_submit=False):
        col_id, col_name = st.columns([1, 2])
        with col_id:
            student_id = st.text_input(
                "Student Number (NRP) *",
                value="",
                max_chars=20,
                placeholder="e.g. 221050",
                help="Digits only. This is the primary key used for "
                     "duplicate detection (Stage 1) and as the pickle "
                     "registry key.",
                key="_enroll_student_id",
            )
        with col_name:
            student_name = st.text_input(
                "Student Name (optional)",
                value="",
                max_chars=80,
                placeholder="e.g. Jane Doe (defaults to NRP if blank)",
                key="_enroll_student_name",
            )

        st.markdown("**Photo 1 — Flat / Neutral Expression** *(required)*")
        src1 = st.radio(
            "Source for Photo 1",
            options=["Gallery", "Camera"],
            index=0,
            horizontal=True,
            key="_enroll_src1",
        )
        photo1_file = None
        if src1 == "Gallery":
            photo1_file = st.file_uploader(
                "Upload Photo 1 (flat)",
                type=["jpg", "jpeg", "png"],
                key="_enroll_photo1_upload",
                label_visibility="collapsed",
            )
        else:
            photo1_file = st.camera_input(
                "Capture Photo 1 (flat)",
                key="_enroll_photo1_camera",
                label_visibility="collapsed",
            )

        st.markdown("**Photo 2 — Subtle Expression** *(required)*")
        src2 = st.radio(
            "Source for Photo 2",
            options=["Gallery", "Camera"],
            index=0,
            horizontal=True,
            key="_enroll_src2",
        )
        photo2_file = None
        if src2 == "Gallery":
            photo2_file = st.file_uploader(
                "Upload Photo 2 (subtle)",
                type=["jpg", "jpeg", "png"],
                key="_enroll_photo2_upload",
                label_visibility="collapsed",
            )
        else:
            photo2_file = st.camera_input(
                "Capture Photo 2 (subtle)",
                key="_enroll_photo2_camera",
                label_visibility="collapsed",
            )

        submitted = st.form_submit_button(
            "Submit Enrollment",
            type="primary",
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # 3. On submit: validate inputs, save photos, spawn subprocess.
    # ------------------------------------------------------------------
    if submitted:
        # Validate ID: digits only (loose check -- allow letters too
        # but warn on whitespace).
        sid_clean = (student_id or "").strip()
        if not sid_clean:
            st.toast("Student number is required.", icon="⚠️")
            return
        if any(ch.isspace() for ch in sid_clean):
            st.toast("Student number must not contain whitespace.", icon="⚠️")
            return
        if photo1_file is None or photo2_file is None:
            st.toast("Both photos are required (flat + subtle).", icon="⚠️")
            return

        # Save uploaded photos to a tmpdir.
        tmpdir = tempfile.mkdtemp(prefix="sortendance_enroll_")
        p1_path = _save_uploaded_photo(photo1_file, tmpdir, "photo1")
        p2_path = _save_uploaded_photo(photo2_file, tmpdir, "photo2")
        if not p1_path or not p2_path:
            st.toast("Failed to save one or both photos. Try again.", icon="❌")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        # Spawn the subprocess (NON-blocking -- we stash the Popen
        # handle in session_state and poll on the next rerun).
        try:
            proc, enroll_tmpdir = _spawn_enroll_subprocess(
                student_id=sid_clean,
                student_name=(student_name or "").strip(),
                photo1_path=p1_path,
                photo2_path=p2_path,
                config_path=config_path,
            )
        except Exception as exc:
            logger.error("Enroll subprocess spawn failed: %s\n%s",
                         exc, traceback.format_exc())
            st.toast(f"Spawn failed: {exc}", icon="❌")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        st.session_state[_ENROLL_SESSION_KEY] = {
            "proc": proc,
            "student_id": sid_clean,
            "student_name": (student_name or "").strip(),
            "started_at": time.time(),
            "enroll_tmpdir": enroll_tmpdir,
        }
        st.session_state[_ENROLL_TMPDIR_KEY] = enroll_tmpdir
        st.toast(
            f"Enrollment started for {sid_clean}...",
            icon="⏳",
        )
        # Force a rerun so we enter the "subprocess running" branch above.
        st.rerun()


def _display_enroll_result(
    container,
    rc: Optional[int],
    last_line: str,
    state: Dict[str, Any],
) -> None:
    """
    Surface the enrollment subprocess result to the user via a status
    panel + a dynamic toast notification (top-right, auto-dismiss).

    Exit codes (defined in enroll.py):
        0  = RC_OK             -> enrolled successfully
       10  = RC_RATE_LIMITED   -> throttled (wait <60s since last enroll)
       20  = RC_DUPLICATE_ID   -> student_id already in DB
       21  = RC_DUPLICATE_FACE -> face already in DB (cosine >=0.6)
       30  = RC_BAD_INPUT      -> missing fields / unreadable photos
       40  = RC_ENGINE_ERROR   -> InsightFace stack failed
      -1   = TIMEOUT           -> subprocess killed after 180s
    """
    sid = state.get("student_id", "?")
    name = state.get("student_name", "") or sid

    container.markdown("### Enrollment Result")

    if rc == 0:
        container.success(
            f"✅ **Enrolled successfully.**\n\n"
            f"- Student ID: `{sid}`\n"
            f"- Name: **{name}**\n"
            f"- Status line: `{last_line or 'ENROLLED'}`\n\n"
            f"The supervisor will restart `main.py` in ~15s so the "
            f"newly-enrolled face is loaded into the live recognition "
            f"pipeline. You do not need to do anything."
        )
        st.toast(f"✅ Enrolled {sid} ({name})! main.py restart in 15s.",
                 icon="✅")

    elif rc == 10:
        # Parse the wait time from the last_line: "RATE_LIMITED wait_s=42.3"
        wait_s = 0.0
        try:
            for tok in (last_line or "").split():
                if tok.startswith("wait_s="):
                    wait_s = float(tok.split("=", 1)[1])
        except ValueError:
            pass
        container.warning(
            f"⏳ **Rate-limited.**\n\n"
            f"The last enrollment was less than 60s ago. Try again in "
            f"**{wait_s:.1f}s**.\n\n"
            f"Status: `{last_line}`"
        )
        st.toast(f"⏳ Rate-limited. Try again in {wait_s:.0f}s.", icon="⏳")

    elif rc == 20:
        # Parse: "DUPLICATE_ID student_id=X existing_name=Y enrolled_at=Z"
        existing_name = "?"
        enrolled_at = "?"
        for tok in (last_line or "").split():
            if tok.startswith("existing_name="):
                existing_name = tok.split("=", 1)[1]
            elif tok.startswith("enrolled_at="):
                enrolled_at = tok.split("=", 1)[1]
        container.error(
            f"🚫 **Duplicate student ID.**\n\n"
            f"- Submitted ID: `{sid}`\n"
            f"- Already enrolled as: **{existing_name}**\n"
            f"- Originally enrolled at: `{enrolled_at}`\n\n"
            f"Use a different student number, or remove the existing "
            f"record from `data/student_db.pickle` if this is a "
            f"re-enrollment."
        )
        st.toast(f"🚫 Duplicate ID: {sid} already exists.", icon="🚫")

    elif rc == 21:
        # Parse: "DUPLICATE_FACE matched_id=X matched_name=Y cosine=Z"
        matched_id = "?"
        matched_name = "?"
        cosine = 0.0
        for tok in (last_line or "").split():
            if tok.startswith("matched_id="):
                matched_id = tok.split("=", 1)[1]
            elif tok.startswith("matched_name="):
                matched_name = tok.split("=", 1)[1]
            elif tok.startswith("cosine="):
                try:
                    cosine = float(tok.split("=", 1)[1])
                except ValueError:
                    pass
        container.error(
            f"🚫 **Duplicate face detected.**\n\n"
            f"- Submitted ID: `{sid}`\n"
            f"- Closest match: `{matched_id}` (**{matched_name}**)\n"
            f"- Cosine similarity: **{cosine:.4f}** (threshold 0.60)\n\n"
            f"This face is already enrolled under a different ID. "
            f"Verify the student number and confirm with the subject "
            f"before re-attempting."
        )
        st.toast(
            f"🚫 Duplicate face: matches {matched_name} (cos={cosine:.3f}).",
            icon="🚫",
        )

    elif rc == 30:
        container.error(
            f"⚠️ **Bad input.**\n\n"
            f"One or more inputs were missing or unreadable.\n\n"
            f"Status: `{last_line}`"
        )
        st.toast("⚠️ Bad input. Check ID + photos.", icon="⚠️")

    elif rc == 40:
        container.error(
            f"❌ **Engine error.**\n\n"
            f"The InsightFace stack failed to initialize or warm up. "
            f"Check the supervisor / main console for traceback.\n\n"
            f"Status: `{last_line}`"
        )
        st.toast("❌ Engine error during enrollment.", icon="❌")

    elif rc == -1:
        container.error(
            f"⏱️ **Timeout.**\n\n"
            f"The enrollment subprocess did not finish within 180s. "
            f"It was force-killed.\n\n"
            f"Status: `{last_line}`"
        )
        st.toast("⏱️ Enrollment timed out after 180s.", icon="⏱️")

    else:
        container.error(
            f"❓ **Unknown result.** (rc={rc})\n\n"
            f"Status: `{last_line or '(empty)'}`"
        )
        st.toast(f"❓ Unknown enrollment result (rc={rc}).", icon="❓")


# ============================================================================
# End of Student Enrollment Page patch.
# ============================================================================


# ============================================================================
# Patch: Face Database Page
# ----------------------------------------------------------------------------
# A read-only browser for data/student_db.pickle + an "Add Photos" form
# that appends more embeddings to an EXISTING student (no dedup, honors
# profile_capacity). Spawns `python scripts/enroll.py --add-to <id>
# --photo p1 --photo p2 ...` as a non-blocking subprocess, mirroring the
# enroll page's pattern. NO delete button -- only adding is supported.
# ============================================================================
def _load_face_db_pickle(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read data/student_db.pickle directly (no engine init).

    Returns {} if the file is missing or unreadable. The pickle is a
    dict[student_id -> profile_dict]; we don't touch face_embeddings
    here (just count them), so no numpy allocation pressure.
    """
    enr = config.get("enrollment", {}) or {}
    db_path = enr.get("db_pickle_path") or "data/student_db.pickle"
    if not os.path.isabs(db_path):
        db_path = os.path.join(_PROJECT_ROOT, db_path)
    if not os.path.isfile(db_path):
        return {}
    try:
        import pickle as _pickle
        with open(db_path, "rb") as fh:
            return _pickle.load(fh) or {}
    except Exception as exc:
        logger.warning("Face DB pickle load failed (%s): %s", db_path, exc)
        return {}


def _spawn_add_photos_subprocess(
    student_id: str,
    photo_paths: List[str],
    config_path: str,
) -> Tuple[subprocess.Popen, str]:
    """
    Spawn `python scripts/enroll.py --add-to <id> --photo p1 --photo p2 ...`
    as a NON-blocking subprocess. Returns (Popen, tmpdir_path).

    The tmpdir holds COPIES of the uploaded photos so original
    BytesIO/file objects can be GC'd by Streamlit between reruns.
    """
    tmpdir = tempfile.mkdtemp(prefix="sortendance_add_")
    new_paths: List[str] = []
    for i, p in enumerate(photo_paths, start=1):
        ext = os.path.splitext(p)[1] or ".jpg"
        dest = os.path.join(tmpdir, f"photo{i}{ext}")
        shutil.copy2(p, dest)
        new_paths.append(dest)

    py_cmd = [
        sys.executable, "-u",
        os.path.join("scripts", "enroll.py"),
        "--add-to", student_id,
        "--config", config_path,
    ]
    for p in new_paths:
        py_cmd += ["--photo", p]
    logger.info("Face DB: spawning add-photos subprocess: %s", " ".join(py_cmd))
    proc = subprocess.Popen(
        py_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, tmpdir


def _display_add_photos_result(
    container, rc: Optional[int], last_line: str, state: dict,
) -> None:
    """Map add-photos subprocess exit code -> markdown + toast."""
    sid = state.get("student_id", "?")
    container.markdown("### Add Photos Result")

    if rc == 0:
        # Parse: "ADDED_EMBEDDINGS student_id=X student_name=Y
        #         total_embeddings=N capacity=M"
        total_emb = "?"
        capacity = "?"
        for tok in (last_line or "").split():
            if tok.startswith("total_embeddings="):
                total_emb = tok.split("=", 1)[1]
            elif tok.startswith("capacity="):
                capacity = tok.split("=", 1)[1]
        container.markdown(
            f"✅ **Added successfully.**\n\n"
            f"- Student: **{sid}**\n"
            f"- Total embeddings now: **{total_emb} / {capacity}**\n\n"
            f"`main.py` will reload the expanded embeddings in ~15 seconds."
        )
        st.toast(f"✅ Added photos to {sid}! Restart in 15s.", icon="✅")
    elif rc == 10:  # RC_RATE_LIMITED
        wait_s = "?"
        for tok in (last_line or "").split():
            if tok.startswith("wait_s="):
                wait_s = tok.split("=", 1)[1]
        container.markdown(
            f"⏱️ **Rate-limited.**\n\n"
            f"The last enrollment/add was less than 60s ago. "
            f"Try again in **{wait_s}s**."
        )
        st.toast(f"⏱️ Try again in {wait_s}s.", icon="⏱️")
    elif rc == 30:  # RC_BAD_INPUT
        container.markdown(
            f"❌ **Bad input.**\n\n"
            f"```\n{last_line}\n```"
        )
        st.toast("❌ Bad input.", icon="❌")
    elif rc == 40:  # RC_ENGINE_ERROR
        container.markdown(
            f"💥 **Engine error.**\n\n"
            f"```\n{last_line}\n```"
        )
        st.toast("❌ Engine error.", icon="❌")
    elif rc == -1:
        container.markdown("⏱️ **Timed out** after 180s.")
        st.toast("⏱️ Timed out.", icon="⏱️")
    else:
        container.markdown(
            f"❓ **Unknown result** (rc={rc}).\n\n"
            f"```\n{last_line}\n```"
        )
        st.toast(f"❓ Unknown (rc={rc}).", icon="❓")


def _request_main_restart(
    reason: str = "student_info_updated",
    delay_s: float = 15.0,
) -> bool:
    """
    Write the supervisor restart-request flag file so start_sortendance.py
    gracefully restarts ONLY the `main` child after `delay_s` seconds.

    Mirrors the payload format written by scripts/enroll.py
    `_request_supervisor_restart()`. The supervisor polls for this file
    every 5s, so the restart will fire ~delay_s + 5s after this call.

    Returns True on success, False on OSError.
    """
    flag_rel = "data/.restart_main_requested"
    flag_path = os.path.join(_PROJECT_ROOT, flag_rel)
    os.makedirs(os.path.dirname(flag_path), exist_ok=True)
    payload = json.dumps({
        "requested_at": time.time(),
        "delay_s": delay_s,
        "reason": reason,
    })
    tmp = flag_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, flag_path)
        logger.info(
            "Dashboard: supervisor restart flag written -> %s | "
            "main.py restart in %.0fs (reason=%s)",
            flag_path, delay_s, reason,
        )
        return True
    except OSError as exc:
        logger.warning(
            "Dashboard: failed to write restart-request flag: %s", exc,
        )
        return False


def _render_edit_student_info(
    container, config: Dict[str, Any], config_path: str,
) -> None:
    """
    Render the "Edit Student Info" panel on the Face Database page.

    Two-stage flow per operator spec:
      1. Show the FULL student table as an editable st.data_editor with
         two columns: Student Number (= student_id) and Name.
            - DTI-1 / DTI-2 rows are shown in a SEPARATE editor where
              Student Number is locked (codename protection) and only
              Name is editable.
      2. On "Review Changes" click, show a confirmation dialog listing
         every row that changed (old -> new for both fields).
      3. On "Confirm & Save" click:
            - Call EnrollmentService.update_student_profile() for each
              changed row.
            - Write data/.restart_main_requested so main.py reloads
              the renamed pickle.
            - Show a success toast + result summary.
            - st.rerun() to refresh the table.
    """
    container.subheader("✏️ Edit Student Info")
    container.caption(
        "Edit the **Student Number** and **Name** of any enrolled "
        "student in-place. `DTI-1` and `DTI-2` are codenames -- their "
        "Student Number is locked, but you can still update the Name "
        "(e.g. `DTI-1_Budi`). On save, the on-disk raw-photos folder "
        "is renamed to match the new `StudentNumber_Name`, and `main.py` "
        "is restarted to reload the pickle."
    )

    # ---- Load the current registry fresh on every render -------------
    registry = _load_face_db_pickle(config)
    if not registry:
        container.info(
            "No students enrolled yet. Nothing to edit. Use the "
            "**Enroll Student** tab to add the first one."
        )
        return

    # Split into regular + codename rows.
    codename_ids = {"DTI-1", "DTI-2"}
    regular_rows = []
    codename_rows = []
    for sid in sorted(registry.keys()):
        prof = registry[sid]
        name = str(prof.get("student_name", sid))
        row = {"Student Number": sid, "Name": name}
        if sid in codename_ids:
            codename_rows.append(row)
        else:
            regular_rows.append(row)

    # ---- Stage key: which mode are we in? ---------------------------
    # "_edit_stage" in {None, "review", "saving"}
    stage = st.session_state.get("_edit_student_stage", None)

    # We always render the editors so the operator can keep editing.
    # If we're in "review" stage, the editors are rendered read-only
    # (via disabled=True) so the diff stays consistent.

    try:
        import pandas as pd
    except ImportError:
        container.error("pandas is required for the Edit Student Info panel.")
        return

    # Stash the *original* snapshot so we can compute the diff even
    # after the editor has been edited in-place.
    #
    # BUGFIX (Update-Student-2): The snapshot was previously cached
    # ONCE per session and never refreshed. If the registry changed
    # out-of-band after the cache was populated -- e.g. a new student
    # was enrolled via the Enroll Student tab, OR a previous edit was
    # just saved and the pickle was re-keyed -- the cached `original`
    # became STALE and was missing the new IDs. The diff loop below
    # (`original[old_id]`) then raised `KeyError('5027221006')` (which
    # str() renders as `'5027221006'` with quotes -- the exact error
    # the operator saw), crashing the whole panel.
    #
    # Fix: rebuild `original` whenever the current registry's ID set
    # differs from the cached key set. Skip the refresh while the user
    # is in the "review" / "saving" stages -- they have pending edits
    # and we must preserve the diff baseline until they confirm or
    # cancel.
    current_ids_map: Dict[str, str] = {
        r["Student Number"]: r["Name"] for r in (regular_rows + codename_rows)
    }
    cached_original = st.session_state.get("_edit_student_original")
    needs_refresh = (
        cached_original is None
        or set(cached_original.keys()) != set(current_ids_map.keys())
    )
    if needs_refresh and stage not in ("review", "saving"):
        if cached_original is not None:
            added = set(current_ids_map.keys()) - set(cached_original.keys())
            removed = set(cached_original.keys()) - set(current_ids_map.keys())
            logger.info(
                "Edit Student Info: refreshing stale original snapshot "
                "(added=%s removed=%s)", added, removed,
            )
        st.session_state["_edit_student_original"] = current_ids_map

    # If user clicked "Cancel" in review, reset stage to None.
    if stage == "review":
        # Show the diff first, then editors below.
        container.markdown("#### ⚠️ Confirm Changes")
        container.warning(
            "You are about to permanently rename student profiles in the "
            "database AND rename the on-disk raw-photos folders. Please "
            "review the diff carefully before confirming."
        )

    # ---- Editors ----------------------------------------------------
    editor_disabled = (stage == "review")

    container.markdown("**Regular students** (Student Number + Name editable)")
    reg_df = pd.DataFrame(regular_rows) if regular_rows else pd.DataFrame(
        [{"Student Number": "", "Name": ""}]
    )
    edited_reg_df = container.data_editor(
        reg_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="_edit_student_reg_editor",
        disabled=editor_disabled,
    )

    if codename_rows:
        container.markdown(
            "**Codename students** (`DTI-1` / `DTI-2` -- Student Number "
            "locked, Name editable)"
        )
        # For codename rows, disable the Student Number column entirely.
        cn_df = pd.DataFrame(codename_rows)
        edited_cn_df = container.data_editor(
            cn_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="_edit_student_cn_editor",
            disabled=["Student Number"] if not editor_disabled else True,
        )
    else:
        edited_cn_df = pd.DataFrame(columns=["Student Number", "Name"])

    # ---- Build the merged "edited" snapshot -------------------------
    # Use the original key set to preserve row identity. We match by
    # row POSITION (the editors don't reorder rows).
    original = st.session_state["_edit_student_original"]

    reg_ids = [r["Student Number"] for r in regular_rows]
    cn_ids = [r["Student Number"] for r in codename_rows]

    # Map edited values back by position.
    edited_map: Dict[str, Tuple[str, str]] = {}  # old_id -> (new_id, new_name)
    if regular_rows:
        for i, old_id in enumerate(reg_ids):
            if i < len(edited_reg_df):
                new_id = str(edited_reg_df.iloc[i]["Student Number"]).strip()
                new_name = str(edited_reg_df.iloc[i]["Name"]).strip()
                edited_map[old_id] = (new_id, new_name)
    if codename_rows:
        for i, old_id in enumerate(cn_ids):
            if i < len(edited_cn_df):
                # Student Number column is disabled for codenames, so
                # new_id == old_id (read it back to be safe).
                new_id = str(edited_cn_df.iloc[i]["Student Number"]).strip()
                new_name = str(edited_cn_df.iloc[i]["Name"]).strip()
                edited_map[old_id] = (new_id, new_name)

    # ---- Compute diff -----------------------------------------------
    changes: List[Dict[str, str]] = []
    validation_errors: List[str] = []
    seen_new_ids: set = set()
    for old_id, (new_id, new_name) in edited_map.items():
        # BUGFIX (Update-Student-2): defensive .get() -- if the
        # original snapshot is somehow still stale (e.g. the registry
        # changed mid-review), fall back to treating the cached name
        # as equal to the current name so we don't crash with
        # KeyError. The needs_refresh block above should have already
        # rebuilt `original`, but this belt-and-suspenders guard
        # guarantees the panel never crashes on a missing key.
        old_name = original.get(old_id, new_name)
        id_changed = (new_id != old_id)
        name_changed = (new_name != old_name)
        if not id_changed and not name_changed:
            continue
        # Validation: codename protection.
        if old_id in codename_ids and id_changed:
            validation_errors.append(
                f"Row '{old_id}': Student Number is locked (codename). "
                f"Cannot change to '{new_id}'."
            )
            continue
        # Validation: empty.
        if not new_id:
            validation_errors.append(
                f"Row '{old_id}': Student Number cannot be empty."
            )
            continue
        # Validation: collision with another existing ID that is NOT
        # the same row's old ID.
        if new_id != old_id and new_id in registry:
            validation_errors.append(
                f"Row '{old_id}': new Student Number '{new_id}' is "
                f"already used by another student in the database."
            )
            continue
        # Validation: duplicate new IDs WITHIN this batch.
        if new_id != old_id and new_id in seen_new_ids:
            validation_errors.append(
                f"Row '{old_id}': new Student Number '{new_id}' collides "
                f"with another row's new Student Number in this batch."
            )
            continue
        seen_new_ids.add(new_id)
        changes.append({
            "old_id": old_id,
            "old_name": old_name,
            "new_id": new_id,
            "new_name": new_name,
            "id_changed": id_changed,
            "name_changed": name_changed,
        })

    # ---- Stage: REVIEW ----------------------------------------------
    if stage == "review":
        if changes:
            diff_rows = []
            for c in changes:
                id_cell = (
                    f"`{c['old_id']}` → `{c['new_id']}`" if c["id_changed"]
                    else f"`{c['old_id']}` (unchanged)"
                )
                name_cell = (
                    f"`{c['old_name']}` → `{c['new_name']}`" if c["name_changed"]
                    else f"`{c['old_name']}` (unchanged)"
                )
                folder_old = f"{c['old_id']}_{c['old_name']}"
                folder_new = f"{c['new_id']}_{c['new_name']}"
                folder_cell = (
                    f"`{folder_old}` → `{folder_new}`"
                    if (c["id_changed"] or c["name_changed"]) else "(no change)"
                )
                diff_rows.append({
                    "Student Number": id_cell,
                    "Name": name_cell,
                    "Folder (data/student_faces/)": folder_cell,
                })
            container.markdown("##### Changes to apply:")
            container.dataframe(
                pd.DataFrame(diff_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            container.info("No changes detected. You can click 'Cancel' to go back.")

        if validation_errors:
            container.error(
                "**Validation errors -- cannot save:**\n\n"
                + "\n".join(f"- {e}" for e in validation_errors)
            )

        col_cancel, col_save = container.columns([1, 1])
        if col_cancel.button(
            "Cancel", key="_edit_student_cancel_btn",
            use_container_width=True,
        ):
            st.session_state["_edit_student_stage"] = None
            st.session_state.pop("_edit_student_original", None)
            st.session_state.pop("_edit_student_reg_editor", None)
            st.session_state.pop("_edit_student_cn_editor", None)
            st.rerun()
        if col_save.button(
            "Confirm & Save", key="_edit_student_confirm_btn",
            use_container_width=True, type="primary",
            disabled=bool(validation_errors) or not changes,
        ):
            # ---- Stage: SAVING --------------------------------------
            st.session_state["_edit_student_stage"] = "saving"
            try:
                # Lazy-import EnrollmentService.
                sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
                from src.utils.database_manager import EnrollmentService  # type: ignore
                svc = EnrollmentService(config=config)
                results: List[Dict[str, Any]] = []
                for c in changes:
                    res = svc.update_student_profile(
                        old_student_id=c["old_id"],
                        new_student_id=c["new_id"],
                        new_student_name=c["new_name"],
                        rename_folder=True,
                    )
                    results.append(res)
                try:
                    svc.close()
                except Exception:
                    pass

                # Write the supervisor restart flag.
                _request_main_restart(reason="student_info_updated")

                # Build a one-line summary for the toast.
                n = len(results)
                id_changes = sum(1 for r in results if r.get("id_changed"))
                name_changes = sum(1 for r in results if r.get("name_changed"))
                folders_renamed = sum(
                    1 for r in results if r.get("folder_renamed")
                )
                container.success(
                    f"✅ **Saved {n} change(s).** "
                    f"ID renamed: {id_changes} | Name changed: {name_changes} | "
                    f"Folders renamed: {folders_renamed}."
                )
                st.toast(
                    f"✅ Saved {n} change(s). main.py restarting in ~15s.",
                    icon="✅",
                )
                # Reset state + rerun.
                st.session_state["_edit_student_stage"] = None
                st.session_state.pop("_edit_student_original", None)
                # Clear the data_editor cached edits so the next render
                # starts fresh from the just-saved registry values.
                st.session_state.pop("_edit_student_reg_editor", None)
                st.session_state.pop("_edit_student_cn_editor", None)
                st.rerun()
            except Exception as exc:
                logger.error(
                    "Edit Student Info save failed: %s\n%s",
                    exc, traceback.format_exc(),
                )
                container.error(f"Save failed: {exc}")
                st.session_state["_edit_student_stage"] = None
                st.session_state.pop("_edit_student_original", None)
                st.session_state.pop("_edit_student_reg_editor", None)
                st.session_state.pop("_edit_student_cn_editor", None)
        return  # End of review stage.

    # ---- Stage: DEFAULT (editing) -----------------------------------
    # Show validation errors live if any.
    if validation_errors:
        container.error(
            "**Cannot review -- fix these first:**\n\n"
            + "\n".join(f"- {e}" for e in validation_errors)
        )

    # "Review Changes" button is disabled if there are no changes OR
    # if there are validation errors.
    if container.button(
        "Review Changes",
        key="_edit_student_review_btn",
        type="primary",
        use_container_width=True,
        disabled=(not changes) or bool(validation_errors),
    ):
        st.session_state["_edit_student_stage"] = "review"
        st.rerun()

    if changes:
        container.caption(
            f"{len(changes)} pending change(s). Click 'Review Changes' "
            f"to see the diff and confirm."
        )
    else:
        container.caption(
            "Edit any cell above to start. Both Student Number and Name "
            "are editable for regular students; only Name is editable for "
            "DTI codenames."
        )


def _render_face_db_page(
    container, config: Dict[str, Any], config_path: str,
) -> None:
    """
    Render the Face Database page. Read-only browse of
    data/student_db.pickle + "Edit Student Info" panel + "Add Photos to
    Existing Student" form that appends embeddings via
    `python scripts/enroll.py --add-to`.
    """
    container.markdown("## Face Database")
    container.caption(
        "Every enrolled student is stored as a set of face embeddings in "
        "`data/student_db.pickle`. Use this page to browse the registry and "
        "append more photos to an existing student to improve recognition. "
        "No delete button -- only adding is supported."
    )

    # Top bar: metric + refresh.
    col_m, col_r = container.columns([3, 1])
    registry = _load_face_db_pickle(config)
    total = len(registry)
    with col_m:
        st.metric("Total Enrolled", total)
    with col_r:
        if st.button(
            "🔄 Refresh", key="_face_db_refresh_btn",
            use_container_width=True,
        ):
            st.rerun()

    if total == 0:
        container.info(
            "No students enrolled yet. Use the **Enroll Student** tab to "
            "add the first one."
        )
        return

    # Build the table.
    rows = []
    for sid in sorted(registry.keys()):
        prof = registry[sid]
        try:
            n_emb = int(prof["face_embeddings"].shape[0])
        except Exception:
            n_emb = 0
        capacity = int(prof.get("profile_capacity", 25))
        rows.append({
            "Student ID": str(sid),
            "Name": str(prof.get("student_name", sid)),
            "Embeddings": f"{n_emb} / {capacity}",
            "Enrolled At": str(prof.get("enrollment_timestamp", "?"))[:19],
            "Anchor": str(prof.get("anchor_image_hash", "?"))[:12],
        })
    try:
        import pandas as pd
        container.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        for r in rows:
            container.write(r)

    container.divider()

    # "Edit Student Info" panel -- rename Student Number / Name in-place.
    try:
        _render_edit_student_info(container, config, config_path)
    except Exception as exc:
        logger.error(
            "Edit Student Info panel crashed: %s\n%s",
            exc, traceback.format_exc(),
        )
        container.error(f"Edit Student Info panel failed: {exc}")

    container.divider()

    # "Add Photos" form.
    container.subheader("➕ Add Photos to Existing Student")
    container.caption(
        "Pick a student, upload 1-5 new photos (different angles, lighting, "
        "or expressions help the most). New embeddings are appended to the "
        "student's existing set. After a successful add, `main.py` restarts "
        "in ~15s to load the updated embeddings."
    )

    # If an add-photos subprocess is in-flight or just finished, show status.
    add_state = st.session_state.get("_add_photos_subprocess_state")
    if add_state and add_state.get("proc") is not None:
        proc: subprocess.Popen = add_state["proc"]
        if proc.poll() is None:
            container.info(
                f"⏳ Adding photos to student "
                f"{add_state.get('student_id', '?')}... "
                f"(running in background)"
            )
            return
        # Finished -- show result + cleanup.
        rc, last_line = _read_enroll_status(proc)
        _display_add_photos_result(container, rc, last_line, add_state)
        try:
            shutil.rmtree(add_state.get("tmpdir", ""), ignore_errors=True)
        except Exception:
            pass
        st.session_state["_add_photos_subprocess_state"] = None
        if st.button("Add More Photos", key="_add_photos_another_btn"):
            st.rerun()
        return

    # Fresh form.
    sorted_ids = sorted(registry.keys())
    with container.form("add_photos_form", clear_on_submit=False):
        selected_id = st.selectbox(
            "Student",
            options=sorted_ids,
            format_func=lambda sid: (
                f"{sid} — {registry[sid].get('student_name', sid)} "
                f"({int(registry[sid].get('face_embeddings').shape[0]) if hasattr(registry[sid].get('face_embeddings'), 'shape') else 0}/"
                f"{int(registry[sid].get('profile_capacity', 25))})"
            ),
            key="_face_db_add_target",
        )
        uploaded = st.file_uploader(
            "New Photos (1-5)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="_face_db_add_files",
        )
        submitted = st.form_submit_button("Add Photos", type="primary")

        if submitted:
            if not selected_id:
                st.error("Pick a student first.")
            elif not uploaded or len(uploaded) == 0:
                st.error("Upload at least 1 photo.")
            elif len(uploaded) > 5:
                st.error("Max 5 photos at a time.")
            else:
                # Save uploads to a tmpdir, spawn subprocess.
                upload_tmpdir = tempfile.mkdtemp(prefix="sortendance_add_upload_")
                saved: List[str] = []
                ok = True
                for i, up in enumerate(uploaded, start=1):
                    ext = os.path.splitext(up.name)[1] or ".jpg"
                    dest = os.path.join(upload_tmpdir, f"photo{i}{ext}")
                    try:
                        with open(dest, "wb") as fh:
                            fh.write(up.getvalue())
                        saved.append(dest)
                    except OSError as exc:
                        st.error(f"Failed to save {up.name}: {exc}")
                        ok = False
                        break
                if ok:
                    try:
                        proc, enroll_tmpdir = _spawn_add_photos_subprocess(
                            student_id=selected_id,
                            photo_paths=saved,
                            config_path=config_path,
                        )
                    except Exception as exc:
                        logger.error(
                            "Add-photos subprocess spawn failed: %s\n%s",
                            exc, traceback.format_exc(),
                        )
                        st.error(f"Failed to start: {exc}")
                        return
                    st.session_state["_add_photos_subprocess_state"] = {
                        "proc": proc,
                        "student_id": selected_id,
                        "tmpdir": enroll_tmpdir,
                    }
                    st.rerun()


# ============================================================================
# End of Face Database Page patch.
# ============================================================================


def _dashboard_render_loop(
    receiver: "UDPTelemetryReceiver",
    poller: "CSVAttendancePoller",
    scanner: "StrangerGalleryScanner",
    refresh_interval_ms: int,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config/config.yaml",
) -> None:
    """
    Patch 44 :: Extracted the render block into a separate function so
    it can be wrapped in a top-level try/except by main(). Every major
    step is logged so the operator can see EXACTLY where the dashboard
    dies on the next silent crash.
    """
    # Patch 51 :: One-time warning about streamlit_autorefresh.
    global _AUTOREFRESH_WARNING_PRINTED
    if not _AUTOREFRESH_WARNING_PRINTED:
        try:
            from streamlit_autorefresh import st_autorefresh  # noqa: F401
        except ImportError:
            logger.warning(
                "streamlit-autorefresh is NOT installed. The dashboard "
                "will use time.sleep + st.rerun() for refreshes, which "
                "re-executes the entire script every cycle. Install it "
                "for smoother operation:  pip install streamlit-autorefresh"
            )
        _AUTOREFRESH_WARNING_PRINTED = True

    _log_step = 0
    # Patch 51 :: Rate-limit step logging to 1/sec to avoid spam
    # when DEBUG is enabled. Without this, 2 browser tabs at 250ms
    # rerun interval = 80 log lines/sec.
    _last_step_log_ts: float = 0.0

    def _step(label: str) -> None:
        nonlocal _log_step, _last_step_log_ts
        _log_step += 1
        # Patch 51 :: Demoted from INFO to DEBUG. The step-by-step
        # logging was added in Patch 44 for diagnosing silent crashes.
        # It served its purpose (Patch 48 faulthandler gave us the
        # real crash location). Now demote to DEBUG so it doesn't
        # spam the console at default INFO level. Enable with:
        #   logging.getLogger("sortendance.dashboard").setLevel(logging.DEBUG)
        _now = time.time()
        if _now - _last_step_log_ts >= 1.0:
            logger.debug("Dashboard render step %d: %s", _log_step, label)
            _last_step_log_ts = _now

    _step("entering render block")

    # Top-bar controls: Force Refresh only (manual immediate re-render).
    _step("rendering Force Refresh button")
    # Patch 63 (hotfix) :: Pass an explicit unique key to the Force
    # Refresh button. Without `key=`, Streamlit auto-generates the
    # widget ID from (label, parent_container, position). When a rerun
    # is triggered (auto-refresh every 2s, OR a previous Force Refresh
    # click), Streamlit's delta path may still hold the previous
    # widget registration, causing:
    #   StreamlitDuplicateElementId: There are multiple `button`
    #   elements with the same auto-generated ID.
    # An explicit key stabilises the widget identity across reruns.
    ctl = st.columns([1, 6])
    if ctl[0].button("Force Refresh", key="_patch63_force_refresh_top"):
        logger.info("Dashboard: Force Refresh clicked")
        st.rerun()

    # ----------------------------------------------------------------
    # Patch 38 :: Performance time-series charts at the TOP.
    # ----------------------------------------------------------------
    _step("rendering performance charts (Patch 38/41)")
    st.divider()
    try:
        _render_performance_charts(st.container(), receiver)
    except Exception as exc:
        logger.error(
            "Dashboard: _render_performance_charts crashed: %s\n%s",
            exc, traceback.format_exc(),
        )
        st.warning(f"Performance charts failed: {exc}")

    # ----------------------------------------------------------------
    # Patch 60 (hotfix L) :: Sidebar nav buttons.
    #
    # Previously used st.radio, which the operator reported felt like a
    # "radio design stuff" rather than a proper clickable sidebar nav.
    # Replaced with full-width st.button widgets -- one per page -- so
    # each nav item is a large, obvious click target. The active page
    # is rendered as a `type="primary"` button (filled with the theme
    # accent color) so the operator can see at a glance which page is
    # currently displayed.
    #
    # The selected page is stored in st.session_state["_patch60_page"]
    # and persists across reruns. Clicking a different nav button
    # updates the state and triggers a rerun so the new page renders.
    # ----------------------------------------------------------------
    _step("rendering sidebar nav buttons")
    _NAV_PAGES: List[str] = [
        "Main",
        "Live Attendance",
        "Schedule",
        "Students",
        "Enroll Student",
        "Face Database",
        "Reports",
        "Event Log",
        "Stranger Gallery",
    ]
    # Initialize once. Default to "Main".
    if "_patch60_page" not in st.session_state:
        st.session_state["_patch60_page"] = "Main"
    _page_selector: str = st.session_state["_patch60_page"]
    # Defensive clamp -- if the cached page is somehow not in the list
    # (e.g. after a code change), fall back to "Main".
    if _page_selector not in _NAV_PAGES:
        _page_selector = "Main"
        st.session_state["_patch60_page"] = "Main"

    with st.sidebar:
        st.markdown("### SORT-tendance")
        st.caption("Navigation")
        # Render one full-width button per page. The active page uses
        # `type="primary"` so it's visually filled with the accent
        # color, making the current page obvious.
        for _page_name in _NAV_PAGES:
            _is_active = (_page_name == _page_selector)
            _btn_label = f"▶  {_page_name}" if _is_active else _page_name
            if st.button(
                _btn_label,
                key=f"_patch60_nav_btn_{_page_name}",
                use_container_width=True,
                type="primary" if _is_active else "secondary",
                help=f"Switch to the {_page_name} page",
            ):
                if not _is_active:
                    st.session_state["_patch60_page"] = _page_name
                    logger.info(
                        "Dashboard: sidebar nav clicked -> %s", _page_name,
                    )
                    st.rerun()
        st.divider()
        # Patch 63 (hotfix) :: Use the function parameter
        # `refresh_interval_ms`, NOT the module-level name
        # `DASHBOARD_REFRESH_INTERVAL_MS` (which is a local variable
        # in main() and is NOT in scope here). The previous reference
        # caused:
        #   NameError: name 'DASHBOARD_REFRESH_INTERVAL_MS' is not
        #   defined
        # at dashboard.py line 3311.
        st.caption(
            f"Refresh: **{refresh_interval_ms/1000:.1f}s** | "
            f"Build: {_dt.datetime.now(tz=_dt.timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        st.divider()

    # ----------------------------------------------------------------
    # Patch 60 :: Page dispatch.
    # ----------------------------------------------------------------
    if _page_selector == "Main":
        # Render the 3-column enterprise dashboard.
        _step("rendering 3-column block")
        col1, col2, col3 = st.columns([1.2, 1.5, 1.3], gap="medium")

        _step("rendering col1 (performance metrics)")
        with col1:
            try:
                _render_performance_column(col1, receiver)
            except Exception as exc:
                logger.error(
                    "Dashboard: _render_performance_column crashed: %s\n%s",
                    exc, traceback.format_exc(),
                )
                col1.error(f"Performance column failed: {exc}")

        _step("rendering col2 (attendance table)")
        with col2:
            try:
                _render_attendance_column(col2, poller)
            except Exception as exc:
                logger.error(
                    "Dashboard: _render_attendance_column crashed: %s\n%s",
                    exc, traceback.format_exc(),
                )
                col2.error(f"Attendance column failed: {exc}")

        _step("rendering col3 (stranger gallery)")
        with col3:
            try:
                _render_stranger_gallery_column(col3, scanner)
            except Exception as exc:
                logger.error(
                    "Dashboard: _render_stranger_gallery_column crashed: %s\n%s",
                    exc, traceback.format_exc(),
                )
                col3.error(f"Stranger gallery failed: {exc}")

        # Footer.
        _step("rendering footer")
        st.divider()
        try:
            st.caption(
                f"Receiver: {receiver.telemetry()} | "
                f"Poller: {poller.telemetry()} | "
                f"Scanner: {scanner.telemetry()}",
            )
        except Exception as exc:
            logger.error("Dashboard: footer caption crashed: %s", exc)

    elif _page_selector == "Event Log":
        # Patch 63 :: Event Log page (full implementation).
        _step("rendering Event Log page")
        try:
            _render_event_log_page(st.container(), scanner)
        except Exception as exc:
            logger.error(
                "Dashboard: _render_event_log_page crashed: %s\n%s",
                exc, traceback.format_exc(),
            )
            st.error(f"Event Log render failed: {exc}")

    elif _page_selector == "Stranger Gallery":
        # Patch 60 :: Stranger Gallery page (reuses the existing
        # scanner column logic as a full page).
        _step("rendering Stranger Gallery page")
        st.subheader("Stranger Gallery")
        st.caption(
            "Full-page stranger gallery view (same data as the Main "
            "page's right column, but with more room)."
        )
        try:
            _render_stranger_gallery_column(st.container(), scanner)
        except Exception as exc:
            st.warning(f"Stranger gallery render failed: {exc}")

    # ----------------------------------------------------------------
    # Patch :: Class Scheduling & Attendance pages.
    # Each page is implemented in ui/scheduling_pages.py and takes the
    # config + config_path. They construct their own ScheduleManager
    # + AttendanceEngine instances (read-only on the dashboard side;
    # main.py does the writes).
    # ----------------------------------------------------------------
    elif _page_selector == "Schedule":
        _step("rendering Schedule page")
        if not _SCHED_PAGES_AVAILABLE:
            st.error(f"Scheduling pages unavailable: {_SCHED_PAGES_ERROR}")
        else:
            try:
                render_schedule_page(config, config_path)
            except Exception as exc:
                logger.error("Schedule page crashed: %s\n%s", exc, traceback.format_exc())
                st.error(f"Schedule page failed: {exc}")

    elif _page_selector == "Students":
        _step("rendering Students page")
        if not _SCHED_PAGES_AVAILABLE:
            st.error(f"Scheduling pages unavailable: {_SCHED_PAGES_ERROR}")
        else:
            try:
                render_students_page(config, config_path)
            except Exception as exc:
                logger.error("Students page crashed: %s\n%s", exc, traceback.format_exc())
                st.error(f"Students page failed: {exc}")

    elif _page_selector == "Enroll Student":
        # Patch: Student Enrollment page. Self-contained in this module
        # (no scheduling_pages dependency). Spawns enroll.py as a
        # non-blocking subprocess; see _render_enroll_page() above.
        _step("rendering Enroll Student page")
        try:
            _render_enroll_page(st.container(), config, config_path)
        except Exception as exc:
            logger.error("Enroll page crashed: %s\n%s",
                         exc, traceback.format_exc())
            st.error(f"Enroll page failed: {exc}")

    elif _page_selector == "Face Database":
        # Patch: Face Database page. Reads data/student_db.pickle
        # directly + "Add Photos" form that spawns enroll.py --add-to.
        _step("rendering Face Database page")
        try:
            _render_face_db_page(st.container(), config, config_path)
        except Exception as exc:
            logger.error("Face DB page crashed: %s\n%s",
                         exc, traceback.format_exc())
            st.error(f"Face DB page failed: {exc}")

    elif _page_selector == "Live Attendance":
        _step("rendering Live Attendance page")
        if not _SCHED_PAGES_AVAILABLE:
            st.error(f"Scheduling pages unavailable: {_SCHED_PAGES_ERROR}")
        else:
            try:
                render_live_attendance_page(config, config_path)
            except Exception as exc:
                logger.error("Live Attendance page crashed: %s\n%s", exc, traceback.format_exc())
                st.error(f"Live Attendance page failed: {exc}")

    elif _page_selector == "Reports":
        _step("rendering Reports page")
        if not _SCHED_PAGES_AVAILABLE:
            st.error(f"Scheduling pages unavailable: {_SCHED_PAGES_ERROR}")
        else:
            try:
                render_reports_page(config, config_path)
            except Exception as exc:
                logger.error("Reports page crashed: %s\n%s", exc, traceback.format_exc())
                st.error(f"Reports page failed: {exc}")

    # ----------------------------------------------------------------
    # Patch 34/44 :: Auto-refresh trigger (placed AFTER rendering).
    #
    # Patch 44 :: Broader exception handling. The original code only
    # caught ImportError from streamlit_autorefresh. If st_autorefresh
    # itself raised any other exception (StreamlitAPIException,
    # RuntimeError, etc.), it would propagate and crash the Streamlit
    # server process. Now we catch ALL exceptions, log them, and fall
    # back to time.sleep + st.rerun().
    #
    # Patch :: Class Scheduling -- the Live Attendance page installs
    # its OWN auto-refresh (every 10s, matching the schedule's
    # poll_interval_s). When the user is on that page, we SKIP the
    # main 2s auto-refresh so the two don't fight (and so the Live
    # Attendance page doesn't get re-rendered 5x more often than
    # necessary, which would just burn CPU for no benefit).
    # ----------------------------------------------------------------
    if _page_selector == "Live Attendance":
        # The page already called st_autorefresh() internally. Skip
        # the main auto-refresh to avoid double-scheduling.
        _refresh_scheduled = True
        logger.debug(
            "Dashboard: skipping main auto-refresh (Live Attendance "
            "page has its own)."
        )
    else:
        _step("scheduling auto-refresh")
        _refresh_scheduled = False
        _refresh_error: Any = None
        try:
            from streamlit_autorefresh import st_autorefresh
            # No key= parameter (Patch 34).
            st_autorefresh(interval=refresh_interval_ms)
            _refresh_scheduled = True
            logger.info("Dashboard: st_autorefresh scheduled (interval=%dms)", refresh_interval_ms)
        except ImportError:
            # Patch 51 :: Demoted to DEBUG -- this fires on EVERY rerun
            # cycle (every 1s with the new interval), so INFO-level
            # logging produces ~1 line/sec just for this. The startup
            # warning below handles the user-facing notification.
            logger.debug(
                "Dashboard: streamlit_autorefresh not installed -- "
                "using fallback (sleep + st.rerun)"
            )
            _refresh_scheduled = False
        except Exception as exc:
            # Patch 44 :: Catch ALL exceptions, not just ImportError.
            _refresh_error = exc
            logger.error(
                "Dashboard: st_autorefresh raised %s: %s -- "
                "falling back to sleep + st.rerun()",
                type(exc).__name__, exc,
            )
            _refresh_scheduled = False

    if not _refresh_scheduled:
        # Patch 51 :: _step already demoted to DEBUG + rate-limited.
        _step("running fallback refresh (sleep + st.rerun)")
        try:
            time.sleep(refresh_interval_ms / 1000.0)
            st.rerun()
        except Exception as exc:
            logger.error(
                "Dashboard: fallback refresh (st.rerun) raised %s: %s",
                type(exc).__name__, exc,
            )
            # If st.rerun() itself fails, we can't do much. Let it
            # propagate so Streamlit's script runner handles it.

    _step("render block completed")
    logger.info("Dashboard: main() render cycle completed successfully")

    # ----------------------------------------------------------------
    # Patch 57 :: periodic gc.collect() at end of every render.
    # The render loop materializes ~17 MB of transient objects per
    # rerun (history list, window list, rows list, pandas df, 6
    # matplotlib figures). Python's cyclic GC eventually frees them,
    # but on a 1-sec rerun cadence the allocator can fall behind,
    # causing sustained high-water-mark memory. A forced gc.collect()
    # at the end of each render keeps the working set tight.
    # ----------------------------------------------------------------
    try:
        gc.collect()
    except Exception:
        pass


# Patch the UDPTelemetryReceiver to expose a public `is_running` accessor
# (added at module-load time so the dashboard's main() can query state).
def _receiver_is_running_get(self: UDPTelemetryReceiver) -> bool:
    return self._running

UDPTelemetryReceiver.is_running_get = _receiver_is_running_get  # type: ignore


# ---------------------------------------------------------------------------
# Patch 63 (hotfix F) :: Module-level SIGINT/SIGTERM force-exit handler.
#
# This is installed at MODULE LOAD TIME (not inside __main__) so it works
# BOTH when run via `python dashboard.py` AND when run via
# `streamlit run dashboard.py`. In the streamlit case, Streamlit's own
# signal handler catches Ctrl+C first, but during its shutdown sequence it
# re-executes the script multiple times. Each re-run calls main(), which
# checks _DASHBOARD_SHUTTING_DOWN and returns early. This breaks the
# start/stop cycle that produced 40+ seconds of "loop exited / thread
# joined cleanly" spam.
#
# When run via `python dashboard.py` (the __main__ block below also
# installs this handler, but installing it here too is harmless and
# ensures coverage even if the __main__ block is never reached).
# ---------------------------------------------------------------------------
def _patch63_force_exit_handler(signum, frame):
    # Patch 63 (hotfix J) :: Write to _DASH_STATE (persistent on sys)
    # instead of the module-level bool, which gets wiped on every
    # Streamlit re-run. This ensures the shutdown flag survives across
    # re-executions and main()'s early-return guard works correctly.
    _DASH_STATE["dashboard_shutting_down"] = True
    try:
        logger.info(
            "Dashboard: signal %d received -- force-exiting immediately.",
            signum,
        )
    except Exception:
        pass
    os._exit(0)


try:
    import signal as _patch63_signal
    _patch63_signal.signal(
        _patch63_signal.SIGINT, _patch63_force_exit_handler,
    )
    if hasattr(_patch63_signal, "SIGTERM"):
        _patch63_signal.signal(
            _patch63_signal.SIGTERM, _patch63_force_exit_handler,
        )
except (ValueError, OSError, ImportError):
    # signal.signal() raises ValueError if not in the main thread
    # (e.g. when Streamlit imports this module from a worker thread).
    # In that case, the __main__ block's handler will cover the
    # `python dashboard.py` case, and the _DASHBOARD_SHUTTING_DOWN
    # flag covers the `streamlit run` case.
    pass


# ============================================================================
# Module Entry Point (Patch 45 :: Auto-restart on crash, mirror of Patch 39)
# ============================================================================
# NOTE: When dashboard.py is launched via `streamlit run`, this __main__
# block does NOT execute -- streamlit imports dashboard.py as a module
# and calls main() itself. For `streamlit run` crashes, use the
# generated `run_dashboard.bat` (Windows) or `run_dashboard.sh` (Linux)
# wrapper script, which restarts the `streamlit run` command itself.
#
# This in-process restart loop only activates when dashboard.py is run
# directly via `python dashboard.py` (e.g. for debugging).
if __name__ == "__main__":
    import time as _time

    # Note: The SIGINT/SIGTERM force-exit handler is installed at
    # module-load time (see _patch63_force_exit_handler above), so it
    # is already active by the time __main__ runs. No need to reinstall.

    MAX_RESTARTS_DASH: int = 100
    RESTART_DELAY_S_DASH: float = 2.0
    STABLE_RESET_S_DASH: float = 60.0
    restart_count_dash: int = 0
    first_run_start_dash: float = _time.time()

    while True:
        run_start_dash: float = _time.time()
        try:
            main()
            # Normal return -- clean exit (Q key, browser close, etc.)
            logger.info(
                "Dashboard main() returned normally -- exiting restart loop."
            )
            break
        except KeyboardInterrupt:
            logger.info(
                "KeyboardInterrupt (Ctrl+C) received -- exiting. No auto-restart."
            )
            break
        except SystemExit as exc:
            logger.info(
                "SystemExit (code=%s) -- treating as clean shutdown. No auto-restart.",
                exc.code,
            )
            break
        except Exception as exc:
            restart_count_dash += 1
            runtime_s_dash = _time.time() - run_start_dash
            logger.error(
                "Dashboard CRASHED (#%d): %s\n%s",
                restart_count_dash, exc, traceback.format_exc(),
            )
            # Reset counter after STABLE_RESET_S_DASH seconds of stable runtime.
            if runtime_s_dash >= STABLE_RESET_S_DASH:
                if restart_count_dash > 1:
                    logger.info(
                        "Dashboard stable for %.1fs -- resetting restart counter "
                        "(was %d)",
                        runtime_s_dash, restart_count_dash,
                    )
                restart_count_dash = 1
                first_run_start_dash = _time.time()

            if restart_count_dash >= MAX_RESTARTS_DASH:
                logger.error(
                    "Dashboard: MAX_RESTARTS (%d) reached -- giving up. "
                    "Total runtime: %.1fs",
                    MAX_RESTARTS_DASH,
                    _time.time() - first_run_start_dash,
                )
                break

            logger.warning(
                "Dashboard: restarting in %.1fs (attempt %d/%d)...",
                RESTART_DELAY_S_DASH, restart_count_dash, MAX_RESTARTS_DASH,
            )
            _time.sleep(RESTART_DELAY_S_DASH)
            # Loop back to the top -- main() will be called again.
            continue
        break

    # ----------------------------------------------------------------
    # Patch 63 (hotfix F, revised) :: NO cleanup here.
    #
    # The previous code (hotfix E + hotfix F v1) stopped cached
    # receivers and called os._exit(0) here. This was catastrophically
    # wrong for the `streamlit run` case:
    #
    #   When Streamlit runs dashboard.py, it sets __name__ to
    #   "__main__", so this block executes. The restart loop calls
    #   main(), which does ONE render cycle and returns (Streamlit's
    #   normal execution model — the script runs once per render,
    #   then Streamlit waits for the next autorefresh/user interaction
    #   to re-run it). The loop sees "returned normally" -> breaks ->
    #   cleanup ran -> os._exit(0) -> PROCESS DIES INSTANTLY after
    #   the first render.
    #
    # The fix: do NOTHING here. The receivers must stay alive across
    # Streamlit re-runs (they're cached as singletons in
    # _COMPONENTS_CACHE precisely for this purpose). The SIGINT/SIGTERM
    # handler (_patch63_force_exit_handler, installed at module load)
    # is the ONLY place that force-exits, and it only fires on Ctrl+C
    # or kill -- not on normal render completion.
    # ----------------------------------------------------------------
    logger.debug("Dashboard: __main__ block completed -- returning to Streamlit.")
