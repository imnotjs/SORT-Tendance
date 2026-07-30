"""
SORT-tendance :: main.py

Root Orchestrator :: Quad-Threaded Execution Architecture.

THIS FILE MUST BE THE FIRST USER-MODULE IMPORTED IN THE SORT-TENDANCE
PROCESS. It imports `src/utils/gpu_linker.py` at the absolute top of its
own import block (BEFORE importing torch / onnxruntime / ultralytics /
insightface / numpy-with-MKL) so that Windows DLL directories for CUDA,
cuBLAS, cuDNN, and ONNX Runtime's `capi` layer are registered via
`os.add_dll_directory(...)` before any lazy dlopen() can raise Win32
Error 126.

Thread Architecture (CPU affinity pinned via psutil):

  +--------------------------------------------------------------------+
  |                       main.py orchestrator                         |
  +--------------------------------------------------------------------+
  |                                                                    |
  |  +--------------+        +-----------------+       +-------------+ |
  |  | Thread 1     | frame  | Thread 3        | bbox  | Main Thread | |
  |  | Camera       | -----> | Isolated AI     | ----> | Graphics    | |
  |  | Capture      |  q(2)  | (YOLO+BoTSORT+  | clean | GUI         | |
  |  | (CAP_DSHOW)  |        |  InsightFace)   | bbox  | (cv2 window)| |
  |  +--------------+        +-----------------+       +-------------+ |
  |         |                         ^                      ^         |
  |         |                         |                      |         |
  |         v                         |                      |         |
  |  +--------------+        +-----------------+              |         |
  |  | Thread 2     | <----- | VideoRecorder   | <------------+         |
  |  | Async        | push   | Engine (PyAV /  |  push (str.  bboxes)   |
  |  | Recorder     | frame  | NVENC / libx264)|                        |
  |  +--------------+        +-----------------+                        |
  |                                                                    |
  +--------------------------------------------------------------------+

  * Thread 1 (Camera Capture):
      OpenCV with cv2.CAP_DSHOW backend (DirectShow -- selected over MSMF
      after benchmarking because it honors CAP_PROP_AUTOFOCUS / CAP_PROP_FOCUS
      on this hardware; MSMF silently no-ops those properties), MJPG FourCC,
      1280x720 @ 50 FPS.
      IMPORTANT property-set order: WIDTH -> HEIGHT -> FPS -> FOURCC. Setting
      FOURCC before FPS/DIMENSIONS causes DSHOW to ignore the MJPG override
      and silently select YUY2 (uncompressed), which USB-2.0 bandwidth caps
      at ~11 FPS. See _open_capture() for the exact ordering contract.
      Enforces hardware driver verification checks to log and mitigate
      silent stream degradation. Pushes raw frames onto a strict
      queue.Queue(maxsize=2); on queue saturation, drops the OLDEST
      frame to eliminate bounding box lagging.

  * Thread 2 (Asynchronous Recorder):
      Owns and drives the background VideoRecorderEngine worker loop.
      Pulls frames from the AI thread (after anonymization overlay
      decisions have been computed) and pushes them to the recorder
      queue.

  * Thread 3 (Isolated AI Thread):
      Feeds from the strict queue.Queue(maxsize=2). Runs the full
      computer-vision stack: YOLOv8 detection + BoTSORT tracking +
      InsightFace LightFaceEngine matching. Projects CleanBBox
      objects onto the GUI queue. Enforces the strict frame-dropping
      policy on queue saturation.

  * Main Thread (Graphics GUI):
      High-frequency cv2.imshow loop. Receives CleanBBox objects and
      draws anti-aliased boxes containing ONLY high-level resolved
      tags ('[NRP / Student Name]', '[Stranger_XX]', '[ANOMALY]').
      Rigid Display Decoupling: internal BoTSORT track IDs, hashes,
      or Kalman metrics are NEVER rendered onto the live screen.

Runtime State Alignment Layer:
      Integrates the `verified_nrp_state` persistence mechanism from
      `gating_opt.py`. If a student track approaches the lens and
      causes a sudden bounding box scale expansion that drops the
      tracking connection, the orchestrator enforces the immediate
      inheritance path: the newly generated track ID instantly
      inherits the verified identity of the student via body Re-ID
      matching, suppressing re-identification overhead and accidental
      stranger alarms.

Deterministic BBox Timestamping:
      Every frame fed to the tracking and gating engine extracts the
      exact hardware capture index marker at the precise microsecond
      of birth (capture_us), pinning this metadata to all logging
      transactions and CSV records.

Author: SORT-tendance Engineering
"""

# ============================================================================
# CRITICAL IMPORT ORDER INVARIANT
# ============================================================================
# The GPULinker MUST be imported and invoked BEFORE any machine-learning
# framework. We import it at the absolute top of this module and call
# link_dlls() immediately.
from __future__ import annotations

# Patch 48 :: Enable faulthandler for native-crash diagnostics.
# This installs a SIGSEGV/SIGABRT handler that prints the Python
# stack trace to stderr when a C extension access-violates. Without
# this, native crashes (0xC0000005) die silently with no stack trace.
import faulthandler
faulthandler.enable()

# Patch 62 [10] :: atexit cleanup. Even on sys.exit() / normal Python exit,
# we want cv2.destroyAllWindows() + torch.cuda.empty_cache() to run so the
# CUDA driver doesn't have to forcibly reclaim resources during process
# teardown (which is exactly when 0x50 BSODs occur).
import atexit

def _patch62_atexit_cleanup() -> None:
    try:
        import cv2 as _cv2
        _cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
    except Exception:
        pass

atexit.register(_patch62_atexit_cleanup)

import os
import sys
import time

# Resolve project root.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# --- GPULinker: Windows DLL registration (MUST run before ML imports) ---
try:
    from utils.gpu_linker import GPULinker, link_dlls
    _GPU_LINKER = link_dlls()
except Exception as _gpu_exc:                       # pragma: no cover
    import logging as _logging
    _logging.basicConfig(level=_logging.CRITICAL)
    _logging.critical(
        "GPULinker import or link() failed: %s -- continuing, but CUDA "
        "session creation may fail with Win32 Error 126 on Windows.",
        _gpu_exc,
    )
    _GPU_LINKER = None

# ============================================================================
# Standard library imports.
# ============================================================================
import gc
import json
import time
import queue
import socket
import threading
import logging
import traceback
import datetime as _dt
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

# ============================================================================
# Third-party imports (now safe; DLL directories are registered).
# ============================================================================
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2
    _CV2_AVAILABLE = True
    # Patch 28 :: Silence OpenCV's MSMF/DShow warning spam.
    # When the camera is unplugged or in use by another process,
    # cv2.VideoCapture.read() returns (False, None) AND OpenCV's
    # internal logger prints a WARN-level line on EVERY call:
    #   [ WARN:1@348.053] global cap_msmf.cpp:1816
    #   CvCapture_MSMF::grabFrame videoio(MSMF): can't grab
    #   frame. Error: -2147023832
    # At 200 calls/sec this floods the console and hides real
    # errors. Setting LOG_LEVEL_ERROR keeps genuine backend init
    # failures visible while suppressing the per-frame grab
    # warnings. The capture thread's own _read_errors counter
    # still tracks the failures for our telemetry.
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except (AttributeError, TypeError):
        # Older OpenCV builds may not expose setLogLevel.
        pass
except ImportError:                         # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _YAML_AVAILABLE = False
    yaml = None  # type: ignore

# ============================================================================
# Local SORT-tendance imports.
# ============================================================================
try:
    from utils.database_manager import (
        ConfigRegistry, _LightFaceEngine, ArcFaceAligner,
    )
except ImportError:                         # pragma: no cover
    ConfigRegistry = None  # type: ignore
    _LightFaceEngine = None  # type: ignore
    ArcFaceAligner = None  # type: ignore

try:
    from core.tracking_engine import TrackingEngine, CleanBBox, InternalTrack
except ImportError:                         # pragma: no cover
    TrackingEngine = None  # type: ignore
    CleanBBox = None  # type: ignore
    InternalTrack = None  # type: ignore

try:
    from core.identity_matcher import IdentityMatcher, FaceMatchResult, BodyReIDResult
except ImportError:                         # pragma: no cover
    IdentityMatcher = None  # type: ignore
    FaceMatchResult = None  # type: ignore
    BodyReIDResult = None  # type: ignore

try:
    from core.res_opt_engine import ResourceOptEngine, ThrottleMode
except ImportError:                         # pragma: no cover
    ResourceOptEngine = None  # type: ignore
    ThrottleMode = None  # type: ignore

try:
    from core.gating_opt import GatingEngine, FrameTimestamp, EntityState
except ImportError:                         # pragma: no cover
    GatingEngine = None  # type: ignore
    FrameTimestamp = None  # type: ignore
    EntityState = None  # type: ignore

try:
    from utils.async_logger import AsyncLoggingEngine
except ImportError:                         # pragma: no cover
    AsyncLoggingEngine = None  # type: ignore

try:
    from utils.snap_strangers import (
        SnapStrangersEngine, TriggerReason,
    )
except ImportError:                         # pragma: no cover
    SnapStrangersEngine = None  # type: ignore
    TriggerReason = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.main")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
# CRITICAL: disable propagation. Every sortendance.* logger in this project
# installs its own StreamHandler; if propagation is left at the default True,
# every record gets emitted TWICE -- once by our handler, and again by the
# root logger's handler (installed by `logging.basicConfig` in main()).
# This was the root cause of the duplicate-log-line bug observed when
# running `python main.py`.
logger.propagate = False


def _silence_sortendance_propagation() -> None:
    """
    Walk the entire `sortendance.*` logger tree and disable propagation
    to the root logger.

    Each named logger in this project already has its own StreamHandler
    installed at module-load time. When `main()` later calls
    `logging.basicConfig(...)`, the root logger gets ANOTHER StreamHandler.
    With the default `propagate=True`, every record emitted by a
    `sortendance.*` logger would be printed twice: once by the named
    logger's own handler, and once by the root handler.

    This function is called early in `main()` AFTER basicConfig has run,
    so it can reliably traverse the now-fully-populated logger tree.
    """
    root = logging.getLogger()
    for child_name, child_logger in root.manager.loggerDict.items():
        if not child_name.startswith("sortendance"):
            continue
        if isinstance(child_logger, logging.PlaceHolder):
            continue
        # `child_logger` is a real Logger instance.
        child_logger.propagate = False


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class CapturedFrame:
    """
    A single frame captured by Thread 1, enqueued onto the AI queue.

    `capture_us` is the microsecond-of-epoch at the precise instant
    cv2.VideoCapture.read() returned the frame; this is the
    deterministic hardware-frame-index marker that flows downstream
    to the gating engine and CSV logger.
    """
    frame: Any                    # np.ndarray (H, W, 3) uint8 BGR
    frame_index: int              # Monotonic hardware capture index
    capture_us: int               # Microsecond-of-epoch at frame grab
    # Performance telemetry fields (populated by the capture thread).
    capture_latency_ms: float = 0.0


@dataclass
class RenderPackage:
    """
    A package sent from the AI thread to the GUI main thread.

    Contains the annotated frame (with bbox overlays already drawn
    on the AI thread to keep the GUI loop purely display-focused)
    plus telemetry metadata for the on-screen HUD.
    """
    frame: Any                    # np.ndarray (H, W, 3) uint8 BGR (annotated)
    frame_index: int
    capture_us: int
    clean_bboxes: List[Any]       # List[CleanBBox]
    ai_latency_ms: float = 0.0
    active_track_count: int = 0
    pending_track_count: int = 0
    verified_track_count: int = 0
    stranger_track_count: int = 0
    anomaly_count: int = 0
    throttle_mode: str = "IDLE"


# ============================================================================
# CPU Affinity Helper
# ============================================================================
class AffinityManager:
    """
    Pins the calling thread (and optionally the whole process) to a
    configured CPU core block via psutil.

    On platforms where psutil is unavailable or the affinity call
    fails, the manager logs a warning and continues without pinning.
    """

    # ------------------------------------------------------------------
    def __init__(self, config: Dict[str, Any]) -> None:
        hw_cfg = config.get("hardware", {}).get("cpu", {})
        self._masks: Dict[str, List[int]] = dict(
            hw_cfg.get("affinity_masks", {})
        )
        self._enable_lock: bool = bool(hw_cfg.get("enable_affinity_lock", True))
        self._physical_total: int = int(hw_cfg.get("physical_cores_total", 0))
        self._logical_total: int = int(hw_cfg.get("logical_threads_total", 0))

    # ------------------------------------------------------------------
    def apply_to_process(self, core_list: List[int]) -> bool:
        """Pin the entire process to the given core list.

        P2-M13 fix: the `return False` short-circuit that previously
        occupied the first line of this method (and apply_to_current_thread)
        was dead-code-disabling the entire affinity subsystem. The
        orchestrator's docstring at the top of main.py claims "CPU affinity
        pinned via psutil" which was false. Removing the short-circuit
        restores the documented behavior. If affinity pinning causes issues
        on a specific production box, set `main.affinity.enable_lock: false`
        in config.yaml instead of hard-disabling here.
        """
        if not _PSUTIL_AVAILABLE or not self._enable_lock:
            return False
        try:
            proc = psutil.Process()
            proc.cpu_affinity(core_list)
            logger.info(
                "AffinityManager: process pinned to cores %s", core_list,
            )
            return True
        except (OSError, ValueError, psutil.Error) as exc:
            logger.warning(
                "AffinityManager: process pinning failed for %s: %s",
                core_list, exc,
            )
            return False

    # ------------------------------------------------------------------
    def apply_to_current_thread(self, scope_name: str) -> bool:
        """
        Pin the calling thread to the core block configured for the
        given scope name (e.g. 'capture_thread', 'ai_inference_thread').

        P2-M13 fix: dead `return False` short-circuit removed (see
        apply_to_process docstring above for rationale).
        """
        if not _PSUTIL_AVAILABLE or not self._enable_lock:
            return False
        cores = self._masks.get(scope_name)
        if not cores:
            logger.warning(
                "AffinityManager: no affinity mask for scope '%s'", scope_name,
            )
            return False
        try:
            # On Windows + Linux, psutil.Process() reflects the calling
            # thread's affinity when invoked from a worker thread, but
            # to be safe we use psutil.Process().cpu_affinity() which
            # operates on the current thread on Linux (via sched_setaffinity)
            # and on the whole process on Windows. For per-thread pinning
            # on Windows we would need win32api; we accept process-level
            # pinning as a reasonable approximation for the quad-threaded
            # architecture (each thread tends to inherit the process mask).
            proc = psutil.Process()
            proc.cpu_affinity(cores)
            logger.info(
                "AffinityManager: scope '%s' pinned to cores %s",
                scope_name, cores,
            )
            return True
        except (OSError, ValueError, psutil.Error) as exc:
            logger.warning(
                "AffinityManager: thread pinning failed for scope '%s': %s",
                scope_name, exc,
            )
            return False

    # ------------------------------------------------------------------
    def get_scope(self, scope_name: str) -> Optional[List[int]]:
        return list(self._masks.get(scope_name, []))

    # ------------------------------------------------------------------
    def all_scopes(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self._masks.items()}

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "enable_lock": self._enable_lock,
            "scopes": self.all_scopes(),
            "physical_cores_total": self._physical_total,
            "logical_threads_total": self._logical_total,
            "psutil_available": _PSUTIL_AVAILABLE,
        }


# ============================================================================
# Performance Broadcaster (UDP)
# ============================================================================
class PerformanceBroadcaster:
    """
    Non-blocking UDP broadcaster that emits a TelemetryPacket JSON
    payload to the dashboard's UDPTelemetryReceiver on each AI cycle.

    The broadcast is fire-and-forget: if the dashboard is not running
    or the receiver's socket buffer is full, packets are silently
    dropped (no back-pressure on the orchestrator).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        enabled: bool = True,
    ) -> None:
        self._host: str = str(host)
        self._port: int = int(port)
        self._enabled: bool = bool(enabled)
        self._socket: Optional[socket.socket] = None
        self._lock: threading.Lock = threading.Lock()
        self._packets_sent: int = 0
        self._send_errors: int = 0
        self._last_send_us: int = 0

        if self._enabled:
            try:
                self._socket = socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP,
                )
                self._socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, 65536,
                )
                logger.info(
                    "PerformanceBroadcaster: targeting udp://%s:%d",
                    self._host, self._port,
                )
            except OSError as exc:
                logger.warning(
                    "PerformanceBroadcaster: socket creation failed: %s",
                    exc,
                )
                self._socket = None
                self._enabled = False

    # ------------------------------------------------------------------
    def broadcast(self, packet_dict: Dict[str, Any]) -> bool:
        if not self._enabled or self._socket is None:
            return False
        try:
            payload = json.dumps(packet_dict, default=str).encode("utf-8")
            with self._lock:
                self._socket.sendto(payload, (self._host, self._port))
                self._packets_sent += 1
                self._last_send_us = int(time.time() * 1_000_000)
            return True
        except OSError as exc:
            self._send_errors += 1
            # Patch 63 (hotfix G) :: Log the first 5 send errors at
            # WARNING level (visible at default INFO log config) so
            # the operator can see if sendto() is failing. After the
            # first 5, drop to DEBUG to avoid spam.
            if self._send_errors <= 5:
                logger.warning(
                    "PerformanceBroadcaster: send error #%d: %s "
                    "(target=udp://%s:%d)",
                    self._send_errors, exc,
                    self._host, self._port,
                )
            else:
                logger.debug(
                    "PerformanceBroadcaster: send error (%d total): %s",
                    self._send_errors, exc,
                )
            return False
        except (TypeError, ValueError) as exc:
            self._send_errors += 1
            logger.warning(
                "PerformanceBroadcaster: serialization failed: %s", exc,
            )
            return False

    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None
        self._enabled = False

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "host": self._host,
            "port": self._port,
            "enabled": self._enabled,
            "packets_sent": self._packets_sent,
            "send_errors": self._send_errors,
            "last_send_age_us": (
                int(time.time() * 1_000_000) - self._last_send_us
                if self._last_send_us > 0 else -1
            ),
        }


# ============================================================================
# Session Boundary Watcher (Patch 20 :: 12h CSV / snapshot rotation)
# ----------------------------------------------------------------------------
# Polls local time every 60 s. On every 06:00 and 18:00 crossing, fires
# rotation hooks on:
#   - async_logger   : closes the current CSV handle, opens a new one for
#                      the new half-day session (logs/YYYY-MM-DD_6AM.csv
#                      or logs/YYYY-MM-DD_6PM.csv).
#   - snap_engine    : switches the active write directory to
#                      snap_strangers/YYYY-MM-DD/6AM_Session/ (or 6PM_Session/).
#                      Existing files on disk are NEVER deleted or moved;
#                      only the write pointer moves forward.
#   - identity_matcher: clears the OSNet body-Re-ID dynamic ring buffer.
#                      The static enrolled-student USearch index is untouched.
#   - broadcaster    : pushes a session-change notification packet so the
#                      dashboard can refresh its session selector.
#
# All engine hooks are optional (None-safe). If an engine hasn't been
# constructed yet (e.g. import failed), the watcher just skips that hook.
# ============================================================================
class SessionBoundaryWatcher(threading.Thread):
    """
    Polls local time every 60 s. On every 06:00 and 18:00 crossing,
    fires rotation hooks on async_logger / snap_engine / identity_matcher.
    """

    POLL_INTERVAL_S: float = 60.0

    def __init__(
        self,
        async_logger: Any = None,
        snap_engine: Any = None,
        identity_matcher: Any = None,
        broadcaster: Any = None,
        am_hour: int = 6,
        pm_hour: int = 18,
        tz_local: bool = True,
        # Patch 57 :: gating_engine is needed so we can clear the
        # attendance_final_log and anomaly_log at the session boundary.
        # These audit logs were growing unbounded across sessions
        # (~0.5-2 MB per 12h session, never freed). The CSV log is the
        # canonical audit record, so clearing the in-RAM copy at the
        # boundary is safe.
        gating_engine: Any = None,
    ) -> None:
        super().__init__(daemon=True, name="SessionBoundaryWatcher")
        self._async_logger = async_logger
        self._snap_engine = snap_engine
        self._identity_matcher = identity_matcher
        self._gating_engine = gating_engine
        # P1-H6 fix: late-bound reference to the AI thread, used to reset
        # the AI thread's _clearshot_state mirror at session boundaries.
        # Set by the orchestrator AFTER IsolatedAIThread is constructed
        # (which happens after this watcher is created).
        self._ai_thread = None
        # Patch 35 :: Cache psutil.Process() handle for per-frame
        # CPU% and RSS sampling. cpu_percent() requires repeated
        # calls on the SAME Process object to produce a meaningful
        # delta; constructing a fresh Process() each frame would
        # always return 0.0. When psutil is unavailable, the
        # handle is None and the broadcast emits 0.0/0 (the
        # dashboard renders an empty chart rather than crashing).
        self._psutil_proc = None
        if _PSUTIL_AVAILABLE:
            try:
                self._psutil_proc = psutil.Process()
                # Prime cpu_percent() so the first broadcast call
                # returns a real delta rather than 0.0.
                self._psutil_proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._psutil_proc = None
        self._broadcaster = broadcaster
        self._am_hour = int(am_hour)
        self._pm_hour = int(pm_hour)
        self._tz_local = bool(tz_local)
        self._stop_event = threading.Event()
        # Compute the initial session label so we don't fire a spurious
        # rotation on startup -- the engines are already initialised for
        # the current session by their constructors.
        self._last_session_label: str = self._compute_session_label()
        self._last_session_date: str = self._compute_session_date()

    # ------------------------------------------------------------------
    @staticmethod
    def _now_struct() -> Any:
        # time.localtime() respects the OS timezone (Windows: control
        # panel setting; Linux: TZ env / /etc/localtime). This is what
        # the user means by "6AM local".
        return time.localtime()

    def _compute_session_label(self) -> str:
        """Return '6AM' for the 06:00-17:59 window, '6PM' for 18:00-05:59."""
        t = self._now_struct()
        h = t.tm_hour
        if self._am_hour <= h < self._pm_hour:
            return "6AM"
        return "6PM"

    def _compute_session_date(self) -> str:
        """
        Return the calendar date of the CURRENT session's START boundary.

        - In the AM window (06:00-17:59): session started TODAY at 06:00.
        - In the PM window (18:00-23:59): session started TODAY at 18:00.
        - In the PM-overhang (00:00-05:59): session started YESTERDAY at 18:00.
        """
        import datetime as _dt
        t = self._now_struct()
        today = _dt.date(t.tm_year, t.tm_mon, t.tm_mday)
        if t.tm_hour < self._am_hour:
            # We're in the overhang of yesterday's 6PM session.
            today = today - _dt.timedelta(days=1)
        return today.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    def current_session_label(self) -> str:
        return self._last_session_label

    def current_session_date(self) -> str:
        return self._last_session_date

    # ------------------------------------------------------------------
    def _fire_rotation(self, new_label: str, new_date: str) -> None:
        """Call the rotation hook on every available engine."""
        logger.info(
            "SessionBoundaryWatcher: rotating session %s_%s -> %s_%s",
            self._last_session_date, self._last_session_label,
            new_date, new_label,
        )

        # 1) AsyncLogger -- close current CSV, open the new half-day file.
        if self._async_logger is not None:
            try:
                rotate = getattr(self._async_logger, "rotate_session", None)
                if callable(rotate):
                    rotate(session_label=new_label, session_date=new_date)
                else:
                    logger.warning(
                        "SessionBoundaryWatcher: async_logger has no "
                        "rotate_session() method -- CSV will keep writing "
                        "to the old file. Update async_logger.py."
                    )
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: async_logger.rotate_session "
                    "failed: %s", exc, exc_info=True,
                )

        # 2) SnapStrangersEngine -- switch active write directory.
        #    Existing files on disk are NOT touched.
        if self._snap_engine is not None:
            try:
                rotate = getattr(self._snap_engine, "rotate_session", None)
                if callable(rotate):
                    rotate(session_label=new_label, session_date=new_date)
                else:
                    logger.warning(
                        "SessionBoundaryWatcher: snap_engine has no "
                        "rotate_session() method -- snapshots will keep "
                        "writing to the old folder. Update snap_strangers.py."
                    )
                # Patch 65 :: Also reset the clearshot state (per-track
                # counters + cached stranger labels). New strangers in
                # the new session get fresh Stranger_XX IDs, so the
                # cached labels from the previous session are stale.
                # Existing clearshot PNGs on disk are retained (they're
                # organized by date+session folder via _session_subdir_for_ts).
                reset_cs = getattr(self._snap_engine, "reset_clearshot_state", None)
                if callable(reset_cs):
                    reset_cs()
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: snap_engine.rotate_session "
                    "failed: %s", exc, exc_info=True,
                )

        # 3) IdentityMatcher -- reset OSNet dynamic memory only.
        #    Static enrolled-student USearch index is preserved.
        if self._identity_matcher is not None:
            try:
                reset = getattr(self._identity_matcher, "reset_dynamic_memory", None)
                if callable(reset):
                    reset()
                    logger.info(
                        "SessionBoundaryWatcher: OSNet dynamic memory "
                        "reset at %s_%s boundary.", new_date, new_label,
                    )
                else:
                    logger.warning(
                        "SessionBoundaryWatcher: identity_matcher has no "
                        "reset_dynamic_memory() method -- OSNet memory will "
                        "keep accumulating. Update identity_matcher.py."
                    )
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: identity_matcher."
                    "reset_dynamic_memory failed: %s", exc, exc_info=True,
                )

            # P1-H1 fix: AFTER wiping the OSNet stranger cache, repopulate
            # it from disk clearshots. Without this, a stranger seen at
            # 5:50 PM who re-appears at 6:10 PM gets a NEW Stranger_XX ID
            # (no body-Re-ID match) -- defeating the entire disk-recall
            # system, which previously only fired at process startup.
            try:
                recalled = self._identity_matcher.recall_strangers_from_disk()
                logger.info(
                    "SessionBoundaryWatcher: post-reset recall rebuilt "
                    "%d strangers from disk clearshots.", recalled,
                )
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: post-reset recall failed "
                    "(non-fatal, stranger cache will be empty for new "
                    "session): %s", exc, exc_info=True,
                )

        # P1-H6 fix: reset the AI thread's clearshot mirror state. The
        # snap_engine.reset_clearshot_state() call below clears the
        # engine's _clearshot_counters, but the AI thread keeps its own
        # _clearshot_state[tid]["count"] mirror. Without this reset, a
        # stranger that hit max_per_track=20 in the AM session is
        # permanently blocked from clearshots in the PM session.
        ai_thread = getattr(self, '_ai_thread', None)
        if ai_thread is not None:
            try:
                reset_mirror = getattr(ai_thread, 'reset_clearshot_mirror', None)
                if callable(reset_mirror):
                    reset_mirror()
                    logger.info(
                        "SessionBoundaryWatcher: AI thread clearshot "
                        "mirror reset for new session.",
                    )
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: AI clearshot mirror reset "
                    "failed (non-fatal): %s", exc, exc_info=True,
                )

        # 3b) Patch 57 :: GatingEngine audit-log cleanup + GC + CUDA
        #     cache flush. The attendance_final_log and anomaly_log
        #     were growing unbounded across sessions (~0.5-2 MB per
        #     12h session, never freed). The CSV log is the canonical
        #     audit record, so clearing the in-RAM copy at the boundary
        #     is safe. We also force gc.collect() to defragment the
        #     allocator after the OSNet dynamic index rebuild, and
        #     torch.cuda.empty_cache() to return unused GPU memory
        #     back to the OS (otherwise the CUDA caching allocator
        #     holds it for the process lifetime).
        if self._gating_engine is not None:
            try:
                # P1-H5 fix: atomic dict/list swap instead of in-place
                # .clear(). The AI thread iterates attendance_final_log
                # every frame in telemetry() and evaluate_track(); an
                # in-place .clear() during that iteration raises
                # RuntimeError: dictionary changed size during iteration.
                # Replacing the attribute reference is atomic in CPython
                # (GIL-protected attribute assignment) -- the AI thread
                # sees either the old dict or the new empty one, never
                # a half-cleared one.
                _att_log = getattr(self._gating_engine, 'attendance_final_log', None)
                if isinstance(_att_log, dict):
                    n = len(_att_log)
                    setattr(self._gating_engine, 'attendance_final_log', {})
                    logger.info(
                        "SessionBoundaryWatcher: gating attendance_final_log "
                        "cleared (%d entries dropped).", n,
                    )
                _anom_log = getattr(self._gating_engine, 'anomaly_log', None)
                if isinstance(_anom_log, list):
                    n = len(_anom_log)
                    setattr(self._gating_engine, 'anomaly_log', [])
                    logger.info(
                        "SessionBoundaryWatcher: gating anomaly_log "
                        "cleared (%d entries dropped).", n,
                    )
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: gating audit-log clear "
                    "failed: %s", exc, exc_info=True,
                )

        # Patch 57 :: force Python cyclic GC after the OSNet reset +
        # audit-log clear. This defragments the allocator and frees
        # any circular references the engines may have left behind.
        try:
            import gc as _gc
            collected = _gc.collect()
            logger.info(
                "SessionBoundaryWatcher: gc.collect() freed %d objects.",
                collected,
            )
        except Exception as exc:
            logger.debug(
                "SessionBoundaryWatcher: gc.collect() failed: %s", exc,
            )

        # Patch 57 :: torch.cuda.empty_cache() returns unused GPU
        # memory back to the OS. The CUDA caching allocator otherwise
        # holds it for the process lifetime, which can OOM on long
        # runs. Safe to call even if CUDA is not in use (no-op).
        try:
            import torch as _torch  # type: ignore
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                logger.info(
                    "SessionBoundaryWatcher: torch.cuda.empty_cache() "
                    "called (CUDA cached memory returned to OS)."
                )
        except ImportError:
            pass  # torch not installed -- skip
        except Exception as exc:
            logger.debug(
                "SessionBoundaryWatcher: torch.cuda.empty_cache() "
                "failed: %s", exc,
            )

        # 4) Broadcast a session-change notification so the dashboard
        #    can refresh its session selector immediately.
        if self._broadcaster is not None:
            try:
                broadcast = getattr(self._broadcaster, "broadcast", None)
                if callable(broadcast):
                    broadcast({
                        "session_change": True,
                        "new_session_label": new_label,
                        "new_session_date": new_date,
                        "broadcast_us": int(time.time() * 1_000_000),
            # Patch 35 :: Process-scoped CPU% + RSS bytes for the
            # dashboard's performance graphs. cpu_percent() is
            # normalised across all cores (so on an 8-core box a
            # fully-saturated single thread reads ~12.5%, fully
            # saturated 4 threads reads ~50%). RSS is the non-
            # swapped physical memory this process is using.
            "cpu_percent": (
                float(self._psutil_proc.cpu_percent(interval=None))
                if self._psutil_proc is not None else 0.0
            ),
            "rss_bytes": (
                int(self._psutil_proc.memory_info().rss)
                if self._psutil_proc is not None else 0
            ),
                    })
            except Exception as exc:
                logger.debug(
                    "SessionBoundaryWatcher: broadcaster.broadcast failed: %s",
                    exc,
                )

        self._last_session_label = new_label
        self._last_session_date = new_date

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            "SessionBoundaryWatcher entered | start_session=%s_%s | "
            "am_hour=%d pm_hour=%d",
            self._last_session_date, self._last_session_label,
            self._am_hour, self._pm_hour,
        )
        while not self._stop_event.is_set():
            try:
                new_label = self._compute_session_label()
                new_date = self._compute_session_date()
                if new_label != self._last_session_label or \
                   new_date != self._last_session_date:
                    self._fire_rotation(new_label, new_date)
            except Exception as exc:
                logger.error(
                    "SessionBoundaryWatcher: top-level exception: %s",
                    exc, exc_info=True,
                )
            # Poll every 60 s. Boundary detection latency <= 60 s, which
            # is acceptable for a half-day rotation cadence.
            self._stop_event.wait(self.POLL_INTERVAL_S)

        logger.info(
            "SessionBoundaryWatcher exited | final_session=%s_%s",
            self._last_session_date, self._last_session_label,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()


# ============================================================================
# Thread 1: Camera Capture Thread
# ============================================================================
class CameraCaptureThread(threading.Thread):
    """
    Thread 1 :: DirectShow MJPEG capture at 1280x720 @ 50 FPS.

    Enforces hardware driver verification checks to log and mitigate
    silent stream degradation (e.g. backend falling back to 30 FPS
    without notice). Pushes CapturedFrame instances onto the AI queue
    with strict latest-frame-wins dropping on saturation.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Dict[str, Any],
        ai_queue: "queue.Queue[Optional[CapturedFrame]]",
        stop_event: threading.Event,
        affinity_manager: AffinityManager,
    ) -> None:
        super().__init__(name="sortendance.capture", daemon=True)
        self._config: Dict[str, Any] = config
        self._ai_queue: "queue.Queue[Optional[CapturedFrame]]" = ai_queue
        self._stop_event: threading.Event = stop_event
        self._affinity: AffinityManager = affinity_manager

        cam_cfg = config.get("camera", {})
        # Backend selection. Default is CAP_DSHOW (DirectShow) -- chosen after
        # benchmarking because it honors CAP_PROP_AUTOFOCUS / CAP_PROP_FOCUS
        # on this hardware. The OpenCV MSMF backend silently no-ops those
        # properties on Windows (verified via set/readback: AUTOFOCUS and
        # FOCUS write calls return True but the values are discarded).
        # MSMF still works as a fallback if DSHOW is unavailable.
        self._backend: str = str(cam_cfg.get("backend", "CAP_DSHOW"))
        self._device_index: int = int(cam_cfg.get("device_index", 0))
        self._width: int = int(cam_cfg.get("width", 1280))
        self._height: int = int(cam_cfg.get("height", 720))
        self._target_fps: int = int(cam_cfg.get("target_fps", 50))
        self._fourcc: str = str(cam_cfg.get("fourcc", "MJPG"))
        # Focus control (DSHOW only). On MSMF these properties are silently
        # ignored. `autofocus=False` + a fixed `focus` value is the recommended
        # operating mode for attendance scenarios -- continuous AF causes
        # periodic bbox-scale jitter when subjects approach the lens, which
        # breaks BoTSORT track continuity. Focus range is 0..1023 on this
        # driver; value 0 disables autofocus and locks the lens at the
        # nearest focal plane.
        self._autofocus: bool = bool(cam_cfg.get("autofocus", False))
        self._focus: int = int(cam_cfg.get("focus", 0))
        # Patch 46 :: buffer_size and auto_exposure are intentionally NOT
        # read or applied. Touching CAP_PROP_BUFFERSIZE>1 or
        # CAP_PROP_AUTO_EXPOSURE on Windows DSHOW / MSMF frequently triggers
        # silent FPS degradation (driver AE loop, slower polling).
        self._flip_h: bool = bool(cam_cfg.get("flip_horizontal", False))
        self._flip_v: bool = bool(cam_cfg.get("flip_vertical", False))
        self._fps_verify_window_ms: int = int(cam_cfg.get("fps_verify_window_ms", 2000))

        self._cap: Any = None                     # cv2.VideoCapture
        self._frame_index: int = 0
        self._fps_history: Deque[float] = deque(maxlen=60)
        self._last_frame_us: int = 0
        self._dropped_full_queue: int = 0
        self._read_errors: int = 0
        self._actual_fps: float = 0.0
        self._warmup_complete: bool = False

        # Patch 28 :: Camera reconnect-with-backoff state.
        # When cap.read() fails consecutively for too long, we close
        # the device and attempt to reopen it with exponential backoff.
        # This handles transient camera disconnects (USB cable jiggle,
        # another process briefly grabbing the camera, driver reset)
        # without spinning at 200 Hz and flooding the log.
        # _camera_lost_logged ensures we log the camera-loss event
        # ONCE per outage, not every 100 read errors.
        self._consecutive_read_failures: int = 0
        self._camera_lost_logged: bool = False
        self._reopen_backoff_s: float = 0.5
        self._reopen_backoff_max_s: float = 5.0

    # ------------------------------------------------------------------
    def _resolve_backend(self) -> int:
        if not _CV2_AVAILABLE:
            return 0
        # Default to CAP_DSHOW (DirectShow) -- chosen after benchmarking
        # because it honors CAP_PROP_AUTOFOCUS / CAP_PROP_FOCUS on this
        # hardware. MSMF silently no-ops those properties on Windows.
        # MSMF remains the fallback if the named backend attr is missing.
        return getattr(cv2, self._backend, cv2.CAP_DSHOW)

    # ------------------------------------------------------------------
    def _open_capture(self) -> bool:
        if not _CV2_AVAILABLE:
            logger.error("CameraCaptureThread: cv2 not available.")
            return False
        backend = self._resolve_backend()
        try:
            self._cap = cv2.VideoCapture(self._device_index, backend)
        except cv2.error as exc:
            logger.error(
                "CameraCaptureThread: VideoCapture open failed: %s", exc,
            )
            return False

        if self._cap is None or not self._cap.isOpened():
            logger.error(
                "CameraCaptureThread: VideoCapture did not open "
                "(device=%d backend=%s).",
                self._device_index, self._backend,
            )
            return False

        # ----------------------------------------------------------------------
        # CRITICAL: Property-set ORDER.
        # ----------------------------------------------------------------------
        # On this hardware, DirectShow only honors FOURCC=MJPG when the
        # properties are applied in this exact order:
        #
        #     WIDTH  ->  HEIGHT  ->  FPS  ->  FOURCC  ->  AUTOFOCUS  ->  FOCUS
        #
        # Setting FOURCC *before* FPS/DIMENSIONS causes DSHOW to ignore
        # the MJPG override and silently select YUY2 (uncompressed),
        # which USB-2.0 bandwidth caps at ~11 FPS at 1280x720. This was
        # confirmed empirically via bench_dshow_force_mjpg.py and
        # bench_dshow_1280x720_final.py.
        #
        # AUTOFOCUS / FOCUS only work on DSHOW; on MSMF they are silently
        # ignored (the setter returns True but the value is discarded).
        # We set them unconditionally -- harmless on MSMF, essential on
        # DSHOW for fixed-focus attendance scenarios.
        #
        # We deliberately do NOT touch:
        #   - CAP_PROP_BUFFERSIZE      (DShow/MSMF >1 forces slower polling)
        #   - CAP_PROP_AUTO_EXPOSURE   (driver AE loop tanks FPS to ~10.7)
        #   - any other CAP_PROP_AUTO_* (driver-specific, often destructive)
        # ----------------------------------------------------------------------
        try:
            fourcc_code = cv2.VideoWriter_fourcc(*self._fourcc)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS,          self._target_fps)
            self._cap.set(cv2.CAP_PROP_FOURCC,       fourcc_code)   # MUST be after W/H/FPS
            # Focus control: disable continuous AF first, then set the
            # fixed focus position. On this driver the AUTOFOCUS=0 set
            # may report readback=2 (driver quirk) but the subsequent
            # FOCUS=0 set still locks the lens at the nearest focal
            # plane -- visually verified.
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self._autofocus else 0)
            self._cap.set(cv2.CAP_PROP_FOCUS,      self._focus)
        except cv2.error as exc:
            logger.warning(
                "CameraCaptureThread: property set failed (continuing): %s",
                exc,
            )

        # Verify the driver actually accepted our settings.
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        actual_fcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        try:
            actual_fcc_str = "".join(
                chr((actual_fcc_int >> (8 * k)) & 0xFF) for k in range(4)
            ).strip() or "(empty)"
        except Exception:
            actual_fcc_str = "(?)"
        actual_focus = self._cap.get(cv2.CAP_PROP_FOCUS)

        logger.info(
            "CameraCaptureThread: opened | backend=%s | requested=%dx%d@%dfps %s | "
            "actual=%dx%d@%.1ffps %s | focus=%s (set=%d) | af_mode=%s",
            self._backend, self._width, self._height, self._target_fps, self._fourcc,
            actual_w, actual_h, actual_fps, actual_fcc_str,
            actual_focus, self._focus, "AUTO" if self._autofocus else "MANUAL",
        )

        if actual_w != self._width or actual_h != self._height:
            logger.warning(
                "CameraCaptureThread: SILENT RESOLUTION DEGRADATION detected "
                "(requested %dx%d, got %dx%d).",
                self._width, self._height, actual_w, actual_h,
            )
        if actual_fps > 0 and actual_fps < self._target_fps * 0.8:
            logger.warning(
                "CameraCaptureThread: SILENT FPS DEGRADATION detected "
                "(requested %d, got %.1f).",
                self._target_fps, actual_fps,
            )
        # FOURCC mismatch is the smoking gun for the DSHOW YUY2 fallback.
        # If we requested MJPG and got something else, the property-set
        # order in _open_capture() has been broken.
        if (
            self._fourcc
            and actual_fcc_str.upper() != self._fourcc.upper()
            and actual_fcc_str != "(empty)"
        ):
            logger.error(
                "CameraCaptureThread: FOURCC MISMATCH detected "
                "(requested %s, got %s). DSHOW likely fell back to YUY2 "
                "and the stream will be USB-bandwidth-limited to ~11 FPS. "
                "Check property-set order in _open_capture(): must be "
                "W -> H -> FPS -> FOURCC (FOURCC LAST).",
                self._fourcc, actual_fcc_str,
            )
        # Focus readback verification (DSHOW only). On MSMF this check is
        # a no-op (readback returns the same default value regardless of
        # what was set).
        if not self._autofocus and self._backend == "CAP_DSHOW":
            if abs(actual_focus - self._focus) > 1.0:
                logger.warning(
                    "CameraCaptureThread: FOCUS SET DID NOT TAKE "
                    "(requested %d, read back %.1f). Lens may still be "
                    "in autofocus mode.",
                    self._focus, actual_focus,
                )
        return True

    # ------------------------------------------------------------------
    def _verify_fps_window(self) -> None:
        """Verify the actual achieved FPS over a verification window."""
        if not _CV2_AVAILABLE or self._cap is None:
            return
        window_start = time.time()
        frames_seen = 0
        while (
            time.time() - window_start < self._fps_verify_window_ms / 1000.0
            and not self._stop_event.is_set()
        ):
            ok, _ = self._cap.read()
            if ok:
                frames_seen += 1
            else:
                time.sleep(0.001)
        elapsed = time.time() - window_start
        if elapsed > 0:
            self._actual_fps = frames_seen / elapsed
        logger.info(
            "CameraCaptureThread: FPS verification window | %.2fs | "
            "%d frames | actual=%.1f fps | target=%d fps",
            elapsed, frames_seen, self._actual_fps, self._target_fps,
        )
        if self._actual_fps < self._target_fps * 0.7:
            logger.warning(
                "CameraCaptureThread: ACHIEVED FPS < 70%% of target -- "
                "consider reducing target_fps or verifying driver support.",
            )

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("CameraCaptureThread entered.")
        self._affinity.apply_to_current_thread("capture_thread")

        if not self._open_capture():
            self._stop_event.set()
            return

        # FPS verification window.
        self._verify_fps_window()
        self._warmup_complete = True

        last_read_us = int(time.time() * 1_000_000)
        while not self._stop_event.is_set():
            try:
                if self._cap is None or not self._cap.isOpened():
                    logger.error(
                        "CameraCaptureThread: capture closed unexpectedly; "
                        "attempting reopen.",
                    )
                    if not self._open_capture():
                        time.sleep(0.5)
                        continue

                t_read_start = time.time()
                ok, frame = self._cap.read()
                t_read_end = time.time()
                if not ok or frame is None:
                    # Patch 28 :: Reconnect-with-backoff instead of
                    # spin-looping at 200 Hz. After 30 consecutive
                    # read failures (~150 ms at the 5 ms sleep below),
                    # we close the device and attempt to reopen it
                    # with exponential backoff (0.5s -> 1s -> 2s ->
                    # 4s -> 5s cap). This handles transient camera
                    # disconnects without flooding the log.
                    self._read_errors += 1
                    self._consecutive_read_failures += 1

                    if self._consecutive_read_failures < 30:
                        # Brief sleep, then retry -- handles single-
                        # frame dropouts without a full reopen.
                        time.sleep(0.005)
                        continue

                    # We've lost the camera. Log ONCE per outage.
                    if not self._camera_lost_logged:
                        logger.error(
                            "CameraCaptureThread: camera lost "
                            "(read failure #%d, %d consecutive) -- "
                            "entering reconnect-with-backoff mode. "
                            "Check: camera plugged in? Another app "
                            "(Zoom/Teams/OBS) using it? USB cable?",
                            self._read_errors,
                            self._consecutive_read_failures,
                        )
                        self._camera_lost_logged = True

                    # Close the defunct capture handle.
                    try:
                        if self._cap is not None:
                            self._cap.release()
                            self._cap = None
                    except Exception:
                        pass

                    # Backoff before reopen.
                    time.sleep(self._reopen_backoff_s)
                    # Exponential backoff, capped.
                    self._reopen_backoff_s = min(
                        self._reopen_backoff_s * 2.0,
                        self._reopen_backoff_max_s,
                    )

                    # Attempt to reopen. _open_capture() returns
                    # False on failure; we just loop and try again
                    # after the next backoff tick.
                    if not self._open_capture():
                        continue

                    # Reopen succeeded -- reset state and continue.
                    logger.info(
                        "CameraCaptureThread: camera reopened "
                        "successfully after %d consecutive failures. "
                        "Resuming capture.",
                        self._consecutive_read_failures,
                    )
                    self._consecutive_read_failures = 0
                    self._camera_lost_logged = False
                    self._reopen_backoff_s = 0.5
                    continue

                # Patch 28 :: Reset backoff state on a successful read.
                if self._consecutive_read_failures > 0:
                    self._consecutive_read_failures = 0
                    self._camera_lost_logged = False
                    self._reopen_backoff_s = 0.5

                # Apply flips if configured.
                if self._flip_h and _CV2_AVAILABLE:
                    frame = cv2.flip(frame, 1)
                if self._flip_v and _CV2_AVAILABLE:
                    frame = cv2.flip(frame, 0)

                now_us = int(time.time() * 1_000_000)
                capture_latency_ms = (t_read_end - t_read_start) * 1000.0

                # FPS history (rolling).
                if self._last_frame_us > 0:
                    delta_s = (now_us - self._last_frame_us) / 1_000_000.0
                    if delta_s > 0:
                        self._fps_history.append(1.0 / delta_s)
                self._last_frame_us = now_us

                self._frame_index += 1
                captured = CapturedFrame(
                    frame=frame,
                    frame_index=self._frame_index,
                    capture_us=now_us,
                    capture_latency_ms=capture_latency_ms,
                )

                # Non-blocking enqueue with latest-frame-wins dropping.
                try:
                    self._ai_queue.put_nowait(captured)
                except queue.Full:
                    self._dropped_full_queue += 1
                    try:
                        self._ai_queue.get_nowait()  # Drop oldest.
                        self._ai_queue.put_nowait(captured)
                    except (queue.Empty, queue.Full):
                        pass

            except Exception as exc:
                logger.error(
                    "CameraCaptureThread: top-level exception: %s\n%s",
                    exc, traceback.format_exc(),
                )
                time.sleep(0.05)

        # Cleanup.
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info(
            "CameraCaptureThread exited | frames=%d | dropped=%d | "
            "read_errors=%d | avg_fps=%.1f",
            self._frame_index, self._dropped_full_queue,
            self._read_errors, self.current_fps(),
        )

    # ------------------------------------------------------------------
    def current_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return float(np.mean(list(self._fps_history))) if _NUMPY_AVAILABLE else (
            sum(self._fps_history) / len(self._fps_history)
        )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "frame_index": self._frame_index,
            "actual_fps": self.current_fps(),
            "dropped_full_queue": self._dropped_full_queue,
            "read_errors": self._read_errors,
            "warmup_complete": self._warmup_complete,
            "resolution": [self._width, self._height],
            "target_fps": self._target_fps,
        }


# ============================================================================
# Thread 2: Isolated AI Thread
# ============================================================================
class IsolatedAIThread(threading.Thread):
    """
    Thread 2 :: Isolated computer-vision stack.

    Drives YOLOv8 + BoTSORT + LightFaceEngine matching, fed from the
    strict queue.Queue(maxsize=2). Enforces the latest-frame-wins
    dropping policy on queue saturation. Projects CleanBBox objects
    onto the GUI queue.

    Responsibilities per frame:
      1. Drain the AI queue (drop stale frames if >1 queued).
      2. Run YOLOv8 + BoTSORT via TrackingEngine.process().
      3. For each track with active face scanning, run LightFaceEngine
         detection + embedding extraction.
      4. Batch-match face embeddings via IdentityMatcher.batch_match_faces_raw().
      5. Update ResourceOptEngine (TTFM, Pose-EMA, Velocity) per track.
      6. Run GatingEngine.evaluate_track() for the cascade decision.
      7. Apply Runtime State Alignment: if a track ID is new AND body
         Re-ID matches a previously-verified track, inherit the label.
      8. Project CleanBBox objects via TrackingEngine.to_clean_bboxes().
      9. Annotate the frame with anti-aliased bboxes + resolved labels.
     10. Push RenderPackage to the GUI queue.
     11. Broadcast telemetry via PerformanceBroadcaster.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Dict[str, Any],
        ai_queue: "queue.Queue[Optional[CapturedFrame]]",
        gui_queue: "queue.Queue[Optional[RenderPackage]]",
        stop_event: threading.Event,
        affinity_manager: AffinityManager,
        tracking_engine: Any,
        identity_matcher: Any,
        res_opt_engine: Any,
        gating_engine: Any,
        face_engine: Any,
        arcface_aligner: Any,
        recorder_thread: AsyncRecorderThread,
        broadcaster: PerformanceBroadcaster,
    ) -> None:
        super().__init__(name="sortendance.ai", daemon=True)
        self._config: Dict[str, Any] = config
        self._ai_queue: "queue.Queue[Optional[CapturedFrame]]" = ai_queue
        self._gui_queue: "queue.Queue[Optional[RenderPackage]]" = gui_queue
        self._stop_event: threading.Event = stop_event
        self._affinity: AffinityManager = affinity_manager
        self._tracking: Any = tracking_engine
        # Patch 43 :: Cached latency telemetry for the anomaly-scan
        # path (so ANOMALY CSV rows get realistic latency values).
        self._last_body_reid_latency_ms: float = 0.0
        self._last_tracking_latency_ms: float = 0.0
        self._matcher: Any = identity_matcher
        self._res_opt: Any = res_opt_engine
        self._gating: Any = gating_engine
        self._face: Any = face_engine
        self._aligner: Any = arcface_aligner
        self._recorder_thread: AsyncRecorderThread = recorder_thread
        self._broadcaster: PerformanceBroadcaster = broadcaster
        self._track_last_face_scan: Dict[int, int] = {}
        self._track_face_cache: Dict[int, Dict[str, Any]] = {}
        # Track-birth body Re-ID inheritance: tracks that have been seen
        # at least once. A track_id absent from this set on the current
        # frame triggers a one-shot body Re-ID inheritance attempt.
        # If the body matches an existing verified student or registered
        # stranger, the track inherits that identity immediately and
        # skips all face scanning -- TTFM drops to ~30ms (one OSNet call).
        self._seen_track_ids: set = set()
        self._resolved_track_ids: set = set()

        main_cfg = config.get("main", {})
        self._render_fps_target: int = int(main_cfg.get("render_fps_target", 60))
        self._enable_deterministic_ts: bool = bool(
            main_cfg.get("enable_deterministic_timestamping", True)
        )
        self._freeze_verified: bool = bool(main_cfg.get("freeze_verified_tracks", True))
        self._freeze_stranger: bool = bool(main_cfg.get("freeze_stranger_tracks", True))

        render_cfg = main_cfg.get("render", {})
        self._box_thickness: int = int(render_cfg.get("box_thickness", 2))
        self._font_scale: float = float(render_cfg.get("font_scale", 0.6))
        self._anti_alias: bool = bool(render_cfg.get("anti_alias", True))
        # Patch 59 :: Bbox render throttle. Draw bboxes on every Nth
        # frame only; in-between frames reuse the last annotated frame
        # from cache. Reduces cv2.rectangle/putText workload by 1/N.
        self._bbox_render_every_n_frames: int = max(1, int(
            render_cfg.get("bbox_render_every_n_frames", 1)
        ))
        self._bbox_render_frame_counter: int = 0
        self._last_annotated_frame: Optional["np.ndarray"] = None

        # Patch 61 :: TTFM recovery throttles.
        # All four throttles below are read from main.render in config.yaml.
        # See config.yaml comments for the per-throttle rationale.
        self._anomaly_scan_every_n_cycles: int = max(1, int(
            render_cfg.get("anomaly_scan_every_n_cycles", 10)
        ))
        self._telemetry_broadcast_every_n_cycles: int = max(1, int(
            render_cfg.get("telemetry_broadcast_every_n_cycles", 5)
        ))
        self._telemetry_vram_every_n_broadcasts: int = max(1, int(
            render_cfg.get("telemetry_vram_every_n_broadcasts", 6)
        ))
        self._face_retry_every_n_scans: int = max(1, int(
            render_cfg.get("face_retry_every_n_scans", 3)
        ))
        # Counters.
        self._broadcast_counter: int = 0
        self._vram_query_counter: int = 0
        self._cached_gpu_vram_used: int = 0
        self._cached_gpu_vram_total: int = 0
        # Patch 63 (hotfix B) :: Heartbeat state. Tracks whether the
        # AI thread is LIVE (processing frames), IDLE (no frames from
        # camera), or ERROR (processing crashes). Broadcast in every
        # telemetry packet so the dashboard can show main.py's status
        # even when no frames are being processed.
        self._heartbeat_state: str = "STARTUP"
        self._idle_cycles: int = 0  # incremented when no frame available
        self._last_heartbeat_broadcast_us: int = 0
        # Birth-frame body feature cache (Patch 61 [C]). Populated when
        # a track is born; consumed by the batched body pass to avoid
        # double OSNet extraction. Cleared at end of _process_frame.
        self._birth_body_cache: Dict[int, Any] = {}

        # Telemetry state.
        self._fps_history: Deque[float] = deque(maxlen=60)
        self._last_ai_us: int = 0
        self._frames_processed: int = 0
        self._frames_dropped_full_gui: int = 0
        self._ai_errors: int = 0
        # Patch 62 [2] :: CUDA consecutive-failure counter.
        # Reset to 0 on every successful _cuda_sync_check(); incremented
        # on every CUDA fault. At >=5, forces a full context reset.
        self._cuda_consecutive_failures: int = 0
        self._last_processing_latency_ms: float = 0.0
        self._last_tracking_latency_ms: float = 0.0

        # Track stranger bboxes for the recorder anonymization hook.
        self._stranger_bboxes_by_track: Dict[int, Tuple[Tuple[int, int, int, int], ...]] = {}

        # ------------------------------------------------------------------
        # Snapshot engine state (Patch: snap_strangers migration).
        # ------------------------------------------------------------------
        # Track IDs born THIS frame. Reset every frame after the snapshot
        # capture pass. Used so we can capture a birth snapshot AFTER
        # _annotate_frame() has drawn the bbox (the bbox is what we want
        # visible in the snapshot).
        self._birth_tids_this_frame: set = set()
        # Track IDs whose snapshot has been finalized (renamed to stranger
        # or deleted as verified). Prevents duplicate finalize calls across
        # frames. Pruned when a track disappears.
        self._snapshot_finalized_tids: set = set()
        self._stranger_bboxes_lock: threading.Lock = threading.Lock()

        # ------------------------------------------------------------------
        # Patch 65 :: CLEARSHOT snapshot state.
        #
        # For each STRANGER-locked track, the AI thread periodically
        # captures an additional "clearshot" PNG when:
        #   * YOLO det_conf >= clearshot.min_yolo_conf (default 0.70)
        #   * bbox width AND height >= clearshot.min_bbox_size (default 80px)
        #   * cooldown (clearshot.cooldown_s, default 30s) elapsed since
        #     the last clearshot for this track
        #   * per-track clearshot count < clearshot.max_per_track (default 20)
        #
        # These serve as OSNet "memory recall" reference frames -- each
        # clearshot is a clean, high-confidence capture that the operator
        # (or a downstream re-extraction pipeline) can use as an
        # additional body-feature reference for the same stranger.
        #
        # _clearshot_last_ts_us[tid] = wall-clock microseconds of the
        #   last clearshot captured for this track. Used for cooldown.
        # _clearshot_state[tid] = {"label": str, "count": int}
        #   Cached stranger label + local count mirror (the snap_strangers
        #   engine also maintains its own counter; this is for fast
        #   cooldown checks without acquiring the engine's lock).
        # ------------------------------------------------------------------
        self._clearshot_last_ts_us: Dict[int, int] = {}
        self._clearshot_state: Dict[int, Dict[str, Any]] = {}
        # Read the clearshot config once at init (cheap).
        _cs_cfg = (
            self._config.get("snap_strangers", {}).get("clearshot") or {}
        )
        self._clearshot_enabled: bool = bool(_cs_cfg.get("enabled", True))
        self._clearshot_min_yolo_conf: float = float(
            _cs_cfg.get("min_yolo_conf", 0.70)
        )
        self._clearshot_min_bbox_size: int = int(
            _cs_cfg.get("min_bbox_size", 80)
        )
        self._clearshot_cooldown_s: float = float(
            _cs_cfg.get("cooldown_s", 30.0)
        )
        self._clearshot_max_per_track: int = int(
            _cs_cfg.get("max_per_track", 20)
        )
        logger.info(
            "IsolatedAIThread: Patch 65 clearshot config | "
            "enabled=%s | min_yolo_conf=%.2f | min_bbox=%dpx | "
            "cooldown=%.1fs | max_per_track=%d",
            self._clearshot_enabled, self._clearshot_min_yolo_conf,
            self._clearshot_min_bbox_size, self._clearshot_cooldown_s,
            self._clearshot_max_per_track,
        )

        # Patch 56 :: Track the set of active tids from the PREVIOUS
        # frame, so we can compute dead_tids (tracks that disappeared)
        # and call drop_track() on the 3 sub-engines to prevent
        # unbounded per-track state growth.
        self._last_active_tids: set = set()
        # Patch 56/57 :: Track the set of active tids from the PREVIOUS
        # frame, so we can compute dead_tids (tracks that disappeared)
        # and call drop_track() on the 3 sub-engines to prevent
        # unbounded per-track state growth. (Patch 57 removed a
        # duplicate assignment that was left here by Patch 56.)

        # Patch 13: Adaptive YOLO interval. When ResOptEngine signals
        # IDLE mode (all tracks resolved), we skip tracking.process()
        # on N-1 of every N frames and reuse the cached track boxes.
        # BoTSORT track_buffer=30 keeps lost tracks alive for 30 frames.
        # New track births force BURST (Patch 7) which re-engages YOLO.
        self._cached_tracks: List[Any] = []
        self._yolo_skip_count: int = 0
        self._yolo_executed_count: int = 0

        # Patch 20: Current session context for telemetry. Updated by
        # the orchestrator's SessionBoundaryWatcher. If the watcher
        # isn't running, _broadcast_telemetry falls back to computing
        # these inline from time.localtime().
        self._current_session_label: Optional[str] = None
        self._current_session_date: Optional[str] = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("IsolatedAIThread entered.")
        self._affinity.apply_to_current_thread("ai_inference_thread")

        if self._tracking is None or self._matcher is None or self._gating is None:
            logger.error(
                "IsolatedAIThread: missing core engine(s) -- "
                "tracking=%s matcher=%s gating=%s",
                self._tracking is not None,
                self._matcher is not None,
                self._gating is not None,
            )
            self._stop_event.set()
            return

        # Patch 63 (hotfix B) :: Startup heartbeat. Fire a telemetry
        # packet immediately so the dashboard knows main.py's AI
        # thread is alive, BEFORE the first frame arrives. Without
        # this, the dashboard shows "main.py has not sent any telemetry
        # yet" for the entire model-loading + camera-warmup period
        # (which can be 5-15 seconds on cold start).
        self._heartbeat_state = "STARTUP"
        try:
            self._broadcast_telemetry(None, None)
            self._last_heartbeat_broadcast_us = int(time.time() * 1_000_000)
            logger.info("IsolatedAIThread: startup heartbeat sent.")
        except Exception as exc:
            logger.warning(
                "IsolatedAIThread: startup heartbeat failed: %s", exc,
            )

        while not self._stop_event.is_set():
            try:
                # 1) Drain the AI queue with latest-frame-wins policy.
                captured: Optional[CapturedFrame] = None
                dropped_count = 0
                while True:
                    try:
                        nxt = self._ai_queue.get_nowait()
                    except queue.Empty:
                        break
                    if captured is not None:
                        dropped_count += 1
                    captured = nxt
                    # If the queue still has items, we keep draining so
                    # we always process the latest frame.
                    if self._ai_queue.empty():
                        break

                if captured is None:
                    # Block briefly on a timeout-bound get to avoid spin.
                    try:
                        captured = self._ai_queue.get(timeout=0.5)
                    except queue.Empty:
                        # Patch 63 (hotfix B) :: Idle heartbeat.
                        # No frame available -- camera is disconnected,
                        # not yet started, or producing frames slower
                        # than the AI thread can consume. Previously
                        # this just `continue`d, meaning the dashboard
                        # would NEVER receive telemetry and would show
                        # "main.py has not sent any telemetry yet"
                        # indefinitely.
                        #
                        # Now: broadcast an IDLE heartbeat every ~1s
                        # (every 2nd idle cycle, since each timeout is
                        # 0.5s). The dashboard can then show
                        # "main.py is running but camera is idle."
                        self._idle_cycles += 1
                        self._heartbeat_state = "IDLE"
                        _now_us = int(time.time() * 1_000_000)
                        if (_now_us - self._last_heartbeat_broadcast_us) >= 1_000_000:
                            try:
                                self._broadcast_telemetry(None, None)
                                self._last_heartbeat_broadcast_us = _now_us
                            except Exception:
                                pass
                        continue

                if dropped_count > 0:
                    logger.debug(
                        "IsolatedAIThread: dropped %d stale frame(s) before "
                        "processing frame_index=%d",
                        dropped_count, captured.frame_index,
                    )

                # 2) Process the frame.
                t_start = time.time()
                render_pkg = self._process_frame(captured)
                t_end = time.time()
                self._last_processing_latency_ms = (t_end - t_start) * 1000.0

                # FPS history.
                now_us = int(time.time() * 1_000_000)
                if self._last_ai_us > 0:
                    delta_s = (now_us - self._last_ai_us) / 1_000_000.0
                    if delta_s > 0:
                        self._fps_history.append(1.0 / delta_s)
                self._last_ai_us = now_us
                self._frames_processed += 1
                # Patch 63 (hotfix B) :: We got a frame -- transition
                # heartbeat from STARTUP/IDLE/ERROR back to LIVE.
                if self._heartbeat_state != "LIVE":
                    self._heartbeat_state = "LIVE"
                    logger.info(
                        "IsolatedAIThread: heartbeat -> LIVE "
                        "(frame_index=%d)",
                        captured.frame_index,
                    )

                # 3) Push to GUI queue (latest-frame-wins dropping).
                if render_pkg is not None:
                    try:
                        self._gui_queue.put_nowait(render_pkg)
                    except queue.Full:
                        self._frames_dropped_full_gui += 1
                        try:
                            self._gui_queue.get_nowait()
                            self._gui_queue.put_nowait(render_pkg)
                        except (queue.Empty, queue.Full):
                            pass

                # 4) Broadcast telemetry.
                # Patch 61 [G] :: Throttle to every Nth AI cycle.
                # Dashboard graphs don't need 50Hz updates; 10Hz is
                # plenty for human-readable trends. Saves ~0.5-1.5ms
                # per skipped cycle (json.dumps + UDP sendto + CUDA
                # API call for VRAM).
                if (self._frames_processed % self._telemetry_broadcast_every_n_cycles) == 0:
                    self._broadcast_telemetry(captured, render_pkg)
                    self._last_heartbeat_broadcast_us = int(time.time() * 1_000_000)

            except Exception as exc:
                self._ai_errors += 1
                logger.error(
                    "IsolatedAIThread: top-level exception (#%d): %s\n%s",
                    self._ai_errors, exc, traceback.format_exc(),
                )
                # Patch 63 (hotfix B) :: Error heartbeat. If
                # _process_frame crashes every cycle, telemetry never
                # fires and the dashboard can't tell that main.py is
                # crashing. Broadcast an ERROR heartbeat every ~1s so
                # the dashboard can show "main.py is crashing repeatedly."
                self._heartbeat_state = "ERROR"
                _now_us = int(time.time() * 1_000_000)
                if (_now_us - self._last_heartbeat_broadcast_us) >= 1_000_000:
                    try:
                        self._broadcast_telemetry(None, None)
                        self._last_heartbeat_broadcast_us = _now_us
                    except Exception:
                        pass
                time.sleep(0.05)

        logger.info(
            "IsolatedAIThread exited | processed=%d | gui_drops=%d | errors=%d",
            self._frames_processed, self._frames_dropped_full_gui, self._ai_errors,
        )

    # ------------------------------------------------------------------
    def _process_frame(self, captured: CapturedFrame) -> Optional[RenderPackage]:
        """Run the full per-frame CV stack on one captured frame."""
        if captured.frame is None:
            return None

        # Build the deterministic FrameTimestamp at the bbox-generation
        # microsecond (i.e. NOW, in the AI thread, NOT at capture time).
        # The capture_us is preserved on FrameTimestamp.capture_us for
        # queue-delay diagnostics.
        bbox_us = int(time.time() * 1_000_000)
        frame_ts: Any = FrameTimestamp(
            frame_index=captured.frame_index,
            capture_us=captured.capture_us,
            bbox_us=bbox_us,
        )

        # --- 2a) YOLO + BoTSORT tracking ---
        #
        # Patch 13: Adaptive YOLO interval. Ask the resource optimizer
        # whether we should skip YOLO on this frame. If yes AND we have
        # cached tracks from a prior frame, reuse them. Otherwise run
        # the full YOLO+BoTSORT inference and refresh the cache.
        #
        # Skip is only triggered when current_mode == IDLE (set at the
        # END of the previous frame by decide_mode). On the very first
        # frame, current_mode defaults to IDLE but _cached_tracks is
        # empty, so we always run YOLO on frame 0.
        skip_yolo_this_frame = (
            self._res_opt is not None
            and self._res_opt.should_skip_yolo(captured.frame_index)
        )

        if skip_yolo_this_frame and self._cached_tracks:
            # Reuse cached tracks from the last YOLO frame. BoTSORT
            # track_buffer=30 keeps lost tracks alive for 30 frames,
            # so reusing boxes for 2 of every 3 frames is safe.
            tracks = self._cached_tracks
            yolo_latency_ms = 0.0
            self._yolo_skip_count += 1
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "IsolatedAIThread: YOLO skipped on frame %d "
                    "(reusing %d cached tracks)",
                    captured.frame_index, len(tracks),
                )
        else:
            t_track_start = time.perf_counter()
            try:
                tracks: List[Any] = self._tracking.process(captured.frame)
                self._cuda_sync_check("tracking.process")
            except Exception as exc:
                logger.error(
                    "IsolatedAIThread: tracking.process failed: %s", exc
                )
                tracks = []
            yolo_latency_ms = (time.perf_counter() - t_track_start) * 1000.0
            self._last_tracking_latency_ms = yolo_latency_ms
            # Refresh the cache with the latest YOLO results so future
            # skip-frames can reuse them.
            self._cached_tracks = tracks
            self._yolo_executed_count += 1

        if not tracks:
            # Empty scene -- still push a render package so the GUI updates.
            annotated = self._annotate_frame_throttled(
                captured.frame, [], frame_ts, None,
            )
            return RenderPackage(
                frame=annotated,
                frame_index=captured.frame_index,
                capture_us=captured.capture_us,
                clean_bboxes=[],
                ai_latency_ms=self._last_processing_latency_ms,
                active_track_count=0,
            )

        # --- 2b) Per-track face detection + embedding extraction ---
        # Build a batch of face embeddings for all tracks with active
        # face scanning.
        #
        # LATENCY ATTRIBUTION (per-track):
        #   yolo_ms        -- global per-frame, same value for every track
        #   face_det_ms    -- only nonzero on the frame we actually ran det
        #   arcface_ms     -- only nonzero on the frame we actually ran rec
        #   usearch_ms     -- filled in later by the batch-match loop
        #
        # For cached-path tracks (face throttle hit): face_det/arcface = 0.0
        # because no detection ran this frame. For skipped tracks (locked
        # or too-small bbox): all-zero, but they don't enter the cascade
        # anyway.
        track_ids_to_match: List[int] = []
        face_embeddings: List[Any] = []
        track_meta: Dict[int, Dict[str, Any]] = {}

        for trk in tracks:
            tid = int(trk.track_id)
            # Patch 63 (hotfix D) :: Skip invalid track IDs.
            # BoTSORT uses -1 as a sentinel for "unassigned" or
            # "invalid detection". Processing these pollutes
            # _seen_track_ids with -1, wastes OSNet extraction on
            # non-existent tracks, and creates bogus stranger
            # snapshots (track=-1_BIRTH.png). Skip any tid < 0.
            if tid < 0:
                continue

            # --- Track-birth body Re-ID inheritance ---
            # If this is the first frame we've seen this track_id, try to
            # inherit an identity from the body Re-ID cache BEFORE doing
            # any face scanning. This handles:
            #   (a) Returning students -- body matches a verified student,
            #       track is immediately resolved, no face scanning needed.
            #   (b) Returning strangers -- body matches an existing
            #       Stranger_XX record, track inherits that Stranger ID,
            #       stranger counter doesn't inflate.
            # TTFM for inheritance case: ~30ms (one OSNet call). Without
            # this patch: full face cascade (80-300ms).
            is_track_birth = tid not in self._seen_track_ids
            if is_track_birth:
                self._seen_track_ids.add(tid)
                # Patch 7 is handled internally by res_opt_engine.decide_mode()
                # (it detects new track_ids by diffing against
                # _seen_track_ids_for_burst and forces BURST for that frame).
                # No explicit register_track_birth() call needed here.
                #
                # Snapshot migration :: mark this tid for birth-snapshot
                # capture. The actual capture happens AFTER _annotate_frame()
                # so the bbox is visible in the snapshot.
                self._birth_tids_this_frame.add(tid)
                if self._matcher is not None and self._gating is not None:
                    # Compute centroid for spatial matching.
                    birth_centroid = (
                        float(trk.x1 + trk.x2) * 0.5,
                        float(trk.y1 + trk.y2) * 0.5,
                    )
                    # Crop the person region. Same guard as the main path.
                    bx1 = max(0, int(trk.x1))
                    by1 = max(0, int(trk.y1))
                    bx2 = min(captured.frame.shape[1], int(trk.x2))
                    by2 = min(captured.frame.shape[0], int(trk.y2))
                    if bx2 - bx1 >= 16 and by2 - by1 >= 16:
                        birth_crop = captured.frame[by1:by2, bx1:bx2]
                        if birth_crop.size > 0:
                            try:
                                birth_body_list = (
                                    self._matcher.extract_body_features(
                                        crops_bgr=[birth_crop],
                                    )
                                )
                                self._cuda_sync_check("extract_body_features.birth")
                                # Patch 63 (hotfix D) :: extract_body_features()
                                # returns np.ndarray of shape (N, 512), NOT a
                                # Python list. The previous code used
                                #   `if birth_body_list and len(...) > 0`
                                # which evaluates the numpy array in a boolean
                                # context. For an array with >1 element (512
                                # here), Python raises:
                                #   ValueError: The truth value of an array
                                #   with more than one element is ambiguous.
                                #   Use a.any() or a.all()
                                # This caused EVERY track birth's body feature
                                # extraction to fail, which meant:
                                #   1. _birth_body_cache was never populated
                                #   2. Body Re-ID at birth never ran
                                #   3. The batched body pass always re-extracted
                                #      (double OSNet work, TTFM regression)
                                # Fix: check len() directly without using the
                                # array in a boolean context. len() on a 2D
                                # numpy array returns the first dimension size.
                                if birth_body_list is not None and len(birth_body_list) > 0:
                                    birth_body = birth_body_list[0]
                                else:
                                    birth_body = None
                            except Exception as exc:
                                # Patch 62 [6/7] :: Elevated DEBUG -> ERROR.
                                logger.error(
                                    "IsolatedAIThread: birth body feat "
                                    "failed for track %d: %s", tid, exc,
                                )
                                birth_body = None

                            if birth_body is not None:
                                # Patch 61 [C] :: Cache the birth body
                                # feature so the batched body pass below
                                # can reuse it instead of re-extracting
                                # via a second OSNet forward pass.
                                self._birth_body_cache[tid] = birth_body
                                try:
                                    birth_reid = self._matcher.match_body_reid(
                                        track_id=tid,
                                        body_feature=birth_body,
                                        centroid=birth_centroid,
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "IsolatedAIThread: birth body_reid "
                                        "match failed for track %d: %s",
                                        tid, exc,
                                    )
                                    birth_reid = None

                                if (
                                    birth_reid is not None
                                    and birth_reid.inheritance_applied
                                ):
                                    try:
                                        inherited = (
                                            self._gating.inherit_verified_state(
                                                new_track_id=tid,
                                                body_reid_result=birth_reid,
                                                frame_ts=frame_ts,
                                                bbox=(
                                                    int(trk.x1), int(trk.y1),
                                                    int(trk.x2), int(trk.y2),
                                                ),
                                            )
                                        )
                                        if inherited:
                                            # Inheritance succeeded -- mark
                                            # the track as resolved and skip
                                            # all face scanning. TTFM for
                                            # this track: ~30ms (one OSNet).
                                            self._update_stranger_bbox_cache(
                                                tid, trk,
                                                EntityState.VERIFIED_STUDENT,
                                            )
                                            track_meta[tid] = {
                                                "track": trk,
                                                "face_result": None,
                                                "skip_match": True,
                                                "inherited_at_birth": True,
                                                "yolo_ms": yolo_latency_ms,
                                                "face_det_ms": 0.0,
                                                "arcface_ms": 0.0,
                                                "usearch_ms": 0.0,
                                            }
                                            continue
                                    except Exception as exc:
                                        logger.warning(
                                            "IsolatedAIThread: birth "
                                            "inherit_verified_state failed "
                                            "for track %d: %s", tid, exc,
                                        )

            # Priority-aware throttle (restores old always-scan-while-unresolved
            # behavior for TTFM-critical tracks): an unresolved/newly-born track
            # scans every frame, same as evaluate_gatekeepers()'s HIGH-priority path
            # in the old pipeline. Only already-matched/long-lived tracks fall back
            # to the interval throttle, since their identity no longer needs racing.
            is_unresolved = (
                self._matcher is None
                or self._matcher.is_face_scanning_active(tid)
            ) and tid not in self._resolved_track_ids  # track your own "resolved" set
            scan_interval = 1 if is_unresolved else 3
            last_scan = self._track_last_face_scan.get(tid, -99)
            if (captured.frame_index - last_scan) < scan_interval:
                cached = self._track_face_cache.get(tid)
                if cached is not None:
                    track_meta[tid] = {
                        "track": trk,
                        "face_result": None,
                        "skip_match": False,
                        "best_face": cached["best_face"],
                        "face_bbox_frame": cached["face_bbox_frame"],
                        "landmarks": cached["landmarks"],
                        # Cache hit -- no detection ran this frame.
                        "yolo_ms": yolo_latency_ms,
                        "face_det_ms": 0.0,
                        "arcface_ms": 0.0,
                        # usearch_ms will be filled in by the batch-match loop.
                    }
                    track_ids_to_match.append(tid)
                    face_embeddings.append(cached["embedding"])
                    self._tracking.set_face_bbox(tid, cached["face_bbox_frame"])
                    continue

            # Skip tracks whose face scanning is already terminated.
            if self._matcher is not None and not self._matcher.is_face_scanning_active(tid):
                track_meta[tid] = {
                    "track": trk,
                    "face_result": None,
                    "skip_match": True,
                    # Locked track -- no per-track inference at all.
                    "yolo_ms": yolo_latency_ms,
                    "face_det_ms": 0.0,
                    "arcface_ms": 0.0,
                    "usearch_ms": 0.0,
                }
                continue

            # Crop the person bbox region and run face detection inside it.
            x1 = max(0, int(trk.x1))
            y1 = max(0, int(trk.y1))
            x2 = min(captured.frame.shape[1], int(trk.x2))
            y2 = min(captured.frame.shape[0], int(trk.y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                track_meta[tid] = {
                    "track": trk,
                    "face_result": None,
                    "skip_match": True,
                    "yolo_ms": yolo_latency_ms,
                    "face_det_ms": 0.0,
                    "arcface_ms": 0.0,
                    "usearch_ms": 0.0,
                }
                continue

            person_crop = captured.frame[y1:y2, x1:x2]

            # --- Tiered crop upscaling for small/far persons (fixed-shape version) ---
            # Same tier thresholds as before, but each tier now resizes to ONE fixed
            # canonical size instead of a per-bbox scaled size. This keeps the set of
            # distinct tensor shapes hitting InsightFace/cuDNN closed and small (4
            # shapes total instead of effectively unbounded), so cuDNN's conv-algo
            # cache actually gets reused instead of re-profiling almost every call.
            # Aspect ratio is preserved via letterbox padding so faces aren't warped.
            _TIER_SIZES = {
                "far":    (256, 256),   # was 4x cubic
                "mid":    (224, 224),   # was 3x cubic
                "near":   (192, 192),   # was 2x linear
                "close":  (160, 160),   # was no upscale
            }

            def _letterbox_to(img: np.ndarray, target_wh: Tuple[int, int],
                            interp: int) -> np.ndarray:
                """Resize preserving aspect ratio into a fixed target canvas."""
                th, tw = target_wh[1], target_wh[0]
                h, w = img.shape[:2]
                scale = min(tw / w, th / h)
                new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
                canvas = np.zeros((th, tw, 3), dtype=img.dtype)
                pad_x = (tw - new_w) // 2
                pad_y = (th - new_h) // 2
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                return canvas, scale, pad_x, pad_y

            crop_h = y2 - y1
            crop_w = x2 - x1
            if crop_h < 40 and crop_h >= 12 and crop_w >= 10:
                crop_for_face, _lb_scale, _lb_px, _lb_py = _letterbox_to(
                    person_crop, _TIER_SIZES["far"], cv2.INTER_CUBIC,
                )
            elif crop_h < 60:
                crop_for_face, _lb_scale, _lb_px, _lb_py = _letterbox_to(
                    person_crop, _TIER_SIZES["mid"], cv2.INTER_CUBIC,
                )
            elif crop_h < 120:
                crop_for_face, _lb_scale, _lb_px, _lb_py = _letterbox_to(
                    person_crop, _TIER_SIZES["near"], cv2.INTER_LINEAR,
                )
            else:
                crop_for_face, _lb_scale, _lb_px, _lb_py = _letterbox_to(
                    person_crop, _TIER_SIZES["close"], cv2.INTER_LINEAR,
                )

            face_results: List[Dict[str, Any]] = []
            face_det_ms = 0.0
            arcface_ms = 0.0
            if self._face is not None:
                try:
                    import time as _diag_time
                    _diag_t0 = _diag_time.perf_counter()
                    face_results = self._face.detect_and_embed(crop_for_face)
                    _diag_wall_ms = (_diag_time.perf_counter() - _diag_t0) * 1000.0
                    # Patch 24 :: demoted WARNING -> DEBUG. This per-frame
                    # diagnostic was leftover from the Patch 16 SCRFD
                    # regression investigation; the regression is fixed
                    # and the spam was drowning out real INFO messages.
                    # Set logger level to DEBUG to re-enable.
                    logger.debug(
                        "DIAG detect_and_embed | track=%d | shape=%s | wall_ms=%.1f",
                        tid, crop_for_face.shape[:2], _diag_wall_ms,
                    )
                except Exception as exc:
                    logger.debug(
                        "IsolatedAIThread: face detect failed for track %d: %s",
                        tid, exc,
                    )
                # Read the real per-stage latencies from the engine.
                face_det_ms = float(self._face.last_det_latency_ms)
                arcface_ms = float(self._face.last_rec_latency_ms)

            # --- Patch 4a: Side-view / hard-pose fallback retry ---
            # InsightFace det_10g at default det_thresh=0.5 misses many
            # side-profile and partially-occluded faces even when the
            # person crop is large. The tiered upscaling above is sized
            # by person crop_h, so a CLOSE side-view person (crop_h>=120)
            # gets NO upscale and the face stays undetected frame after
            # frame -- the user's own side-profile became Stranger_1 in
            # the test exactly because of this.
            #
            # Fix: if the first pass returned no faces AND the person
            # crop is at least 60px tall (worth retrying -- below that
            # the face is too small for any tier to help), retry the
            # detection at 4x INTER_CUBIC. The 4x retry costs one extra
            # ~80ms InsightFace call ONLY on frames where the first pass
            # failed -- zero cost on success frames, and it converts
            # long-running "no-face" tracks into resolved students on
            # the very next frame.
            # Patch 61 [D] :: 4x retry throttle.
            # The retry costs ~80ms InsightFace call per failed-detect
            # track. Gate to every Nth AI cycle so the retry fires on
            # the first scan of a new track (preserves TTFM recovery)
            # but skips 2/3 of subsequent failed scans. The first-scan
            # path is preserved because frames_processed starts at 0.
            if (
                not face_results
                and self._face is not None
                and crop_h >= 60
                and crop_for_face.shape[0] < crop_h * 4
                and (self._frames_processed % self._face_retry_every_n_scans) == 0
            ):
                try:
                    retry_crop = cv2.resize(
                        person_crop,
                        (crop_w * 4, crop_h * 4),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    retry_results = self._face.detect_and_embed(retry_crop)
                except Exception as exc:
                    logger.debug(
                        "IsolatedAIThread: 4x retry face detect failed "
                        "for track %d: %s", tid, exc,
                    )
                    retry_results = []
                if retry_results:
                    # Accept the retry result and repoint crop_for_face
                    # so the bbox back-projection below uses the 4x scale.
                    face_results = retry_results
                    crop_for_face = retry_crop
                    face_det_ms += float(self._face.last_det_latency_ms)
                    arcface_ms += float(self._face.last_rec_latency_ms)
                    logger.debug(
                        "IsolatedAIThread: 4x retry recovered face for "
                        "track %d (crop_h=%d)", tid, crop_h,
                    )

            if not face_results:
                track_meta[tid] = {
                    "track": trk,
                    "face_result": None,
                    "skip_match": True,
                    # Detection ran (face_det_ms is meaningful) but no face
                    # was found in the crop, so no embedding -> no matching.
                    # Flag for Patch 3a: body Re-ID should still run on
                    # this track so it can be recognized via body inheritance
                    # (handles side-view / occluded-face recovery).
                    "no_face_detected": True,
                    "yolo_ms": yolo_latency_ms,
                    "face_det_ms": face_det_ms,
                    "arcface_ms": 0.0,
                    "usearch_ms": 0.0,
                }
                continue

            # Take the largest face by bbox area.
            best_face = max(
                face_results,
                key=lambda f: (
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                ),
            )
            emb = best_face.get("embedding")
            if emb is None:
                track_meta[tid] = {
                    "track": trk,
                    "face_result": None,
                    "skip_match": True,
                    "yolo_ms": yolo_latency_ms,
                    "face_det_ms": face_det_ms,
                    "arcface_ms": arcface_ms,
                    "usearch_ms": 0.0,
                }
                continue

            # Translate the face bbox from crop_for_face coords to frame coords.
            # crop_for_face may have been upscaled relative to person_crop,
            # so we need to scale the bbox back down before adding x1/y1.
            fb = best_face["bbox"]
            scale_x = crop_w / float(crop_for_face.shape[1])
            scale_y = crop_h / float(crop_for_face.shape[0])
            face_bbox_frame = (
                int(fb[0] * scale_x) + x1, int(fb[1] * scale_y) + y1,
                int(fb[2] * scale_x) + x1, int(fb[3] * scale_y) + y1,
            )
            self._tracking.set_face_bbox(tid, face_bbox_frame)

            track_ids_to_match.append(tid)
            face_embeddings.append(emb)
            # Stamp latencies NOW, while face_det_ms/arcface_ms are still
            # the values for THIS track (not the next iteration's).
            track_meta[tid] = {
                "track": trk,
                "face_result": None,
                "skip_match": False,
                "best_face": best_face,
                "face_bbox_frame": face_bbox_frame,
                "landmarks": (
                    (np.asarray(best_face["kps"], dtype=np.float32)
                     * np.array([scale_x, scale_y], dtype=np.float32))
                    if best_face.get("kps") is not None else None
                ),
                "yolo_ms": yolo_latency_ms,
                "face_det_ms": face_det_ms,
                "arcface_ms": arcface_ms,
                # usearch_ms filled in by the batch-match loop below.
            }
            self._track_last_face_scan[tid] = captured.frame_index
            # Cache the scale factors so the cached face_bbox_frame and
            # landmarks stay valid even if the person moves slightly between
            # frames. The cached embedding is resolution-independent so it
            # doesn't need rescaling.
            self._track_face_cache[tid] = {
                "best_face": best_face,
                "face_bbox_frame": face_bbox_frame,
                "landmarks": (
                    (np.asarray(best_face["kps"], dtype=np.float32)
                     * np.array([scale_x, scale_y], dtype=np.float32))
                    if best_face.get("kps") is not None else None
                ),
                "embedding": emb,
            }

        # Prune _seen_track_ids so that tracks which disappeared from the
        # frame are forgotten. This ensures that a person who leaves and
        # re-enters the frame (BoTSort will assign a new track_id) gets a
        # fresh inheritance attempt -- otherwise we'd skip the inheritance
        # path and force them through the full face cascade.
        active_tids_this_frame = {int(t.track_id) for t in tracks}
        self._seen_track_ids &= active_tids_this_frame
        # Patch 61 [C] :: Clear any unconsumed birth body cache entries.
        # The cache is per-frame (a track is only "born" on its first
        # frame); entries that weren't consumed by the batched body
        # pass are stale by next cycle.
        if self._birth_body_cache:
            self._birth_body_cache.clear()
        # Patch 7 pruning is handled internally by res_opt_engine
        # (_prune_seen_track_ids inside decide_mode).
        #
        # Snapshot migration :: prune finalized-tids set for tracks that
        # have disappeared (prevents unbounded set growth).
        self._snapshot_finalized_tids &= active_tids_this_frame
        #
        # Patch 56 :: Prune ALL per-track state dicts/sets that previously
        # grew unbounded. drop_track() was defined in identity_matcher,
        # gating_opt, and res_opt_engine but NEVER called -- every track
        # ever seen accumulated state in 4 separate dicts across 3
        # modules for the entire process lifetime. Over a 12-hour session
        # with thousands of track births/deaths, this caused steady RSS
        # growth (each TrackInferenceState holds body_feature_history =
        # list of 10x 512-dim float32 numpy arrays = ~20KB per track).
        #
        # We compute the dead_tids set (tracks present last frame but not
        # this frame) and call drop_track() on each engine. We also prune
        # _track_last_face_scan / _track_face_cache / _resolved_track_ids
        # which had the same leak.
        if hasattr(self, '_last_active_tids') and self._last_active_tids:
            dead_tids = self._last_active_tids - active_tids_this_frame
            if dead_tids:
                # Prune per-track dicts in this thread.
                for tid in dead_tids:
                    self._track_last_face_scan.pop(tid, None)
                    self._track_face_cache.pop(tid, None)
                    # Patch 57 :: also prune _stranger_bboxes_by_track.
                    # Without this, a stranger track that disappeared
                    # without being reclassified left a 4-int tuple
                    # (~80 B) in this dict forever. Small per-entry
                    # leak, but it never shrank.
                    self._stranger_bboxes_by_track.pop(tid, None)
                    # Patch 65 :: Prune clearshot per-track state.
                    # _clearshot_last_ts_us and _clearshot_state are
                    # small (one int + one small dict per stranger),
                    # but they never shrank without this prune. The
                    # snapshot engine's _stranger_labels /
                    # _clearshot_counters dicts are NOT pruned here --
                    # they're reset wholesale at session rollover via
                    # reset_clearshot_state().
                    self._clearshot_last_ts_us.pop(tid, None)
                    self._clearshot_state.pop(tid, None)
                self._resolved_track_ids &= active_tids_this_frame
                # Prune per-track state in the 3 sub-engines.
                if self._matcher is not None:
                    _drop = getattr(self._matcher, 'drop_track', None)
                    if callable(_drop):
                        for tid in dead_tids:
                            try:
                                _drop(tid)
                            except Exception:
                                pass
                if self._gating is not None:
                    _drop = getattr(self._gating, 'drop_track', None)
                    if callable(_drop):
                        for tid in dead_tids:
                            try:
                                _drop(tid)
                            except Exception:
                                pass
                if self._res_opt is not None:
                    _drop = getattr(self._res_opt, 'drop_track', None)
                    if callable(_drop):
                        for tid in dead_tids:
                            try:
                                _drop(tid)
                            except Exception:
                                pass
                # Prune orphaned pending snapshot paths.
                if self._recorder_thread is not None:
                    _prune = getattr(self._recorder_thread, 'prune_dead_tracks', None)
                    if callable(_prune):
                        try:
                            _prune(active_tids_this_frame)
                        except Exception as exc:
                            logger.debug(
                                "recorder.prune_dead_tracks failed: %s", exc,
                            )
        self._last_active_tids = set(active_tids_this_frame)

        # --- 2c) Batch face matching via BLAS ---
        # USearch is one batched call for the whole frame -- the same
        # usearch_latency_ms applies to every track that participated.
        usearch_latency_ms = 0.0
        if track_ids_to_match and self._matcher is not None:
            t_us_start = time.perf_counter()
            try:
                if _NUMPY_AVAILABLE:
                    emb_matrix = np.stack(face_embeddings, axis=0).astype(np.float32)
                else:
                    emb_matrix = face_embeddings[0] if face_embeddings else None
                match_results = self._matcher.batch_match_faces_raw(
                    track_ids=track_ids_to_match,
                    face_embeddings=emb_matrix,
                    top_k=5,
                )
                self._cuda_sync_check("batch_match_faces_raw")
                usearch_latency_ms = (time.perf_counter() - t_us_start) * 1000.0
                for tid, res in zip(track_ids_to_match, match_results):
                    if tid in track_meta:
                        track_meta[tid]["face_result"] = res
                        track_meta[tid]["usearch_ms"] = usearch_latency_ms
            except Exception as exc:
                logger.error(
                    "IsolatedAIThread: batch_match_faces_raw failed: %s", exc,
                )

        # --- 2c-bis) Batched body Re-ID feature extraction ---
        # OSNet AIN at FP32 is ~30ms per forward pass (10ms fixed CUDA
        # overhead + 20ms variable compute). Calling it once per track
        # costs N x 30ms; calling it once with N crops costs ~10 + N x 20ms.
        # For 3 unresolved tracks: 90ms -> 70ms. For 5: 150ms -> 110ms.
        #
        # We extract body features for tracks that:
        #   (a) are not yet locked (verified student or registered stranger)
        #   (b) actually entered the matching cascade (skip_match=False), OR
        #       were skipped ONLY because no face was detected this frame
        #       (Patch 3a: no_face_detected=True). Side-view / occluded-face
        #       tracks can still be recognized via body Re-ID inheritance,
        #       which is the only fallback when face detection fails for
        #       multiple consecutive frames.
        # Tracks that are skipped because they were inherited-at-birth or
        # because their bbox is too small (<8px) are still excluded.
        body_feature_by_tid: Dict[int, Optional[np.ndarray]] = {}
        if self._matcher is not None:
            body_batch_tids: List[int] = []
            body_batch_crops: List[np.ndarray] = []
            for tid, meta in track_meta.items():
                if self._gating.is_track_locked(tid):
                    continue
                if meta.get("skip_match"):
                    # Patch 3a: only allow no-face-detected tracks through;
                    # inherited-at-birth and too-small-bbox tracks stay skipped.
                    if not meta.get("no_face_detected"):
                        continue
                    if meta.get("inherited_at_birth"):
                        continue
                # For normal-cascade tracks (skip_match=False), we
                # previously also required face_result is not None.
                # That gate is now redundant: a skip_match=False track
                # always has face_result set (post batch_match in 2c).
                trk_meta = meta["track"]
                crop_b = captured.frame[
                    max(0, int(trk_meta.y1)):min(captured.frame.shape[0], int(trk_meta.y2)),
                    max(0, int(trk_meta.x1)):min(captured.frame.shape[1], int(trk_meta.x2)),
                ]
                if crop_b.size == 0:
                    continue
                # Patch 61 [C] :: Skip OSNet re-extraction for tracks
                # whose body feature was already extracted at birth.
                # Reuse the cached vector directly so the batched OSNet
                # forward pass only processes tracks that actually need
                # a fresh vector.
                if tid in self._birth_body_cache:
                    body_feature_by_tid[tid] = self._birth_body_cache.pop(tid)
                    continue
                body_batch_tids.append(tid)
                body_batch_crops.append(crop_b)

            if body_batch_crops:
                # Patch 43 :: Time the OSNet body Re-ID batch call.
                # The batch latency is distributed evenly across all
                # tracks that participated in the batch (per-track
                # share = batch_ms / n_tracks). This is approximate
                # but accurate enough for telemetry purposes.
                _t_body_start = time.perf_counter()
                try:
                    body_features_list = self._matcher.extract_body_features(
                        crops_bgr=body_batch_crops,
                    )
                    self._cuda_sync_check("extract_body_features.batch")
                    _body_batch_ms = (time.perf_counter() - _t_body_start) * 1000.0
                    # Per-track share (avoid div-by-zero).
                    _n_body = max(1, len(body_batch_tids))
                    _body_per_track_ms = _body_batch_ms / float(_n_body)
                    for tid_b, feat_b in zip(body_batch_tids, body_features_list):
                        body_feature_by_tid[tid_b] = feat_b
                        # Stamp into track_meta if the entry exists.
                        if tid_b in track_meta:
                            track_meta[tid_b]["body_reid_ms"] = _body_per_track_ms
                    self._last_body_reid_latency_ms = _body_batch_ms
                except Exception as exc:
                    _body_batch_ms = (time.perf_counter() - _t_body_start) * 1000.0
                    logger.debug(
                        "IsolatedAIThread: batched body feature extraction "
                        "failed (took %.2f ms): %s", _body_batch_ms, exc,
                    )

        # --- 2d) Per-track TTFM / Pose-EMA / Velocity updates ---
        for tid, meta in track_meta.items():
            if meta.get("skip_match"):
                # --- Patch 3a: body-Re-ID-only path for no-face tracks ---
                # Side-view / occluded-face tracks (no_face_detected=True)
                # were extracted into the body Re-ID batch in 2c-bis.
                # If body inheritance succeeds, resolve the track now --
                # this is the only path that can recover a side-view
                # track whose face is undetectable for many frames.
                if not meta.get("no_face_detected"):
                    continue
                if meta.get("inherited_at_birth"):
                    continue
                trk = meta["track"]
                body_feature = body_feature_by_tid.get(tid)
                if body_feature is None or self._matcher is None:
                    # No body feature was extracted (e.g. empty crop);
                    # nothing we can do for this track this frame.
                    continue
                centroid_nf = (
                    float(trk.x1 + trk.x2) * 0.5,
                    float(trk.y1 + trk.y2) * 0.5,
                )
                try:
                    nf_reid = self._matcher.match_body_reid(
                        track_id=tid,
                        body_feature=body_feature,
                        centroid=centroid_nf,
                    )
                except Exception as exc:
                    logger.debug(
                        "IsolatedAIThread: no-face body_reid match failed "
                        "for track %d: %s", tid, exc,
                    )
                    nf_reid = None
                if nf_reid is not None and nf_reid.inheritance_applied:
                    try:
                        inherited_nf = self._gating.inherit_verified_state(
                            new_track_id=tid,
                            body_reid_result=nf_reid,
                            frame_ts=frame_ts,
                            bbox=(
                                int(trk.x1), int(trk.y1),
                                int(trk.x2), int(trk.y2),
                            ),
                        )
                        if inherited_nf:
                            self._update_stranger_bbox_cache(
                                tid, trk, EntityState.VERIFIED_STUDENT,
                            )
                            logger.info(
                                "IsolatedAIThread: no-face track %d "
                                "resolved via body inheritance (sim=%.4f, "
                                "dist=%.0fpx)", tid,
                                nf_reid.body_similarity,
                                nf_reid.spatial_distance_px,
                            )
                    except Exception as exc:
                        logger.warning(
                            "IsolatedAIThread: no-face inherit_verified_state "
                            "failed for track %d: %s", tid, exc,
                        )
                # Either way, the no-face track can't enter the normal
                # cascade (no face_similarity to evaluate). Continue.
                continue

            trk = meta["track"]
            face_res: Optional[Any] = meta.get("face_result")
            if face_res is None:
                continue

            # Register the raw similarity with the TTFM tracker.
            effective_threshold = self._res_opt.register_face_similarity(
                track_id=tid, raw_similarity=face_res.best_similarity,
            )

            # Update the Pose-Weighted EMA accumulator.
            landmarks = meta.get("landmarks")
            ema_score, _ = self._res_opt.update_ema(
                track_id=tid,
                raw_similarity=face_res.best_similarity,
                landmarks=landmarks if landmarks is not None else None,
            )

            # Update the Velocity tracker.
            centroid = (
                float(trk.x1 + trk.x2) * 0.5,
                float(trk.y1 + trk.y2) * 0.5,
            )
            self._res_opt.update_velocity(
                track_id=tid,
                centroid=centroid,
                frame_index=captured.frame_index,
                frame_width=captured.frame.shape[1],
            )

            ema_score_val, ema_cluster = self._res_opt.get_ema_state(tid)

            # --- 2e) Gating cascade evaluation ---
            # Body feature was already extracted in the batched pre-pass
            # (section 2c-bis). Just look it up -- no per-track OSNet call.
            # If the track was skipped, locked, or had no face_result,
            # body_feature_by_tid won't have an entry and we get None.
            body_feature = body_feature_by_tid.get(tid)

            # Run body Re-ID matching for runtime state alignment.
            body_reid_result = None
            if body_feature is not None and self._matcher is not None:
                try:
                    body_reid_result = self._matcher.match_body_reid(
                        track_id=tid,
                        body_feature=body_feature,
                        centroid=centroid,
                    )
                except Exception as exc:
                    logger.debug(
                        "IsolatedAIThread: body_reid match failed for track %d: %s",
                        tid, exc,
                    )

            # Attempt verified-state inheritance (track-ID-swap recovery).
            if body_reid_result is not None and body_reid_result.inheritance_applied:
                try:
                    inherited = self._gating.inherit_verified_state(
                        new_track_id=tid,
                        body_reid_result=body_reid_result,
                        frame_ts=frame_ts,
                        bbox=(
                            int(trk.x1), int(trk.y1),
                            int(trk.x2), int(trk.y2),
                        ),
                    )
                    if inherited:
                        # Skip the normal cascade -- the track is already
                        # resolved via inheritance.
                        self._update_stranger_bbox_cache(tid, trk, EntityState.VERIFIED_STUDENT)
                        continue
                except Exception as exc:
                    logger.warning(
                        "IsolatedAIThread: inherit_verified_state failed for "
                        "track %d: %s", tid, exc,
                    )

            # Normal cascade evaluation.
            try:
                self._gating.evaluate_track(
                    track_id=tid,
                    frame_ts=frame_ts,
                    bbox=(
                        int(trk.x1), int(trk.y1),
                        int(trk.x2), int(trk.y2),
                    ),
                    centroid=centroid,
                    face_similarity=face_res.best_similarity,
                    best_student_id=face_res.best_student_id,
                    best_student_name=face_res.best_student_name,
                    body_feature=body_feature,
                    effective_threshold=effective_threshold,
                    ema_score=ema_score_val,
                    ema_cluster_size=ema_cluster,
                    yolo_latency_ms=float(meta.get("yolo_ms", 0.0)),
                    face_det_latency_ms=float(meta.get("face_det_ms", 0.0)),
                    arcface_latency_ms=float(meta.get("arcface_ms", 0.0)),
                    usearch_latency_ms=float(meta.get("usearch_ms", 0.0)),
                    # Patch 42/43 :: OSNet body Re-ID inference latency.
                    body_reid_latency_ms=float(meta.get("body_reid_ms", 0.0)),
                    # Patch 37 :: Pass YOLO det_conf so the gating engine
                    # can skip Re-ID when the bbox is unreliable.
                    yolo_conf=float(getattr(trk, "det_conf", 1.0)),
                )
            except Exception as exc:
                logger.error(
                    "IsolatedAIThread: gating.evaluate_track failed for "
                    "track %d: %s", tid, exc,
                )

            # Update the recorder's stranger bbox cache for anonymization.
            self._update_stranger_bbox_cache(tid, trk, None)

        # --- 2f) Project to CleanBBox + annotate ---
        # Patch 61 [H] :: Removed the first to_clean_bboxes() call --
        # it was immediately overwritten by the second call below after
        # the label-update loop. The variable is declared here for
        # readability but only populated after the label loop.
        clean_bboxes: List[Any] = []

        # Anomaly detection: faces detected outside any track's person bbox.
        # Patch 23 :: forward `captured` so capture_anomaly_snapshot()
        # can access frame_index and capture_us (was raising
        # NameError: name 'captured' is not defined).
        # Patch 61 [A] :: Anomaly scan throttle. _scan_for_anomalies
        # runs InsightFace detect_and_embed on the FULL 1280x720 frame
        # (~38ms) -- fires on every Nth AI cycle to avoid doubling
        # face-det cost in sparse scenes (the TTFM-critical case).
        if (self._frames_processed % self._anomaly_scan_every_n_cycles) == 0:
            self._scan_for_anomalies(captured.frame, tracks, frame_ts, captured)

        # Update resolved labels on the tracking engine's cache.
        for trk in tracks:
            label = self._gating.get_resolved_label(int(trk.track_id))
            self._tracking.set_resolved_label(int(trk.track_id), label)

        # Re-project after label updates.
        clean_bboxes = self._tracking.to_clean_bboxes(tracks)

        # P2-M12 fix :: removed dead `stranger_bboxes_tuple` block.
        # Previously this code took a snapshot of _stranger_bboxes_by_track
        # under a lock and assigned it to a local `stranger_bboxes_tuple`
        # variable -- but that variable was NEVER read downstream. The
        # recorder (snap_engine) is called per-track via
        # capture_birth_snapshot() / _maybe_capture_clearshot() further
        # below, none of which consume stranger_bboxes_tuple. Pure dead
        # code that held a lock for no reason on every AI cycle.

        # Annotate the frame with clean bboxes.
        annotated = self._annotate_frame_throttled(
            captured.frame, clean_bboxes, frame_ts, None,
        )

        # ------------------------------------------------------------------
        # Snapshot migration :: capture birth snapshots for tracks born
        # this frame. The annotated frame now has the bbox overlay drawn
        # (yellow/green/orange depending on resolved state at this instant).
        # At track birth, the label is typically empty or PENDING, which
        # is exactly what we want -- the snapshot shows the moment the
        # person FIRST appeared, before recognition completed.
        # ------------------------------------------------------------------
        if self._birth_tids_this_frame and self._recorder_thread is not None:
            # Patch 21 :: _recorder_thread IS the SnapStrangersEngine now.
            snap_engine = self._recorder_thread
            if hasattr(snap_engine, "capture_birth_snapshot"):
                for birth_tid in self._birth_tids_this_frame:
                    # Find the bbox for this track from the clean_bboxes list.
                    birth_bbox = (0, 0, 0, 0)
                    birth_det_conf: float = 0.0
                    for cb in clean_bboxes:
                        try:
                            # CleanBBox has a track_id attribute; fall back
                            # to matching by iterating tracks.
                            if int(getattr(cb, "track_id", -1)) == int(birth_tid):
                                birth_bbox = (
                                    int(cb.x1), int(cb.y1),
                                    int(cb.x2), int(cb.y2),
                                )
                                birth_det_conf = float(getattr(cb, "det_conf", 0.0))
                                break
                        except Exception:
                            continue
                    # Fallback: look up the raw track bbox.
                    if birth_bbox == (0, 0, 0, 0):
                        for trk in tracks:
                            if int(trk.track_id) == int(birth_tid):
                                birth_bbox = (
                                    int(trk.x1), int(trk.y1),
                                    int(trk.x2), int(trk.y2),
                                )
                                birth_det_conf = float(getattr(trk, "det_conf", 0.0))
                                break
                    # Patch 37 :: Gate snapshot capture on YOLO confidence.
                    # When det_conf is below the configured threshold, the
                    # bbox is unreliable (partial occlusion / motion blur)
                    # and the snapshot would be unusable. Skip it.
                    _snap_threshold = float(
                        self._config.get("gating", {}).get("stranger", {})
                        .get("snapshot_min_yolo_conf", 0.50)
                    )
                    if birth_det_conf < _snap_threshold:
                        logger.debug(
                            "IsolatedAIThread: birth snapshot SKIPPED for "
                            "track %d (det_conf=%.3f < threshold=%.2f)",
                            birth_tid, birth_det_conf, _snap_threshold,
                        )
                        continue
                    try:
                        # Patch 62 [5] :: Pass a COPY to the snapshot engine.
                        # The SnapStrangersEngine worker thread reads the buffer
                        # asynchronously to encode a PNG. Meanwhile the AI
                        # thread pushes the SAME buffer as RenderPackage.frame
                        # to the GUI queue. Passing a copy breaks the 3-way
                        # cross-thread race on the annotated buffer.
                        snap_engine.capture_birth_snapshot(
                            annotated_frame=annotated.copy(),
                            track_id=int(birth_tid),
                            frame_index=int(captured.frame_index),
                            capture_us=int(captured.capture_us),
                            bbox=birth_bbox,
                        )
                    except Exception as exc:
                        logger.debug(
                            "IsolatedAIThread: capture_birth_snapshot "
                            "failed for track %d: %s", birth_tid, exc,
                        )
            # Clear the set for the next frame.
            self._birth_tids_this_frame.clear()

        # ------------------------------------------------------------------
        # Patch 65 :: CLEARSHOT capture for STRANGER-locked tracks.
        #
        # For each track that's been locked as STRANGER, check the
        # clearshot gating conditions (YOLO conf, bbox size, cooldown,
        # max-per-track) and, if all pass, queue a WRITE_CLEARSHOT op.
        # The snapshot engine handles the actual PNG write off-thread.
        #
        # This runs AFTER _annotate_frame_throttled() so the bbox
        # overlay (cyan border + "CLEARSHOT #YY" caption) is visible
        # in the snapshot. We pass `annotated` (the AI thread's working
        # buffer); capture_clearshot() makes a defensive deep copy.
        # ------------------------------------------------------------------
        if self._clearshot_enabled and self._recorder_thread is not None:
            for trk in tracks:
                tid = int(trk.track_id)
                # Patch 63 (hotfix D) :: Skip invalid track IDs.
                if tid < 0:
                    continue
                label = self._gating.get_resolved_label(tid) if self._gating else None
                if not label or "Stranger" not in label:
                    continue  # Not a locked stranger.
                try:
                    self._maybe_capture_clearshot(
                        track_id=tid,
                        trk=trk,
                        annotated_frame=annotated,
                        captured=captured,
                        state=label,
                    )
                except Exception as exc:
                    # Patch 62 [6/7] :: Elevated DEBUG -> ERROR.
                    logger.error(
                        "IsolatedAIThread: _maybe_capture_clearshot "
                        "failed for track %d: %s", tid, exc,
                    )

        # Throttle mode decision (IDLE / BURST)
        active_tracks_for_throttle: List[Dict[str, Any]] = []
        for trk in tracks:
            tid = int(trk.track_id)
            # Patch 63 (hotfix D) :: Skip invalid track IDs.
            # BoTSORT uses -1 as a sentinel for "unassigned" or
            # "invalid detection". Processing these pollutes
            # _seen_track_ids with -1, wastes OSNet extraction on
            # non-existent tracks, and creates bogus stranger
            # snapshots (track=-1_BIRTH.png). Skip any tid < 0.
            if tid < 0:
                continue
            label = self._gating.get_resolved_label(tid) if self._gating else None
            is_pending = (
                label is None
                or label == ""
                or "PENDING" in label.upper()
            )
            active_tracks_for_throttle.append({
                "track_id": tid,
                "resolved_state": "PENDING" if is_pending else "RESOLVED",
                "face_scanning_active": (
                    self._matcher.is_face_scanning_active(tid)
                    if self._matcher is not None else True
                ),
            })
        if self._res_opt is not None:
            mode, sleep_ms = self._res_opt.decide_mode(
                active_tracks=active_tracks_for_throttle,
                current_frame_index=captured.frame_index,
            )
            self._res_opt.apply_sleep(sleep_ms)

        # Telemetry counts.
        gate_telem = self._gating.telemetry() if self._gating is not None else {}
        return RenderPackage(
            frame=annotated,
            frame_index=captured.frame_index,
            capture_us=captured.capture_us,
            clean_bboxes=clean_bboxes,
            ai_latency_ms=self._last_processing_latency_ms,
            active_track_count=gate_telem.get("active_track_count", 0),
            pending_track_count=gate_telem.get("pending_track_count", 0),
            verified_track_count=gate_telem.get("verified_count", 0),
            stranger_track_count=gate_telem.get("stranger_count", 0),
            anomaly_count=gate_telem.get("anomaly_count", 0),
            throttle_mode=(
                self._res_opt.current_mode.value
                if self._res_opt is not None else "IDLE"
            ),
        )

    # ------------------------------------------------------------------
    def _update_stranger_bbox_cache(
        self,
        track_id: int,
        trk: Any,
        forced_state: Optional[Any],
    ) -> None:
        """Update the per-track stranger bbox cache + snapshot finalize."""
        with self._stranger_bboxes_lock:
            state = self._gating.get_resolved_label(track_id)
            # Only keep bboxes for STRANGER tracks (the anonymization
            # target is the person bbox, not the face bbox).
            if "Stranger" in state:
                self._stranger_bboxes_by_track[track_id] = (
                    (int(trk.x1), int(trk.y1), int(trk.x2), int(trk.y2)),
                )
            else:
                self._stranger_bboxes_by_track.pop(track_id, None)

            # Also push the update to the recorder's stranger cache.
            # Patch 21 :: _recorder_thread IS the SnapStrangersEngine now.
            if self._recorder_thread is not None:
                try:
                    bboxes = self._stranger_bboxes_by_track.get(track_id, ())
                    self._recorder_thread.update_stranger_bboxes(
                        track_id, bboxes,
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Snapshot migration :: finalize the birth snapshot when the
        # track's resolved state first becomes locked.
        #   - STRANGER locked  -> rename PNG to include the label.
        #   - VERIFIED locked  -> delete PNG (no forensic value).
        # Uses _snapshot_finalized_tids to ensure exactly-once finalize.
        # ------------------------------------------------------------------
        if track_id in self._snapshot_finalized_tids:
            return
        if not state:
            return  # Still pending or empty.
        # Patch 21 :: _recorder_thread IS the SnapStrangersEngine now.
        snap_engine = self._recorder_thread
        if snap_engine is None:
            return
        if not hasattr(snap_engine, "finalize_stranger"):
            return  # Old VideoRecorderEngine -- skip.

        if "Stranger" in state:
            try:
                snap_engine.finalize_stranger(
                    track_id=int(track_id),
                    stranger_label=state,
                )
                self._snapshot_finalized_tids.add(track_id)
                # Patch 65 :: Register the stranger label with the
                # snapshot engine so subsequent capture_clearshot()
                # calls can build the STRANGER_{label}_CLEARSHOT_YY.png
                # filename. Also cache it locally for the cooldown /
                # max-per-track checks in _maybe_capture_clearshot().
                if hasattr(snap_engine, "register_stranger_label"):
                    snap_engine.register_stranger_label(
                        track_id=int(track_id),
                        stranger_label=state,
                    )
                self._clearshot_state[int(track_id)] = {
                    "label": state,
                    "count": 0,
                }
                # Initialize last_ts to 0 so the first clearshot fires
                # immediately (subject to YOLO conf + bbox size gates).
                self._clearshot_last_ts_us[int(track_id)] = 0
            except Exception as exc:
                # Patch 62 [6/7] :: Elevated DEBUG -> ERROR.
                logger.error(
                    "finalize_stranger failed for track %d: %s",
                    track_id, exc,
                )
        elif "ANOMALY" in state.upper():
            # Anomaly tracks don't get a snapshot finalize; the anomaly
            # snapshot is captured separately via capture_anomaly_snapshot().
            pass
        elif state.startswith("[") and "PENDING" not in state.upper():
            # Verified student (label like "[221050 / 221050]").
            try:
                # Patch 63 :: Pass student_label so the snapshot
                # engine can MOVE the birth PNG into identified/
                # with the student label in the filename (instead
                # of deleting it). This preserves the snapshot for
                # the Event Log page's hourly history view.
                snap_engine.finalize_verified(
                    track_id=int(track_id),
                    student_label=state,
                )
                self._snapshot_finalized_tids.add(track_id)
            except Exception as exc:
                # Patch 62 [6/7] :: Elevated DEBUG -> ERROR.
                logger.error(
                    "finalize_verified failed for track %d: %s",
                    track_id, exc,
                )

    # ------------------------------------------------------------------
    # Patch 65 :: _maybe_capture_clearshot()
    #
    # Called once per STRANGER-locked track per frame. Checks all the
    # clearshot gating conditions and, if they're all met, calls
    # snap_engine.capture_clearshot() to queue a WRITE_CLEARSHOT op.
    #
    # Gating conditions (ALL must be true):
    #   1. self._clearshot_enabled is True (global enable).
    #   2. snap_engine has capture_clearshot() method.
    #   3. Track is locked as STRANGER (state contains "Stranger").
    #   4. YOLO det_conf >= self._clearshot_min_yolo_conf.
    #   5. bbox width AND height >= self._clearshot_min_bbox_size.
    #   6. Cooldown elapsed: now - last_clearshot_ts >= cooldown_s.
    #   7. Per-track count < self._clearshot_max_per_track.
    #
    # The actual PNG write is asynchronous (queued to the snapshot
    # worker thread). This method only does the cheap gating checks +
    # the enqueue; the expensive cv2.imwrite happens off the AI thread.
    # ------------------------------------------------------------------
    def reset_clearshot_mirror(self) -> None:
        """P1-H6 fix: reset the AI thread's clearshot mirror state.

        Called by SessionBoundaryWatcher at every 6AM/6PM boundary,
        AFTER the snap_engine.reset_clearshot_state() call. Without
        this, a stranger that hit max_per_track=20 in the AM session
        is permanently blocked from clearshots in the PM session
        because the AI thread's local _clearshot_state[tid]["count"]
        mirror is never reset.

        Also clears _snapshot_finalized_tids so tracks can re-finalize
        in the new session (the snap engine's pending-paths dict was
        cleared by prune_dead_tracks at the boundary, so re-finalize
        will not collide with stale entries).

        Thread-safety: this method is called from the watcher thread
        while the AI thread may be reading these dicts. We use atomic
        reassignment (CPython GIL-protected attribute assignment)
        instead of in-place .clear() to avoid iteration races.
        """
        try:
            # Atomic swap: reassign to fresh empty containers.
            # The AI thread reads self._clearshot_state.get(tid) and
            # self._clearshot_last_ts_us.get(tid, 0) -- both .get()
            # calls are atomic in CPython and tolerate the swap.
            self._clearshot_state = {}
            self._clearshot_last_ts_us = {}
            self._snapshot_finalized_tids = set()
        except Exception as exc:
            logger.error(
                "IsolatedAIThread.reset_clearshot_mirror failed: %s",
                exc, exc_info=True,
            )

    def _maybe_capture_clearshot(
        self,
        track_id: int,
        trk: Any,
        annotated_frame: Any,
        captured: Any,
        state: str,
    ) -> None:
        """Gate + enqueue a CLEARSHOT snapshot for a STRANGER track.

        Args:
            track_id: The track ID (must be STRANGER-locked).
            trk: The track object (with x1/y1/x2/y2/det_conf attrs).
            annotated_frame: The annotated GUI frame (will be deep-copied).
            captured: The CapturedFrame (for frame_index + capture_us).
            state: The resolved label (must contain "Stranger").
        """
        # Gate 1+2: global enable + engine method availability.
        if not self._clearshot_enabled:
            return
        snap_engine = self._recorder_thread
        if snap_engine is None or not hasattr(snap_engine, "capture_clearshot"):
            return

        # Gate 3: track must be locked as STRANGER. The caller already
        # guarantees this, but we re-check defensively (the state could
        # have changed between the caller's check and this call).
        if not state or "Stranger" not in state:
            return

        # Gate 4: YOLO det_conf.
        det_conf = float(getattr(trk, "det_conf", 0.0))
        if det_conf < self._clearshot_min_yolo_conf:
            return

        # Gate 5: bbox size.
        try:
            bw = int(trk.x2) - int(trk.x1)
            bh = int(trk.y2) - int(trk.y1)
        except (AttributeError, TypeError, ValueError):
            return
        if bw < self._clearshot_min_bbox_size or bh < self._clearshot_min_bbox_size:
            return

        # Gate 6: cooldown.
        now_us = int(time.time() * 1_000_000)
        last_us = self._clearshot_last_ts_us.get(int(track_id), 0)
        elapsed_s = (now_us - last_us) / 1_000_000.0
        if elapsed_s < self._clearshot_cooldown_s:
            return

        # Gate 7: max-per-track.
        cs_state = self._clearshot_state.get(int(track_id))
        if cs_state is None:
            # Track was registered via register_stranger_label but we
            # don't have local state (e.g. process restart). Rebuild it.
            cs_state = {"label": state, "count": 0}
            self._clearshot_state[int(track_id)] = cs_state
        if (
            self._clearshot_max_per_track > 0
            and cs_state.get("count", 0) >= self._clearshot_max_per_track
        ):
            return

        # All gates passed -- enqueue the clearshot.
        bbox = (
            int(trk.x1), int(trk.y1),
            int(trk.x2), int(trk.y2),
        )
        try:
            ok = snap_engine.capture_clearshot(
                annotated_frame=annotated_frame,
                track_id=int(track_id),
                frame_index=int(getattr(captured, "frame_index", -1)),
                capture_us=int(getattr(captured, "capture_us", now_us)),
                bbox=bbox,
                stranger_label=state,
            )
        except Exception as exc:
            logger.debug(
                "IsolatedAIThread: capture_clearshot failed for "
                "track %d: %s", track_id, exc,
            )
            return

        if ok:
            # Update local cooldown + count state.
            self._clearshot_last_ts_us[int(track_id)] = now_us
            cs_state["count"] = int(cs_state.get("count", 0)) + 1
            # Refresh the cached label (in case it changed, e.g. stranger
            # re-locked with a different ID after a brief unlock).
            cs_state["label"] = state
            logger.debug(
                "IsolatedAIThread: clearshot queued for track %d "
                "(label=%s, idx=%d, det_conf=%.3f, bbox=%dx%d)",
                track_id, state, cs_state["count"], det_conf, bw, bh,
            )

    # ------------------------------------------------------------------
    def _scan_for_anomalies(
        self,
        frame: Any,
        tracks: List[Any],
        frame_ts: Any,
        captured: Any = None,
    ) -> None:
        """
        Lightweight anomaly scan: if LightFaceEngine detects a face
        with no overlapping person bbox, fire evaluate_anomaly on the
        gating engine.

        For performance, we only run this scan if there are few or
        zero tracks (the common anomaly case).

        Patch 23 :: The `captured` parameter is the CapturedFrame
        object from the surrounding _process_frame() call. It is
        forwarded to capture_anomaly_snapshot() so the anomaly PNG
        carries the correct frame_index and capture_us. If None
        (legacy caller), anomaly snapshot capture is skipped but
        evaluate_anomaly still fires.
        """
        if self._face is None or self._gating is None:
            return
        # Only scan for anomalies when the scene is sparse (avoids
        # doubling face-detection cost on crowded scenes).
        if len(tracks) > 3:
            return
        # Patch 42/43 :: Time the anomaly-scan face detection so we can
        # populate yolo/face_det/arcface latencies on the ANOMALY CSV row.
        # YOLO latency for this frame is the last tracking inference time.
        _anom_yolo_ms = float(getattr(self, "_last_tracking_latency_ms", 0.0))
        _t_anom_face_start = time.perf_counter()
        try:
            faces = self._face.detect_and_embed(frame)
        except Exception as exc:
            logger.debug("IsolatedAIThread: anomaly scan face detect failed: %s", exc)
            return
        _anom_face_ms = (time.perf_counter() - _t_anom_face_start) * 1000.0
        # arcface_ms is included in detect_and_embed when embed=True;
        # LightFaceEngine exposes last_det_latency_ms + last_rec_latency_ms.
        _anom_face_det_ms = float(getattr(self._face, "last_det_latency_ms", 0.0))
        _anom_arcface_ms = float(getattr(self._face, "last_rec_latency_ms", 0.0))
        # If LightFaceEngine collapsed both stages into one timer, fall
        # back to the wall-clock time.
        if _anom_face_det_ms <= 0.0 and _anom_arcface_ms <= 0.0 and _anom_face_ms > 0.0:
            _anom_face_det_ms = _anom_face_ms

        if not faces:
            return

        for face in faces:
            fb = face.get("bbox")
            if fb is None:
                continue
            face_x1, face_y1, face_x2, face_y2 = (
                int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]),
            )
            # Check overlap with any person track.
            overlap_found = False
            for trk in tracks:
                ix1 = max(face_x1, int(trk.x1))
                iy1 = max(face_y1, int(trk.y1))
                ix2 = min(face_x2, int(trk.x2))
                iy2 = min(face_y2, int(trk.y2))
                if ix2 > ix1 and iy2 > iy1:
                    overlap_found = True
                    break

            if not overlap_found:
                # Anomaly: face without body bbox.
                try:
                    self._gating.evaluate_anomaly(
                        face_bbox=(face_x1, face_y1, face_x2, face_y2),
                        frame_ts=frame_ts,
                        body_track_ids=[int(t.track_id) for t in tracks],
                        yolo_latency_ms=_anom_yolo_ms,
                        face_det_latency_ms=_anom_face_det_ms,
                        arcface_latency_ms=_anom_arcface_ms,
                        usearch_latency_ms=0.0,
                    )
                    # Snapshot migration :: capture a one-off anomaly
                    # snapshot (no track_id association). The frame
                    # passed in is the raw captured frame (no bbox
                    # overlay yet); the worker will write it as-is.
                    # Patch 23 :: guard against captured=None (legacy
                    # callers that did not forward the CapturedFrame).
                    if self._recorder_thread is not None and captured is not None:
                        # Patch 21 :: _recorder_thread IS the SnapStrangersEngine now.
                        snap_eng = self._recorder_thread
                        if hasattr(
                            snap_eng, "capture_anomaly_snapshot",
                        ):
                            snap_eng.capture_anomaly_snapshot(
                                annotated_frame=captured.frame,
                                frame_index=int(captured.frame_index),
                                capture_us=int(captured.capture_us),
                                anomaly_label="[ANOMALY]",
                            )
                        else:
                            # Fallback: legacy trigger_segment (no-op on
                            # the new engine, but keeps the call site
                            # valid during migration).
                            self._recorder_thread.trigger_segment(
                                TriggerReason.ANOMALY, label="[ANOMALY]",
                            )
                except Exception as exc:
                    logger.warning(
                        "IsolatedAIThread: anomaly evaluation failed: %s", exc,
                    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Patch 62 [2] :: CUDA synchronization + fault escalation.
    #
    # All CUDA-touching sub-engine calls (YOLO, OSNet, InsightFace, USearch)
    # are enqueued asynchronously. Without explicit torch.cuda.synchronize(),
    # a kernel fault surfaces on the NEXT CUDA API call (or never, if the
    # driver's TDR watchdog fires first). This helper forces faults to
    # surface synchronously in the AI thread's Python stack where they can
    # be caught, logged at CRITICAL, and trigger a context reset BEFORE the
    # driver's TDR fires (which is the most likely direct cause of the
    # PAGE_FAULT_IN_NONPAGED_AREA BSOD).
    #
    # Cost: ~0.1-0.5ms per call. At ~10 CUDA calls/frame * 50fps = 500 calls/s,
    # overhead is ~50-250ms/s = 5-25% of one CPU core. Acceptable for stability.
    # ------------------------------------------------------------------
    def _cuda_sync_check(self, context: str) -> None:
        """Drain the CUDA queue and surface any async fault as a Python exception.

        Args:
            context: short string identifying the call site (for logging).
                e.g. "tracking.process", "extract_body_features.birth".

        On a CUDA fault:
          - Logs at CRITICAL level (always visible, unlike DEBUG).
          - Calls torch.cuda.empty_cache() to release corrupted allocations.
          - Increments _cuda_consecutive_failures; on >=5, forces a full
            context reset via torch.cuda.synchronize() + empty_cache() x2.
          - Re-raises the RuntimeError so the caller's except block can
            decide whether to skip the frame or abort the track.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
        except RuntimeError as exc:
            self._cuda_consecutive_failures += 1
            logger.critical(
                "CUDA fault in %s (#%d consecutive): %s",
                context, self._cuda_consecutive_failures, exc,
            )
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self._cuda_consecutive_failures >= 5:
                        logger.critical(
                            "5+ consecutive CUDA failures -- forcing full reset"
                        )
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        self._cuda_consecutive_failures = 0
            except Exception:
                pass
            raise
        # Successful sync resets the failure counter.
        self._cuda_consecutive_failures = 0

    # ------------------------------------------------------------------
    # Patch 59 :: Throttled bbox annotation wrapper.
    #
    # Returns the cached annotated frame on skip frames (every N-1 of N
    # frames), calls the real _annotate_frame() on the Nth frame only.
    # This halves/thirds the cv2.rectangle + cv2.putText workload on
    # the AI thread without breaking the display (bboxes are still
    # shown, just slightly stale on skip frames).
    # ------------------------------------------------------------------
    def _annotate_frame_throttled(
        self,
        frame: Any,
        clean_bboxes: List[Any],
        frame_ts: Any,
        extra: Any = None,
    ) -> Any:
        """Throttled bbox annotation. Patch 61 [B] fix.

        Draws bboxes on every Nth AI cycle (N = bbox_render_every_n_frames).
        Skip cycles return the cached annotated frame directly -- NO copy.

        The GUI loop already does its own copy/resize before display, so
        returning a shared reference here is safe. The AI thread does NOT
        mutate the cached frame after this call returns.
        """
        if self._bbox_render_every_n_frames <= 1:
            # Throttle disabled -- always annotate.
            return self._annotate_frame(frame, clean_bboxes, frame_ts, extra)
        self._bbox_render_frame_counter += 1
        if (self._bbox_render_frame_counter % self._bbox_render_every_n_frames) == 0 \
           or self._last_annotated_frame is None \
           or self._last_annotated_frame.shape != frame.shape:
            # Annotate this frame + cache it.
            annotated = self._annotate_frame(frame, clean_bboxes, frame_ts, extra)
            if annotated is not None:
                # Store reference (NOT a copy). Callers do not mutate
                # the cached frame; the GUI loop does its own copy.
                self._last_annotated_frame = annotated
            return annotated
        else:
            # Skip frame -- return the cached annotated frame.
            # No .copy() here: a 1280x720x3 BGR frame is ~2.7MB, and
            # copying it on every skip frame would defeat the throttle.
            if self._last_annotated_frame is not None:
                return self._last_annotated_frame
            return frame

    def _annotate_frame(
        self,
        frame: Any,
        clean_bboxes: List[Any],
        frame_ts: Any,
        hud: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Draw anti-aliased bboxes with high-level resolved labels onto
        the frame.

        Rigid Display Decoupling: NEVER render internal track IDs, hashes,
        or Kalman metrics. Only the resolved label is drawn.
        """
        if not _CV2_AVAILABLE or frame is None:
            return frame

        annotated = frame.copy()
        for cb in clean_bboxes:
            try:
                x1, y1, x2, y2 = int(cb.x1), int(cb.y1), int(cb.x2), int(cb.y2)
                label = str(cb.resolved_label)

                # Color by state.
                if label.startswith("[ANOMALY]"):
                    color = (0, 0, 255)         # Red
                elif "Stranger" in label:
                    color = (0, 165, 255)       # Orange
                elif label.startswith("[PENDING]"):
                    color = (128, 128, 128)     # Gray
                else:
                    color = (0, 255, 0)         # Green (verified)

                # Anti-aliased rectangle (cv2.LINE_AA on the outline).
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self._box_thickness, cv2.LINE_AA)

                # Label background + text.
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 1,
                )
                # Position the label just above the bbox, or just inside
                # the top if the bbox is too close to the frame top.
                ly = y1 - 4 if y1 - 4 - th >= 0 else y1 + th + 4
                cv2.rectangle(
                    annotated,
                    (x1, ly - th - 4),
                    (x1 + tw + 4, ly + 2),
                    color, -1, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )
            except Exception as exc:
                logger.debug("IsolatedAIThread: annotate failed: %s", exc)

        return annotated

    # ------------------------------------------------------------------
    def _broadcast_telemetry(
        self,
        captured: Optional[CapturedFrame],
        render_pkg: Optional[RenderPackage],
    ) -> None:
        """Broadcast a telemetry packet to the dashboard."""
        if self._broadcaster is None:
            return

        # Compute GUI latency if we have a render package.
        gui_latency_ms = 0.0
        if render_pkg is not None:
            now_us = int(time.time() * 1_000_000)
            gui_latency_ms = max(0.0, (now_us - render_pkg.capture_us) / 1000.0)

        # Patch 61 [G] :: GPU VRAM query throttle.
        # torch.cuda.memory_allocated + get_device_properties are CUDA
        # runtime API calls that can stutter to 5-10ms under GPU
        # contention. Query them on every Mth *broadcast* only (default
        # M=6 -> ~1Hz VRAM updates when broadcast_every=5). Between
        # queries we reuse the cached values.
        self._broadcast_counter += 1
        should_query_vram = (
            (self._broadcast_counter % self._telemetry_vram_every_n_broadcasts) == 0
            or self._cached_gpu_vram_total == 0
        )
        if should_query_vram:
            try:
                import torch
                if torch.cuda.is_available():
                    self._cached_gpu_vram_total = int(
                        torch.cuda.get_device_properties(0).total_memory
                    )
                    self._cached_gpu_vram_used = int(
                        torch.cuda.memory_allocated(0)
                    )
            except Exception:
                pass
        gpu_vram_used = self._cached_gpu_vram_used
        gpu_vram_total = self._cached_gpu_vram_total

        # Patch 22 :: capture_fps / inference_fps REMOVED from the
        # telemetry packet per operator request ("we don't need the
        # fps #"). The dashboard's FPS metric cards have also been
        # removed (Patch 22 dashboard). The TelemetryPacket dataclass
        # still carries the fields (defaulting to 0.0 via .get()) for
        # backward compatibility, but they are no longer computed or
        # sent. The _fps_history deque and current_fps() method are
        # kept for internal orchestrator telemetry (not the UDP
        # broadcast).

        # Patch 20: Compute current session label + date for the
        # dashboard's session selector. If the watcher has already
        # set these attributes, use them (authoritative); otherwise
        # compute inline from time.localtime() as a fallback so the
        # telemetry packet always carries a valid session context.
        session_label: str
        session_date: str
        if self._current_session_label is not None and \
           self._current_session_date is not None:
            session_label = self._current_session_label
            session_date = self._current_session_date
        else:
            _now_t = time.localtime()
            _now_h = _now_t.tm_hour
            session_label = "6AM" if 6 <= _now_h < 18 else "6PM"
            import datetime as _dt
            _today = _dt.date(_now_t.tm_year, _now_t.tm_mon, _now_t.tm_mday)
            if _now_h < 6:
                _today = _today - _dt.timedelta(days=1)
            session_date = _today.strftime("%Y-%m-%d")

        # Patch 22 :: capture_fps / inference_fps omitted from the
        # packet (see note above). The dashboard's TelemetryPacket
        # .from_dict() defaults them to 0.0 via .get().
        packet = {
            "processing_latency_ms": self._last_processing_latency_ms,
            "tracking_latency_ms": self._last_tracking_latency_ms,
            "gui_latency_ms": gui_latency_ms,
            "ai_queue_depth": self._ai_queue.qsize(),
            "ai_queue_maxsize": self._ai_queue.maxsize,
            "current_session_label": session_label,
            "current_session_date": session_date,
            "active_track_count": (
                render_pkg.active_track_count if render_pkg else 0
            ),
            "pending_track_count": (
                render_pkg.pending_track_count if render_pkg else 0
            ),
            "verified_track_count": (
                render_pkg.verified_track_count if render_pkg else 0
            ),
            "stranger_track_count": (
                render_pkg.stranger_track_count if render_pkg else 0
            ),
            "anomaly_count": (
                render_pkg.anomaly_count if render_pkg else 0
            ),
            "gpu_vram_used_bytes": int(gpu_vram_used),
            "gpu_vram_total_bytes": int(gpu_vram_total),
            "throttle_mode": (
                render_pkg.throttle_mode if render_pkg else "IDLE"
            ),
            "encoder_kind": "h264_nvenc",  # Patched by orchestrator from recorder.
            "thread_affinity": self._affinity.all_scopes(),
            "frame_index": captured.frame_index if captured else -1,
            "broadcast_us": int(time.time() * 1_000_000),
            # Patch 63 (hotfix B) :: Heartbeat fields. Lets the
            # dashboard distinguish "main.py is alive but camera is
            # disconnected" (heartbeat=IDLE) from "main.py is alive
            # but processing crashes every cycle" (heartbeat=ERROR)
            # from "main.py is running normally" (heartbeat=LIVE).
            "heartbeat": self._heartbeat_state,
            "ai_errors": self._ai_errors,
            "frames_processed": self._frames_processed,
        }
        self._broadcaster.broadcast(packet)

    # ------------------------------------------------------------------
    def current_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return float(np.mean(list(self._fps_history))) if _NUMPY_AVAILABLE else (
            sum(self._fps_history) / len(self._fps_history)
        )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "frames_processed": self._frames_processed,
            "frames_dropped_full_gui": self._frames_dropped_full_gui,
            "ai_errors": self._ai_errors,
            "current_fps": self.current_fps(),
            "last_processing_latency_ms": self._last_processing_latency_ms,
            "last_tracking_latency_ms": self._last_tracking_latency_ms,
            "ai_queue_depth": self._ai_queue.qsize(),
            "gui_queue_depth": self._gui_queue.qsize(),
        }


# ============================================================================
# Main Orchestrator
# ============================================================================

# ---------------------------------------------------------------------------
# Patch 62 [8] :: VRAMWatchdog daemon thread.
#
# Polls torch.cuda.memory_allocated(0) every 60s. If usage > 85% of total,
# forces torch.cuda.empty_cache() + gc.collect() to release cached
# allocations BEFORE the GPU OOMs. Without this, the only empty_cache call
# is at the 12h session boundary -- a fast VRAM leak (e.g. from the
# _birth_body_cache exception path) can OOM the GPU within minutes,
# triggering driver-level error paths that contribute to BSODs.
# ---------------------------------------------------------------------------
class VRAMWatchdog(threading.Thread):
    """Daemon thread that monitors VRAM and forces cleanup on growth."""

    def __init__(
        self,
        threshold_pct: float = 85.0,
        interval_s: float = 60.0,
        name: str = "VRAMWatchdog",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._threshold_pct = threshold_pct
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._cleanup_count = 0

    def run(self) -> None:
        logger.info(
            "VRAMWatchdog: started (threshold=%.1f%%, interval=%.0fs)",
            self._threshold_pct, self._interval_s,
        )
        while not self._stop_event.is_set():
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    if total > 0:
                        pct = (used / total) * 100.0
                        if pct > self._threshold_pct:
                            logger.warning(
                                "VRAMWatchdog: usage %.1f%% (%d/%d MB) -- "
                                "forcing empty_cache + gc.collect",
                                pct, used // (1024*1024), total // (1024*1024),
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            self._cleanup_count += 1
            except Exception as exc:
                logger.debug("VRAMWatchdog: poll failed: %s", exc)
            self._stop_event.wait(self._interval_s)
        logger.info("VRAMWatchdog: exited (cleanups=%d)", self._cleanup_count)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Patch 63 (hotfix C) :: Orchestrator Heartbeat Thread.
#
# ROOT CAUSE of "main.py has not sent any telemetry yet":
#   The previous heartbeat (Patch 63 hotfix B) was inside
#   IsolatedAIThread.run(). But the AI thread only starts AFTER
#   initialize() completes -- and initialize() loads YOLO, InsightFace,
#   OSNet, ArcFace, etc., which can take 10-30 seconds OR crash
#   entirely (missing model file, CUDA OOM, etc.). If initialize()
#   crashes, the AI thread never starts, no heartbeat fires, and the
#   dashboard shows "no telemetry" forever.
#
# FIX: A dedicated daemon thread that starts as soon as the broadcaster
# is created (in __init__) and broadcasts a minimal heartbeat packet
# every 1 second. This thread is completely independent of the AI
# thread and the frame processing pipeline. It survives:
#   - initialize() crashes (YOLO/CUDA/InsightFace init failure)
#   - AI thread crashes (_process_frame exceptions)
#   - Camera disconnection (no frames available)
#   - Long model loading times (10-30 seconds at cold start)
#
# The heartbeat packet includes the orchestrator's current state string
# so the dashboard can show WHERE main.py is (e.g. "BOOTING",
# "INITIALIZING_TRACKING", "RUNNING", "CRASHED").
# ---------------------------------------------------------------------------
class HeartbeatThread(threading.Thread):
    """Daemon thread that broadcasts orchestrator state every 1 second."""

    def __init__(
        self,
        broadcaster: "PerformanceBroadcaster",
        state_getter: Any,
        stop_event: threading.Event,
        interval_s: float = 1.0,
        name: str = "HeartbeatThread",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._broadcaster = broadcaster
        self._state_getter = state_getter
        self._stop_event = stop_event
        self._interval_s = interval_s
        self._packets_sent: int = 0

    def run(self) -> None:
        logger.info(
            "HeartbeatThread: started (interval=%.1fs, target=udp://%s:%d)",
            self._interval_s,
            self._broadcaster._host, self._broadcaster._port,
        )
        while not self._stop_event.is_set():
            try:
                state: str = "UNKNOWN"
                try:
                    state = str(self._state_getter())
                except Exception:
                    pass
                packet = {
                    "processing_latency_ms": 0.0,
                    "tracking_latency_ms": 0.0,
                    "gui_latency_ms": 0.0,
                    "ai_queue_depth": 0,
                    "ai_queue_maxsize": 2,
                    "current_session_label": "",
                    "current_session_date": "",
                    "active_track_count": 0,
                    "pending_track_count": 0,
                    "verified_track_count": 0,
                    "stranger_track_count": 0,
                    "anomaly_count": 0,
                    "gpu_vram_used_bytes": 0,
                    "gpu_vram_total_bytes": 0,
                    "throttle_mode": "IDLE",
                    "encoder_kind": "h264_nvenc",
                    "thread_affinity": {},
                    "frame_index": -1,
                    "broadcast_us": int(time.time() * 1_000_000),
                    "heartbeat": state,
                    "ai_errors": 0,
                    "frames_processed": 0,
                }
                self._broadcaster.broadcast(packet)
                self._packets_sent += 1
                # Patch 63 (hotfix G) :: Log the first 3 packets sent,
                # then every 30th, so the operator can confirm the
                # heartbeat is actually broadcasting (not just that the
                # thread started). This is critical for diagnosing "no
                # telemetry" issues on the dashboard side.
                if self._packets_sent <= 3 or self._packets_sent % 30 == 0:
                    logger.info(
                        "HeartbeatThread: packet #%d sent (state=%s, "
                        "broadcaster_total=%d)",
                        self._packets_sent, state,
                        self._broadcaster._packets_sent,
                    )
            except Exception as exc:
                logger.debug("HeartbeatThread: broadcast failed: %s", exc)
            self._stop_event.wait(self._interval_s)
        logger.info("HeartbeatThread: exited (packets_sent=%d)", self._packets_sent)

    def stop(self) -> None:
        self._stop_event.set()


class SORTtendanceOrchestrator:
    """
    Top-level orchestrator that wires together the quad-threaded
    execution architecture.

    Public API:
        orch = SORTtendanceOrchestrator(config_path="config/config.yaml")
        orch.initialize()
        orch.start()                       # Blocks on the GUI loop.
        orch.shutdown()                    # Cleanup.
    """

    # ------------------------------------------------------------------
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path: str = config_path
        self.config: Dict[str, Any] = (
            ConfigRegistry.load(config_path) if ConfigRegistry else {}
        )

        # CPU affinity manager.
        self._affinity: AffinityManager = AffinityManager(self.config)

        # Queues.
        main_cfg = self.config.get("main", {})
        self._ai_queue_maxsize: int = int(main_cfg.get("ai_queue_maxsize", 2))
        self._ai_queue: "queue.Queue[Optional[CapturedFrame]]" = queue.Queue(
            maxsize=self._ai_queue_maxsize,
        )
        self._gui_queue: "queue.Queue[Optional[RenderPackage]]" = queue.Queue(
            maxsize=2,
        )

        # Stop event (shared across all threads).
        self._stop_event: threading.Event = threading.Event()

        # Core engines (constructed in initialize()).
        self._tracking_engine: Any = None
        self._identity_matcher: Any = None
        self._res_opt_engine: Any = None
        self._gating_engine: Any = None
        self._face_engine: Any = None
        self._arcface_aligner: Any = None
        self._async_logger: Any = None
        self._video_recorder: Any = None
        self._broadcaster: PerformanceBroadcaster = PerformanceBroadcaster(
            host=self.config.get("dashboard", {}).get("host", "127.0.0.1"),
            port=int(self.config.get("dashboard", {}).get("udp_metrics_port", 9999)),
            enabled=True,
        )

        # Threads.
        self._capture_thread: Optional[CameraCaptureThread] = None
        self._recorder_thread: Optional[AsyncRecorderThread] = None
        self._ai_thread: Optional[IsolatedAIThread] = None

        # Patch 20: Session boundary watcher (12h CSV/snapshot rotation).
        # Constructed in initialize() after the engines it talks to.
        self._session_watcher: Optional[SessionBoundaryWatcher] = None

        # State.
        self._initialized: bool = False
        self._running: bool = False

        # Patch 63 (hotfix C) :: Orchestrator state string.
        # Updated at each major lifecycle step so the HeartbeatThread
        # can broadcast WHERE main.py is. The dashboard shows this
        # string so the operator can see if main.py is stuck in
        # "INITIALIZING_TRACKING" (YOLO load), "INITIALIZING_FACES"
        # (InsightFace load), etc.
        self._orchestrator_state: str = "BOOTING"

        # Patch 63 (hotfix C) :: Start the heartbeat thread IMMEDIATELY
        # after the broadcaster is created. This ensures the dashboard
        # receives telemetry even if initialize() crashes or takes 30
        # seconds to load models. The thread reads
        # self._orchestrator_state and broadcasts it every 1 second.
        self._heartbeat_thread: Optional[HeartbeatThread] = HeartbeatThread(
            broadcaster=self._broadcaster,
            state_getter=lambda: self._orchestrator_state,
            stop_event=self._stop_event,
            interval_s=1.0,
        )
        self._heartbeat_thread.start()

        # Signal handler registration.
        self._signal_handlers_installed: bool = False

    # ==================================================================
    # Lifecycle.
    # ==================================================================
    def initialize(self) -> None:
        if self._initialized:
            logger.warning("SORTtendanceOrchestrator already initialized.")
            return

        self._orchestrator_state = "INITIALIZING"
        logger.info(
            "SORT-tendance orchestrator initializing | config=%s | "
            "ai_queue_maxsize=%d",
            self._config_path, self._ai_queue_maxsize,
        )

        # Pin the main process to its dedicated core block.
        main_cores = self._affinity.get_scope("main_orchestrator") or [0, 1]
        self._affinity.apply_to_process(main_cores)

        # --- Construct the async logger (first, so other engines can log) ---
        self._orchestrator_state = "INITIALIZING_LOGGER"
        if AsyncLoggingEngine is not None:
            self._async_logger = AsyncLoggingEngine(config=self.config)
            self._async_logger.initialize()
            self._async_logger.start()

        # --- Construct the stranger snapshot engine (replaces video recorder) ---
        self._orchestrator_state = "INITIALIZING_SNAP_ENGINE"
        if SnapStrangersEngine is not None:
            self._video_recorder = SnapStrangersEngine(config=self.config)
            self._video_recorder.initialize()
            self._video_recorder.start()
            # Patch 21 :: Wire the SnapStrangersEngine into the AI thread's
            # recorder slot. The AI thread calls capture_birth_snapshot() /
            # finalize_stranger() / finalize_verified() / capture_anomaly_snapshot()
            # on this object. Without this assignment, self._recorder_thread
            # stays None and ALL snapshot operations are silently skipped
            # (no PNGs ever get queued -> empty stranger gallery).
            self._recorder_thread = self._video_recorder

        # --- Construct the resource optimization engine ---
        self._orchestrator_state = "INITIALIZING_RES_OPT"
        if ResourceOptEngine is not None:
            self._res_opt_engine = ResourceOptEngine(config=self.config)

        # --- Construct the tracking engine ---
        self._orchestrator_state = "INITIALIZING_TRACKING"
        if TrackingEngine is not None:
            self._tracking_engine = TrackingEngine(config=self.config)
            self._tracking_engine.initialize()
            self._tracking_engine.warmup()

        # --- Construct the identity matcher ---
        self._orchestrator_state = "INITIALIZING_IDENTITY"
        if IdentityMatcher is not None:
            self._identity_matcher = IdentityMatcher(config=self.config)
            self._identity_matcher.initialize()
            self._identity_matcher.warmup()

            # Patch 67 :: OSNet stranger memory recall from disk.
            # Rebuild the in-memory stranger cache from clearshot PNGs
            # saved on disk. This lets the tracker "remember" strangers
            # across the 12-hour scheduled restart (6AM/6PM) and
            # mid-session crash recovery. Must run AFTER initialize()
            # (which builds the empty dynamic index) + warmup() (which
            # pre-compiles the OSNet CUDA graphs).
            try:
                recalled = self._identity_matcher.recall_strangers_from_disk()
                logger.info(
                    "Stranger OSNet memory recall complete | "
                    "strangers_recalled=%d",
                    recalled,
                )
            except Exception as recall_exc:
                logger.error(
                    "Stranger OSNet memory recall failed (non-fatal, "
                    "continuing with empty stranger cache): %s",
                    recall_exc,
                )
                logger.error(traceback.format_exc())

        # --- Construct the LightFaceEngine + ArcFaceAligner ---
        self._orchestrator_state = "INITIALIZING_FACES"
        if _LightFaceEngine is not None:
            self._face_engine = _LightFaceEngine(config=self.config)
            self._face_engine.initialize()
            self._face_engine.warmup()
        if ArcFaceAligner is not None:
            self._arcface_aligner = ArcFaceAligner(config=self.config)

        # --- Construct the gating engine (wires all the above) ---
        self._orchestrator_state = "INITIALIZING_GATING"
        if GatingEngine is not None:
            self._gating_engine = GatingEngine(
                config=self.config,
                identity_matcher=self._identity_matcher,
                res_opt_engine=self._res_opt_engine,
                async_logger=self._async_logger,
                video_recorder=self._video_recorder,
            )
            self._gating_engine.initialize()

        # Install signal handlers.
        self._install_signal_handlers()

        # Patch 20: Construct the session-boundary watcher.
        # Fires rotation hooks at 06:00 and 18:00 local time on the
        # async_logger, snap_engine, and identity_matcher.
        # All hooks are None-safe: if an engine failed to construct,
        # the watcher simply skips it.
        dashboard_cfg = self.config.get("dashboard", {})
        session_cfg = self.config.get("session_rotation", {})
        self._session_watcher = SessionBoundaryWatcher(
            async_logger=self._async_logger,
            snap_engine=self._video_recorder,
            identity_matcher=self._identity_matcher,
            broadcaster=self._broadcaster,
            am_hour=int(session_cfg.get("am_hour", 6)),
            pm_hour=int(session_cfg.get("pm_hour", 18)),
            tz_local=True,
            # Patch 57 :: pass the gating engine so the watcher can
            # clear the audit logs at the session boundary.
            # Patch 62 [1] :: Fix typo. self._gating is an attribute of
            # IsolatedAIThread, not SORTtendanceOrchestrator. The
            # orchestrator only has self._gating_engine (set at line 3121).
            # Without this fix, SessionBoundaryWatcher is constructed with
            # gating_engine=None, which silently disables the 6AM/6PM audit
            # log cleanup (attendance_final_log + anomaly_log grow unbounded).
            gating_engine=self._gating_engine,
        )

        # Patch 62 [8] :: Start the VRAM watchdog.
        self._vram_watchdog = VRAMWatchdog(
            threshold_pct=float(self.config.get("main", {}).get("vram_watchdog_threshold_pct", 85.0)),
            interval_s=float(self.config.get("main", {}).get("vram_watchdog_interval_s", 60.0)),
        )
        self._vram_watchdog.start()
        self._initialized = True
        self._orchestrator_state = "INITIALIZED"
        logger.info(
            "SORT-tendance orchestrator initialized successfully. "
            "Initial session: %s_%s",
            self._session_watcher.current_session_date(),
            self._session_watcher.current_session_label(),
        )

    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        if self._signal_handlers_installed:
            return
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._signal_handlers_installed = True
            logger.info("Signal handlers installed (SIGINT, SIGTERM).")
        except (ValueError, OSError) as exc:
            # Signal handlers can only be installed on the main thread.
            logger.warning("Failed to install signal handlers: %s", exc)

    # ------------------------------------------------------------------
    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info("Signal %d received; initiating graceful shutdown.", signum)
        self._stop_event.set()

    # ==================================================================
    # Thread startup.
    # ==================================================================
    def start(self) -> None:
        if not self._initialized:
            self.initialize()
        if self._running:
            logger.warning("SORTtendanceOrchestrator already running.")
            return

        self._orchestrator_state = "STARTING_THREADS"
        logger.info("Starting quad-threaded execution architecture...")

        # Construct the threads.
        self._capture_thread = CameraCaptureThread(
            config=self.config,
            ai_queue=self._ai_queue,
            stop_event=self._stop_event,
            affinity_manager=self._affinity,
        )
        self._ai_thread = IsolatedAIThread(
            config=self.config,
            ai_queue=self._ai_queue,
            gui_queue=self._gui_queue,
            stop_event=self._stop_event,
            affinity_manager=self._affinity,
            tracking_engine=self._tracking_engine,
            identity_matcher=self._identity_matcher,
            res_opt_engine=self._res_opt_engine,
            gating_engine=self._gating_engine,
            face_engine=self._face_engine,
            arcface_aligner=self._arcface_aligner,
            recorder_thread=self._recorder_thread,
            broadcaster=self._broadcaster,
        )

        # Start in order: recorder -> capture -> AI.
        self._capture_thread.start()
        self._ai_thread.start()

        # P1-H6 fix: late-bind the AI thread reference into the watcher.
        # The watcher is constructed before _ai_thread exists, so we
        # pass None in its __init__ and set the real reference here.
        # The watcher uses this to call ai_thread.reset_clearshot_mirror()
        # at every 6AM/6PM boundary.
        if self._session_watcher is not None and self._ai_thread is not None:
            self._session_watcher._ai_thread = self._ai_thread

        # Patch 20: Start the session-boundary watcher.
        # Daemon thread; polls every 60 s.
        if self._session_watcher is not None:
            self._session_watcher.start()
            # Seed the AI thread's session-context attributes so the
            # telemetry packet reflects the watcher's authoritative
            # value instead of the inline fallback. The watcher is a
            # daemon and won't update these directly (we keep the
            # coupling one-way: watcher fires rotation hooks only,
            # and the AI thread reads time.localtime() if the watcher
            # hasn't been started).
            if self._ai_thread is not None:
                self._ai_thread._current_session_label = (
                    self._session_watcher.current_session_label()
                )
                self._ai_thread._current_session_date = (
                    self._session_watcher.current_session_date()
                )

        self._running = True
        self._orchestrator_state = "RUNNING"
        logger.info("Quad-threaded execution architecture started.")

        # Main thread enters the GUI loop (blocks until stop).
        try:
            self._gui_loop()
        finally:
            self.shutdown()

    # ==================================================================
    # Main Thread: Graphics GUI Loop.
    # ==================================================================
    def _gui_loop(self) -> None:
        """
        Main-thread GUI loop: drains the GUI queue and renders frames
        via cv2.imshow.

        Rigid Display Decoupling: the GUI loop only consumes
        RenderPackage objects, which contain CleanBBox instances with
        ONLY resolved labels. Internal track IDs are NEVER rendered.
        """
        if not _CV2_AVAILABLE:
            logger.error(
                "GUI loop: cv2 not available -- running headless. The "
                "system will still capture + process + record, but no "
                "preview window will be shown.",
            )
            while not self._stop_event.is_set():
                # Drain the GUI queue to prevent back-pressure on the AI thread.
                try:
                    self._gui_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.05)
            return

        window_name = "SORT-tendance :: Live Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, 1280, 720)
        except cv2.error:
            pass

        render_fps_target = int(
            self.config.get("main", {}).get("render_fps_target", 60)
        )
        frame_interval_s = 1.0 / max(1, render_fps_target)

        last_render_us = 0
        frames_rendered = 0
        # Patch 62 [9] :: AI-thread hang detection.
        # If render_pkg.frame_index hasn't advanced for > 10s, the AI
        # thread is hung inside a CUDA call (e.g. waiting on a driver
        # mutex). Force shutdown to avoid the driver's TDR path that
        # triggers 0x50 BSODs.
        _last_ai_frame_index: int = -1
        _last_ai_progress_us: int = int(time.time() * 1_000_000)
        _ai_hang_threshold_s: float = float(
            self.config.get("main", {}).get("ai_hang_threshold_s", 10.0)
        )
        _ai_hang_triggered: bool = False

        logger.info("GUI loop entered (target %d FPS).", render_fps_target)

        while not self._stop_event.is_set():
            try:
                # Drain the GUI queue (latest-frame-wins).
                render_pkg: Optional[RenderPackage] = None
                while True:
                    try:
                        nxt = self._gui_queue.get_nowait()
                    except queue.Empty:
                        break
                    render_pkg = nxt
                    if self._gui_queue.empty():
                        break

                if render_pkg is None:
                    try:
                        render_pkg = self._gui_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                if render_pkg.frame is None:
                    continue

                # Patch 62 [9] :: Check for AI-thread hang.
                if render_pkg.frame_index == _last_ai_frame_index:
                    _stall_ms = (int(time.time() * 1_000_000) - _last_ai_progress_us) / 1000.0
                    if _stall_ms > _ai_hang_threshold_s * 1000.0 and not _ai_hang_triggered:
                        _ai_hang_triggered = True
                        logger.critical(
                            "AI thread hung for %.1fs (frame_index stuck at %d) "
                            "-- forcing shutdown to avoid driver TDR / BSOD",
                            _stall_ms / 1000.0, _last_ai_frame_index,
                        )
                        self._stop_event.set()
                        break
                else:
                    _last_ai_frame_index = render_pkg.frame_index
                    _last_ai_progress_us = int(time.time() * 1_000_000)
                    _ai_hang_triggered = False

                # Draw a minimal HUD (AI latency + active tracks + throttle
                # mode). Patch 33 :: "Frame #N" prefix removed per
                # operator request -- the frame index is internal debug
                # info that doesn't belong on the live camera feed.
                # This is the ONLY on-screen metadata; no internal IDs.
                # Patch 62 [3] :: Do NOT mutate render_pkg.frame in-place.
                # The AI thread holds a reference to the same buffer (via
                # _last_annotated_frame when bbox throttle is active). NumPy
                # buffers are not thread-safe under concurrent read+write;
                # on Windows this race can corrupt the buffer's internal
                # metadata, leading to heap corruption that manifests hours
                # later as a 0x50 BSOD. Copy first, then draw HUD on the copy.
                if _CV2_AVAILABLE:
                    display_frame = render_pkg.frame.copy()
                    hud_text = (
                        f"AI: {render_pkg.ai_latency_ms:.1f}ms | "
                        f"Active: {render_pkg.active_track_count} | "
                        f"Mode: {render_pkg.throttle_mode}"
                    )
                    cv2.putText(
                        display_frame, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2, cv2.LINE_AA,
                    )
                else:
                    display_frame = render_pkg.frame

                # Upscale 640x360 -> 1280x720 for display only (AI ran on the smaller frame).
                display_w = self.config["camera"].get("display_width", 1280)
                display_h = self.config["camera"].get("display_height", 720)
                if display_frame.shape[1] != display_w or display_frame.shape[0] != display_h:
                    display_frame = cv2.resize(
                        display_frame, (display_w, display_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                cv2.imshow(window_name, display_frame)

                # WaitKey with the frame interval (in ms) to bound CPU usage.
                # 1ms minimum so OpenCV can pump events.
                wait_ms = max(1, int(frame_interval_s * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("GUI loop: quit key pressed; initiating shutdown.")
                    self._stop_event.set()
                    break

                frames_rendered += 1
                last_render_us = int(time.time() * 1_000_000)

            except Exception as exc:
                logger.error(
                    "GUI loop: top-level exception: %s\n%s",
                    exc, traceback.format_exc(),
                )
                time.sleep(0.05)

        # Cleanup.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        logger.info("GUI loop exited | frames_rendered=%d", frames_rendered)

    # ==================================================================
    # Shutdown.
    # ==================================================================
    def shutdown(self) -> None:
        if not self._running:
            logger.info("SORTtendanceOrchestrator: shutdown called but not running.")
            # Patch 63 (hotfix C) :: Even if not running, stop the
            # heartbeat thread if it's still alive (e.g. initialize()
            # crashed before start() was called).
            self._orchestrator_state = "SHUTTING_DOWN"
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                try:
                    self._heartbeat_thread.stop()
                    self._heartbeat_thread.join(timeout=2.0)
                except Exception:
                    pass
            return

        self._orchestrator_state = "SHUTTING_DOWN"
        logger.info("SORT-tendance orchestrator shutting down...")
        self._stop_event.set()

        # Patch 63 (hotfix C) :: Stop the heartbeat thread early so it
        # doesn't try to broadcast on a closed socket. Give it a moment
        # to send one final "SHUTTING_DOWN" packet before stopping.
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            try:
                # Allow one final heartbeat (1s interval) to fire.
                time.sleep(0.1)
                self._heartbeat_thread.stop()
                self._heartbeat_thread.join(timeout=2.0)
            except Exception as exc:
                logger.debug("HeartbeatThread stop failed: %s", exc)

        # Patch 20: Stop the session-boundary watcher FIRST so it
        # doesn't fire rotation hooks on engines that are mid-shutdown.
        # Patch 62 [8] :: Stop the VRAM watchdog first (before any
        # CUDA teardown so it doesn't poll a torn-down context).
        if getattr(self, "_vram_watchdog", None) is not None:
            try:
                self._vram_watchdog.stop()
                self._vram_watchdog.join(timeout=2.0)
            except Exception as exc:
                logger.warning("VRAMWatchdog stop failed: %s", exc)

        if self._session_watcher is not None and self._session_watcher.is_alive():
            self._session_watcher.stop()
            self._session_watcher.join(timeout=5.0)

        # Give the threads a moment to observe the stop event.
        time.sleep(0.2)

        # Stop threads in reverse order: AI -> capture -> recorder.
        if self._ai_thread is not None and self._ai_thread.is_alive():
            self._ai_thread.join(timeout=5.0)
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
        # Patch 21 :: self._recorder_thread is now the SnapStrangersEngine
        # (not a threading.Thread), so is_alive()/join() would raise
        # AttributeError. The engine's internal worker thread is already
        # joined by self._video_recorder.shutdown(timeout_s=8.0) below, so
        # no explicit join is needed here.

        # Shut down engines.
        if self._gating_engine is not None:
            try:
                self._gating_engine.shutdown()
            except Exception as exc:
                logger.warning("GatingEngine shutdown failed: %s", exc)

        if self._identity_matcher is not None:
            try:
                self._identity_matcher.close()
            except Exception as exc:
                logger.warning("IdentityMatcher close failed: %s", exc)

        if self._tracking_engine is not None:
            try:
                self._tracking_engine.shutdown()
            except Exception as exc:
                logger.warning("TrackingEngine shutdown failed: %s", exc)

        if self._face_engine is not None:
            try:
                self._face_engine.close()
            except Exception as exc:
                logger.warning("FaceEngine close failed: %s", exc)

        if self._res_opt_engine is not None:
            try:
                self._res_opt_engine.shutdown()
            except Exception as exc:
                logger.warning("ResourceOptEngine shutdown failed: %s", exc)

        if self._video_recorder is not None:
            try:
                self._video_recorder.shutdown(timeout_s=8.0)
            except Exception as exc:
                logger.warning("VideoRecorder shutdown failed: %s", exc)

        if self._async_logger is not None:
            try:
                self._async_logger.shutdown(timeout_s=5.0)
            except Exception as exc:
                logger.warning("AsyncLogger shutdown failed: %s", exc)

        if self._broadcaster is not None:
            try:
                self._broadcaster.close()
            except Exception as exc:
                logger.warning("Broadcaster close failed: %s", exc)

        self._running = False
        gc.collect()
        # Patch 62 [4] :: CUDA + cv2 cleanup at shutdown.
        # Without this, the CUDA driver must forcibly reclaim resources
        # during process teardown -- which is exactly when BSODs occur if
        # the driver is in any state of corruption. synchronize() drains
        # the queue; empty_cache() releases the caching allocator's pool;
        # ipc_collect() releases any inter-process handles. destroyAllWindows
        # is a belt-and-suspenders call in case _gui_loop didn't reach it.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("CUDA cleanup at shutdown: OK")
        except Exception as exc:
            logger.warning("CUDA cleanup at shutdown failed: %s", exc)
        try:
            if _CV2_AVAILABLE:
                cv2.destroyAllWindows()
        except Exception as exc:
            logger.warning("cv2.destroyAllWindows at shutdown failed: %s", exc)
        logger.info("SORT-tendance orchestrator shutdown complete.")

    # ==================================================================
    # Telemetry.
    # ==================================================================
    def telemetry(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "stop_event_set": self._stop_event.is_set(),
            "ai_queue_depth": self._ai_queue.qsize(),
            "gui_queue_depth": self._gui_queue.qsize(),
            "capture_thread": (
                self._capture_thread.telemetry()
                if self._capture_thread is not None else None
            ),
            "ai_thread": (
                self._ai_thread.telemetry()
                if self._ai_thread is not None else None
            ),
            "recorder_thread": (
                self._recorder_thread.telemetry()
                if self._recorder_thread is not None else None
            ),
            "affinity": self._affinity.telemetry(),
            "broadcaster": self._broadcaster.telemetry(),
            "gating": (
                self._gating_engine.telemetry()
                if self._gating_engine is not None else None
            ),
            "res_opt": (
                self._res_opt_engine.telemetry()
                if self._res_opt_engine is not None else None
            ),
        }


# ============================================================================
# Module Entry Point
# ============================================================================
def main() -> None:
    """Top-level entry point for `python main.py`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable propagation on all sortendance.* loggers AFTER basicConfig
    # has wired up the root handler. This prevents the duplicate-log-line
    # bug where every record gets emitted once by the named logger's own
    # handler and again by the root handler.
    _silence_sortendance_propagation()

    config_path = "config/config.yaml"
    # Allow CLI override.
    argv = sys.argv[1:]
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            config_path = argv[idx + 1]

    logger.info("========================================")
    logger.info("SORT-tendance :: Booting orchestrator")
    logger.info("Config: %s", config_path)
    logger.info("========================================")

    if _GPU_LINKER is not None:
        logger.info("GPU Linker: %s", _GPU_LINKER.telemetry())

    # Patch 39 :: Auto-restart on uncaught exceptions.
    # The orchestrator is wrapped in a retry loop. On any uncaught
    # Exception (NOT KeyboardInterrupt = Ctrl+C, and NOT the clean Q
    # key path which returns normally from orch.start()), the
    # orchestrator is shut down and a fresh instance is constructed +
    # started. A 2-second sleep between restarts prevents a tight
    # crash loop. The restart counter resets after 60 seconds of
    # stable operation, so "rapid" crashes (boot-loop) hit the 100-
    # restart cap and exit, while a crash after hours of stable
    # operation is treated as a fresh incident.
    MAX_RESTARTS: int = 100
    RESTART_DELAY_S: float = 2.0
    STABLE_RUNTIME_RESET_S: float = 60.0
    restart_count: int = 0
    while True:
        orch = SORTtendanceOrchestrator(config_path=config_path)
        boot_time = time.time()
        try:
            orch.start()
            # Normal exit (Q key pressed in the OpenCV window ->
            # orch.start() returns cleanly). Don't restart.
            logger.info("Orchestrator exited normally; not restarting.")
            break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt (Ctrl+C) received; shutting down. No auto-restart.")
            orch.shutdown()
            break
        except SystemExit as exc:
            # Q key path may call sys.exit() in some code paths.
            # Treat as clean exit -- don't restart.
            logger.info("SystemExit (code=%s) -- treating as clean shutdown. No auto-restart.", exc.code)
            orch.shutdown()
            break
        except Exception as exc:
            runtime_s = time.time() - boot_time
            # If the orchestrator ran stably for > 60s before crashing,
            # reset the restart counter (this is a "fresh incident",
            # not a boot-loop).
            if runtime_s > STABLE_RUNTIME_RESET_S:
                restart_count = 0
            restart_count += 1
            logger.critical(
                "Orchestrator crashed (restart %d/%d, runtime=%.1fs): %s\n%s",
                restart_count, MAX_RESTARTS, runtime_s, exc,
                traceback.format_exc(),
            )
            # Patch 63 (hotfix C) :: Set CRASHED state so the
            # heartbeat thread can broadcast the crash to the
            # dashboard BEFORE shutdown() stops it. This lets the
            # dashboard show "main.py CRASHED -- auto-restarting"
            # instead of just going silent.
            try:
                orch._orchestrator_state = "CRASHED"
                # Give the heartbeat thread a moment to broadcast
                # the CRASHED state before shutdown stops it.
                time.sleep(1.5)
            except Exception:
                pass
            try:
                orch.shutdown()
            except Exception as shutdown_exc:
                logger.error(
                    "Shutdown after crash failed: %s", shutdown_exc,
                )
            if restart_count >= MAX_RESTARTS:
                logger.critical(
                    "Max restarts (%d) reached -- exiting to prevent "
                    "infinite crash loop. Fix the root cause and relaunch.",
                    MAX_RESTARTS,
                )
                sys.exit(1)
            logger.info(
                "Auto-restarting in %.1fs (restart %d/%d)...",
                RESTART_DELAY_S, restart_count, MAX_RESTARTS,
            )
            time.sleep(RESTART_DELAY_S)
            # Re-emit the boot banner so the operator can see the
            # restart in the console log.
            logger.info("========================================")
            logger.info("SORT-tendance :: Auto-restarting orchestrator")
            logger.info("Config: %s", config_path)
            logger.info("========================================")
            if _GPU_LINKER is not None:
                logger.info("GPU Linker: %s", _GPU_LINKER.telemetry())
            continue


if __name__ == "__main__":
    main()