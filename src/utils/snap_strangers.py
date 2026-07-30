"""
SORT-tendance :: src/utils/snap_strangers.py

Production-grade Stranger Snapshot Engine (Patch 18 :: per-session
subfolder rotation, replaces video_recorder.py).

Design Rationale
================
The previous VideoRecorderEngine ran a 25 fps H.264 encoding pipeline
(NVENC / libx264) on EVERY captured frame, backed by a 120-frame RAM
rolling buffer, a StrangerAnonymizer overlay, and a PurgeWorker. The
continuous encoding consumed:
  * ~331 MB of pinned RAM (120 frames x 1280x720x3 BGR),
  * 1 dedicated worker thread + 1 PurgeWorker thread,
  * GPU NVENC duty cycle (competing with face/recognition inference),
  * ~25 disk writes/sec of H.264 segments on every trigger.

For a security attendance system whose ONLY forensic need is "show me
the moment a stranger first appeared", this is overkill. The snapshot
engine:
  * Captures EXACTLY ONE PNG per stranger event (at track birth).
  * Renames the PNG once the stranger is formally locked (so the
    filename carries the Stranger_XX label).
  * Deletes the PNG if the track turns out to be a verified student
    (no forensic value -- the student is known).
  * Runs ONE lightweight worker thread + ONE PurgeWorker thread.
  * Uses ~0 GPU cycles, ~0 continuous RAM (PNG is encoded by cv2.imwrite
    on the worker thread, ~50 ms per snapshot, no rolling buffer).

Snapshot Lifecycle
==================
  Track birth (YOLO detects a new person):
    -> capture_birth_snapshot(annotated_frame, tid, ...)
       -> worker writes: {timestamp_ms}_track{tid}_BIRTH.png
       -> pending_path[tid] = path

  Track locked as STRANGER:
    -> finalize_stranger(tid, "[Stranger_03]")
       -> worker renames: ..._BIRTH.png -> ..._STRANGER_[Stranger_03].png
       -> pending_path[tid] = renamed path

  Track locked as VERIFIED_STUDENT:
    -> finalize_verified(tid)
       -> worker deletes: ..._BIRTH.png
       -> pending_path.pop(tid, None)

  Track disappears (no lock reached):
    -> The PNG stays on disk as ..._BIRTH.png with the track_id.
       Per Patch 18 the PurgeWorker is DISABLED by default -- snapshots
       are retained indefinitely (organized by date+session folder) so
       the operator can browse the full forensic history via the
       dashboard. Set `snap_strangers.purge_enabled: true` in
       config.yaml to re-enable the legacy retention-based purging.

Patch 18 :: Per-Session Subfolder Layout
========================================
As of Patch 18, snapshots are written into a date+session hierarchy:

    {output_dir}/{YYYY-MM-DD}/{6AM Session | 6PM Session}/...png

The session key is computed from the snapshot's capture timestamp
using the same `compute_session_key()` helper as the async_logger
(modules share the LOCAL 06:00 / 18:00 boundary convention). This
means:
    * 06:00..17:59 local  -> {output_dir}/{today}/6AM Session/...
    * 18:00..23:59 local  -> {output_dir}/{today}/6PM Session/...
    * 00:00..05:59 local  -> {output_dir}/{yesterday}/6PM Session/...

The dashboard's StrangerGalleryScanner walks this hierarchy directly
(no coupling to the engine itself) so it can render historical
sessions even while the engine is offline.

Note on OSNet memory reset (Patch 18):
The user has explicitly requested that snap_strangers itself should
NOT purge any files -- only the OSNet re-identification feature
gallery should be reset every 06:00 / 18:00. That OSNet reset is
performed by main.py (or identity_matcher.py) via the
`AsyncLoggingEngine.register_session_rollover_observer()` hook
exposed in async_logger.py -- the snap_strangers engine is unaware
of OSNet and only handles PNG writes.

API Compatibility
=================
The SnapStrangersEngine exposes the same public method names as the
old VideoRecorderEngine (initialize, start, shutdown, push_frame,
trigger_segment, update_stranger_bboxes, clear_stranger_bboxes,
end_current_segment, telemetry). The continuous-frame methods
(push_frame, trigger_segment for ANOMALY) are accepted as no-ops so
that main.py's call sites can be left structurally intact while we
migrate. The NEW methods are:
  * capture_birth_snapshot(annotated_frame, track_id, frame_index, ...)
  * finalize_stranger(track_id, stranger_label)
  * finalize_verified(track_id)
  * get_snapshot_path(track_id)  -- used by the logger to fill the
    snapshot_path CSV column.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import re
import sys
import gc
import time
import queue
import shutil
import threading
import logging
import traceback
import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _YAML_AVAILABLE = False
    yaml = None  # type: ignore

# Local config registry import (absolute path resolution).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.database_manager import ConfigRegistry
except ImportError:                         # pragma: no cover
    ConfigRegistry = None  # type: ignore

# Patch 18 :: Import the shared 12-hour session helpers from async_logger.
# These provide `compute_session_key(ts_us) -> (date_str, "06AM"|"06PM")`
# and `session_label_to_dir(label) -> "6AM Session"|"6PM Session"` so that
# snap_strangers, async_logger, and dashboard all agree on the same
# session boundary (LOCAL 06:00 / 18:00) and folder naming.
try:
    from utils.async_logger import (
        compute_session_key,
        session_label_to_dir,
        SESSION_LABEL_AM,
        SESSION_LABEL_PM,
        SESSION_DIR_AM,
        SESSION_DIR_PM,
    )
    _SESSION_HELPERS_AVAILABLE = True
except ImportError:                         # pragma: no cover
    # Fallback: re-implement the helpers locally so the module can still
    # operate standalone (e.g. in unit tests that don't have async_logger
    # on the import path). The semantics are identical to async_logger.
    _SESSION_HELPERS_AVAILABLE = False
    SESSION_AM_START_HOUR = 6
    SESSION_PM_START_HOUR = 18
    SESSION_LABEL_AM = "06AM"
    SESSION_LABEL_PM = "06PM"
    SESSION_DIR_AM = "6AM Session"
    SESSION_DIR_PM = "6PM Session"

    def compute_session_key(ts_us: int) -> Tuple[str, str]:
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

    def session_label_to_dir(session_label: str) -> str:
        if session_label == SESSION_LABEL_AM:
            return SESSION_DIR_AM
        if session_label == SESSION_LABEL_PM:
            return SESSION_DIR_PM
        return session_label


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.snap_strangers")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Sentinels
# ============================================================================
class _ShutdownSentinel:
    """Queue sentinel signaling worker drain-and-exit."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<SnapStrangersShutdownSentinel>"


_SHUTDOWN = _ShutdownSentinel()


# ============================================================================
# Enums
# ============================================================================
class TriggerReason(str, Enum):
    """
    Kept for API compatibility with main.py's existing call sites.
    The snapshot engine ignores these -- it only acts on track-birth
    and track-lock events, not on segment triggers.
    """
    ANOMALY = "ANOMALY"
    STRANGER = "STRANGER"
    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"


class _SnapshotOp(str, Enum):
    """Internal op tag for the worker queue."""
    WRITE_BIRTH = "WRITE_BIRTH"          # Write a new birth snapshot.
    RENAME_STRANGER = "RENAME_STRANGER"  # Rename to include label.
    DELETE_VERIFIED = "DELETE_VERIFIED"  # Delete (track was verified).
    # Patch 63 :: MOVE_VERIFIED moves the birth PNG into the
    # identified/ subfolder instead of deleting it, so the Event
    # Log page can show verified-person snapshots with hourly history.
    MOVE_VERIFIED = "MOVE_VERIFIED"      # Move to identified/ subfolder.
    ANOMALY_SNAPSHOT = "ANOMALY_SNAPSHOT"  # One-off anomaly capture.
    # Patch 65 :: WRITE_CLEARSHOT writes an additional high-quality PNG
    # for a STRANGER-locked track, captured periodically when YOLO
    # confidence is high. These serve as OSNet "memory recall" reference
    # frames for the stranger -- the snapshot itself is the memory
    # artifact that operators (and downstream OSNet re-extraction) can
    # use to re-identify the same stranger in future sessions.
    WRITE_CLEARSHOT = "WRITE_CLEARSHOT"  # Periodic stranger clearshot.


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class SnapshotTicket:
    """One unit of work for the snapshot worker thread."""
    op: _SnapshotOp
    track_id: int = -1
    # For WRITE_BIRTH.
    frame: Any = None              # np.ndarray (BGR uint8) -- the annotated frame.
    frame_index: int = -1
    capture_us: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # For RENAME_STRANGER.
    stranger_label: str = ""
    # For ANOMALY_SNAPSHOT.
    anomaly_label: str = ""
    # Patch 65 :: For WRITE_CLEARSHOT. Per-stranger clearshot counter
    # (1-indexed). Baked into the filename as _CLEARSHOT_{YY:02d} so
    # operators can browse the sequence of clearshots per stranger.
    clearshot_idx: int = 0
    enqueue_wall_us: int = 0


@dataclass
class SnapshotStats:
    """Per-snapshot statistics (returned by the worker for telemetry)."""
    track_id: int
    op: str
    path: Optional[str]
    bytes_written: int = 0
    success: bool = True
    error: str = ""


# ============================================================================
# Purge Worker (retention enforcement)
# ============================================================================
class PurgeWorker:
    """
    Background thread that scans the snapshot output directory and
    purges PNG files older than the configured retention window.

    Runs on a coarse cadence (default: every 5 minutes) to avoid
    disk I/O contention with the snapshot writer.
    """

    def __init__(
        self,
        output_dir: str,
        retention_hours: int,
        scan_interval_s: int,
        file_extension: str = ".png",
    ) -> None:
        self._output_dir: str = output_dir
        self._retention_s: float = float(retention_hours) * 3600.0
        self._scan_interval_s: int = max(60, int(scan_interval_s))
        self._file_extension: str = file_extension
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._files_purged: int = 0
        self._bytes_freed: int = 0
        self._scan_count: int = 0
        self._running: bool = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sortendance.snap_strangers.purge",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info(
            "PurgeWorker started | dir=%s | retention=%.1fh | scan=%ds",
            self._output_dir,
            self._retention_s / 3600.0,
            self._scan_interval_s,
        )

    # ------------------------------------------------------------------
    def stop(self, timeout_s: float = 5.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._running = False
        logger.info(
            "PurgeWorker joined cleanly | purged=%d | freed=%d bytes | scans=%d",
            self._files_purged, self._bytes_freed, self._scan_count,
        )

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            # Initial sleep so the recorder has a chance to start before
            # the first purge scan.
            time.sleep(2.0)
            while not self._stop_event.is_set():
                try:
                    self._scan_once()
                except Exception as exc:
                    logger.warning(
                        "PurgeWorker scan error: %s", exc,
                    )
                # Wait for the scan interval, but wake up early if
                # shutdown is requested.
                self._stop_event.wait(self._scan_interval_s)
        except Exception as exc:
            logger.error(
                "PurgeWorker crashed: %s\n%s",
                exc, traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    def _scan_once(self) -> None:
        if not os.path.isdir(self._output_dir):
            return
        self._scan_count += 1
        now_s = time.time()
        purged_this_scan = 0
        bytes_this_scan = 0
        try:
            for fname in os.listdir(self._output_dir):
                if not fname.lower().endswith(self._file_extension):
                    continue
                fpath = os.path.join(self._output_dir, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    age_s = now_s - mtime
                    if age_s > self._retention_s:
                        fsize = os.path.getsize(fpath)
                        os.remove(fpath)
                        purged_this_scan += 1
                        bytes_this_scan += fsize
                except OSError:
                    # File may have been removed between listdir and
                    # getmtime; ignore.
                    continue
        except OSError as exc:
            logger.warning("PurgeWorker listdir failed: %s", exc)
            return

        if purged_this_scan > 0:
            self._files_purged += purged_this_scan
            self._bytes_freed += bytes_this_scan
            logger.info(
                "PurgeWorker scan #%d purged %d files (%.2f MB freed) | "
                "total_purged=%d",
                self._scan_count, purged_this_scan,
                bytes_this_scan / (1024.0 * 1024.0),
                self._files_purged,
            )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "files_purged": self._files_purged,
            "bytes_freed": self._bytes_freed,
            "scan_count": self._scan_count,
            "running": self._running,
        }


# ============================================================================
# Snapshot Engine
# ============================================================================
class SnapStrangersEngine:
    """
    Top-level stranger snapshot orchestrator.

    Owns:
      * A bounded `queue.Queue` for non-blocking snapshot hand-off.
      * A daemon worker thread driving PNG writes / renames / deletes.
      * A `PurgeWorker` for retention enforcement.
      * A per-track-id pending-path map (lock-protected) so the
        orchestrator can query the snapshot path for a given track
        (used by the logger to fill the snapshot_path CSV column).

    Public API for the orchestrator:
        engine = SnapStrangersEngine(config)
        engine.initialize()
        engine.start()
        engine.capture_birth_snapshot(annotated, tid, frame_index, ...)
        engine.finalize_stranger(tid, "[Stranger_03]")
        engine.finalize_verified(tid)
        path = engine.get_snapshot_path(tid)
        engine.shutdown()

    Backward-compatible no-ops (so main.py's existing call sites keep
    working during the migration):
        engine.push_frame(...)               # no-op
        engine.trigger_segment(reason, label)  # captures anomaly snapshot
        engine.update_stranger_bboxes(...)   # no-op
        engine.clear_stranger_bboxes(...)    # no-op
        engine.end_current_segment()         # no-op
    """

    MAX_CONSECUTIVE_ERRORS: int = 64

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        # Read the snap_strangers section. Fall back to the legacy
        # video_recorder section if the user hasn't migrated config yet
        # (graceful degradation during the transition).
        rcfg = self.config.get("snap_strangers")
        if rcfg is None:
            rcfg = self.config.get("video_recorder", {})
            if rcfg:
                logger.warning(
                    "snap_strangers config section not found; falling "
                    "back to legacy video_recorder section. Please "
                    "rename 'video_recorder:' to 'snap_strangers:' in "
                    "config.yaml.",
                )

        self._output_dir: str = str(rcfg.get("output_dir", "storage/snap_strangers"))
        self._retention_hours: int = int(rcfg.get("retention_hours", 72))
        self._purge_interval_s: int = int(rcfg.get("purge_check_interval_s", 300))
        self._png_compression: int = int(rcfg.get("png_compression", 6))
        self._queue_maxsize: int = int(rcfg.get("queue_maxsize", 64))
        # Patch 18 :: PurgeWorker is DISABLED by default. The operator
        # explicitly requested that snap_strangers NOT purge any files
        # (snapshots are retained indefinitely, organized by date+session
        # folder, and browsable via the dashboard). Only the OSNet
        # re-identification feature gallery is reset every 06:00 / 18:00
        # (handled by main.py via the AsyncLoggingEngine session-rollover
        # observer hook, NOT by snap_strangers itself).
        self._purge_enabled: bool = bool(rcfg.get("purge_enabled", False))
        # Patch 63 :: Verified-snapshot retention. When True (default),
        # finalize_verified() MOVES the birth PNG into the identified/
        # subfolder instead of deleting it. When False, falls back to
        # the old delete behavior.
        self._retain_verified_snapshots: bool = bool(
            rcfg.get("retain_verified_snapshots", True)
        )
        self._identified_subdir: str = str(
            rcfg.get("identified_subdir", "identified")
        )
        # Patch 63 (hotfix K) :: Unresolved birth snapshot retention.
        # When False (default), prune_dead_tracks() keeps the BIRTH.png
        # on disk (only removes the _pending_paths dict entry). This
        # lets operators review everyone who appeared, even briefly.
        # When True, the old behavior is restored (BIRTH.png deleted).
        self._prune_unresolved_births: bool = bool(
            rcfg.get("prune_unresolved_births", False)
        )

        # ----------------------------------------------------------------
        # Patch 65 :: CLEARSHOT snapshot config.
        #
        # After a track is locked as STRANGER, the AI thread periodically
        # requests an additional "clearshot" -- a high-quality PNG
        # captured when YOLO confidence on the person is high. These
        # clearshots serve as OSNet "memory recall" reference frames:
        # each one is a clean, frontal-ish capture that the operator
        # (or a downstream re-extraction pipeline) can use as an
        # additional body-feature reference for the same stranger.
        #
        # The cooldown, max-per-track, and YOLO confidence threshold
        # are enforced by main.py (it has access to the track state +
        # det_conf per frame). The snapshot engine only handles the
        # file write -- it just needs the subdir + the per-track
        # counter (so filenames don't collide).
        # ----------------------------------------------------------------
        cs_cfg = rcfg.get("clearshot") or {}
        self._clearshot_enabled: bool = bool(cs_cfg.get("enabled", True))
        self._clearshot_subdir: str = str(cs_cfg.get("subdir", "clearshots"))
        # Min YOLO confidence and min bbox size are enforced by main.py
        # (they're passed through via config.get("snap_strangers.clearshot.*")),
        # but we also read them here for telemetry / self-test purposes.
        self._clearshot_min_yolo_conf: float = float(
            cs_cfg.get("min_yolo_conf", 0.70)
        )
        self._clearshot_min_bbox_size: int = int(
            cs_cfg.get("min_bbox_size", 80)
        )
        self._clearshot_cooldown_s: float = float(
            cs_cfg.get("cooldown_s", 30.0)
        )
        self._clearshot_max_per_track: int = int(
            cs_cfg.get("max_per_track", 20)
        )

        # Camera frame dimensions (from camera block if present).
        cam_cfg = self.config.get("camera", {})
        self._width: int = int(cam_cfg.get("width", width))
        self._height: int = int(cam_cfg.get("height", height))

        # Core components.
        self._queue: "queue.Queue[Union[SnapshotTicket, _ShutdownSentinel]]" = queue.Queue(
            maxsize=self._queue_maxsize,
        )
        self._purge_worker: PurgeWorker = PurgeWorker(
            output_dir=self._output_dir,
            retention_hours=self._retention_hours,
            scan_interval_s=self._purge_interval_s,
            file_extension=".png",
        )

        # Worker thread state.
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event: threading.Event = threading.Event()
        self._initialized: bool = False
        self._running: bool = False

        # Per-track pending snapshot path map.
        # tid -> absolute path of the most recent snapshot for this track.
        # Lock-protected so the AI thread (writer) and the logger thread
        # (reader) can access it concurrently.
        self._pending_paths: Dict[int, str] = {}
        self._pending_paths_lock: threading.RLock = threading.RLock()

        # Patch 65 :: Per-stranger state for CLEARSHOT snapshots.
        # _stranger_labels[tid] = stranger_label (e.g. "[Stranger_03]")
        #   - populated by register_stranger_label() when main.py calls
        #     finalize_stranger(). Used to build clearshot filenames.
        # _clearshot_counters[tid] = next YY index (1-indexed)
        #   - incremented each time capture_clearshot() is called for
        #     this track. Reset on session rollover (via
        #     reset_clearshot_counters()).
        # Both dicts are protected by _clearshot_lock.
        self._stranger_labels: Dict[int, str] = {}
        self._clearshot_counters: Dict[int, int] = {}
        self._clearshot_lock: threading.RLock = threading.RLock()

        # Telemetry counters.
        self._births_captured: int = 0
        self._strangers_finalized: int = 0
        self._verified_deleted: int = 0
        self._anomaly_snapshots: int = 0
        self._ops_dropped_full_queue: int = 0
        self._worker_errors: int = 0
        self._consecutive_errors: int = 0
        self._max_observed_queue_depth: int = 0
        # Patch 65 :: Clearshot telemetry.
        self._clearshots_captured: int = 0
        self._clearshots_skipped_disabled: int = 0
        self._clearshots_skipped_max: int = 0

    # ==================================================================
    # Lifecycle.
    # ==================================================================
    def initialize(self) -> None:
        if self._initialized:
            logger.warning("SnapStrangersEngine already initialized; skipping.")
            return

        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "SnapStrangersEngine: failed to create output_dir %s: %s",
                self._output_dir, exc,
            )

        logger.info(
            "SnapStrangersEngine initializing (Patch 18 :: per-session subfolder, "
            "purge_enabled=%s) | out=%s | %dx%d | retention=%dh | purge=%ds | "
            "png_compression=%d | queue_max=%d",
            self._purge_enabled, self._output_dir, self._width, self._height,
            self._retention_hours, self._purge_interval_s,
            self._png_compression, self._queue_maxsize,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._initialized:
            self.initialize()
        if self._running:
            logger.warning("SnapStrangersEngine worker already running.")
            return

        # Patch 18 :: PurgeWorker is conditionally started. The operator
        # has explicitly requested that snap_strangers NOT purge any
        # files (so the full forensic history is browsable via the
        # dashboard). When `purge_enabled: false` (the new default), the
        # PurgeWorker thread is left dormant.
        if self._purge_enabled:
            self._purge_worker.start()
        else:
            logger.info(
                "SnapStrangersEngine: PurgeWorker DISABLED (purge_enabled=false) "
                "-- snapshots are retained indefinitely."
            )

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="sortendance.snap_strangers.worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._running = True
        logger.info("SnapStrangersEngine worker thread started.")

    # ------------------------------------------------------------------
    def shutdown(self, timeout_s: float = 8.0) -> None:
        if not self._running:
            logger.info("SnapStrangersEngine shutdown: worker not running.")
            self._purge_worker.stop(timeout_s=timeout_s)
            return

        logger.info(
            "SnapStrangersEngine shutdown initiated | queue_depth=%d",
            self._queue.qsize(),
        )

        # Push shutdown sentinel.
        try:
            self._queue.put(_SHUTDOWN, timeout=1.0)
        except queue.Full:
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                logger.warning(
                    "SnapStrangersEngine: queue full at shutdown; forcing "
                    "event-based wakeup.",
                )
                self._shutdown_event.set()

        # Join the worker thread.
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout_s)
        self._running = False

        # Stop the purge worker.
        self._purge_worker.stop(timeout_s=timeout_s)

        logger.info(
            "SnapStrangersEngine shutdown complete | births=%d | "
            "strangers_finalized=%d | verified_deleted=%d | "
            "anomaly_snapshots=%d | dropped_full=%d | worker_errors=%d",
            self._births_captured, self._strangers_finalized,
            self._verified_deleted, self._anomaly_snapshots,
            self._ops_dropped_full_queue, self._worker_errors,
        )

    # ==================================================================
    # Producer API -- new methods.
    # ==================================================================
    def capture_birth_snapshot(
        self,
        annotated_frame: Any,
        track_id: int,
        frame_index: int,
        capture_us: int,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Queue a WRITE_BIRTH op for a newly-born track.

        Called by the AI thread at the moment a new track_id is first
        observed. The annotated_frame is the FULL GUI frame (with bbox
        overlays already drawn by _annotate_frame). The worker will
        write it to disk as a PNG.

        Returns True if the op was admitted to the queue, False if
        dropped due to queue saturation.
        """
        if not self._running:
            logger.warning(
                "SnapStrangersEngine.capture_birth_snapshot called before "
                "start() -- op will be dropped.",
            )
            return False

        if annotated_frame is None or not _CV2_AVAILABLE:
            return False

        # Defensive deep copy -- the AI thread reuses the frame buffer
        # across frames, so we must snapshot the pixel data NOW.
        try:
            frame_copy = annotated_frame.copy()
        except Exception as exc:
            logger.warning(
                "capture_birth_snapshot: frame copy failed: %s", exc,
            )
            return False

        ticket = SnapshotTicket(
            op=_SnapshotOp.WRITE_BIRTH,
            track_id=int(track_id),
            frame=frame_copy,
            frame_index=int(frame_index),
            capture_us=int(capture_us),
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            enqueue_wall_us=int(time.time() * 1_000_000),
        )
        return self._enqueue(ticket)

    # ------------------------------------------------------------------
    def finalize_stranger(
        self,
        track_id: int,
        stranger_label: str,
    ) -> bool:
        """
        Queue a RENAME_STRANGER op. The worker will rename the pending
        birth-snapshot PNG to include the stranger label in the filename.

        Returns True if the op was admitted to the queue.
        """
        if not self._running:
            return False
        ticket = SnapshotTicket(
            op=_SnapshotOp.RENAME_STRANGER,
            track_id=int(track_id),
            stranger_label=str(stranger_label),
            enqueue_wall_us=int(time.time() * 1_000_000),
        )
        return self._enqueue(ticket)

    # ------------------------------------------------------------------
    def finalize_verified(self, track_id: int, student_label: str = "") -> bool:
        """
        Patch 63 :: Queue a MOVE_VERIFIED op (or DELETE_VERIFIED if
        retain_verified_snapshots is False).

        When retain_verified_snapshots=True (default), the worker MOVES
        the pending birth-snapshot PNG into the identified/ subfolder
        with the student label baked into the filename. This preserves
        the snapshot for the Event Log page's hourly history view.

        When retain_verified_snapshots=False, the worker DELETES the
        PNG (old behavior -- no forensic value for known students).

        Args:
            track_id: The track ID whose birth snapshot should be finalized.
            student_label: The resolved student label (e.g. "[221050 / 221050]"
                or "221050"). Used as the filename suffix when moving.

        Returns True if the op was admitted to the queue.
        """
        if not self._running:
            return False
        # Patch 63 :: Choose MOVE vs DELETE based on config.
        op = (
            _SnapshotOp.MOVE_VERIFIED
            if self._retain_verified_snapshots
            else _SnapshotOp.DELETE_VERIFIED
        )
        ticket = SnapshotTicket(
            op=op,
            track_id=int(track_id),
            stranger_label=str(student_label) if student_label else "",
            enqueue_wall_us=int(time.time() * 1_000_000),
        )
        return self._enqueue(ticket)

    # ------------------------------------------------------------------
    def get_snapshot_path(self, track_id: int) -> Optional[str]:
        """
        Return the on-disk path of the snapshot for the given track,
        or None if no snapshot exists / has been finalized.

        Thread-safe. Used by the async logger to fill the snapshot_path
        CSV column for STRANGER entries.
        """
        with self._pending_paths_lock:
            return self._pending_paths.get(int(track_id))

    # ==================================================================
    # Patch 65 :: CLEARSHOT API.
    #
    # After a track is locked as STRANGER, the AI thread calls
    # register_stranger_label(tid, label) so the snapshot engine knows
    # the stranger label to bake into clearshot filenames. Then, on
    # every frame where the clearshot gating conditions are met (YOLO
    # conf >= threshold, cooldown elapsed, max not reached), the AI
    # thread calls capture_clearshot(...) to queue a WRITE_CLEARSHOT op.
    #
    # The worker writes the PNG to:
    #   {output_dir}/{YYYY-MM-DD}/{6AM Session|6PM Session}/clearshots/
    #     {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY:02d}.png
    #
    # The YY counter is per-track and 1-indexed, so the first clearshot
    # for Stranger_03 is _CLEARSHOT_01.png, the second is _CLEARSHOT_02,
    # etc. This guarantees unique filenames even if multiple clearshots
    # are captured within the same millisecond (rare but possible under
    # high FPS).
    # ==================================================================
    def register_stranger_label(
        self,
        track_id: int,
        stranger_label: str,
    ) -> None:
        """Register the stranger label for a track (used for clearshots).

        Called by main.py after finalize_stranger(). The label is
        cached so subsequent capture_clearshot() calls can build the
        STRANGER_{label}_CLEARSHOT_{YY}.png filename without re-prompting
        the caller.

        Thread-safe. Also resets the per-track clearshot counter to 0
        (the first capture_clearshot() call will increment it to 1).

        Args:
            track_id: The track ID that was just locked as STRANGER.
            stranger_label: The label assigned by the gating engine
                (e.g. "[Stranger_03]"). Brackets will be stripped
                before being used in the filename.
        """
        with self._clearshot_lock:
            self._stranger_labels[int(track_id)] = str(stranger_label)
            # Reset the counter so the first clearshot is _CLEARSHOT_01.
            # If the track was previously registered (e.g. re-locked
            # after a brief unlock), the counter starts fresh.
            self._clearshot_counters[int(track_id)] = 0

    # ------------------------------------------------------------------
    def capture_clearshot(
        self,
        annotated_frame: Any,
        track_id: int,
        frame_index: int,
        capture_us: int,
        bbox: Tuple[int, int, int, int],
        stranger_label: Optional[str] = None,
    ) -> bool:
        """Queue a WRITE_CLEARSHOT op for a locked STRANGER track.

        The clearshot is a high-quality PNG captured periodically (every
        `clearshot.cooldown_s` seconds by default) when YOLO confidence
        on the person is high. It serves as an OSNet "memory recall"
        reference frame for the stranger.

        Pre-conditions (enforced by main.py BEFORE calling this method):
          * Track is locked as STRANGER (state contains "Stranger").
          * YOLO det_conf >= clearshot.min_yolo_conf.
          * Bbox width AND height >= clearshot.min_bbox_size.
          * Cooldown (clearshot.cooldown_s) has elapsed since the last
            clearshot for this track.
          * Per-track clearshot count < clearshot.max_per_track.

        This method does NOT re-check those conditions -- it trusts the
        caller. The only check it performs is the global enabled flag
        (clearshot.enabled) and the max-per-track cap (defensive; in
        case the caller's count is stale).

        Args:
            annotated_frame: Full GUI frame (with bbox overlays). A
                defensive deep copy is made before enqueueing.
            track_id: The STRANGER track ID.
            frame_index: AI-thread frame index at capture time.
            capture_us: Hardware-frame timestamp (microseconds).
            bbox: Person bbox (x1, y1, x2, y2) in source-frame pixels.
            stranger_label: Optional stranger label. If provided,
                overrides the cached label from register_stranger_label().
                If None and no label was registered, the clearshot is
                SKIPPED (we can't build the filename without a label).

        Returns True if the op was admitted to the queue, False if
        dropped (queue full, disabled, max-per-track reached, or no
        stranger label available).
        """
        if not self._running:
            return False

        # Global enable check.
        if not self._clearshot_enabled:
            self._clearshots_skipped_disabled += 1
            return False

        if annotated_frame is None or not _CV2_AVAILABLE:
            return False

        # Resolve the stranger label (caller override > cached).
        tid = int(track_id)
        with self._clearshot_lock:
            label = stranger_label or self._stranger_labels.get(tid, "")
            if not label:
                logger.debug(
                    "capture_clearshot: no stranger label for track %d -- "
                    "call register_stranger_label() first. Skipping.",
                    tid,
                )
                return False

            # Defensive max-per-track check (the caller should have
            # already enforced this, but we re-check here in case the
            # caller's count is stale or another thread raced ahead).
            current_count = self._clearshot_counters.get(tid, 0)
            if self._clearshot_max_per_track > 0 and current_count >= self._clearshot_max_per_track:
                self._clearshots_skipped_max += 1
                logger.debug(
                    "capture_clearshot: track %d already has %d clearshots "
                    "(max=%d). Skipping.",
                    tid, current_count, self._clearshot_max_per_track,
                )
                return False

            # Increment the per-track counter (1-indexed).
            next_idx = current_count + 1
            self._clearshot_counters[tid] = next_idx

        # Defensive deep copy -- the AI thread reuses the frame buffer.
        try:
            frame_copy = annotated_frame.copy()
        except Exception as exc:
            logger.warning(
                "capture_clearshot: frame copy failed: %s", exc,
            )
            # Roll back the counter increment since we're dropping the op.
            with self._clearshot_lock:
                self._clearshot_counters[tid] = current_count
            return False

        ticket = SnapshotTicket(
            op=_SnapshotOp.WRITE_CLEARSHOT,
            track_id=tid,
            frame=frame_copy,
            frame_index=int(frame_index),
            capture_us=int(capture_us),
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            stranger_label=str(label),
            clearshot_idx=next_idx,
            enqueue_wall_us=int(time.time() * 1_000_000),
        )
        return self._enqueue(ticket)

    # ------------------------------------------------------------------
    def reset_clearshot_state(self) -> None:
        """Reset all clearshot state (counters + labels).

        Called by main.py's SessionBoundaryWatcher at every 06:00 /
        18:00 LOCAL boundary, AFTER identity_matcher.reset_dynamic_memory()
        has cleared the OSNet stranger cache. New strangers in the new
        session will get fresh Stranger_XX IDs, so the cached labels and
        counters from the previous session are stale and must be cleared.

        Does NOT delete any clearshot PNGs from disk -- those are
        organized by date+session folder (via _session_subdir_for_ts)
        and are retained indefinitely (per the operator's request:
        "snap_strangers doesn't need to purge anything").
        """
        with self._clearshot_lock:
            n_labels = len(self._stranger_labels)
            n_counters = len(self._clearshot_counters)
            self._stranger_labels.clear()
            self._clearshot_counters.clear()
        if n_labels or n_counters:
            logger.info(
                "SnapStrangersEngine.reset_clearshot_state(): cleared "
                "%d stranger labels + %d clearshot counters (session "
                "rollover). Existing PNGs on disk are retained.",
                n_labels, n_counters,
            )

    # ------------------------------------------------------------------
    def get_clearshot_count(self, track_id: int) -> int:
        """Return the number of clearshots captured so far for this track.

        Thread-safe. Used by main.py to enforce the max-per-track cap
        without maintaining a duplicate counter in the AI thread.
        """
        with self._clearshot_lock:
            return int(self._clearshot_counters.get(int(track_id), 0))

    # ------------------------------------------------------------------
    # Patch 56 :: Prune orphaned pending paths for tracks that have
    # exited the frame WITHOUT being resolved (neither verified nor
    # stranger). Without this, _pending_paths grows unbounded over a
    # 12-hour session as every track birth creates an entry that's
    # only removed on rename_stranger or delete_verified.
    #
    # Patch 63 (hotfix K) :: The _pending_paths dict entry is ALWAYS
    # removed (to prevent unbounded growth), but the BIRTH.png file
    # on disk is only deleted if self._prune_unresolved_births is True.
    # When False (the new default), the file is KEPT so operators can
    # review everyone who appeared, even briefly. The file will appear
    # in the dashboard as "Pending_Track_NN".
    # ------------------------------------------------------------------
    def prune_dead_tracks(self, active_track_ids: set) -> None:
        """Remove pending paths for tracks no longer in the frame.

        Patch 63 (hotfix K) :: The _pending_paths dict entry is always
        removed, but the on-disk BIRTH.png is only deleted if
        self._prune_unresolved_births is True. When False (default),
        the file is kept for forensic review.
        """
        if not active_track_ids:
            return
        with self._pending_paths_lock:
            dead_keys = [
                tid for tid in self._pending_paths.keys()
                if tid not in active_track_ids
            ]
            _deleted_count = 0
            _kept_count = 0
            for tid in dead_keys:
                path = self._pending_paths.pop(tid, None)
                # P2-M10 fix: only delete *_BIRTH.png files. The previous
                # logic would also delete *_STRANGER_*.png files (set by
                # _handle_rename_stranger), destroying forensic stranger
                # snapshots when the operator enabled this flag to clean
                # up orphaned BIRTH captures.
                if (path and self._prune_unresolved_births
                        and path.endswith("_BIRTH.png")):
                    # Old behavior: delete the BIRTH.png from disk.
                    # Best-effort -- if the file is locked or already
                    # gone, just drop the dict entry.
                    try:
                        import os as _os
                        if _os.path.exists(path):
                            _os.remove(path)
                            _deleted_count += 1
                    except OSError:
                        pass
                elif path:
                    # Hotfix K: keep the file on disk for forensic
                    # review. Only the dict entry is removed.
                    _kept_count += 1
            if dead_keys:
                logger.debug(
                    "AsyncRecorderThread.prune_dead_tracks: processed %d "
                    "orphaned pending paths (deleted=%d, kept_on_disk=%d, "
                    "active=%d, remaining=%d)",
                    len(dead_keys), _deleted_count, _kept_count,
                    len(active_track_ids), len(self._pending_paths),
                )

    # ==================================================================
    # Producer API -- backward-compatible no-ops.
    # ==================================================================
    def push_frame(
        self,
        frame: Any,
        frame_index: int,
        capture_us: int,
        stranger_bboxes: Optional[Tuple[Tuple[int, int, int, int], ...]] = None,
    ) -> bool:
        """
        No-op. The snapshot engine does NOT do continuous recording.
        Accepted for API compatibility with the old VideoRecorderEngine
        so that main.py's existing AsyncRecorderThread can be left in
        place during the migration.
        """
        return True

    # ------------------------------------------------------------------
    def trigger_segment(
        self,
        reason: TriggerReason,
        label: Optional[str] = None,
    ) -> bool:
        """
        For ANOMALY triggers, capture a one-off anomaly snapshot (no
        track_id association -- the anomaly is a face-without-body
        event). For all other trigger reasons, this is a no-op.
        """
        if not self._running:
            return False
        # We don't have a frame here -- anomaly snapshots are captured
        # by the AI thread directly via capture_anomaly_snapshot().
        # This method is kept for API compat but logs the event.
        if reason == TriggerReason.ANOMALY:
            logger.info(
                "SnapStrangersEngine: ANOMALY trigger received (label=%s) -- "
                "anomaly snapshot should be captured via capture_anomaly_snapshot()",
                label,
            )
        return True

    # ------------------------------------------------------------------
    def capture_anomaly_snapshot(
        self,
        annotated_frame: Any,
        frame_index: int,
        capture_us: int,
        anomaly_label: str = "[ANOMALY]",
    ) -> bool:
        """
        Capture a one-off anomaly snapshot (face detected outside any
        person bbox). Anomalies are NOT associated with a track_id, so
        the filename uses a synthetic negative track_id (-1).
        """
        if not self._running or annotated_frame is None or not _CV2_AVAILABLE:
            return False
        try:
            frame_copy = annotated_frame.copy()
        except Exception as exc:
            logger.warning(
                "capture_anomaly_snapshot: frame copy failed: %s", exc,
            )
            return False
        ticket = SnapshotTicket(
            op=_SnapshotOp.ANOMALY_SNAPSHOT,
            track_id=-1,
            frame=frame_copy,
            frame_index=int(frame_index),
            capture_us=int(capture_us),
            anomaly_label=str(anomaly_label),
            enqueue_wall_us=int(time.time() * 1_000_000),
        )
        return self._enqueue(ticket)

    # ------------------------------------------------------------------
    def update_stranger_bboxes(
        self,
        track_id: int,
        bboxes: Tuple[Tuple[int, int, int, int], ...],
    ) -> None:
        """No-op for API compatibility."""
        pass

    # ------------------------------------------------------------------
    def clear_stranger_bboxes(self, track_id: int) -> None:
        """No-op for API compatibility."""
        pass

    # ------------------------------------------------------------------
    def end_current_segment(self) -> bool:
        """No-op for API compatibility."""
        return True

    # ==================================================================
    # Internal: queue + worker.
    # ==================================================================
    def _enqueue(self, ticket: SnapshotTicket) -> bool:
        try:
            self._queue.put_nowait(ticket)
            depth = self._queue.qsize()
            if depth > self._max_observed_queue_depth:
                self._max_observed_queue_depth = depth
            return True
        except queue.Full:
            self._ops_dropped_full_queue += 1
            # Drop oldest to make room for newest (latest-op-wins).
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(ticket)
                return True
            except (queue.Empty, queue.Full):
                return False

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        logger.info("SnapStrangersEngine worker loop entered.")
        try:
            while True:
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    if self._consecutive_errors > 0:
                        self._consecutive_errors = 0
                    continue

                if item is _SHUTDOWN:
                    self._drain_remaining()
                    logger.info(
                        "SnapStrangersEngine worker observed shutdown "
                        "sentinel; exiting.",
                    )
                    return

                self._process_item(item)

        except Exception as exc:
            self._worker_errors += 1
            logger.critical(
                "SnapStrangersEngine worker loop crashed: %s\n%s",
                exc, traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    def _drain_remaining(self) -> None:
        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                continue
            self._process_item(item)
            drained += 1
        if drained > 0:
            logger.info(
                "SnapStrangersEngine drained %d residual tickets on shutdown.",
                drained,
            )

    # ------------------------------------------------------------------
    def _process_item(self, ticket: SnapshotTicket) -> None:
        try:
            if ticket.op == _SnapshotOp.WRITE_BIRTH:
                self._handle_write_birth(ticket)
            elif ticket.op == _SnapshotOp.RENAME_STRANGER:
                self._handle_rename_stranger(ticket)
            # Patch 63 :: MOVE_VERIFIED dispatch.
            elif ticket.op == _SnapshotOp.MOVE_VERIFIED:
                self._handle_move_verified(ticket)
            elif ticket.op == _SnapshotOp.DELETE_VERIFIED:
                self._handle_delete_verified(ticket)
            elif ticket.op == _SnapshotOp.ANOMALY_SNAPSHOT:
                self._handle_anomaly_snapshot(ticket)
            # Patch 65 :: WRITE_CLEARSHOT dispatch.
            elif ticket.op == _SnapshotOp.WRITE_CLEARSHOT:
                self._handle_write_clearshot(ticket)
            else:
                logger.warning(
                    "SnapStrangersEngine: unknown op %s; dropping.",
                    ticket.op,
                )
        except Exception as exc:
            self._worker_errors += 1
            self._consecutive_errors += 1
            logger.error(
                "SnapStrangersEngine: op %s failed for track %d: %s\n%s",
                ticket.op, ticket.track_id, exc, traceback.format_exc(),
            )
            if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "SnapStrangersEngine: %d consecutive errors -- "
                    "subsequent ops may be unreliable.",
                    self._consecutive_errors,
                )

    # ------------------------------------------------------------------
    def _session_subdir_for_ts(self, ts_us: int) -> str:
        """
        Patch 18 :: Resolve (and create) the per-date+session subdirectory
        for the given microsecond timestamp.

        Returns the absolute path:
            {output_dir}/{YYYY-MM-DD}/{6AM Session | 6PM Session}/

        The session key is computed via `compute_session_key(ts_us)`
        (shared with async_logger.py), so the 06:00 / 18:00 boundary
        semantics are identical across modules. The date reflects the
        day the session STARTED on, not the wall-clock date at write
        time (so 00:00-05:59 writes are filed under the previous day's
        6PM Session folder).
        """
        date_str, session_label = compute_session_key(ts_us)
        session_dir_name = session_label_to_dir(session_label)
        subdir = os.path.join(self._output_dir, date_str, session_dir_name)
        try:
            os.makedirs(subdir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "SnapStrangersEngine: failed to create session subdir %s: %s",
                subdir, exc,
            )
        return subdir

    # ------------------------------------------------------------------
    # Patch 20 :: rotate_session()
    #
    # Called by main.py's SessionBoundaryWatcher at every 06:00 / 18:00
    # LOCAL boundary. The snapshot engine ALREADY routes each PNG to
    # the correct per-session subfolder via _session_subdir_for_ts()
    # (which uses the ticket's capture_us timestamp, not the worker
    # loop's wall clock). So this method is effectively a NO-OP -- it
    # does not need to move files, switch directories, or reset any
    # internal state.
    #
    # We log the boundary crossing for operational visibility and
    # touch nothing on disk. Existing files in the old session folder
    # are NEVER moved or deleted (per the operator's explicit request:
    # "snap_strangers doesn't need to purge anything on the folder").
    # ------------------------------------------------------------------
    def rotate_session(
        self,
        session_label: str,
        session_date: str,
    ) -> None:
        """No-op adapter for the session-boundary watcher.

        Per-session folder routing is already handled per-snapshot
        via _session_subdir_for_ts(capture_us). This method exists
        solely so main.py's SessionBoundaryWatcher can call a
        uniform `rotate_session(label, date)` interface across all
        three engines (async_logger / snap_strangers / identity_matcher).

        Args:
            session_label: The new session label (advisory; ignored).
            session_date: The new session date (advisory; ignored).
        """
        logger.info(
            "SnapStrangersEngine.rotate_session() :: boundary crossed "
            "-> new session %s_%s | per-snapshot routing is automatic; "
            "no files moved or deleted.",
            session_date, session_label,
        )
        # Intentionally empty. The next capture_birth_snapshot() call
        # will compute its own session subfolder from its capture_us
        # timestamp and write into the new session folder.

    # ------------------------------------------------------------------
    def _handle_write_birth(self, ticket: SnapshotTicket) -> None:
        if not _CV2_AVAILABLE or ticket.frame is None:
            return

        # Patch 18 :: Resolve the per-date+session subfolder for this
        # snapshot. We use the ticket's capture_us (the hardware-frame
        # timestamp) rather than the worker-loop wall clock so the
        # snapshot lands in the session the frame was actually captured
        # in -- important at the 06:00 / 18:00 boundary where the
        # worker may drain the queue a few hundred ms after capture.
        capture_us = ticket.capture_us if ticket.capture_us else int(time.time() * 1_000_000)
        session_subdir = self._session_subdir_for_ts(capture_us)

        # Filename: {timestamp_ms}_track{tid}_BIRTH.png
        # P2-M11 fix: prefer capture_us (hardware capture time) over worker
        # wall-clock. A clearshot captured at 17:59:59.900 could land in
        # 6AM Session/ (correct, routed by capture_us) but get a filename
        # ts_ms of 18:00:00.x (wrong, from worker wall-clock) -- making
        # the filename inconsistent with the session folder. Fall back to
        # wall-clock only if capture_us is missing.
        ts_ms = (int(ticket.capture_us // 1000) if getattr(ticket, 'capture_us', 0)
                 else int(time.time() * 1000.0))
        fname = f"{ts_ms}_track{ticket.track_id}_BIRTH.png"
        fpath = os.path.join(session_subdir, fname)

        # Optional: draw a thin yellow border around the bbox region
        # to make the stranger visually prominent in the snapshot.
        # We do NOT modify the annotated frame in-place (it's already a
        # copy). We add a small "[BIRTH]" caption below the existing
        # label so reviewers can identify the capture moment.
        try:
            x1, y1, x2, y2 = ticket.bbox
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                cv2.rectangle(
                    ticket.frame,
                    (max(0, x1 - 2), max(0, y1 - 2)),
                    (min(self._width, x2 + 2), min(self._height, y2 + 2)),
                    (0, 255, 255),  # Yellow border
                    1, cv2.LINE_AA,
                )
        except Exception:
            pass

        # Patch 49 :: Atomic write -- encode to a .tmp file, then
        # os.replace() atomically swaps it into place. Readers
        # (dashboard.py PIL.Image.open()) either see the complete
        # file or no file at all -- never a partial write. This
        # eliminates the cv2.imwrite() ↔ PIL race that caused
        # STATUS_ACCESS_VIOLATION (0xC0000005) dashboard crashes.
        # Patch 52 :: Temp file must end with .png so cv2.imwrite()
        # recognizes the extension and uses the PNG encoder. Patch 49
        # used ".tmp" which cv2.imwrite() cannot encode -- it returns
        # False immediately without writing. This caused 37 worker
        # errors and 0 successful snapshots in production.
        tmp_path = fpath + ".tmp.png"
        ok = cv2.imwrite(
            tmp_path, ticket.frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression],
        )
        if not ok:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: cv2.imwrite failed for track %d -> %s",
                ticket.track_id, fpath,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return
        # Atomic rename: .tmp -> final path. On Windows and POSIX,
        # os.replace() is atomic when source and destination are
        # on the same filesystem.
        try:
            os.replace(tmp_path, fpath)
        except OSError as replace_exc:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: atomic rename failed for track %d "
                "(tmp=%s -> final=%s): %s",
                ticket.track_id, tmp_path, fpath, replace_exc,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return

        # Record the path so the logger can query it.
        with self._pending_paths_lock:
            self._pending_paths[ticket.track_id] = fpath

        self._births_captured += 1
        self._consecutive_errors = 0
        logger.info(
            "Stranger birth snapshot captured | track=%d | frame=%d | "
            "bbox=(%d,%d,%d,%d) | path=%s",
            ticket.track_id, ticket.frame_index,
            ticket.bbox[0], ticket.bbox[1], ticket.bbox[2], ticket.bbox[3],
            fpath,
        )

    # ------------------------------------------------------------------
    def _handle_rename_stranger(self, ticket: SnapshotTicket) -> None:
        with self._pending_paths_lock:
            old_path = self._pending_paths.get(ticket.track_id)
        if old_path is None or not os.path.isfile(old_path):
            logger.warning(
                "SnapStrangersEngine: rename_stranger -- no pending snapshot "
                "for track %d (already finalized or never captured).",
                ticket.track_id,
            )
            return

        # Sanitize the stranger label for use in a filename.
        # Labels look like "[Stranger_03]" -- strip brackets and
        # replace any chars that are illegal on Windows.
        safe_label = ticket.stranger_label
        for ch in ('[', ']', '<', '>', ':', '"', '/', '\\', '|', '?', '*'):
            safe_label = safe_label.replace(ch, '_')
        safe_label = safe_label.strip().strip('_') or f"track{ticket.track_id}"

        # New filename: keep the original timestamp, swap _BIRTH -> _STRANGER_{label}
        old_fname = os.path.basename(old_path)
        # old_fname = "{ts_ms}_track{tid}_BIRTH.png"
        try:
            ts_part, track_part, _ = old_fname.split('_', 2)
            new_fname = f"{ts_part}_{track_part}_STRANGER_{safe_label}.png"
        except ValueError:
            # Fallback if the filename format is unexpected.
            new_fname = old_fname.replace("_BIRTH", f"_STRANGER_{safe_label}")

        # Patch 18 :: Keep the renamed file in the SAME session subfolder
        # as the original birth snapshot. The original file's parent
        # directory is the per-date+session folder; we do NOT recompute
        # the session from the rename timestamp (which would risk
        # crossing a session boundary mid-finalization).
        new_path = os.path.join(os.path.dirname(old_path), new_fname)

        # Patch 63 (hotfix I) :: Robust rename with retry + copy+delete
        # fallback. On Windows, os.rename() fails with WinError 32
        # ("The process cannot access the file because it is being used
        # by another process") when:
        #   - Windows Defender is scanning the newly-created PNG
        #   - Windows Search Indexer has the file open
        #   - The dashboard's StrangerGalleryScanner is reading the
        #     file via PIL.Image.open() at the exact moment of rename
        #   - The OS file system cache hasn't flushed
        #
        # The previous code did a single os.rename() and gave up on
        # failure, leaving the PNG stuck as _BIRTH.png. While the
        # dashboard CAN still display _BIRTH.png files (as "Pending"),
        # the stranger label is lost -- the operator can't tell which
        # stranger it is.
        #
        # Fix: retry the rename up to 5 times with 200ms sleeps. If
        # os.rename still fails, fall back to shutil.move() which
        # does a copy+delete (more robust on Windows because the
        # delete can happen after the file handle is released).
        import shutil as _shutil
        _rename_ok = False
        _last_rename_exc: Optional[OSError] = None
        for _attempt in range(5):
            try:
                os.rename(old_path, new_path)
                _rename_ok = True
                break
            except OSError as exc:
                _last_rename_exc = exc
                # WinError 32 = file in use. Retry after a short sleep.
                # Other errors (e.g. ENOENT = file doesn't exist) are
                # not retryable.
                _winerror = getattr(exc, "winerror", None)
                _errno = getattr(exc, "errno", None)
                if _winerror == 32 or _errno == 13:  # 32=in use, 13=perm
                    time.sleep(0.2 * (_attempt + 1))
                    continue
                else:
                    break  # Non-retryable error.

        if not _rename_ok and _last_rename_exc is not None:
            # Fallback: copy + delete. shutil.move() handles this
            # internally, but on Windows it may still hit the same
            # lock. Use shutil.copy2() + os.remove() with retry on
            # the delete.
            try:
                _shutil.copy2(old_path, new_path)
                # Delete the original with retry (may be locked).
                for _del_attempt in range(5):
                    try:
                        os.remove(old_path)
                        _rename_ok = True
                        break
                    except OSError as del_exc:
                        _winerror = getattr(del_exc, "winerror", None)
                        if _winerror == 32:
                            time.sleep(0.2 * (_del_attempt + 1))
                            continue
                        # Can't delete -- the copy succeeded, so the
                        # new file is in place. The old _BIRTH.png will
                        # remain on disk (harmless duplicate). Log it.
                        logger.warning(
                            "SnapStrangersEngine: rename copy succeeded "
                            "but old file delete failed for track %d "
                            "(old=%s, new=%s): %s -- duplicate _BIRTH.png "
                            "will remain on disk",
                            ticket.track_id, old_path, new_path, del_exc,
                        )
                        _rename_ok = True  # New file exists; that's enough.
                        break
            except OSError as copy_exc:
                logger.warning(
                    "SnapStrangersEngine: rename failed for track %d after "
                    "all retries and copy fallback (%s -> %s): %s",
                    ticket.track_id, old_path, new_path, copy_exc,
                )
                return

        if not _rename_ok:
            logger.warning(
                "SnapStrangersEngine: rename failed for track %d (%s -> %s): %s",
                ticket.track_id, old_path, new_path, _last_rename_exc,
            )
            return

        with self._pending_paths_lock:
            self._pending_paths[ticket.track_id] = new_path

        self._strangers_finalized += 1
        self._consecutive_errors = 0
        logger.info(
            "Stranger snapshot finalized | track=%d | label=%s | path=%s",
            ticket.track_id, ticket.stranger_label, new_path,
        )

    # ------------------------------------------------------------------
    def _handle_move_verified(self, ticket: SnapshotTicket) -> None:
        """Patch 63 :: Move the birth PNG into the identified/ subfolder.

        The new filename encodes the student label so the dashboard's
        StrangerGalleryScanner can parse it via the VERIFIED regex.
        """
        with self._pending_paths_lock:
            old_path = self._pending_paths.pop(ticket.track_id, None)
        if old_path is None:
            return
        if not os.path.isfile(old_path):
            return

        # Sanitize the student label for use as a filename component.
        raw_label = ticket.stranger_label or "VERIFIED"
        safe_label = re.sub(r'[^A-Za-z0-9_\-]', "_", raw_label).strip("_")
        if not safe_label:
            safe_label = "VERIFIED"

        # Build the identified/ subfolder path (same session dir as the
        # original birth PNG).
        old_dir = os.path.dirname(old_path)
        identified_dir = os.path.join(old_dir, self._identified_subdir)
        try:
            os.makedirs(identified_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "SnapStrangersEngine: could not create identified/ "
                "subdir %s: %s", identified_dir, exc,
            )
            # Fall back to delete to avoid leaving the birth PNG around.
            try:
                os.remove(old_path)
            except OSError:
                pass
            return

        # Build the new filename: {ts_ms}_track{tid}_VERIFIED_{label}.png
        # Reuse the timestamp from the original filename so the snapshot
        # retains its original capture time.
        old_basename = os.path.basename(old_path)
        ts_ms = old_basename.split("_")[0] if "_" in old_basename else str(
            int(time.time() * 1000)
        )
        new_filename = f"{ts_ms}_track{ticket.track_id}_VERIFIED_{safe_label}.png"
        new_path = os.path.join(identified_dir, new_filename)

        # Patch 63 (hotfix I) :: Robust move with retry (same fix as
        # _handle_rename_stranger). On Windows, os.replace() can fail
        # with WinError 32 if Defender/Search Indexer/dashboard scanner
        # has the file open. Retry 5 times, then fall back to copy+delete.
        import shutil as _shutil
        _move_ok = False
        _last_move_exc: Optional[OSError] = None
        for _attempt in range(5):
            try:
                os.replace(old_path, new_path)
                _move_ok = True
                break
            except OSError as exc:
                _last_move_exc = exc
                _winerror = getattr(exc, "winerror", None)
                _errno = getattr(exc, "errno", None)
                if _winerror == 32 or _errno == 13:
                    time.sleep(0.2 * (_attempt + 1))
                    continue
                else:
                    break

        if not _move_ok and _last_move_exc is not None:
            try:
                _shutil.copy2(old_path, new_path)
                for _del_attempt in range(5):
                    try:
                        os.remove(old_path)
                        _move_ok = True
                        break
                    except OSError as del_exc:
                        _winerror = getattr(del_exc, "winerror", None)
                        if _winerror == 32:
                            time.sleep(0.2 * (_del_attempt + 1))
                            continue
                        logger.warning(
                            "SnapStrangersEngine: move copy succeeded but "
                            "old file delete failed for track %d: %s",
                            ticket.track_id, del_exc,
                        )
                        _move_ok = True
                        break
            except OSError as copy_exc:
                logger.warning(
                    "SnapStrangersEngine: move failed for track %d after "
                    "all retries (%s -> %s): %s",
                    ticket.track_id, old_path, new_path, copy_exc,
                )
                try:
                    os.remove(old_path)
                except OSError:
                    pass
                return

        if not _move_ok:
            logger.warning(
                "SnapStrangersEngine: move failed for track %d (%s -> %s): %s",
                ticket.track_id, old_path, new_path, _last_move_exc,
            )
            try:
                os.remove(old_path)
            except OSError:
                pass
            return

        self._verified_deleted += 1  # reuse counter (semantically "finalized")
        self._consecutive_errors = 0
        logger.info(
            "Verified-student snapshot retained | track=%d | label=%s | path=%s",
            ticket.track_id, raw_label, new_path,
        )

    # ------------------------------------------------------------------
    def _handle_delete_verified(self, ticket: SnapshotTicket) -> None:
        with self._pending_paths_lock:
            old_path = self._pending_paths.pop(ticket.track_id, None)
        if old_path is None:
            # Already finalized or never captured -- nothing to delete.
            return
        if not os.path.isfile(old_path):
            return
        try:
            os.remove(old_path)
        except OSError as exc:
            logger.warning(
                "SnapStrangersEngine: delete failed for track %d (%s): %s",
                ticket.track_id, old_path, exc,
            )
            return

        self._verified_deleted += 1
        self._consecutive_errors = 0
        logger.info(
            "Verified-student snapshot discarded | track=%d | (file deleted)",
            ticket.track_id,
        )

    # ------------------------------------------------------------------
    def _handle_anomaly_snapshot(self, ticket: SnapshotTicket) -> None:
        if not _CV2_AVAILABLE or ticket.frame is None:
            return

        # Patch 18 :: File the anomaly snapshot into the per-date+session
        # subfolder that matches the capture timestamp.
        capture_us = ticket.capture_us if ticket.capture_us else int(time.time() * 1_000_000)
        session_subdir = self._session_subdir_for_ts(capture_us)

        # P2-M11 fix: prefer capture_us (hardware capture time) over worker
        # wall-clock. A clearshot captured at 17:59:59.900 could land in
        # 6AM Session/ (correct, routed by capture_us) but get a filename
        # ts_ms of 18:00:00.x (wrong, from worker wall-clock) -- making
        # the filename inconsistent with the session folder. Fall back to
        # wall-clock only if capture_us is missing.
        ts_ms = (int(ticket.capture_us // 1000) if getattr(ticket, 'capture_us', 0)
                 else int(time.time() * 1000.0))
        safe_label = ticket.anomaly_label
        for ch in ('[', ']', '<', '>', ':', '"', '/', '\\', '|', '?', '*'):
            safe_label = safe_label.replace(ch, '_')
        safe_label = safe_label.strip().strip('_') or "ANOMALY"
        fname = f"{ts_ms}_{safe_label}.png"
        fpath = os.path.join(session_subdir, fname)

        # Patch 49 :: Atomic write for anomaly snapshots too.
        # Patch 52 :: Same fix as _handle_write_birth -- temp file
        # must end with .png for cv2.imwrite() to recognize it.
        tmp_path = fpath + ".tmp.png"
        ok = cv2.imwrite(
            tmp_path, ticket.frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression],
        )
        if not ok:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: anomaly imwrite failed -> %s", fpath,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return
        try:
            os.replace(tmp_path, fpath)
        except OSError as replace_exc:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: anomaly atomic rename failed "
                "(tmp=%s -> final=%s): %s",
                tmp_path, fpath, replace_exc,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return

        self._anomaly_snapshots += 1
        self._consecutive_errors = 0
        logger.info(
            "Anomaly snapshot captured | frame=%d | path=%s",
            ticket.frame_index, fpath,
        )

    # ------------------------------------------------------------------
    # Patch 65 :: _handle_write_clearshot()
    #
    # Writes a CLEARSHOT PNG for a STRANGER-locked track. Filename:
    #   {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY:02d}.png
    #
    # Filed under the per-date+session subfolder's `clearshots/`
    # subdirectory (configurable via clearshot.subdir). The clearshot
    # is NOT registered in _pending_paths -- it's a standalone capture
    # that doesn't participate in the birth -> stranger/verified
    # rename lifecycle. Each clearshot gets its own unique filename
    # (the YY counter guarantees uniqueness per track).
    # ------------------------------------------------------------------
    def _handle_write_clearshot(self, ticket: SnapshotTicket) -> None:
        if not _CV2_AVAILABLE or ticket.frame is None:
            return

        # Resolve the per-date+session subfolder (same logic as birth).
        capture_us = ticket.capture_us if ticket.capture_us else int(time.time() * 1_000_000)
        session_subdir = self._session_subdir_for_ts(capture_us)

        # Optional clearshots/ subfolder. If clearshot.subdir is empty,
        # clearshots land alongside birth snapshots (mixed folder).
        if self._clearshot_subdir:
            clearshot_dir = os.path.join(session_subdir, self._clearshot_subdir)
            try:
                os.makedirs(clearshot_dir, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "SnapStrangersEngine: could not create clearshot "
                    "subdir %s: %s -- falling back to session root.",
                    clearshot_dir, exc,
                )
                clearshot_dir = session_subdir
        else:
            clearshot_dir = session_subdir

        # Sanitize the stranger label (strip brackets etc.) so it's
        # safe as a filename component.
        safe_label = ticket.stranger_label
        for ch in ('[', ']', '<', '>', ':', '"', '/', '\\', '|', '?', '*'):
            safe_label = safe_label.replace(ch, '_')
        safe_label = safe_label.strip().strip('_') or f"track{ticket.track_id}"

        # Build the filename:
        #   {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY:02d}.png
        # P2-M11 fix: prefer capture_us (hardware capture time) over worker
        # wall-clock. A clearshot captured at 17:59:59.900 could land in
        # 6AM Session/ (correct, routed by capture_us) but get a filename
        # ts_ms of 18:00:00.x (wrong, from worker wall-clock) -- making
        # the filename inconsistent with the session folder. Fall back to
        # wall-clock only if capture_us is missing.
        ts_ms = (int(ticket.capture_us // 1000) if getattr(ticket, 'capture_us', 0)
                 else int(time.time() * 1000.0))
        yy = max(1, int(ticket.clearshot_idx))
        fname = (
            f"{ts_ms}_track{ticket.track_id}_STRANGER_{safe_label}"
            f"_CLEARSHOT_{yy:02d}.png"
        )
        fpath = os.path.join(clearshot_dir, fname)

        # Draw a thin cyan border around the bbox to visually
        # distinguish clearshots from birth snapshots (yellow) and
        # verified (green). Also draw the clearshot index label so
        # the operator can see which capture this is at a glance.
        try:
            x1, y1, x2, y2 = ticket.bbox
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                cv2.rectangle(
                    ticket.frame,
                    (max(0, x1 - 2), max(0, y1 - 2)),
                    (min(self._width, x2 + 2), min(self._height, y2 + 2)),
                    (255, 255, 0),  # Cyan (BGR) -- distinct from yellow birth border.
                    1, cv2.LINE_AA,
                )
                # Caption below the bbox.
                caption = f"CLEARSHOT #{yy:02d}"
                try:
                    cv2.putText(
                        ticket.frame, caption,
                        (max(0, x1), min(self._height - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1, cv2.LINE_AA,
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Atomic write (same pattern as _handle_write_birth).
        tmp_path = fpath + ".tmp.png"
        ok = cv2.imwrite(
            tmp_path, ticket.frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression],
        )
        if not ok:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: clearshot imwrite failed for "
                "track %d -> %s",
                ticket.track_id, fpath,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return
        try:
            os.replace(tmp_path, fpath)
        except OSError as replace_exc:
            self._worker_errors += 1
            logger.warning(
                "SnapStrangersEngine: clearshot atomic rename failed "
                "(tmp=%s -> final=%s): %s",
                tmp_path, fpath, replace_exc,
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return

        # Patch 67 :: Write a sidecar .json with bbox coords next to
        # each clearshot PNG. This lets stranger_recall.py crop the
        # person region deterministically during OSNet memory recall
        # (at every 12-hour restart), without needing to re-run YOLO
        # on the clearshot. The .json is tiny (~100 bytes) and atomic.
        try:
            import json as _json
            sidecar = {
                "bbox": [int(ticket.bbox[0]), int(ticket.bbox[1]),
                         int(ticket.bbox[2]), int(ticket.bbox[3])],
                "track_id": int(ticket.track_id),
                "stranger_label": ticket.stranger_label,
                "clearshot_idx": int(ticket.clearshot_idx),
                "ts_ms": int(ts_ms),
            }
            # P2-M9 fix: was fpath.replace(".png", ".json") which corrupts the path
            # if the stranger label contains ".png". Use splitext instead.
            sidecar_path = os.path.splitext(fpath)[0] + ".json"
            tmp_json = sidecar_path + ".tmp"
            with open(tmp_json, "w", encoding="utf-8") as jf:
                _json.dump(sidecar, jf, separators=(",", ":"))
            os.replace(tmp_json, sidecar_path)
        except Exception as json_exc:
            # Sidecar failure is non-fatal -- the clearshot PNG is
            # already saved. stranger_recall.py will fall back to
            # re-running YOLO if the .json is missing.
            logger.debug(
                "SnapStrangersEngine: clearshot sidecar .json write "
                "failed (non-fatal) for %s: %s",
                fpath, json_exc,
            )

        self._clearshots_captured += 1
        self._consecutive_errors = 0
        logger.info(
            "Stranger clearshot captured | track=%d | label=%s | "
            "idx=%02d | frame=%d | path=%s",
            ticket.track_id, ticket.stranger_label, yy,
            ticket.frame_index, fpath,
        )

    # ==================================================================
    # Telemetry.
    # ==================================================================
    def telemetry(self) -> Dict[str, Any]:
        return {
            "births_captured": self._births_captured,
            "strangers_finalized": self._strangers_finalized,
            "verified_deleted": self._verified_deleted,
            "anomaly_snapshots": self._anomaly_snapshots,
            "ops_dropped_full_queue": self._ops_dropped_full_queue,
            "worker_errors": self._worker_errors,
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self._queue.maxsize,
            "max_observed_queue_depth": self._max_observed_queue_depth,
            "purge_enabled": self._purge_enabled,
            "purge": self._purge_worker.telemetry(),
            # Patch 65 :: Clearshot telemetry.
            "clearshots_captured": self._clearshots_captured,
            "clearshots_skipped_disabled": self._clearshots_skipped_disabled,
            "clearshots_skipped_max": self._clearshots_skipped_max,
            "clearshot_enabled": self._clearshot_enabled,
            "clearshot_min_yolo_conf": self._clearshot_min_yolo_conf,
            "clearshot_cooldown_s": self._clearshot_cooldown_s,
            "clearshot_max_per_track": self._clearshot_max_per_track,
            "clearshot_tracks_registered": len(self._stranger_labels),
        }


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness (no GPU / no camera)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance snap_strangers self-test ===")

    cfg = {
        "snap_strangers": {
            "output_dir": "storage/snap_strangers_selftest",
            "retention_hours": 1,
            "purge_check_interval_s": 3600,
            "png_compression": 6,
            "queue_maxsize": 16,
        },
        "camera": {"width": 640, "height": 480},
    }
    engine = SnapStrangersEngine(config=cfg)
    engine.initialize()
    engine.start()

    # Synthetic frame.
    if _CV2_AVAILABLE:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 400), (0, 255, 0), 2)
    else:
        frame = None

    # Capture a birth snapshot.
    engine.capture_birth_snapshot(
        annotated_frame=frame,
        track_id=42,
        frame_index=1,
        capture_us=int(time.time() * 1_000_000),
        bbox=(100, 100, 300, 400),
    )
    # Wait for the worker to flush.
    time.sleep(0.5)
    p = engine.get_snapshot_path(42)
    logger.info("After birth: pending_path=%s", p)

    # Finalize as stranger.
    engine.finalize_stranger(42, "[Stranger_07]")
    time.sleep(0.5)
    p = engine.get_snapshot_path(42)
    logger.info("After stranger finalize: path=%s", p)

    # Capture another and finalize as verified (should delete).
    engine.capture_birth_snapshot(
        annotated_frame=frame,
        track_id=99,
        frame_index=2,
        capture_us=int(time.time() * 1_000_000),
        bbox=(100, 100, 300, 400),
    )
    time.sleep(0.5)
    engine.finalize_verified(99)
    time.sleep(0.5)
    p = engine.get_snapshot_path(99)
    logger.info("After verified finalize: path=%s (should be None)", p)

    # Anomaly snapshot.
    engine.capture_anomaly_snapshot(
        annotated_frame=frame,
        frame_index=3,
        capture_us=int(time.time() * 1_000_000),
        anomaly_label="[ANOMALY]",
    )
    time.sleep(0.5)

    # Patch 65 :: Clearshot test.
    # Register the stranger label for track 42 (already finalized above).
    engine.register_stranger_label(42, "[Stranger_07]")
    # Capture 3 clearshots -- they should be numbered 01, 02, 03.
    for i in range(3):
        ok = engine.capture_clearshot(
            annotated_frame=frame,
            track_id=42,
            frame_index=10 + i,
            capture_us=int(time.time() * 1_000_000),
            bbox=(100, 100, 300, 400),
        )
        logger.info("Clearshot %d capture admitted=%s", i + 1, ok)
        time.sleep(0.05)
    # Verify the per-track counter.
    cs_count = engine.get_clearshot_count(42)
    logger.info("After 3 clearshots: get_clearshot_count(42)=%d", cs_count)
    assert cs_count == 3, f"Expected 3 clearshots, got {cs_count}"

    # Test: capture_clearshot with no label registered should fail.
    ok = engine.capture_clearshot(
        annotated_frame=frame,
        track_id=999,  # never registered
        frame_index=99,
        capture_us=int(time.time() * 1_000_000),
        bbox=(100, 100, 300, 400),
    )
    logger.info("Clearshot for unregistered track 999 admitted=%s (should be False)", ok)
    assert ok is False, "Expected capture_clearshot to fail for unregistered track"

    # Test: reset_clearshot_state should clear counters + labels.
    engine.reset_clearshot_state()
    cs_count_after_reset = engine.get_clearshot_count(42)
    logger.info("After reset: get_clearshot_count(42)=%d (should be 0)", cs_count_after_reset)
    assert cs_count_after_reset == 0, f"Expected 0 after reset, got {cs_count_after_reset}"

    time.sleep(0.5)
    tele = engine.telemetry()
    logger.info("Telemetry: %s", tele)
    assert tele["clearshots_captured"] == 3, (
        f"Expected 3 clearshots_captured in telemetry, got {tele['clearshots_captured']}"
    )

    engine.shutdown()
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()
