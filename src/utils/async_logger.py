"""
Production-grade Asynchronous Logging Layer (Patch 18 :: 12h Session Rotation).

Responsibilities:
  1. Non-Blocking Background I/O Loop -- owns a dedicated daemon thread
     that drains an internal `queue.Queue` of log entries off the main
     inference / capture / rendering loops. Producers pay only the cost
     of a `queue.put_nowait(...)` call; all disk I/O, CSV serialization,
     session-boundary rotation, and anti-spam evaluation are deferred
     to the background worker.

  2. 12-Hour Session CSV Writer (Patch 18) -- generates one CSV file
     per 12-hour LOCAL session under `storage/logs/` named:
         `{prefix}_{YYYY-MM-DD}_{06AM|06PM}.csv`
     where the date reflects the day the SESSION STARTED on, not the
     wall-clock date at write time. The session boundary is fixed at
     local 06:00 and 18:00 (the operator's "6AM Session" and "6PM
     Session"). Concretely:
         * 06:00..17:59 local  -> belongs to today's 06AM session.
         * 18:00..23:59 local  -> belongs to today's 06PM session.
         * 00:00..05:59 local  -> belongs to YESTERDAY's 06PM session
                                  (the overnight leg of the 18:00-06:00
                                  window -- the file is NOT split at
                                  midnight).
     File handles are kept open for the active session and atomically
     rotated at the 06:00 / 18:00 boundary. A list of session-rollover
     observer callbacks is invoked on every rotation so external systems
     (OSNet feature gallery, stranger OSNet cache, anti-spam filter)
     can perform their own per-session resets in lockstep with the CSV.
     Partial writes are flushed on a configurable cadence
     (`flush_interval_ms`) to bound data loss on crash.

  3. Anti-Spam Verification Filter -- maintains a thread-safe in-memory
     LRU cache that maps deduplication keys (student NRP for
     VERIFIED_STUDENT entries, stranger label for STRANGER entries) to
     the timestamp of the most recent successful registration. Each
     successfully identified student entity registers EXACTLY ONCE per
     active session; all subsequent duplicate validation requests are
     silently dropped at the queue-ingestion boundary to prevent log
     file bloating. ANOMALY events bypass the filter (each is a unique
     security incident).

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import csv
import gc
import time
import queue
import threading
import logging
import traceback
import datetime as _dt
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
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


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.async_logger")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Sentinel for queue shutdown
# ============================================================================
class _ShutdownSentinel:
    """
    Sentinel object pushed onto the queue to signal the worker thread to
    drain remaining entries and exit cleanly. A bare `None` is not safe
    because `None` is a legitimate (if unusual) log payload element.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<AsyncLoggerShutdownSentinel>"


_SHUTDOWN = _ShutdownSentinel()


# ============================================================================
# 12-Hour Session Helpers (Patch 18)
# ============================================================================
# The 12-hour session boundary is fixed at LOCAL 06:00 and 18:00. The
# operator refers to these as the "6AM Session" and "6PM Session".
# These constants are used by:
#   * async_logger.SessionCSVWriter  -> CSV filename + rotation
#   * snap_strangers.SnapStrangersEngine -> per-session PNG subfolder
#   * dashboard.StrangerGalleryScanner -> session dropdown population
#   * main.py (Patch 19) -> OSNet feature-gallery reset hook

SESSION_AM_START_HOUR: int = 6    # 06:00 local
SESSION_PM_START_HOUR: int = 18   # 18:00 local

SESSION_LABEL_AM: str = "06AM"    # filename token for the 06:00-18:00 session
SESSION_LABEL_PM: str = "06PM"    # filename token for the 18:00-06:00 session

# Human-readable folder names used by snap_strangers and the dashboard
# (kept in sync with the filename tokens above).
SESSION_DIR_AM: str = "6AM Session"
SESSION_DIR_PM: str = "6PM Session"


def compute_session_key(ts_us: int) -> Tuple[str, str]:
    """
    Compute the (date_str, session_label) tuple for the given microsecond
    timestamp, using the system LOCAL timezone.

    Returns:
        (date_str, session_label) where:
            date_str       = "YYYY-MM-DD" (the day the session STARTED on)
            session_label  = "06AM" or "06PM"

    The 12h sessions are anchored at local 06:00 and 18:00:
        * 06:00..17:59 local  -> (today,         "06AM")
        * 18:00..23:59 local  -> (today,         "06PM")
        * 00:00..05:59 local  -> (yesterday,     "06PM")  # overnight leg
    """
    local_dt = _dt.datetime.fromtimestamp(ts_us / 1_000_000.0).astimezone()
    hour = local_dt.hour
    if hour < SESSION_AM_START_HOUR:
        # Overnight leg of the previous day's 6PM session.
        session_start_date = (local_dt - _dt.timedelta(days=1)).date()
        session_label = SESSION_LABEL_PM
    elif hour < SESSION_PM_START_HOUR:
        # Today's 6AM session.
        session_start_date = local_dt.date()
        session_label = SESSION_LABEL_AM
    else:
        # Today's 6PM session.
        session_start_date = local_dt.date()
        session_label = SESSION_LABEL_PM
    return (session_start_date.strftime("%Y-%m-%d"), session_label)


def session_label_to_dir(session_label: str) -> str:
    """Translate a filename token ("06AM" / "06PM") to the human-readable
    folder name used by snap_strangers / dashboard ("6AM Session" / "6PM Session")."""
    if session_label == SESSION_LABEL_AM:
        return SESSION_DIR_AM
    if session_label == SESSION_LABEL_PM:
        return SESSION_DIR_PM
    # Defensive fallback -- never raise; the caller will use whatever we return.
    return session_label


def current_active_session_label() -> str:
    """Return the session label that is currently active (LOCAL time)."""
    return compute_session_key(int(time.time() * 1_000_000))[1]


def session_has_started(date_str: str, session_label: str) -> bool:
    """
    Return True if the given session has already STARTED (i.e. the wall
    clock has crossed the session's start boundary). Used by the dashboard
    to hide future sessions from the dropdown.

    A session "starts" at:
        06AM session -> 06:00 local on date_str
        06PM session -> 18:00 local on date_str
    """
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
# Data Classes
# ============================================================================
@dataclass(frozen=True)
class LogEntry:
    """
    Immutable log record. Constructed by the producer thread and drained
    by the background worker.

    Fields mirror the CSV column schema, but with explicit typing so
    serialization cannot silently coerce (e.g.) a None student_name
    into the literal string "None".
    """
    timestamp_us: int                       # Hardware-frame-deterministic microseconds
    frame_index: int                        # Hardware Capture Index (monotonic)
    track_id: int                           # Internal track ID (NOT rendered on feed)
    resolved_label: str                     # [NRP / Name] | [Stranger_XX] | [ANOMALY]
    state: str                              # VERIFIED_STUDENT | STRANGER | ANOMALY
    similarity_score: float                 # Cosine similarity (0.0 - 1.0)
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2) in frame pixel space
    nrp: Optional[str] = None               # Student NRP (None for STRANGER/ANOMALY)
    student_name: Optional[str] = None      # Student Name (None for STRANGER/ANOMALY)
    enqueue_wall_us: int = 0                # Set by log_entry() for latency telemetry
    
    # Per-stage latency telemetry (populated by the inference thread).
    # Empty/0 means that stage did not run for this entry (e.g. stranger
    # entries skip TTFM, frozen tracks skip face_det/arcface/usearch).
    yolo_latency_ms: float = 0.0
    face_det_latency_ms: float = 0.0
    arcface_latency_ms: float = 0.0
    usearch_latency_ms: float = 0.0
    ttfm_ms: float = 0.0       # Time-To-First-Match (verified students only)
    # Patch 42 :: OSNet body Re-ID inference latency (per-frame batch).
    # Empty for ANOMALY entries (anomalies bypass the body-Re-ID path).
    body_reid_latency_ms: float = 0.0

    # Snapshot migration :: path to the stranger birth-snapshot PNG
    # (empty for VERIFIED_STUDENT entries -- their snapshots are deleted;
    # empty for ANOMALY entries -- anomaly snapshots are not track-linked).
    snapshot_path: Optional[str] = None

    # ------------------------------------------------------------------
    def dedup_key(self) -> Optional[str]:
        """
        Compute the anti-spam deduplication key for this entry.

        Returns:
            * student NRP for VERIFIED_STUDENT entries.
            * resolved_label for STRANGER entries (e.g. "[Stranger_03]").
            * None for ANOMALY entries (never deduplicated -- each is a
              unique security incident).
        """
        if self.state == "VERIFIED_STUDENT":
            # Prefer the explicit NRP; fall back to the resolved_label
            # so an entry with a missing NRP still dedups correctly.
            return self.nrp if self.nrp else self.resolved_label
        if self.state == "STRANGER":
            return self.resolved_label
        return None

    # ------------------------------------------------------------------
    def to_csv_row(self, columns: List[str]) -> List[str]:
        """
        Project this entry onto the configured CSV column schema.

        Missing optional fields are rendered as empty strings (CSV-safe),
        not the Python `None` literal.
        """
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self.bbox
        field_map: Dict[str, str] = {
            "timestamp_us":     str(int(self.timestamp_us)),
            "frame_index":      str(int(self.frame_index)),
            "track_id":         str(int(self.track_id)),
            "nrp":              self.nrp if self.nrp is not None else "",
            "student_name":     self.student_name if self.student_name is not None else "",
            "resolved_label":   self.resolved_label,
            "state":            self.state,
            "similarity_score": f"{float(self.similarity_score):.4f}",
            "bbox_x1":          str(int(bbox_x1)),
            "bbox_y1":          str(int(bbox_y1)),
            "bbox_x2":          str(int(bbox_x2)),
            "bbox_y2":          str(int(bbox_y2)),
            # New latency columns
            "yolo_latency_ms":      f"{float(self.yolo_latency_ms):.2f}" if self.yolo_latency_ms else "",
            "face_det_latency_ms":  f"{float(self.face_det_latency_ms):.2f}" if self.face_det_latency_ms else "",
            "arcface_latency_ms":   f"{float(self.arcface_latency_ms):.2f}" if self.arcface_latency_ms else "",
            "usearch_latency_ms":   f"{float(self.usearch_latency_ms):.2f}" if self.usearch_latency_ms else "",
            "ttfm_ms":              f"{float(self.ttfm_ms):.2f}" if self.ttfm_ms else "",
            "body_reid_latency_ms": f"{float(self.body_reid_latency_ms):.2f}" if self.body_reid_latency_ms else "",
            # Snapshot migration column.
            "snapshot_path":        self.snapshot_path if self.snapshot_path is not None else "",
        }
        return [field_map.get(col, "") for col in columns]


# ============================================================================
# Anti-Spam Verification Filter
# ============================================================================
class AntiSpamVerificationFilter:
    """
    Thread-safe LRU cache that enforces the "exactly once per session"
    registration guarantee for VERIFIED_STUDENT and STRANGER entries.

    The cache maps dedup_key -> registration_wall_us. Subsequent
    registration requests for an existing key within the per-session TTL
    are rejected with `False`; the caller (AsyncLoggingEngine) then
    drops the entry at the queue-ingestion boundary.

    ANOMALY events bypass the filter entirely (each is a unique security
    incident requiring independent audit trail entry).
    """

    def __init__(
        self,
        max_size: int = 256,
        session_ttl_s: int = 3600,
        dedup_states: Optional[List[str]] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._cache: "OrderedDict[str, int]" = OrderedDict()
        self._max_size: int = int(max_size)
        self._session_ttl_s: int = int(session_ttl_s)
        self._dedup_states: set = set(dedup_states or ["VERIFIED_STUDENT", "STRANGER"])
        # Telemetry counters.
        self._accepted: int = 0
        self._rejected: int = 0
        self._evicted: int = 0
        logger.info(
            "AntiSpamVerificationFilter initialized | max_size=%d | ttl=%ds | "
            "dedup_states=%s",
            self._max_size, self._session_ttl_s, sorted(self._dedup_states),
        )

    # ------------------------------------------------------------------
    def should_admit(self, entry: LogEntry) -> bool:
        """
        Evaluate whether the entry should be admitted to the logging
        queue.

        Returns:
            True  -- entry is novel and should be logged.
            False -- entry is a duplicate within the session TTL and
                     must be dropped.
        """
        if entry.state not in self._dedup_states:
            # ANOMALY or unknown state -- always admit.
            with self._lock:
                self._accepted += 1
            return True

        key = entry.dedup_key()
        if key is None:
            with self._lock:
                self._accepted += 1
            return True

        now_us = entry.enqueue_wall_us if entry.enqueue_wall_us else int(time.time() * 1_000_000)
        now_s = now_us / 1_000_000.0

        with self._lock:
            prior_us = self._cache.get(key)
            if prior_us is not None:
                age_s = now_s - (prior_us / 1_000_000.0)
                if age_s < self._session_ttl_s:
                    # Duplicate within TTL -- reject.
                    self._rejected += 1
                    return False
                # TTL expired -- evict and admit as a fresh registration.
                self._cache.pop(key, None)

            # Admit and record.
            self._cache[key] = now_us
            self._cache.move_to_end(key)
            self._accepted += 1

            # LRU eviction if over capacity.
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evicted += 1

            return True

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the cache and reset telemetry counters."""
        with self._lock:
            self._cache.clear()
            self._accepted = 0
            self._rejected = 0
            self._evicted = 0
        logger.info("AntiSpamVerificationFilter reset.")

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "accepted_total": self._accepted,
                "rejected_total": self._rejected,
                "evicted_total": self._evicted,
                "session_ttl_s": self._session_ttl_s,
            }

    # ------------------------------------------------------------------
    def snapshot_keys(self) -> List[str]:
        """Return a copy of the current dedup keys (for dashboard display)."""
        with self._lock:
            return list(self._cache.keys())


# ============================================================================
# 12-Hour Session CSV Writer (Patch 18)
# ============================================================================
class SessionCSVWriter:
    """
    12-hour-session-rotating CSV writer (Patch 18).

    Generates one CSV file per 12-hour LOCAL session at:
        {log_dir}/{prefix}_{YYYY-MM-DD}_{06AM|06PM}.csv

    where the date reflects the day the session STARTED on (so the
    18:00-06:00 overnight session is filed under the date of the 18:00
    boundary, NOT split at midnight).

    The file is opened in append-binary mode with the standard `csv`
    module wrapping a text-mode descriptor (newline="" per RFC 4180).
    The header row is written only on file creation; subsequent appends
    within the same session omit the header. A defensive `os.fsync`
    flush is performed on a configurable cadence to bound data loss on
    crash.

    A list of session-rollover observer callbacks is invoked BEFORE the
    new file handle is opened on every rotation. Each observer receives
    the tuple (old_date, old_session, new_date, new_session) so external
    systems (OSNet feature gallery, stranger OSNet cache, anti-spam
    filter) can perform their own per-session resets in lockstep with
    the CSV rotation.
    """

    def __init__(
        self,
        log_dir: str,
        prefix: str,
        columns: List[str],
        flush_interval_ms: int = 500,
        session_rollover_observers: Optional[List[Callable[[str, str, str, str], None]]] = None,
    ) -> None:
        self._log_dir: str = os.path.abspath(log_dir)
        self._prefix: str = str(prefix)
        self._columns: List[str] = list(columns)
        self._flush_interval_ms: int = int(flush_interval_ms)
        self._lock: threading.RLock = threading.RLock()

        # Session-rollover observers: each is called with
        # (old_date_str, old_session_label, new_date_str, new_session_label)
        # BEFORE the new file is opened. Exceptions are caught + logged so
        # one bad observer cannot block the rotation.
        self._rollover_observers: List[Callable[[str, str, str, str], None]] = (
            list(session_rollover_observers) if session_rollover_observers else []
        )

        # Active file state.
        self._current_date_str: Optional[str] = None
        self._current_session: Optional[str] = None  # "06AM" | "06PM"
        self._file_handle = None            # type: Optional[Any]
        self._csv_writer: Optional[Any] = None
        self._last_flush_wall_s: float = 0.0
        self._rows_written_this_session: int = 0
        self._total_rows_written: int = 0
        self._rotation_count: int = 0

        # Ensure the log directory exists.
        try:
            os.makedirs(self._log_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Failed to create log directory %s: %s",
                self._log_dir, exc,
            )

        logger.info(
            "SessionCSVWriter initialized (Patch 18 :: 12h rotation) | "
            "dir=%s | prefix=%s | columns=%d | flush_ms=%d | observers=%d",
            self._log_dir, self._prefix, len(self._columns),
            self._flush_interval_ms, len(self._rollover_observers),
        )

    # ------------------------------------------------------------------
    def add_rollover_observer(
        self,
        observer: Callable[[str, str, str, str], None],
    ) -> None:
        """
        Register an additional session-rollover observer.

        The observer will be called on every subsequent rotation with
        (old_date, old_session, new_date, new_session). This is the
        primary hook by which main.py registers the OSNet feature-gallery
        reset and the snap_strangers OSNet cache reset.
        """
        with self._lock:
            self._rollover_observers.append(observer)

    # ------------------------------------------------------------------
    def _resolve_path(self, date_str: str, session_label: str) -> str:
        return os.path.join(
            self._log_dir,
            f"{self._prefix}_{date_str}_{session_label}.csv",
        )

    # ------------------------------------------------------------------
    def _open_for_session(self, date_str: str, session_label: str) -> None:
        """
        Open (or rotate to) the CSV file for the given session key.
        Writes the header row only if the file is newly created. Fires
        session-rollover observers BEFORE opening the new file so they
        observe the prior session state.
        """
        # Fire rollover observers BEFORE opening the new file so they
        # see the OLD session as `old_*`.
        if self._current_date_str is not None and self._current_session is not None:
            for observer in list(self._rollover_observers):
                try:
                    observer(
                        self._current_date_str, self._current_session,
                        date_str, session_label,
                    )
                except Exception as exc:
                    logger.warning(
                        "SessionCSVWriter: rollover observer raised: %s", exc,
                    )

        path = self._resolve_path(date_str, session_label)
        file_existed = os.path.exists(path) and os.path.getsize(path) > 0

        try:
            # Use newline="" per csv module RFC 4180 compliance.
            self._file_handle = open(path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._file_handle)
            if not file_existed:
                self._csv_writer.writerow(self._columns)
                self._file_handle.flush()
                logger.info(
                    "SessionCSVWriter created new session file | path=%s | cols=%d",
                    path, len(self._columns),
                )
        except OSError as exc:
            logger.error(
                "SessionCSVWriter failed to open %s: %s -- WRITES SUSPENDED",
                path, exc,
            )
            self._file_handle = None
            self._csv_writer = None
            return

        logger.info(
            "SessionCSVWriter rotated to session %s_%s | path=%s",
            date_str, session_label, path,
        )

        self._current_date_str = date_str
        self._current_session = session_label
        self._rows_written_this_session = 0
        self._rotation_count += 1

    # ------------------------------------------------------------------
    def _rotate_if_needed(self, ts_us: int) -> None:
        """Rotate the file if the session key has changed since last open."""
        date_str, session_label = compute_session_key(ts_us)
        if (
            date_str != self._current_date_str
            or session_label != self._current_session
        ):
            self._close_handle()
            self._open_for_session(date_str, session_label)

    # ------------------------------------------------------------------
    def _close_handle(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                self._file_handle.close()
            except OSError:
                pass
            self._file_handle = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    def write_row(self, entry: LogEntry) -> bool:
        """
        Append one log entry to the active session CSV file.

        Returns True if the write succeeded, False on I/O failure.
        """
        with self._lock:
            try:
                self._rotate_if_needed(entry.timestamp_us)
                if self._csv_writer is None:
                    logger.error(
                        "SessionCSVWriter has no open handle -- dropping row "
                        "state=%s label=%s",
                        entry.state, entry.resolved_label,
                    )
                    return False

                row = entry.to_csv_row(self._columns)
                self._csv_writer.writerow(row)
                self._rows_written_this_session += 1
                self._total_rows_written += 1

                # Periodic flush to bound data loss.
                now_s = time.time()
                if (now_s - self._last_flush_wall_s) * 1000.0 >= self._flush_interval_ms:
                    try:
                        self._file_handle.flush()
                        os.fsync(self._file_handle.fileno())
                    except (OSError, ValueError) as exc:
                        logger.warning(
                            "SessionCSVWriter fsync failed: %s", exc,
                        )
                    self._last_flush_wall_s = now_s
                return True
            except Exception as exc:
                logger.error(
                    "SessionCSVWriter write_row failed for state=%s label=%s: %s",
                    entry.state, entry.resolved_label, exc,
                )
                return False

    # ------------------------------------------------------------------
    def force_flush(self) -> None:
        """Force a synchronous flush + fsync of the active file handle."""
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.flush()
                    os.fsync(self._file_handle.fileno())
                    self._last_flush_wall_s = time.time()
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "SessionCSVWriter force_flush failed: %s", exc,
                    )

    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._close_handle()
            logger.info(
                "SessionCSVWriter closed | total_rows=%d | rotations=%d",
                self._total_rows_written, self._rotation_count,
            )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "log_dir": self._log_dir,
                "current_date": self._current_date_str,
                "current_session": self._current_session,
                "rows_written_this_session": self._rows_written_this_session,
                "total_rows_written": self._total_rows_written,
                "rotation_count": self._rotation_count,
                "rollover_observers": len(self._rollover_observers),
                "columns": list(self._columns),
            }


# Backward-compat alias (older code may have referenced `DailyCSVWriter`).
DailyCSVWriter = SessionCSVWriter


# ============================================================================
# Async Logging Engine
# ============================================================================
class AsyncLoggingEngine:
    """
    Top-level asynchronous logging orchestrator.

    Owns:
      * An internal `queue.Queue` of bounded capacity for non-blocking
        hand-off from producer threads.
      * A daemon worker thread that drains the queue and dispatches
        each entry through the AntiSpamVerificationFilter to the
        SessionCSVWriter (Patch 18: 12-hour session rotation).
      * Per-engine telemetry counters for queue depth, drops, and
        end-to-end latency.

    Public API for the orchestrator:
        engine = AsyncLoggingEngine(config)
        engine.initialize()
        engine.log_entry(timestamp_us=..., track_id=..., resolved_label=...,
                         state=..., similarity_score=..., bbox=...,
                         nrp=..., student_name=..., frame_index=...)
        engine.register_session_rollover_observer(callback)
        engine.shutdown()

    The `register_session_rollover_observer` method is the primary hook
    by which main.py registers the OSNet feature-gallery reset and the
    snap_strangers OSNet cache reset at every 06:00 / 18:00 boundary.
    """

    # Maximum number of consecutive worker errors before emergency
    # shutdown is triggered to prevent tight error loops.
    MAX_CONSECUTIVE_ERRORS: int = 32

    # ------------------------------------------------------------------
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        acfg = self.config.get("async_logger", {})
        self._log_dir: str = str(acfg.get("log_dir", "storage/logs"))
        self._prefix: str = str(acfg.get("daily_csv_prefix", "attendance_"))
        self._columns: List[str] = list(acfg.get(
            "csv_columns",
            [
                "timestamp_us", "frame_index", "track_id", "nrp",
                "student_name", "resolved_label", "state",
                "similarity_score", "bbox_x1", "bbox_y1",
                "bbox_x2", "bbox_y2",
            ],
        ))
        self._queue_maxsize: int = int(acfg.get("queue_maxsize", 1024))
        self._flush_interval_ms: int = int(acfg.get("flush_interval_ms", 500))
        self._anti_spam_size: int = int(acfg.get("anti_spam_cache_size", 256))
        self._anti_spam_ttl_s: int = int(acfg.get("anti_spam_session_ttl_s", 3600))
        self._anti_spam_states: List[str] = list(acfg.get(
            "anti_spam_dedup_states", ["VERIFIED_STUDENT", "STRANGER"],
        ))

        # Core components (constructed in initialize()).
        self._queue: "queue.Queue[Union[LogEntry, _ShutdownSentinel]]" = queue.Queue(
            maxsize=self._queue_maxsize,
        )
        self._filter: AntiSpamVerificationFilter = AntiSpamVerificationFilter(
            max_size=self._anti_spam_size,
            session_ttl_s=self._anti_spam_ttl_s,
            dedup_states=self._anti_spam_states,
        )

        # Patch 18 :: The anti-spam filter is automatically reset on
        # every session rollover (so a student present in BOTH the 06AM
        # and 06PM sessions is logged once per session, not once per
        # day). main.py may register ADDITIONAL observers via
        # `register_session_rollover_observer()` -- e.g. the OSNet
        # feature-gallery reset, the snap_strangers OSNet cache reset.
        self._writer: SessionCSVWriter = SessionCSVWriter(
            log_dir=self._log_dir,
            prefix=self._prefix,
            columns=self._columns,
            flush_interval_ms=self._flush_interval_ms,
            session_rollover_observers=[
                self._on_session_rollover_reset_anti_spam,
            ],
        )

        # Worker thread state.
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event: threading.Event = threading.Event()
        self._initialized: bool = False
        self._running: bool = False

        # Telemetry counters (atomic-ish; GIL-protected int writes).
        self._enqueued: int = 0
        self._dropped_full_queue: int = 0
        self._dropped_anti_spam: int = 0
        self._worker_errors: int = 0
        self._consecutive_errors: int = 0
        self._max_observed_queue_depth: int = 0
        self._last_entry_latency_us: int = 0

    # ==================================================================
    # Session-rollover observer infrastructure (Patch 18).
    # ==================================================================
    def _on_session_rollover_reset_anti_spam(
        self,
        old_date: str,
        old_session: str,
        new_date: str,
        new_session: str,
    ) -> None:
        """
        Built-in observer: reset the anti-spam filter on every session
        rollover so each 12h session starts with a clean dedup cache.
        """
        logger.info(
            "Session rollover %s_%s -> %s_%s :: resetting anti-spam filter.",
            old_date, old_session, new_date, new_session,
        )
        self._filter.reset()

    # ------------------------------------------------------------------
    def register_session_rollover_observer(
        self,
        observer: Callable[[str, str, str, str], None],
    ) -> None:
        """
        Register an external session-rollover observer.

        The observer will be invoked on every 06:00 / 18:00 boundary
        with the signature:
            observer(old_date_str, old_session_label,
                     new_date_str, new_session_label)

        Typical use cases (registered by main.py at startup):
            * OSNet feature-gallery reset (clear the per-stranger
              re-identification embedding cache so a stranger seen in
              the 06AM session and again in the 06PM session gets a
              fresh Stranger_XX label rather than reusing the prior
              session's ID).
            * snap_strangers OSNet memory reset.
            * Dashboard "active session" cache invalidation.

        Exceptions raised by the observer are caught + logged by the
        SessionCSVWriter so one bad observer cannot block rotation.
        """
        self._writer.add_rollover_observer(observer)
        logger.info(
            "Session-rollover observer registered | total_observers=%d",
            len(self._writer._rollover_observers),
        )

    # ------------------------------------------------------------------
    # Patch 20 :: rotate_session()
    #
    # Called by main.py's SessionBoundaryWatcher at every 06:00 / 18:00
    # LOCAL boundary. The logger ALREADY self-rotates on every
    # write_row() call via SessionCSVWriter._rotate_if_needed() (the
    # rotation is driven by the entry's timestamp, not by an explicit
    # call). This adapter forces a PROACTIVE rotation at the boundary
    # so the new CSV file exists immediately on disk -- without this,
    # the new file would only be created when the next log entry
    # arrives (which could be seconds or minutes later, depending on
    # track activity).
    #
    # The session_label / session_date args are ADVISORY ONLY -- the
    # writer computes its own canonical session key from the current
    # timestamp via compute_session_key(). This is intentional: the
    # writer is the single source of truth for session-key computation,
    # and the watcher's label format ("6AM"/"6PM") may differ from
    # the canonical tokens ("06AM"/"06PM") used in filenames.
    # ------------------------------------------------------------------
    def rotate_session(
        self,
        session_label: str,
        session_date: str,
    ) -> None:
        """Force a proactive CSV rotation at the session boundary.

        Args:
            session_label: The new session label (advisory; the writer
                computes its own from the current timestamp). Accepted
                in both "6AM"/"6PM" (watcher format) and "06AM"/"06PM"
                (canonical filename format) for forward-compatibility.
            session_date: The new session date in "YYYY-MM-DD" format
                (advisory; same reason as above).

        This method is safe to call even when no rotation is needed
        (e.g. if the watcher polls slightly before the boundary).
        The writer's _rotate_if_needed() is idempotent -- it only
        rotates when the session key has actually changed.
        """
        now_us = int(time.time() * 1_000_000)
        logger.info(
            "AsyncLoggingEngine.rotate_session() :: proactive rotation "
            "requested | advisory_label=%s | advisory_date=%s | now_us=%d",
            session_label, session_date, now_us,
        )
        # Delegate to the writer's rotation check. If the session key
        # computed from now_us matches the current open session, this
        # is a no-op. Otherwise, it closes the old handle and opens
        # the new file (firing all rollover observers in the process).
        self._writer._rotate_if_needed(now_us)

    # ==================================================================
    # Lifecycle.
    # ==================================================================
    def initialize(self) -> None:
        if self._initialized:
            logger.warning("AsyncLoggingEngine already initialized; skipping.")
            return

        logger.info(
            "AsyncLoggingEngine initializing (Patch 18 :: 12h session rotation) | "
            "log_dir=%s | queue_max=%d | anti_spam=%d | columns=%d",
            self._log_dir, self._queue_maxsize,
            self._anti_spam_size, len(self._columns),
        )
        self._initialized = True

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        if not self._initialized:
            self.initialize()
        if self._running:
            logger.warning("AsyncLoggingEngine worker already running.")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="sortendance.async_logger.worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._running = True
        logger.info("AsyncLoggingEngine worker thread started.")

    # ------------------------------------------------------------------
    def shutdown(self, timeout_s: float = 5.0) -> None:
        """
        Drain the queue, push the shutdown sentinel, and join the worker
        thread. Forces a final flush of the CSV writer.
        """
        if not self._running:
            logger.info("AsyncLoggingEngine shutdown: worker not running.")
            self._writer.close()
            return

        logger.info(
            "AsyncLoggingEngine shutdown initiated | queue_depth=%d",
            self._queue.qsize(),
        )

        # Push the sentinel; this may block if the queue is full, so we
        # use a bounded put with a short timeout and fall back to a
        # non-blocking put.
        try:
            self._queue.put(_SHUTDOWN, timeout=1.0)
        except queue.Full:
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                logger.warning(
                    "AsyncLoggingEngine: queue full at shutdown; forcing "
                    "event-based wakeup.",
                )

        self._shutdown_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout_s)
            if self._worker_thread.is_alive():
                logger.error(
                    "AsyncLoggingEngine worker thread did not exit within %.2fs",
                    timeout_s,
                )
            else:
                logger.info("AsyncLoggingEngine worker thread joined cleanly.")

        self._running = False
        self._writer.force_flush()
        self._writer.close()
        gc.collect()
        logger.info(
            "AsyncLoggingEngine shutdown complete | enqueued=%d | "
            "dropped_full_queue=%d | dropped_anti_spam=%d | worker_errors=%d",
            self._enqueued, self._dropped_full_queue,
            self._dropped_anti_spam, self._worker_errors,
        )

    # ==================================================================
    # Producer API.
    # ==================================================================
    def log_entry(
        self,
        timestamp_us: int,
        track_id: int,
        resolved_label: str,
        state: str,
        similarity_score: float,
        bbox: Tuple[int, int, int, int],
        nrp: Optional[str] = None,
        student_name: Optional[str] = None,
        frame_index: int = 0,
        yolo_latency_ms: float = 0.0,
        face_det_latency_ms: float = 0.0,
        arcface_latency_ms: float = 0.0,
        usearch_latency_ms: float = 0.0,
        ttfm_ms: float = 0.0,
        body_reid_latency_ms: float = 0.0,
    ) -> bool:
        """
        Non-blocking entry point for producer threads.

        Performs anti-spam evaluation INLINE (before queue insertion) so
        that duplicate validation requests are dropped without occupying
        queue capacity. Returns True if the entry was admitted to the
        queue; False if it was dropped (either by the anti-spam filter
        or because the queue was full).
        """
        if not self._running:
            # If the worker is not running, we still attempt the call
            # so that test harnesses that forgot to call start() get a
            # loud warning rather than silent data loss.
            logger.warning(
                "AsyncLoggingEngine.log_entry called before start() -- "
                "entry will be buffered until start() is invoked.",
            )

        enqueue_wall_us = int(time.time() * 1_000_000)
        try:
            entry = LogEntry(
                timestamp_us=int(timestamp_us),
                frame_index=int(frame_index),
                track_id=int(track_id),
                resolved_label=str(resolved_label),
                state=str(state),
                similarity_score=float(similarity_score),
                bbox=(
                    int(bbox[0]), int(bbox[1]),
                    int(bbox[2]), int(bbox[3]),
                ),
                nrp=nrp,
                student_name=student_name,
                enqueue_wall_us=enqueue_wall_us,
                yolo_latency_ms=float(yolo_latency_ms),
                face_det_latency_ms=float(face_det_latency_ms),
                arcface_latency_ms=float(arcface_latency_ms),
                usearch_latency_ms=float(usearch_latency_ms),
                ttfm_ms=float(ttfm_ms),
                body_reid_latency_ms=float(body_reid_latency_ms),
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "AsyncLoggingEngine.log_entry: malformed entry dropped -- "
                "ts=%s track=%s label=%s state=%s sim=%s bbox=%s: %s",
                timestamp_us, track_id, resolved_label, state,
                similarity_score, bbox, exc,
            )
            return False

        # Anti-spam inline evaluation.
        if not self._filter.should_admit(entry):
            self._dropped_anti_spam += 1
            return False

        # Non-blocking queue insertion.
        try:
            self._queue.put_nowait(entry)
            self._enqueued += 1
            depth = self._queue.qsize()
            if depth > self._max_observed_queue_depth:
                self._max_observed_queue_depth = depth
            return True
        except queue.Full:
            self._dropped_full_queue += 1
            # Drop the OLDEST pending entry to make room for the newest,
            # preserving the "latest-frame wins" policy mandated by the
            # orchestrator's ai_queue_maxsize=2 contract.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(entry)
                self._enqueued += 1
                return True
            except queue.Empty:
                return False
            except queue.Full:
                return False

    # ------------------------------------------------------------------
    def force_flush(self) -> None:
        """Public hook to force a synchronous CSV flush."""
        self._writer.force_flush()

    # ==================================================================
    # Background worker loop.
    # ==================================================================
    def _worker_loop(self) -> None:
        """
        Drain the queue and dispatch each entry to the CSV writer.

        The loop exits cleanly when it observes the _SHUTDOWN sentinel,
        after draining any remaining entries ahead of the sentinel in
        the queue.
        """
        logger.info("AsyncLoggingEngine worker loop entered.")
        try:
            while True:
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    # Idle tick -- perform opportunistic flush + reset
                    # the consecutive-error counter on a clean tick.
                    if self._consecutive_errors > 0:
                        self._consecutive_errors = 0
                    self._writer.force_flush()
                    continue

                if item is _SHUTDOWN:
                    # Drain any remaining entries before exiting.
                    self._drain_remaining()
                    logger.info(
                        "AsyncLoggingEngine worker observed shutdown sentinel; "
                        "exiting.",
                    )
                    return

                self._process_item(item)

        except Exception as exc:
            # Catastrophic worker failure -- log and exit so the daemon
            # thread does not silently spin.
            self._worker_errors += 1
            logger.critical(
                "AsyncLoggingEngine worker loop crashed: %s\n%s",
                exc, traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    def _drain_remaining(self) -> None:
        """Drain all entries currently in the queue (best-effort)."""
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
                "AsyncLoggingEngine drained %d residual entries on shutdown.",
                drained,
            )
        self._writer.force_flush()

    # ------------------------------------------------------------------
    def _process_item(self, entry: LogEntry) -> None:
        """Process a single LogEntry: write to CSV, update telemetry."""
        try:
            ok = self._writer.write_row(entry)
            if not ok:
                self._worker_errors += 1
                self._consecutive_errors += 1
            else:
                # Reset consecutive-error counter on a successful write.
                self._consecutive_errors = 0

            # Latency telemetry (enqueue -> process).
            process_us = int(time.time() * 1_000_000)
            latency_us = process_us - entry.enqueue_wall_us
            if latency_us >= 0:
                self._last_entry_latency_us = latency_us

            # Emergency brake on sustained worker errors.
            if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "AsyncLoggingEngine: %d consecutive worker errors -- "
                    "entering emergency drain mode (writes suspended).",
                    self._consecutive_errors,
                )
                # Do NOT exit the loop; keep draining so the queue
                # does not back-pressure the producers. Writes will
                # silently no-op in DailyCSVWriter until the file
                # handle is reopened.
                self._consecutive_errors = 0

        except Exception as exc:
            self._worker_errors += 1
            self._consecutive_errors += 1
            logger.error(
                "AsyncLoggingEngine: failed to process entry "
                "state=%s label=%s: %s\n%s",
                entry.state, entry.resolved_label, exc,
                traceback.format_exc(),
            )

    # ==================================================================
    # Telemetry + read-only views.
    # ==================================================================
    def telemetry(self) -> Dict[str, Any]:
        """Return a comprehensive telemetry snapshot."""
        return {
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self._queue_maxsize,
            "max_observed_queue_depth": self._max_observed_queue_depth,
            "enqueued_total": self._enqueued,
            "dropped_full_queue": self._dropped_full_queue,
            "dropped_anti_spam": self._dropped_anti_spam,
            "worker_errors": self._worker_errors,
            "consecutive_errors": self._consecutive_errors,
            "last_entry_latency_us": self._last_entry_latency_us,
            "running": self._running,
            "writer": self._writer.telemetry(),
            "anti_spam": self._filter.telemetry(),
        }

    # ------------------------------------------------------------------
    def anti_spam_keys(self) -> List[str]:
        """Expose the current anti-spam dedup keys (for dashboard)."""
        return self._filter.snapshot_keys()

    # ------------------------------------------------------------------
    def reset_anti_spam(self) -> None:
        """Clear the anti-spam filter (e.g. on session rollover)."""
        self._filter.reset()

    # ------------------------------------------------------------------
    def columns(self) -> List[str]:
        """Return a copy of the configured CSV column schema."""
        return list(self._columns)


# ============================================================================
# Convenience factory
# ============================================================================
def build_async_logger(
    config_path: Optional[str] = None,
    autostart: bool = True,
) -> AsyncLoggingEngine:
    """
    Construct an AsyncLoggingEngine from the central config registry.

    Args:
        config_path: Optional explicit path to config.yaml. If None,
                     the ConfigRegistry default is used.
        autostart:   If True, initialize() and start() are called
                     before returning.
    """
    cfg: Dict[str, Any] = {}
    if ConfigRegistry is not None:
        try:
            cfg = ConfigRegistry.load(config_path) if config_path else ConfigRegistry.load()
        except Exception as exc:
            logger.error(
                "build_async_logger: ConfigRegistry.load failed: %s -- "
                "falling back to empty config.", exc,
            )

    engine = AsyncLoggingEngine(config=cfg)
    if autostart:
        engine.initialize()
        engine.start()
    return engine


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness (no external engine dependencies)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance async_logger self-test ===")

    cfg: Dict[str, Any] = {}
    if ConfigRegistry is not None:
        try:
            cfg = ConfigRegistry.load("config/config.yaml")
        except Exception as exc:
            logger.warning("self-test: ConfigRegistry.load failed: %s", exc)

    engine = AsyncLoggingEngine(config=cfg)
    engine.initialize()
    engine.start()

    # --- Test 1: VERIFIED_STUDENT (should admit) ---
    ok1 = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=1,
        track_id=10,
        resolved_label="[2024001 / John Doe]",
        state="VERIFIED_STUDENT",
        similarity_score=0.78,
        bbox=(100, 100, 200, 200),
        nrp="2024001",
        student_name="John Doe",
    )
    logger.info("Test 1 VERIFIED_STUDENT admitted=%s", ok1)

    # --- Test 2: duplicate VERIFIED_STUDENT (should drop) ---
    ok2 = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=2,
        track_id=10,
        resolved_label="[2024001 / John Doe]",
        state="VERIFIED_STUDENT",
        similarity_score=0.79,
        bbox=(101, 101, 201, 201),
        nrp="2024001",
        student_name="John Doe",
    )
    logger.info("Test 2 duplicate VERIFIED_STUDENT admitted=%s (expected False)", ok2)

    # --- Test 3: STRANGER (should admit) ---
    ok3 = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=3,
        track_id=11,
        resolved_label="[Stranger_01]",
        state="STRANGER",
        similarity_score=0.32,
        bbox=(300, 100, 400, 200),
    )
    logger.info("Test 3 STRANGER admitted=%s", ok3)

    # --- Test 4: duplicate STRANGER (should drop) ---
    ok4 = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=4,
        track_id=11,
        resolved_label="[Stranger_01]",
        state="STRANGER",
        similarity_score=0.31,
        bbox=(302, 102, 402, 202),
    )
    logger.info("Test 4 duplicate STRANGER admitted=%s (expected False)", ok4)

    # --- Test 5: ANOMALY (always admits) ---
    ok5a = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=5,
        track_id=-1,
        resolved_label="[ANOMALY]",
        state="ANOMALY",
        similarity_score=0.0,
        bbox=(500, 50, 540, 90),
    )
    ok5b = engine.log_entry(
        timestamp_us=int(time.time() * 1_000_000),
        frame_index=6,
        track_id=-1,
        resolved_label="[ANOMALY]",
        state="ANOMALY",
        similarity_score=0.0,
        bbox=(520, 60, 560, 100),
    )
    logger.info(
        "Test 5 ANOMALY: first=%s second=%s (both expected True)",
        ok5a, ok5b,
    )

    # --- Test 6: bulk load to exercise queue backpressure ---
    for i in range(2000):
        engine.log_entry(
            timestamp_us=int(time.time() * 1_000_000),
            frame_index=100 + i,
            track_id=20 + (i % 50),
            resolved_label=f"[2024{(i % 50):04d} / Student_{i % 50}]",
            state="VERIFIED_STUDENT",
            similarity_score=0.70 + (i % 10) * 0.01,
            bbox=(50 + i % 100, 50, 150 + i % 100, 150),
            nrp=f"2024{(i % 50):04d}",
            student_name=f"Student_{i % 50}",
        )

    # Allow the worker to drain.
    time.sleep(2.0)
    logger.info("Telemetry after bulk load: %s", engine.telemetry())

    engine.shutdown(timeout_s=5.0)
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()