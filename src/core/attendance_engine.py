"""
SORT-tendance :: Attendance Engine
===================================
Bridges the vision pipeline's per-track verification events into the
scheduling system's per-slot attendance records.

PUBLIC API (called by main.py's IsolatedAIThread):
  - record_verified(student_id, student_name, track_id, snapshot_path)
      Called when a face match resolves to a known student. The engine
      determines whether the detection falls within any slot's effective
      window and, if so, marks the student as ATTENDED for that slot.
  - record_stranger(stranger_label, track_id, snapshot_path)
      Called when a track is locked as STRANGER. If the detection falls
      within a slot's window, the stranger is recorded in the slot's
      stranger CSV (with snapshot path) for inclusion in the PDF.

PERSISTENCE:
  Writes two CSV files per (date, slot) under ``attend_csv_dir``:
    attendance_{YYYY-MM-DD}_{slot_id}.csv
        Columns: date, slot_id, subject_name, student_id, student_name,
                 status, first_seen_in_slot, last_seen_in_slot
        One row per EXPECTED student. Status is ATTENDED or NOT_ATTENDED.
        NOT_ATTENDED is written when the slot ends (or when the CSV is
        first materialized mid-slot, then updated as detections arrive).
    strangers_{YYYY-MM-DD}_{slot_id}.csv
        Columns: date, slot_id, subject_name, stranger_label,
                 first_seen_in_slot, last_seen_in_slot, snapshot_path
        One row per UNIQUE stranger seen during the slot.

THREAD SAFETY:
  All public methods acquire an internal lock. Safe to call from the AI
  thread, the snapshot finalization path, or the dashboard.

EFFECTIVE WINDOW:
  effective_window = [start_time - grace_minutes, end_time)
  - Detections at t < effective_start: ignored (too early).
  - Detections at effective_start <= t < end_time: count as attended.
  - Detections at t >= end_time: ignored (late entry, no make-up).
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .schedule_manager import ScheduleManager, Slot

logger = logging.getLogger("sortendance.attendance_engine")


# ---------------------------------------------------------------------------
# CSV schema.
# ---------------------------------------------------------------------------
ATTENDANCE_CSV_COLUMNS = [
    "date",
    "slot_id",
    "subject_name",
    "student_id",
    "student_name",
    "status",                # ATTENDED | NOT_ATTENDED
    "first_seen_in_slot",    # HH:MM:SS or empty
    "last_seen_in_slot",     # HH:MM:SS or empty
]

STRANGER_CSV_COLUMNS = [
    "date",
    "slot_id",
    "subject_name",
    "stranger_label",
    "first_seen_in_slot",
    "last_seen_in_slot",
    "snapshot_path",
]


# ---------------------------------------------------------------------------
# AttendanceEngine.
# ---------------------------------------------------------------------------
class AttendanceEngine:
    """Tracks per-slot attendance, persisting to CSV files on each update.

    Construction is cheap. The engine holds an in-memory cache of the
    current day's attendance (per slot, per student) so that re-writes
    are O(1) instead of O(N) re-reads.

    The cache is keyed by (date_str, slot_id, student_id) for attendance
    and (date_str, slot_id, stranger_label) for strangers.
    """

    def __init__(self, schedule_manager: ScheduleManager) -> None:
        self._sched = schedule_manager
        self._lock = threading.RLock()

        # In-memory cache: {(date_str, slot_id): {student_id: {fields}}}
        self._attendance_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        # In-memory cache: {(date_str, slot_id): {stranger_label: {fields}}}
        self._stranger_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

        # Track which (date_str, slot_id) attendance files have been
        # "materialized" (i.e. we've written the full expected-students
        # list with NOT_ATTENDED placeholders). This prevents rewriting
        # the full file on every detection.
        self._materialized: set = set()

        # Cache expected-students list per slot (refreshed on first access
        # per day, or when the slot's student-list file mtime changes).
        self._expected_cache: Dict[str, Tuple[float, List[str]]] = {}

        if not self._sched.enabled:
            logger.info("AttendanceEngine: SCHEDULING DISABLED in config; no-op.")
        else:
            logger.info(
                "AttendanceEngine: ready | attend_csv_dir=%s",
                self._sched.attend_csv_dir,
            )

    # ------------------------------------------------------------------
    # Public API: called by main.py.
    # ------------------------------------------------------------------
    def record_verified(
        self,
        student_id: str,
        student_name: str,
        track_id: int,
        snapshot_path: Optional[str] = None,
        now: Optional[_dt.datetime] = None,
    ) -> Optional[str]:
        """Record a verified-student detection.

        Returns the slot_id if the detection counted toward attendance,
        or None if it fell outside any slot's effective window (free
        period) or the student wasn't expected in any active slot.

        Handles overlapping grace windows correctly: if slot_1 (IoT
        7-10) is still active and slot_2 (AI 10-12) has grace starting
        at 9:50, a student expected in slot_2 arriving at 9:55 will be
        attributed to slot_2 (not rejected as "not in slot_1's list").
        """
        if not self._sched.enabled:
            return None
        if not student_id:
            return None
        if now is None:
            now = _dt.datetime.now()

        # Find the slot where THIS student is expected AND whose window
        # is currently active. Falls back to None if no slot's window
        # contains `now` (free period) OR if the student isn't expected
        # in any currently-active slot.
        slot = self._sched.find_slot_for_student(student_id, now)
        if slot is None:
            # Either free period, or student is recognized but not on
            # any active slot's expected list (recognized "guest").
            # Per spec: recognized but no attendance impact.
            return None

        # Record attendance.
        with self._lock:
            self._ensure_materialized(slot, now.date())
            key = (now.strftime("%Y-%m-%d"), slot.slot_id)
            slot_cache = self._attendance_cache.setdefault(key, {})
            entry = slot_cache.get(student_id)
            time_str = now.strftime("%H:%M:%S")
            if entry is None:
                # First sighting in this slot.
                entry = {
                    "date": key[0],
                    "slot_id": slot.slot_id,
                    "subject_name": slot.subject_name,
                    "student_id": student_id,
                    "student_name": student_name,
                    "status": "ATTENDED",
                    "first_seen_in_slot": time_str,
                    "last_seen_in_slot": time_str,
                }
                slot_cache[student_id] = entry
                logger.info(
                    "AttendanceEngine: %s (%s) ATTENDED %s at %s",
                    student_id, student_name, slot.subject_name, time_str,
                )
            else:
                # Update existing entry: flip status to ATTENDED if it was
                # NOT_ATTENDED (this is the case where the student was
                # materialized as a placeholder before being detected).
                was_not_attended = (entry.get("status") != "ATTENDED")
                if was_not_attended:
                    entry["status"] = "ATTENDED"
                    # Set first_seen_in_slot ONLY if it was empty.
                    if not entry.get("first_seen_in_slot"):
                        entry["first_seen_in_slot"] = time_str
                    if student_name and not entry.get("student_name"):
                        entry["student_name"] = student_name
                    logger.info(
                        "AttendanceEngine: %s (%s) ATTENDED %s at %s (was NOT_ATTENDED)",
                        student_id, student_name, slot.subject_name, time_str,
                    )
                # Always update last_seen_in_slot.
                entry["last_seen_in_slot"] = time_str
                # Refresh student_name in case it was unknown at materialization.
                if not entry.get("student_name") and student_name:
                    entry["student_name"] = student_name

            self._write_attendance_csv(slot, key[0])
        return slot.slot_id

    def record_stranger(
        self,
        stranger_label: str,
        track_id: int,
        snapshot_path: Optional[str] = None,
        now: Optional[_dt.datetime] = None,
    ) -> Optional[str]:
        """Record a stranger detection.

        Returns the slot_id if recorded (stranger appeared during a slot),
        or None if outside any slot's window.

        For strangers, we use `get_active_slot` (first-match) instead of
        `find_slot_for_student` because strangers have no expected list.
        If multiple slots' windows overlap, the stranger is attributed
        to the first active slot (by config order).
        """
        if not self._sched.enabled:
            return None
        if not stranger_label:
            return None
        if now is None:
            now = _dt.datetime.now()

        slot = self._sched.get_active_slot(now)
        if slot is None:
            return None

        with self._lock:
            key = (now.strftime("%Y-%m-%d"), slot.slot_id)
            slot_cache = self._stranger_cache.setdefault(key, {})
            entry = slot_cache.get(stranger_label)
            time_str = now.strftime("%H:%M:%S")
            if entry is None:
                entry = {
                    "date": key[0],
                    "slot_id": slot.slot_id,
                    "subject_name": slot.subject_name,
                    "stranger_label": stranger_label,
                    "first_seen_in_slot": time_str,
                    "last_seen_in_slot": time_str,
                    "snapshot_path": snapshot_path or "",
                }
                slot_cache[stranger_label] = entry
                logger.info(
                    "AttendanceEngine: stranger %s seen in %s at %s",
                    stranger_label, slot.subject_name, time_str,
                )
            else:
                entry["last_seen_in_slot"] = time_str

            # Always try to upgrade the snapshot path. The snapshot
            # worker writes the PNG asynchronously, so the first few
            # frames after a stranger is locked may return None or a
            # stale path. By checking on every call, we pick up the
            # finalized path as soon as the worker has written it.
            #
            # We also verify the file exists on disk -- if the previous
            # path pointed to a file that was rotated/cleaned up, we
            # want to replace it with a valid one.
            if snapshot_path:
                existing = entry.get("snapshot_path", "")
                if not existing or not os.path.isfile(existing):
                    if os.path.isfile(snapshot_path):
                        entry["snapshot_path"] = snapshot_path

            self._write_stranger_csv(slot, key[0])
        return slot.slot_id

    # ------------------------------------------------------------------
    # Public API: called by the dashboard.
    # ------------------------------------------------------------------
    def read_attendance_for_slot(
        self, slot_id: str, date_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the attendance CSV for (slot_id, date_str). Returns list of dicts.

        If date_str is None, uses today's date.
        """
        if date_str is None:
            date_str = _dt.datetime.now().strftime("%Y-%m-%d")
        path = self._attendance_csv_path(slot_id, date_str)
        if not os.path.isfile(path):
            return []
        return self._read_csv(path, ATTENDANCE_CSV_COLUMNS)

    def read_strangers_for_slot(
        self, slot_id: str, date_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the stranger CSV for (slot_id, date_str)."""
        if date_str is None:
            date_str = _dt.datetime.now().strftime("%Y-%m-%d")
        path = self._stranger_csv_path(slot_id, date_str)
        if not os.path.isfile(path):
            return []
        return self._read_csv(path, STRANGER_CSV_COLUMNS)

    def refresh_stranger_snapshots(
        self,
        slot_id: str,
        date_str: Optional[str] = None,
        snapshot_search_roots: Optional[List[str]] = None,
    ) -> int:
        """Re-scan snapshot directories for stranger photos.

        HOTFIX-3 :: Hardened version. Multi-root search, regex-based
        label extraction (no more IndexError on mixed-case filenames),
        and detailed diagnostic logging so operators can see exactly
        what was searched and why matches did or did not happen.

        For each stranger in the CSV whose `snapshot_path` is empty or
        points to a missing file, walk the configured search roots
        looking for PNGs whose filename contains `_STRANGER_{label}`.
        When a match is found, prefer the most-recently-modified file
        (clearshots tend to be sharper than the renamed birth PNG).

        Args:
            slot_id: The slot to refresh.
            date_str: The date (default: today).
            snapshot_search_roots: List of directories to search
                recursively. If None, a MULTI-ROOT default is used:
                  * {project_root}/storage/snap_strangers
                  * {cwd}/storage/snap_strangers
                  * any snap_strangers.output_dir from config

        Returns:
            Number of stranger rows whose snapshot_path was updated.
        """
        if date_str is None:
            date_str = _dt.datetime.now().strftime("%Y-%m-%d")

        rows = self.read_strangers_for_slot(slot_id, date_str)
        if not rows:
            logger.info(
                "refresh_stranger_snapshots: no stranger rows for %s/%s -- nothing to do.",
                slot_id, date_str,
            )
            return 0

        # Determine search roots. Default is MULTI-ROOT to handle the
        # case where the dashboard's _project_root differs from where
        # main.py actually wrote the snapshots (different CWD, deploy
        # subfolder, etc.).
        if not snapshot_search_roots:
            snapshot_search_roots = self._default_snapshot_search_roots()

        # De-duplicate while preserving order.
            seen = set()
            snapshot_search_roots = [
                r for r in snapshot_search_roots
                if not (r in seen or seen.add(r))
            ]

        slot = self._sched.get_slot(slot_id)
        if slot is None:
            logger.warning(
                "refresh_stranger_snapshots: slot %s not found.", slot_id,
            )
            return 0

        # Log the search roots so the operator can verify them.
        existing_roots = [r for r in snapshot_search_roots if os.path.isdir(r)]
        missing_roots = [r for r in snapshot_search_roots if not os.path.isdir(r)]
        logger.info(
            "refresh_stranger_snapshots: searching %d root(s) for %s/%s | "
            "stranger_rows=%d | existing_roots=%d | missing_roots=%d",
            len(snapshot_search_roots), slot_id, date_str, len(rows),
            len(existing_roots), len(missing_roots),
        )
        for r in snapshot_search_roots:
            logger.info("  search_root: %s [%s]", r, "OK" if os.path.isdir(r) else "MISSING")

        # Build a label -> [filepath] index by walking each search root.
        # Regex is case-INSENSITIVE so we handle filenames produced by
        # snap_strangers.py (uppercase _STRANGER_) as well as any
        # legacy / hand-renamed files.
        label_re = re.compile(
            r"_STRANGER_(?P<label>.+?)(?:_CLEARSHOT_\d+)?\.png$",
            re.IGNORECASE,
        )
        label_to_paths: Dict[str, List[str]] = {}
        total_png_scanned = 0
        total_stranger_png = 0
        for root in existing_roots:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.lower().endswith(".png"):
                        continue
                    total_png_scanned += 1
                    m = label_re.search(fname)
                    if not m:
                        continue
                    total_stranger_png += 1
                    raw_label = m.group("label")
                    norm_label = self._normalize_stranger_label(raw_label)
                    if not norm_label:
                        continue
                    full_path = os.path.join(dirpath, fname)
                    label_to_paths.setdefault(norm_label, []).append(full_path)

        logger.info(
            "refresh_stranger_snapshots: index built | total_png=%d | "
            "stranger_png=%d | unique_labels=%d",
            total_png_scanned, total_stranger_png, len(label_to_paths),
        )

        # Update rows that need a snapshot path.
        updated = 0
        unmatched: List[str] = []

        with self._lock:
            key = (date_str, slot_id)
            slot_cache = self._stranger_cache.get(key, {})
            cache_dirty = False
            for row in rows:
                label = row.get("stranger_label", "")
                if not label:
                    continue
                existing_path = row.get("snapshot_path", "")
                if existing_path and os.path.isfile(existing_path):
                    continue  # Already has a valid path.
                norm = self._normalize_stranger_label(label)
                candidates = label_to_paths.get(norm, [])
                if not candidates:
                    unmatched.append(label)
                    continue
                # Prefer the most recently modified file (best quality
                # snapshot is usually the latest clearshot).
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                best_path = candidates[0]
                row["snapshot_path"] = best_path
                entry = slot_cache.get(label)
                if entry is not None:
                    entry["snapshot_path"] = best_path
                    cache_dirty = True
                updated += 1
                logger.info(
                    "  matched: %s -> %s", label, best_path,
                )

            if updated > 0:
                if cache_dirty and slot_cache:
                    self._write_stranger_csv(slot, date_str)
                else:
                    path = self._stranger_csv_path(slot_id, date_str)
                    self._write_csv(path, STRANGER_CSV_COLUMNS, rows)
                logger.info(
                    "refresh_stranger_snapshots: updated %d/%d strangers for %s/%s",
                    updated, len(rows), slot_id, date_str,
                )
            if unmatched:
                logger.warning(
                    "refresh_stranger_snapshots: %d stranger(s) had NO matching "
                    "snapshot file on disk: %s",
                    len(unmatched), unmatched,
                )

        return updated

    @staticmethod
    def _normalize_stranger_label(label: str) -> str:
        """Normalize a stranger label for case-insensitive matching.

        Strips brackets, whitespace, and underscores, then lowercases.
        Handles all of these equivalently:
          "Stranger_07"
          "[Stranger_07]"
          "stranger_07"
          "Stranger_07_"
          "  Stranger_07  "
        """
        if not label:
            return ""
        s = str(label).strip()
        # Strip matching outer brackets.
        while s and s[0] in "[({" and s[-1] in "])}":
            s = s[1:-1].strip()
        # Strip stray bracket characters anywhere.
        for ch in "[](){}":
            s = s.replace(ch, "")
        s = s.strip().strip("_").strip()
        return s.lower()

    def _default_snapshot_search_roots(self) -> List[str]:
        """Build a multi-root default search list.

        HOTFIX-3 :: Previously this returned a single root
        (`{project_root}/storage/snap_strangers`). Now it returns
        multiple candidates so the search succeeds even when the
        dashboard's CWD or _project_root differs from where main.py
        wrote the snapshots.
        """
        roots: List[str] = []
        try:
            project_root = str(self._sched._project_root)  # type: ignore[attr-defined]
        except Exception:
            project_root = os.getcwd()
        cwd = os.getcwd()

        # 1. project_root / storage / snap_strangers
        roots.append(os.path.join(project_root, "storage", "snap_strangers"))
        # 2. cwd / storage / snap_strangers (handles dashboard launched
        #    from a different working directory than main.py).
        if os.path.abspath(cwd) != os.path.abspath(project_root):
            roots.append(os.path.join(cwd, "storage", "snap_strangers"))

        # 3. Any path from config's snap_strangers.output_dir (resolved
        #    relative to project_root AND cwd, in case it's a relative
        #    path).
        try:
            snap_cfg = (self._sched._config or {}).get("snap_strangers", {})
            if not isinstance(snap_cfg, dict):
                snap_cfg = {}
            output_dir = snap_cfg.get("output_dir", "")
            if output_dir:
                if os.path.isabs(output_dir):
                    roots.append(output_dir)
                else:
                    roots.append(os.path.join(project_root, output_dir))
                    roots.append(os.path.join(cwd, output_dir))
        except Exception:
            pass

        # De-duplicate while preserving order.
        seen = set()
        out = []
        for r in roots:
            ar = os.path.abspath(r)
            if ar in seen:
                continue
            seen.add(ar)
            out.append(r)
        return out


    # ------------------------------------------------------------------
    # Internal: expected-students caching.
    # ------------------------------------------------------------------
    def _get_expected_students(self, slot_id: str) -> List[str]:
        """Return the slot's expected student IDs, with mtime-based caching."""
        slot = self._sched.get_slot(slot_id)
        if slot is None:
            return []
        path = self._sched.student_list_path(slot_id) or ""
        try:
            mtime = os.path.getmtime(path) if path and os.path.isfile(path) else 0.0
        except OSError:
            mtime = 0.0
        cached = self._expected_cache.get(slot_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        ids = self._sched.load_expected_students(slot_id)
        self._expected_cache[slot_id] = (mtime, ids)
        return ids

    # ------------------------------------------------------------------
    # Internal: CSV materialization.
    # ------------------------------------------------------------------
    def _ensure_materialized(self, slot: Slot, date: _dt.date) -> None:
        """If the attendance CSV for (slot, date) hasn't been written yet,
        write the full expected-students list with NOT_ATTENDED placeholders.
        """
        date_str = date.strftime("%Y-%m-%d")
        key = (date_str, slot.slot_id)
        if key in self._materialized:
            return

        # Try to load existing CSV from disk (in case the engine was
        # restarted mid-slot).
        path = self._attendance_csv_path(slot.slot_id, date_str)
        if os.path.isfile(path):
            # Pre-populate cache from disk so we don't lose prior detections.
            rows = self._read_csv(path, ATTENDANCE_CSV_COLUMNS)
            slot_cache: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                sid = r.get("student_id", "")
                if sid:
                    slot_cache[sid] = dict(r)
            self._attendance_cache[key] = slot_cache
            self._materialized.add(key)
            return

        # First time: materialize the full expected list.
        expected = self._get_expected_students(slot.slot_id)
        slot_cache = {}
        for sid in expected:
            slot_cache[sid] = {
                "date": date_str,
                "slot_id": slot.slot_id,
                "subject_name": slot.subject_name,
                "student_id": sid,
                "student_name": "",  # Will be filled by record_verified
                "status": "NOT_ATTENDED",
                "first_seen_in_slot": "",
                "last_seen_in_slot": "",
            }
        self._attendance_cache[key] = slot_cache
        self._materialized.add(key)
        self._write_attendance_csv(slot, date_str)
        logger.info(
            "AttendanceEngine: materialized %s attendance CSV with %d expected students",
            slot.slot_id, len(expected),
        )

    # ------------------------------------------------------------------
    # Internal: CSV writing.
    # ------------------------------------------------------------------
    def _attendance_csv_path(self, slot_id: str, date_str: str) -> str:
        return os.path.join(
            self._sched.attend_csv_dir,
            f"attendance_{date_str}_{slot_id}.csv",
        )

    def _stranger_csv_path(self, slot_id: str, date_str: str) -> str:
        return os.path.join(
            self._sched.attend_csv_dir,
            f"strangers_{date_str}_{slot_id}.csv",
        )

    def _write_attendance_csv(self, slot: Slot, date_str: str) -> None:
        """Write the full attendance CSV for (slot, date) from cache."""
        path = self._attendance_csv_path(slot.slot_id, date_str)
        key = (date_str, slot.slot_id)
        slot_cache = self._attendance_cache.get(key, {})
        # Order: ATTENDED first (by first_seen), then NOT_ATTENDED (by student_id).
        attended = sorted(
            [v for v in slot_cache.values() if v.get("status") == "ATTENDED"],
            key=lambda r: r.get("first_seen_in_slot", ""),
        )
        not_attended = sorted(
            [v for v in slot_cache.values() if v.get("status") != "ATTENDED"],
            key=lambda r: r.get("student_id", ""),
        )
        rows = attended + not_attended
        self._write_csv(path, ATTENDANCE_CSV_COLUMNS, rows)

    def _write_stranger_csv(self, slot: Slot, date_str: str) -> None:
        path = self._stranger_csv_path(slot.slot_id, date_str)
        key = (date_str, slot.slot_id)
        slot_cache = self._stranger_cache.get(key, {})
        rows = sorted(
            slot_cache.values(),
            key=lambda r: r.get("first_seen_in_slot", ""),
        )
        self._write_csv(path, STRANGER_CSV_COLUMNS, rows)

    @staticmethod
    def _write_csv(path: str, columns: List[str], rows: List[Dict[str, Any]]) -> None:
        """Atomic CSV write: write to .tmp then rename."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            os.replace(tmp, path)
        except OSError as exc:
            logger.error("AttendanceEngine: failed to write %s: %s", path, exc)
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    @staticmethod
    def _read_csv(path: str, columns: List[str]) -> List[Dict[str, Any]]:
        """Read a CSV into a list of dicts. Missing fields = empty string."""
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    full = {c: row.get(c, "") for c in columns}
                    out.append(full)
        except OSError as exc:
            logger.error("AttendanceEngine: failed to read %s: %s", path, exc)
        return out
