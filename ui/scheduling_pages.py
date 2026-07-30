"""
SORT-tendance :: ui/scheduling_pages.py
========================================
Streamlit pages for the Class Scheduling & Attendance feature.

Pages provided:
  - render_schedule_page()      : CRUD UI for slots (subject, times).
  - render_students_page()      : upload/override/delete per-slot student lists.
  - render_live_attendance_page(): auto-refreshing view of the active slot.
  - render_reports_page()       : date+subject picker -> download PDF.

Each function takes a ScheduleManager + AttendanceEngine (and the config)
and renders into the current Streamlit context. Functions are designed
to be called from dashboard.py's page-dispatch block.

The auto-refresh for the Live Attendance page is driven by
streamlit_autorefresh (if installed) at the schedule's poll_interval_s
(default 10s). Falls back to st.rerun() if streamlit_autorefresh is
unavailable.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import datetime as _dt
import logging
import traceback
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure src/ is on sys.path so we can import core modules when this
# file is loaded by streamlit (which runs from the project root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import streamlit as st

# Optional deps.
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False
    pd = None  # type: ignore

# Core modules.
try:
    from core.schedule_manager import ScheduleManager, Slot
    from core.attendance_engine import AttendanceEngine
except ImportError as _imp_exc:  # pragma: no cover
    ScheduleManager = None  # type: ignore
    Slot = None  # type: ignore
    AttendanceEngine = None  # type: ignore
    _SCHED_IMPORT_ERROR = _imp_exc
else:
    _SCHED_IMPORT_ERROR = None

try:
    from utils.pdf_generator import generate_attendance_pdf, is_available as pdf_is_available
except ImportError as _pdf_exc:  # pragma: no cover
    generate_attendance_pdf = None  # type: ignore
    pdf_is_available = lambda: False  # type: ignore
    _PDF_IMPORT_ERROR = _pdf_exc
else:
    _PDF_IMPORT_ERROR = None

try:
    from utils.xlsx_generator import generate_attendance_xlsx, is_available as xlsx_is_available
except ImportError as _xlsx_exc:  # pragma: no cover
    generate_attendance_xlsx = None  # type: ignore
    xlsx_is_available = lambda: False  # type: ignore
    _XLSX_IMPORT_ERROR = _xlsx_exc
else:
    _XLSX_IMPORT_ERROR = None

logger = logging.getLogger("sortendance.dashboard.scheduling_pages")


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _ensure_imports() -> Optional[str]:
    """Return an error message string if imports failed, else None."""
    if _SCHED_IMPORT_ERROR is not None:
        return (
            f"Scheduling modules could not be imported: {_SCHED_IMPORT_ERROR}. "
            "Make sure src/core/schedule_manager.py and "
            "src/core/attendance_engine.py are present."
        )
    return None


def _build_schedule_manager(config: Dict[str, Any], config_path: str) -> Optional[Any]:
    """Construct a fresh ScheduleManager for the dashboard's use.

    The dashboard's ScheduleManager is READ-MOSTLY. Edits made via the
    UI persist to config.yaml, which main.py will pick up on its next
    schedule reload (currently at startup or 6AM/6PM session boundary).
    """
    if ScheduleManager is None:
        return None
    try:
        return ScheduleManager(
            config=config,
            config_path=config_path,
            project_root=_PROJECT_ROOT,
        )
    except Exception as exc:
        logger.error("Could not build ScheduleManager: %s", exc, exc_info=True)
        return None


def _build_attendance_engine(sched: Any) -> Optional[Any]:
    """Construct a dashboard-side AttendanceEngine for READ-ONLY access
    to the CSV files. Writes happen in main.py; the dashboard only reads.
    """
    if AttendanceEngine is None or sched is None:
        return None
    try:
        return AttendanceEngine(sched)
    except Exception as exc:
        logger.error("Could not build AttendanceEngine: %s", exc, exc_info=True)
        return None


def _parse_hhmm(s: str) -> Optional[_dt.time]:
    """Parse 'HH:MM' -> datetime.time. Returns None on failure."""
    try:
        h, m = s.strip().split(":", 1)
        return _dt.time(int(h), int(m))
    except (ValueError, AttributeError):
        return None


def _time_str_to_dt(t: _dt.time, base_date: _dt.date) -> _dt.datetime:
    return _dt.datetime(base_date.year, base_date.month, base_date.day,
                        t.hour, t.minute, t.second)


# ---------------------------------------------------------------------------
# Page 1: Schedule Management.
# ---------------------------------------------------------------------------
def render_schedule_page(
    config: Dict[str, Any],
    config_path: str,
) -> None:
    """CRUD UI for the daily-recurring class schedule."""
    st.header("📅 Schedule Management")
    st.caption(
        "Configure the daily-recurring class slots. Changes persist to "
        "`config.yaml` and take effect for FUTURE slots (the currently-"
        "active slot is edit-locked)."
    )

    err = _ensure_imports()
    if err:
        st.error(err)
        return

    sm = _build_schedule_manager(config, config_path)
    if sm is None:
        st.error("Could not initialize ScheduleManager.")
        return

    if not sm.enabled:
        st.warning(
            "Class scheduling is DISABLED in config.yaml "
            "(`class_scheduling.enabled: false`). Enable it to use this page."
        )
        return

    # --- Existing slots table ---
    st.subheader("Current Slots")
    slots = sm.list_slots()
    now = _dt.datetime.now()
    if not slots:
        st.info("No slots configured. Add one below.")
    else:
        for slot in slots:
            is_active = sm.is_slot_edit_locked(slot.slot_id, now)
            with st.container(border=True):
                cols = st.columns([2, 2, 2, 3])
                with cols[0]:
                    st.markdown(f"**{slot.subject_name}**")
                    st.caption(f"Slot ID: `{slot.slot_id}`")
                with cols[1]:
                    st.markdown(f"⏰ {slot.start_time} – {slot.end_time}")
                with cols[2]:
                    n_expected = len(sm.load_expected_students(slot.slot_id))
                    st.markdown(f"👥 {n_expected} expected")
                with cols[3]:
                    if is_active:
                        st.warning("🔒 ACTIVE (edit-locked)")
                    else:
                        st.caption("Idle")

                # Edit form (only if not active).
                if not is_active:
                    with st.expander(f"Edit {slot.subject_name}", expanded=False):
                        new_subject = st.text_input(
                            "Subject Name",
                            value=slot.subject_name,
                            key=f"_sched_subject_{slot.slot_id}",
                        )
                        col_a, col_b = st.columns(2)
                        with col_a:
                            new_start = st.text_input(
                                "Start Time (HH:MM)",
                                value=slot.start_time,
                                key=f"_sched_start_{slot.slot_id}",
                            )
                        with col_b:
                            new_end = st.text_input(
                                "End Time (HH:MM)",
                                value=slot.end_time,
                                key=f"_sched_end_{slot.slot_id}",
                            )
                        col_save, col_del = st.columns(2)
                        if col_save.button(
                            "Save Changes",
                            key=f"_sched_save_{slot.slot_id}",
                            type="primary",
                        ):
                            # Validate.
                            t_start = _parse_hhmm(new_start)
                            t_end = _parse_hhmm(new_end)
                            if t_start is None or t_end is None:
                                st.error("Invalid time format. Use HH:MM (24-hour).")
                            elif t_start >= t_end:
                                st.error("Start time must be before end time.")
                            elif not new_subject.strip():
                                st.error("Subject name cannot be empty.")
                            else:
                                try:
                                    sm.update_slot(
                                        slot.slot_id,
                                        subject_name=new_subject.strip(),
                                        start_time=new_start.strip(),
                                        end_time=new_end.strip(),
                                    )
                                    st.success(f"Updated slot {slot.slot_id}.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Update failed: {exc}")
                        if col_del.button(
                            "Delete Slot",
                            key=f"_sched_del_{slot.slot_id}",
                        ):
                            try:
                                if sm.delete_slot(slot.slot_id):
                                    st.success(f"Deleted slot {slot.slot_id}.")
                                    st.rerun()
                                else:
                                    st.error("Delete failed (slot not found).")
                            except Exception as exc:
                                st.error(f"Delete failed: {exc}")
                else:
                    st.info(
                        "This slot is currently active. Edits are blocked "
                        "until the slot's end time. You can still edit "
                        "other slots."
                    )

    # --- Add new slot ---
    st.subheader("Add New Slot")
    with st.form("_sched_add_form", clear_on_submit=True):
        new_id = st.text_input(
            "Slot ID (unique, e.g. slot_4)",
            value=f"slot_{len(slots) + 1}",
            key="_sched_new_id",
        )
        new_subject = st.text_input(
            "Subject Name", value="", key="_sched_new_subject",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            new_start = st.text_input(
                "Start Time (HH:MM)", value="15:00", key="_sched_new_start",
            )
        with col_b:
            new_end = st.text_input(
                "End Time (HH:MM)", value="17:00", key="_sched_new_end",
            )
        submitted = st.form_submit_button("Add Slot", type="primary")
        if submitted:
            t_start = _parse_hhmm(new_start)
            t_end = _parse_hhmm(new_end)
            if not new_id.strip():
                st.error("Slot ID cannot be empty.")
            elif t_start is None or t_end is None:
                st.error("Invalid time format. Use HH:MM (24-hour).")
            elif t_start >= t_end:
                st.error("Start time must be before end time.")
            elif not new_subject.strip():
                st.error("Subject name cannot be empty.")
            else:
                try:
                    new_slot = Slot(
                        slot_id=new_id.strip(),
                        subject_name=new_subject.strip(),
                        start_time=new_start.strip(),
                        end_time=new_end.strip(),
                        expected_students_file=(
                            f"storage/student_lists/{new_id.strip()}.txt"
                        ),
                    )
                    sm.add_slot(new_slot)
                    st.success(f"Added slot {new_id}.")
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Add failed: {exc}")
                except Exception as exc:
                    st.error(f"Add failed: {exc}")

    # --- Grace period ---
    st.subheader("Grace Period")
    st.caption(
        "Minutes of early grace. Students entering up to N minutes BEFORE "
        "the slot starts are counted as attended. Default: 10 minutes."
    )
    cur_grace = sm.grace_minutes
    new_grace = st.number_input(
        "Grace Minutes", min_value=0, max_value=60, value=cur_grace, step=1,
        key="_sched_grace",
    )
    if new_grace != cur_grace:
        # Persist via direct config update -- the ScheduleManager doesn't
        # expose a setter, so we update the in-memory config and persist.
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            raw.setdefault("class_scheduling", {})["grace_minutes"] = int(new_grace)
            tmp = config_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False,
                               allow_unicode=True, width=100)
            os.replace(tmp, config_path)
            st.success(f"Grace period updated to {new_grace} minutes.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to update grace period: {exc}")


# ---------------------------------------------------------------------------
# Page 2: Student List Management.
# ---------------------------------------------------------------------------
def render_students_page(
    config: Dict[str, Any],
    config_path: str,
) -> None:
    """Upload/override/delete per-slot student lists."""
    st.header("👥 Student List Management")
    st.caption(
        "Each slot has a .txt file listing the expected student IDs (one "
        "per line). Upload a file, paste IDs, or delete the list."
    )

    err = _ensure_imports()
    if err:
        st.error(err)
        return

    sm = _build_schedule_manager(config, config_path)
    if sm is None:
        st.error("Could not initialize ScheduleManager.")
        return

    slots = sm.list_slots()
    if not slots:
        st.info("No slots configured. Go to the Schedule page first.")
        return

    # Slot selector.
    slot_options = {s.slot_id: f"{s.subject_name} ({s.start_time}-{s.end_time})"
                    for s in slots}
    selected_slot_id = st.selectbox(
        "Select Slot", options=list(slot_options.keys()),
        format_func=lambda sid: slot_options[sid],
        key="_students_slot_select",
    )
    if not selected_slot_id:
        return

    slot = sm.get_slot(selected_slot_id)
    if slot is None:
        st.error(f"Slot {selected_slot_id} not found.")
        return

    # Edit-lock check.
    is_active = sm.is_slot_edit_locked(selected_slot_id, _dt.datetime.now())
    if is_active:
        st.warning(
            f"🔒 Slot {slot.subject_name} is currently ACTIVE. "
            "Student list edits are blocked until the slot ends."
        )
        # Still show the current list (read-only).
        current_ids = sm.load_expected_students(selected_slot_id)
        st.subheader(f"Current List ({len(current_ids)} students)")
        if current_ids:
            st.text("\n".join(current_ids))
        else:
            st.info("No student IDs in this slot's list.")
        return

    # Current list display.
    current_ids = sm.load_expected_students(selected_slot_id)
    st.subheader(f"Current List for {slot.subject_name} ({len(current_ids)} students)")
    if current_ids:
        if _PANDAS_AVAILABLE:
            st.dataframe(pd.DataFrame({"student_id": current_ids}), use_container_width=True)
        else:
            st.text("\n".join(current_ids))
    else:
        st.info("No student IDs in this slot's list yet.")

    st.divider()

    # --- Upload via file uploader ---
    st.subheader("Upload .txt / .csv File")
    st.caption(
        "File format: one student ID per line. Lines starting with '#' are "
        "treated as comments. For .csv, only the FIRST column is read."
    )
    uploaded = st.file_uploader(
        "Choose a .txt or .csv file",
        type=["txt", "csv"],
        key=f"_students_upload_{selected_slot_id}",
    )
    if uploaded is not None:
        try:
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            ids = _parse_student_ids_text(content)
            st.session_state[f"_students_textarea_{selected_slot_id}"] = "\n".join(ids)
            st.success(f"Parsed {len(ids)} student IDs from uploaded file.")
        except Exception as exc:
            st.error(f"Failed to parse uploaded file: {exc}")

    # --- Paste IDs into textarea ---
    st.subheader("Paste Student IDs")
    st.caption("One ID per line. Lines starting with '#' are comments.")
    textarea_value = st.text_area(
        "Student IDs",
        value=st.session_state.get(f"_students_textarea_{selected_slot_id}", ""),
        height=200,
        key=f"_students_textarea_{selected_slot_id}",
    )

    col_save, col_clear, col_delete = st.columns(3)
    if col_save.button(
        "Save / Override List", type="primary",
        key=f"_students_save_{selected_slot_id}",
    ):
        ids = _parse_student_ids_text(textarea_value)
        if not ids:
            st.warning("No valid IDs to save.")
        else:
            try:
                sm.save_expected_students(
                    selected_slot_id, ids,
                    comment=f"Updated via dashboard at {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                st.success(f"Saved {len(ids)} student IDs to {slot.subject_name}'s list.")
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    if col_clear.button(
        "Clear Textarea",
        key=f"_students_clear_{selected_slot_id}",
    ):
        st.session_state[f"_students_textarea_{selected_slot_id}"] = ""
        st.rerun()

    if col_delete.button(
        "Delete List File",
        key=f"_students_delete_{selected_slot_id}",
    ):
        if sm.delete_expected_students(selected_slot_id):
            st.success(f"Deleted {slot.subject_name}'s student list file.")
            st.rerun()
        else:
            st.info("No file to delete (list was already empty).")


def _parse_student_ids_text(text: str) -> List[str]:
    """Parse a multi-line text block into a list of student IDs."""
    ids: List[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # For CSV-pasted content, take only the first column.
        if "," in line:
            line = line.split(",", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


# ---------------------------------------------------------------------------
# Page 3: Live Attendance (auto-refresh every 10s).
# ---------------------------------------------------------------------------
def render_live_attendance_page(
    config: Dict[str, Any],
    config_path: str,
) -> None:
    """Auto-refreshing view of the currently-active slot's attendance."""
    st.header("📊 Live Attendance")
    st.caption(
        "Auto-refreshes every 10 seconds. Shows the currently-active "
        "slot's attendance in real-time."
    )

    err = _ensure_imports()
    if err:
        st.error(err)
        return

    sm = _build_schedule_manager(config, config_path)
    if sm is None:
        st.error("Could not initialize ScheduleManager.")
        return
    ae = _build_attendance_engine(sm)
    if ae is None:
        st.error("Could not initialize AttendanceEngine.")
        return

    now = _dt.datetime.now()
    active_slot = sm.get_active_slot(now)

    # --- Active slot card ---
    if active_slot is None:
        st.info(
            f"🟢 FREE PERIOD (no active class). "
            f"Current time: {now.strftime('%H:%M:%S')}. "
            "Attendance tracking is idle until the next slot starts."
        )
    else:
        # Compute elapsed / remaining.
        start_dt = active_slot.start_dt(now.date())
        end_dt = active_slot.end_dt(now.date())
        eff_start_dt = active_slot.effective_start_dt(now.date(), sm.grace_minutes)
        elapsed = now - eff_start_dt
        remaining = end_dt - now
        elapsed_str = f"{int(elapsed.total_seconds() // 60)}m {int(elapsed.total_seconds() % 60)}s"
        remaining_str = f"{int(remaining.total_seconds() // 60)}m {int(remaining.total_seconds() % 60)}s"

        # Header card.
        st.markdown(f"### 📍 {active_slot.subject_name}")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Slot", f"{active_slot.start_time}–{active_slot.end_time}")
        col_b.metric("Elapsed", elapsed_str)
        col_c.metric("Remaining", remaining_str)
        col_d.metric("Slot ID", active_slot.slot_id)

        # Attendance summary.
        date_str = now.strftime("%Y-%m-%d")
        att_rows = ae.read_attendance_for_slot(active_slot.slot_id, date_str)
        total = len(att_rows)
        attended = sum(1 for r in att_rows if r.get("status") == "ATTENDED")
        rate = (attended / total * 100.0) if total > 0 else 0.0

        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("Total Expected", total)
        col_f.metric("Attended", attended)
        col_g.metric("Not Attended", total - attended)
        col_h.metric("Rate", f"{rate:.1f}%")

        # Progress bar.
        if total > 0:
            st.progress(attended / total)

        # Detailed table.
        st.subheader("Per-Student Status")
        if not att_rows:
            st.info("No attendance data yet for this slot.")
        else:
            if _PANDAS_AVAILABLE:
                df = pd.DataFrame(att_rows)
                # Reorder columns.
                display_cols = [
                    "student_id", "student_name", "status",
                    "first_seen_in_slot", "last_seen_in_slot",
                ]
                df = df[[c for c in display_cols if c in df.columns]]
                df.columns = [
                    "Student ID", "Student Name", "Status",
                    "First Seen", "Last Seen",
                ]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                # Fallback: plain table.
                st.table(att_rows)

        # Strangers seen during this slot.
        stranger_rows = ae.read_strangers_for_slot(active_slot.slot_id, date_str)
        if stranger_rows:
            st.subheader(f"Strangers Observed ({len(stranger_rows)})")
            if _PANDAS_AVAILABLE:
                sdf = pd.DataFrame(stranger_rows)
                display_cols = [
                    "stranger_label", "first_seen_in_slot",
                    "last_seen_in_slot", "snapshot_path",
                ]
                sdf = sdf[[c for c in display_cols if c in sdf.columns]]
                sdf.columns = ["Label", "First Seen", "Last Seen", "Snapshot"]
                st.dataframe(sdf, use_container_width=True, hide_index=True)
            else:
                st.table(stranger_rows)

    # --- Today's schedule overview ---
    st.divider()
    st.subheader("Today's Schedule")
    for slot in sm.list_slots():
        is_active_now = (active_slot is not None and slot.slot_id == active_slot.slot_id)
        marker = "🟢 " if is_active_now else "⚪ "
        date_str = now.strftime("%Y-%m-%d")
        slot_att = ae.read_attendance_for_slot(slot.slot_id, date_str)
        n_att = sum(1 for r in slot_att if r.get("status") == "ATTENDED")
        n_total = len(slot_att)
        st.markdown(
            f"{marker}**{slot.subject_name}** — {slot.start_time}–{slot.end_time} "
            f"({n_att}/{n_total} attended)"
        )

    # --- Auto-refresh ---
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=sm.poll_interval_s * 1000)
    except ImportError:
        # Fallback: sleep + rerun.
        time.sleep(sm.poll_interval_s)
        st.rerun()
    except Exception as exc:
        logger.warning("Auto-refresh failed: %s -- using sleep+rerun fallback", exc)
        time.sleep(sm.poll_interval_s)
        st.rerun()


# ---------------------------------------------------------------------------
# Page 4: Reports & PDF Download.
# ---------------------------------------------------------------------------
def render_reports_page(
    config: Dict[str, Any],
    config_path: str,
) -> None:
    """Date + subject picker -> preview + download PDF or XLSX."""
    st.header("📋 Reports & PDF Download")
    st.caption(
        "Generate a per-class, per-date attendance overview. Two formats: "
        "**PDF** includes the strangers section (for internal review); "
        "**XLSX** is a clean two-sheet workbook (Summary + Attendance) "
        "with no stranger information — the version to hand to the "
        "lecturer. Filename: `Overview_Class_{Subject}_{Date}.{pdf|xlsx}`"
    )

    err = _ensure_imports()
    if err:
        st.error(err)
        return

    if not pdf_is_available():
        st.error(
            "PDF generation requires `reportlab`. Install with: "
            "`pip install reportlab`"
        )
        return

    sm = _build_schedule_manager(config, config_path)
    if sm is None:
        st.error("Could not initialize ScheduleManager.")
        return
    ae = _build_attendance_engine(sm)
    if ae is None:
        st.error("Could not initialize AttendanceEngine.")
        return

    slots = sm.list_slots()
    if not slots:
        st.info("No slots configured. Go to the Schedule page first.")
        return

    # --- Date picker ---
    today = _dt.date.today()
    selected_date = st.date_input(
        "Select Date", value=today, max_value=today,
        key="_reports_date",
    )
    date_str = selected_date.strftime("%Y-%m-%d")

    # --- Subject dropdown ---
    slot_options = {s.slot_id: f"{s.subject_name} ({s.start_time}-{s.end_time})"
                    for s in slots}
    selected_slot_id = st.selectbox(
        "Select Class", options=list(slot_options.keys()),
        format_func=lambda sid: slot_options[sid],
        key="_reports_slot_select",
    )
    if not selected_slot_id:
        return
    slot = sm.get_slot(selected_slot_id)
    if slot is None:
        st.error(f"Slot {selected_slot_id} not found.")
        return

    # --- Preview ---
    att_rows = ae.read_attendance_for_slot(selected_slot_id, date_str)
    stranger_rows = ae.read_strangers_for_slot(selected_slot_id, date_str)

    total = len(att_rows)
    attended = sum(1 for r in att_rows if r.get("status") == "ATTENDED")
    not_attended = total - attended
    rate = (attended / total * 100.0) if total > 0 else 0.0

    st.subheader("Preview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Expected", total)
    col_b.metric("Attended", attended)
    col_c.metric("Not Attended", not_attended)
    col_d.metric("Rate", f"{rate:.1f}%")

    if att_rows:
        if _PANDAS_AVAILABLE:
            df = pd.DataFrame(att_rows)
            display_cols = [
                "student_id", "student_name", "status",
                "first_seen_in_slot", "last_seen_in_slot",
            ]
            df = df[[c for c in display_cols if c in df.columns]]
            df.columns = [
                "Student ID", "Student Name", "Status",
                "First Seen", "Last Seen",
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.table(att_rows)
    else:
        st.info("No attendance data for this date/slot combination.")

    if stranger_rows:
        st.caption(f"Plus {len(stranger_rows)} strangers observed during this slot.")

    # --- Generate + download ---
    st.subheader("Download Report")
    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        gen_pdf_btn = st.button(
            "📥 Generate PDF",
            type="primary",
            key="_reports_gen_pdf",
            help="Includes the strangers section (for internal review).",
        )
    with col_xlsx:
        gen_xlsx_btn = st.button(
            "📊 Generate XLSX",
            type="secondary",
            key="_reports_gen_xlsx",
            help="Clean two-sheet workbook for the lecturer. No strangers.",
            disabled=not xlsx_is_available(),
        )

    if gen_pdf_btn:
        try:
            # Refresh stranger snapshot paths before generating the PDF.
            # This recovers photos for strangers whose snapshot_path was
            # empty in the CSV (e.g. recorded before the snapshot-path-
            # upgrade fix, or whose snapshot was written by the worker
            # after the cache entry was created).
            try:
                # HOTFIX-3 :: Build explicit multi-root search list so
                # refresh succeeds even if the dashboard's _PROJECT_ROOT
                # differs from where main.py wrote the snapshots.
                search_roots = []
                _snap_root_default = os.path.join(_PROJECT_ROOT, "storage", "snap_strangers")
                search_roots.append(_snap_root_default)
                _cwd = os.getcwd()
                if os.path.abspath(_cwd) != os.path.abspath(_PROJECT_ROOT):
                    search_roots.append(os.path.join(_cwd, "storage", "snap_strangers"))
                # Also honor snap_strangers.output_dir from config if set.
                _snap_cfg = (config or {}).get("snap_strangers", {}) or {}
                _output_dir = _snap_cfg.get("output_dir", "")
                if _output_dir:
                    if os.path.isabs(_output_dir):
                        search_roots.append(_output_dir)
                    else:
                        search_roots.append(os.path.join(_PROJECT_ROOT, _output_dir))
                        search_roots.append(os.path.join(_cwd, _output_dir))
                # De-dup preserving order.
                _seen = set()
                search_roots = [r for r in search_roots
                                if not (os.path.abspath(r) in _seen or _seen.add(os.path.abspath(r)))]

                refreshed = ae.refresh_stranger_snapshots(
                    slot.slot_id, date_str,
                    snapshot_search_roots=search_roots,
                )
                # Re-read the stranger rows with updated paths.
                stranger_rows = ae.read_strangers_for_slot(slot.slot_id, date_str)

                # HOTFIX-3 :: Show the user a clear summary so they can
                # diagnose when photos still don't appear.
                _total_strangers = len(stranger_rows)
                _with_photo = sum(
                    1 for r in stranger_rows
                    if r.get("snapshot_path") and os.path.isfile(r["snapshot_path"])
                )
                _without_photo = _total_strangers - _with_photo
                if refreshed > 0:
                    st.success(
                        f"Recovered {refreshed} stranger photo(s) from disk. "
                        f"Now {_with_photo}/{_total_strangers} strangers have photos."
                    )
                if _without_photo > 0:
                    missing_labels = [
                        r.get("stranger_label", "?")
                        for r in stranger_rows
                        if not (r.get("snapshot_path") and os.path.isfile(r["snapshot_path"]))
                    ]
                    st.warning(
                        f"{_without_photo}/{_total_strangers} strangers still have "
                        f"no photo: {missing_labels}.  "
                        f"Searched: `{search_roots}`. Verify these directories "
                        f"exist on disk and contain `*_STRANGER_*.png` files."
                    )
            except Exception as refresh_exc:
                logger.warning(
                    "refresh_stranger_snapshots failed (non-fatal): %s",
                    refresh_exc,
                    exc_info=True,
                )
                st.warning(
                    f"Stranger photo refresh failed (non-fatal): {refresh_exc}"
                )
                # HOTFIX-3 :: Even if refresh raised, still build a
                # search_roots list so the PDF generator can do its
                # own fallback search.
                search_roots = [os.path.join(_PROJECT_ROOT, "storage", "snap_strangers")]

            # HOTFIX-3 :: Pass search_roots to the PDF generator so it
            # can do a last-ditch filename search for any stranger whose
            # snapshot_path is STILL empty/invalid after refresh. This
            # makes the PDF self-healing even if refresh found nothing.
            pdf_path = generate_attendance_pdf(
                subject_name=slot.subject_name,
                slot_id=slot.slot_id,
                date_str=date_str,
                start_time=slot.start_time,
                end_time=slot.end_time,
                attendance_rows=att_rows,
                stranger_rows=stranger_rows,
                output_dir=sm.reports_dir,
                snapshot_search_roots=search_roots,
            )
            st.success(f"PDF generated: {os.path.basename(pdf_path)}")
            # Read bytes for download.
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                key="_reports_download_pdf",
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    if gen_xlsx_btn:
        try:
            xlsx_path = generate_attendance_xlsx(
                subject_name=slot.subject_name,
                slot_id=slot.slot_id,
                date_str=date_str,
                start_time=slot.start_time,
                end_time=slot.end_time,
                attendance_rows=att_rows,
                output_dir=sm.reports_dir,
                # stranger_rows / include_strangers deliberately omitted --
                # the XLSX generator ignores them anyway (lecturer-facing
                # report). We pass None to make the intent explicit.
                stranger_rows=None,
            )
            st.success(f"XLSX generated: {os.path.basename(xlsx_path)}")
            with open(xlsx_path, "rb") as f:
                xlsx_bytes = f.read()
            st.download_button(
                label="📊 Download XLSX",
                data=xlsx_bytes,
                file_name=os.path.basename(xlsx_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="_reports_download_xlsx",
            )
            st.toast(
                "📊 XLSX ready (lecturer version — no strangers).",
                icon="📊",
            )
        except Exception as exc:
            st.error(f"XLSX generation failed: {exc}")
            logger.error(
                "XLSX generation failed: %s\n%s",
                exc, traceback.format_exc(),
            )

    # --- List existing PDFs + XLSXs ---
    st.divider()
    st.subheader("Previously Generated Reports")
    reports_dir = sm.reports_dir
    if os.path.isdir(reports_dir):
        existing = sorted(
            f for f in os.listdir(reports_dir)
            if f.endswith(".pdf") or f.endswith(".xlsx")
        )
        if not existing:
            st.info("No previously generated reports.")
        else:
            for fname in existing[-20:]:  # show last 20
                fpath = os.path.join(reports_dir, fname)
                try:
                    stat = os.path.isfile(fpath) and os.stat(fpath)
                    if not stat:
                        continue
                    mtime = _dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    size_kb = stat.st_size / 1024.0
                    col_name, col_meta, col_dl = st.columns([3, 2, 1])
                    icon = "📊" if fname.lower().endswith(".xlsx") else "📄"
                    col_name.markdown(f"{icon} `{fname}`")
                    col_meta.caption(f"{mtime} | {size_kb:.1f} KB")
                    with open(fpath, "rb") as f:
                        col_dl.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=fname,
                            mime=(
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                if fname.lower().endswith(".xlsx")
                                else "application/pdf"
                            ),
                            key=f"_reports_existing_dl_{fname}",
                        )
                except OSError:
                    pass
    else:
        st.info("Reports directory does not exist yet.")
