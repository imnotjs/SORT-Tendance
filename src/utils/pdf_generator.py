"""
SORT-tendance :: PDF Report Generator
======================================
Generates per-slot, per-date attendance overview PDFs.

Output filename pattern:
    Overview_Class_{SanitizedSubject}_{YYYY-MM-DD}.pdf
e.g.
    Overview_Class_WebDesign_2026-07-20.pdf

PDF content:
  - Header: SORT-tendance :: Class Attendance Overview
  - Subject / Date / Time Slot / Generated-at metadata
  - Summary block: total expected, attended, not attended, rate %
  - Detailed attendance table (one row per expected student)
  - Strangers section: a table of strangers seen during the slot,
    each row showing the stranger's snapshot photo (if available),
    label, first-seen, last-seen.

The generator reads from:
  - storage/attend_csv/attendance_{date}_{slot_id}.csv
  - storage/attend_csv/strangers_{date}_{slot_id}.csv

Writes to:
  - storage/attendance_reports/Overview_Class_{Subject}_{Date}.pdf

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sortendance.pdf_generator")

# ReportLab imports -- kept inside try/except so the module imports
# cleanly even on systems without reportlab (the dashboard will show
# a "PDF generation unavailable" message instead of crashing).
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    _REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REPORTLAB_AVAILABLE = False
    colors = None  # type: ignore
    SimpleDocTemplate = None  # type: ignore
    Paragraph = None  # type: ignore
    Spacer = None  # type: ignore
    Table = None  # type: ignore
    TableStyle = None  # type: ignore
    RLImage = None  # type: ignore
    getSampleStyleSheet = None  # type: ignore
    ParagraphStyle = None  # type: ignore
    A4 = None  # type: ignore
    cm = mm = 0  # type: ignore
    TA_CENTER = TA_LEFT = 0  # type: ignore


# ---------------------------------------------------------------------------
# Color palette.
# ---------------------------------------------------------------------------
_COLOR_HEADER_BG = colors.HexColor("#1F3A5F") if _REPORTLAB_AVAILABLE else None
_COLOR_HEADER_FG = colors.white if _REPORTLAB_AVAILABLE else None
_COLOR_ATTENDED = colors.HexColor("#D4EDDA") if _REPORTLAB_AVAILABLE else None
_COLOR_NOT_ATTENDED = colors.HexColor("#F8D7DA") if _REPORTLAB_AVAILABLE else None
_COLOR_SUMMARY_BG = colors.HexColor("#F1F3F5") if _REPORTLAB_AVAILABLE else None
_COLOR_TABLE_GRID = colors.HexColor("#CCCCCC") if _REPORTLAB_AVAILABLE else None
_COLOR_STRANGER_BG = colors.HexColor("#FFF3CD") if _REPORTLAB_AVAILABLE else None


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def generate_attendance_pdf(
    subject_name: str,
    slot_id: str,
    date_str: str,
    start_time: str,
    end_time: str,
    attendance_rows: List[Dict[str, Any]],
    stranger_rows: List[Dict[str, Any]],
    output_dir: str,
    snapshot_search_roots: Optional[List[str]] = None,
) -> str:
    """Generate a per-slot, per-date attendance overview PDF.

    Args:
        subject_name: e.g. "AI" or "Web Design".
        slot_id: e.g. "slot_2".
        date_str: e.g. "2026-07-20".
        start_time: e.g. "10:00".
        end_time: e.g. "12:00".
        attendance_rows: List of dicts (one per expected student) with
            keys: student_id, student_name, status, first_seen_in_slot,
            last_seen_in_slot.
        stranger_rows: List of dicts (one per stranger) with keys:
            stranger_label, first_seen_in_slot, last_seen_in_slot,
            snapshot_path.
        output_dir: Directory to write the PDF into (created if missing).

    Returns:
        Absolute path to the generated PDF.
    """
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is not installed. Install with: pip install reportlab"
        )

    os.makedirs(output_dir, exist_ok=True)
    sanitized = _sanitize_subject(subject_name)
    filename = f"Overview_Class_{sanitized}_{date_str}.pdf"
    output_path = os.path.join(output_dir, filename)

    # Build the PDF document.
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"Class Attendance Overview - {subject_name} - {date_str}",
        author="SORT-tendance",
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "SortTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=_COLOR_HEADER_BG,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "SortSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    style_section = ParagraphStyle(
        "SortSection",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=_COLOR_HEADER_BG,
        spaceBefore=12,
        spaceAfter=8,
    )
    style_meta = ParagraphStyle(
        "SortMeta",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
    )
    style_footer = ParagraphStyle(
        "SortFooter",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    story: List[Any] = []

    # --- Header ---
    story.append(Paragraph("SORT-tendance :: Class Attendance Overview", style_title))
    story.append(Paragraph(
        f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        style_subtitle,
    ))

    # --- Metadata block ---
    total = len(attendance_rows)
    attended = sum(1 for r in attendance_rows if r.get("status") == "ATTENDED")
    not_attended = total - attended
    rate = (attended / total * 100.0) if total > 0 else 0.0

    meta_data = [
        ["Subject:", subject_name, "Date:", date_str],
        ["Time Slot:", f"{start_time} - {end_time}", "Slot ID:", slot_id],
        ["Total Expected:", str(total), "Attended:", str(attended)],
        ["Not Attended:", str(not_attended), "Attendance Rate:", f"{rate:.1f}%"],
    ]
    meta_table = Table(meta_data, colWidths=[3 * cm, 5 * cm, 3 * cm, 5 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), _COLOR_HEADER_BG),
        ("TEXTCOLOR", (2, 0), (2, -1), _COLOR_HEADER_BG),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), _COLOR_SUMMARY_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, _COLOR_TABLE_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, _COLOR_TABLE_GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # --- Detailed attendance table ---
    story.append(Paragraph("Detailed Attendance", style_section))

    if total == 0:
        story.append(Paragraph(
            "<i>No students were on the expected list for this slot.</i>",
            style_meta,
        ))
    else:
        header = ["Student ID", "Student Name", "Status", "First Seen", "Last Seen"]
        table_data = [header]
        for r in attendance_rows:
            table_data.append([
                r.get("student_id", ""),
                r.get("student_name", "") or "(unknown)",
                "ATTENDED" if r.get("status") == "ATTENDED" else "NOT ATTENDED",
                r.get("first_seen_in_slot", "") or "—",
                r.get("last_seen_in_slot", "") or "—",
            ])
        att_table = Table(
            table_data,
            colWidths=[3 * cm, 5 * cm, 3 * cm, 2.5 * cm, 2.5 * cm],
            repeatRows=1,
        )
        # Build style commands: header row + per-row background by status.
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 0), (-1, 0), _COLOR_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), _COLOR_HEADER_FG),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.5, _COLOR_TABLE_GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, _COLOR_TABLE_GRID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        # Per-row background by status (skip header row 0).
        for row_idx, r in enumerate(attendance_rows, start=1):
            if r.get("status") == "ATTENDED":
                style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), _COLOR_ATTENDED))
            else:
                style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), _COLOR_NOT_ATTENDED))
        att_table.setStyle(TableStyle(style_cmds))
        story.append(att_table)

    # --- Strangers section ---
    if stranger_rows:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            f"Strangers Observed During Class ({len(stranger_rows)})",
            style_section,
        ))
        stranger_header = ["Photo", "Stranger Label", "First Seen", "Last Seen"]
        stranger_data = [stranger_header]
        stranger_styles = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 0), (-1, 0), _COLOR_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), _COLOR_HEADER_FG),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.5, _COLOR_TABLE_GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, _COLOR_TABLE_GRID),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("BACKGROUND", (0, 1), (-1, -1), _COLOR_STRANGER_BG),
        ]
        # HOTFIX-3 :: Build a label->path index ONCE so each stranger
        # row can do a last-ditch filename search if its snapshot_path
        # is empty or points to a missing file. This makes the PDF
        # self-healing even if the dashboard's refresh step failed.
        _fallback_index: Dict[str, List[str]] = {}
        if snapshot_search_roots:
            import re as _re
            _label_re = _re.compile(
                r"_STRANGER_(?P<label>.+?)(?:_CLEARSHOT_\d+)?\.png$",
                _re.IGNORECASE,
            )
            for _root in snapshot_search_roots:
                if not os.path.isdir(_root):
                    continue
                for _dp, _dn, _fns in os.walk(_root):
                    for _fn in _fns:
                        if not _fn.lower().endswith(".png"):
                            continue
                        _m = _label_re.search(_fn)
                        if not _m:
                            continue
                        _raw = _m.group("label")
                        _norm = _raw.strip("[](){} ").strip("_").strip().lower()
                        if not _norm:
                            continue
                        _fallback_index.setdefault(_norm, []).append(
                            os.path.join(_dp, _fn)
                        )

        for r in stranger_rows:
            photo_path = r.get("snapshot_path", "")
            photo_cell: Any
            # HOTFIX-3 :: If snapshot_path is empty/invalid, try the
            # fallback index before giving up and showing "(no photo)".
            if not (photo_path and os.path.isfile(photo_path)) and _fallback_index:
                _label = str(r.get("stranger_label", "") or "")
                _norm = _label.strip("[](){} ").strip("_").strip().lower()
                _candidates = _fallback_index.get(_norm, [])
                if _candidates:
                    _candidates.sort(
                        key=lambda p: os.path.getmtime(p), reverse=True,
                    )
                    photo_path = _candidates[0]
                    logger.info(
                        "pdf_generator: fallback recovered photo for %s -> %s",
                        _label, photo_path,
                    )
            if photo_path and os.path.isfile(photo_path):
                try:
                    # Scale image to fit ~2.5cm wide while preserving aspect.
                    img = RLImage(photo_path, width=2.5 * cm, height=2.5 * cm)
                    photo_cell = img
                except Exception as exc:
                    logger.warning(
                        "pdf_generator: could not embed image %s: %s",
                        photo_path, exc,
                    )
                    photo_cell = "(image error)"
            else:
                photo_cell = "(no photo)"
            stranger_data.append([
                photo_cell,
                r.get("stranger_label", ""),
                r.get("first_seen_in_slot", "") or "—",
                r.get("last_seen_in_slot", "") or "—",
            ])
        stranger_table = Table(
            stranger_data,
            colWidths=[3 * cm, 5 * cm, 4 * cm, 4 * cm],
            repeatRows=1,
        )
        stranger_table.setStyle(TableStyle(stranger_styles))
        story.append(stranger_table)

    # --- Footer ---
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "Generated by SORT-tendance | Page 1",
        style_footer,
    ))

    # Build the PDF.
    doc.build(story)
    logger.info(
        "pdf_generator: wrote %s | attended=%d/%d | strangers=%d",
        output_path, attended, total, len(stranger_rows),
    )
    return output_path


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _sanitize_subject(name: str) -> str:
    """Filesystem-safe subject name. 'Web Design' -> 'WebDesign'."""
    out = []
    for ch in str(name):
        if ch.isalnum():
            out.append(ch)
    return "".join(out) or "Unknown"


def is_available() -> bool:
    """Return True if reportlab is installed and PDF generation is possible."""
    return _REPORTLAB_AVAILABLE
