# SORT-tendance — Operator How-To Guide

> Face-recognition attendance system for lab / classroom sessions.
> This guide covers day-to-day operation: startup, enrollment, renaming
> students, exporting reports, and common troubleshooting.

---

## 1. Quick Start

### Start the whole system (main + dashboard)

```bash
cd C:\Hezekiah\SORT-tendance
python start_sortendance.py
```

Or double-click **`start_sortendance.bat`**.

This launches **two** child processes in their own console windows:

| Window title              | Process                                | Role                                      |
| ------------------------- | -------------------------------------- | ----------------------------------------- |
| `SORT-tendance :: MAIN`   | `python main.py`                       | Camera + face detection + tracking        |
| `SORT-tendance :: DASHBOARD` | `streamlit run ui/dashboard.py`     | Web UI on http://127.0.0.1:8501           |

Open **http://127.0.0.1:8501** in your browser to use the dashboard.

### Start only one component

```bash
python start_sortendance.py main        # only the camera + AI pipeline
python start_sortendance.py dashboard   # only the web UI
```

### Stop everything

Press `Ctrl+C` in the supervisor console window. The supervisor will
gracefully ask both children to exit (release VRAM, camera, ONNX
sessions) and only force-kill if a child refuses within the timeout.

---

## 2. Dashboard Navigation

The left sidebar has nine pages:

| Page              | What it does                                                |
| ----------------- | ----------------------------------------------------------- |
| **Main**          | 3-column live view: performance, attendance, strangers      |
| **Live Attendance** | Real-time class roster with auto-refresh                  |
| **Schedule**      | Configure class slots (6 AM / 6 PM sessions)               |
| **Students**      | Browse the student roster for scheduling                    |
| **Enroll Student**| Enroll a NEW student from 2 photos                          |
| **Face Database** | Browse DB, add photos to existing student, **rename** students |
| **Reports**       | Export PDF + XLSX attendance reports                        |
| **Event Log**     | Timestamped system events                                   |
| **Stranger Gallery** | All unrecognized faces captured during the session       |

Click any page name to switch. The active page is highlighted with the
accent color.

---

## 3. Main Page (Live Monitoring)

Three columns, auto-refreshing every 2 seconds:

- **Left — Performance Monitor**: FPS, GPU/CPU load, VRAM usage,
  detection latency, queue depth.
- **Middle — Live Attendance Journal**: rolling list of recognized
  students with their last-seen timestamp.
- **Right — Stranger Gallery**: thumbnails of unrecognized faces
  captured in the current session.

No actions required here — it's pure monitoring.

---

## 4. Enroll a New Student

Use this when a student appears for the first time and is NOT in the
database.

1. Go to **Enroll Student** in the sidebar.
2. Fill in:
   - **Student Number (NRP)** — e.g. `2210503008`. Must be unique.
   - **Student Name** — e.g. `Budi Santoso`.
3. Upload **exactly 2 photos**:
   - Photo 1: frontal, neutral expression, good lighting.
   - Photo 2: subtle expression variation (slight smile) or slight
     angle change.
4. Click **Enroll**.

The enrollment runs as a background subprocess (`scripts/enroll.py
--single-student`). Status appears inline:

| Exit code | Meaning           | What to do                              |
| --------- | ----------------- | --------------------------------------- |
| 0         | Success           | Wait ~15s — `main.py` auto-restarts     |
| 10        | Rate-limited      | Last enroll was <60s ago. Wait & retry. |
| 20        | Duplicate ID      | That NRP already exists. Use Face DB.   |
| 21        | Duplicate face    | That face already matches someone.      |
| 30        | Bad input         | Photo too small / no face detected.     |
| 40        | Engine error      | Check logs; restart supervisor.         |

After a successful enroll, `main.py` is automatically restarted by the
supervisor so it picks up the new embeddings. The dashboard stays up
so you see the success toast.

> **Rate limit**: 60 seconds between any two enroll/add operations.
> This prevents VRAM contention between the registration engine
> (640×640 det_size) and the live engine (320×320).

---

## 5. Face Database Page

The most-used page for ongoing maintenance. Three sections:

### 5.1 Browse the registry

Top of the page shows a table of every enrolled student:

| Student ID | Name     | Embeddings | Enrolled At        | Anchor    |
| ---------- | -------- | ---------- | ------------------ | --------- |
| 2210503008 | Budi     | 2 / 25     | 2026-07-01T08:00:00| abc123def456 |
| DTI-1      | Operator | 2 / 25     | 2026-07-01T08:00:00| ghi789... |

Click **🔄 Refresh** to reload after an external change.

### 5.2 Add Photos to an Existing Student

Use this to improve recognition for a student who's already enrolled
(recognize them in more angles / lighting / expressions).

1. Scroll to **➕ Add Photos to Existing Student**.
2. Pick the student from the dropdown.
3. Upload **1–5 new photos** (different angles / lighting / glasses
   on/off help the most).
4. Click **Add Photos**.

Behind the scenes this runs `scripts/enroll.py --add-to <student_id>`,
which appends new embeddings to the student's existing set (up to the
`profile_capacity` of 25). After success, `main.py` auto-restarts in
~15s to load the expanded embeddings.

> If the student is already at capacity (25/25), you'll get a clear
> error. You cannot delete individual embeddings — the system is
> **add-only** by design (deleting risk corrupting the mean vector).

### 5.3 Edit Student Info (Rename)

This is the rename feature. Lets you change **Student Number** and/or
**Name** for any enrolled student, with confirmation.

#### Step 1 — Edit

Scroll to **✏️ Edit Student Info**. Two editable tables appear:

- **Regular students**: both **Student Number** and **Name** columns
  are editable.
- **Codename students** (`DTI-1`, `DTI-2`): **Student Number** column
  is **locked** (codename protection). Only **Name** is editable.

Click any editable cell and type the new value. You can edit multiple
rows at once.

#### Step 2 — Review Changes

Click **Review Changes**. A diff table appears showing, for every
changed row:

| Student Number             | Name                  | Folder (data/student_faces/)              |
| -------------------------- | --------------------- | ----------------------------------------- |
| `2210503008` → `2210503009`| `Budi` → `Budi S.`   | `2210503008_Budi` → `2210503009_Budi S.`  |
| `DTI-1` (unchanged)        | `Operator` → `Bayu`  | `DTI-1_Operator` → `DTI-1_Bayu`           |

A yellow warning reminds you this is permanent.

The **Confirm & Save** button is disabled if there are:
- Empty Student Numbers
- Codename ID-change attempts (shouldn't happen — column is locked,
  but the backend also enforces it)
- ID collisions with another existing student
- Duplicate new IDs within the same batch

#### Step 3 — Confirm & Save

Click **Confirm & Save**. The system:

1. Calls `EnrollmentService.update_student_profile()` for each changed
   row — atomically re-keys the pickle (temp + rename, never corrupts
   on crash).
2. Renames the on-disk raw-photos folder
   `data/student_faces/{old_id}_{old_name}` →
   `data/student_faces/{new_id}_{new_name}` **if it exists**.
   - If the new folder name already exists, the rename is **skipped**
     (never clobbers an existing directory) — the pickle is still
     updated.
   - If there's no raw-photos folder (e.g. enrolled via dashboard
     upload only), the rename is silently skipped.
3. Writes `data/.restart_main_requested` — the supervisor picks this
   up within 5s, waits 15s (for VRAM release), then gracefully
   restarts **only** `main.py`. The dashboard keeps running so you
   see the success toast.
4. Shows: `✅ Saved N change(s). ID renamed: X | Name changed: Y |
   Folders renamed: Z.`

The table refreshes automatically.

#### Codename rules (DTI-1 / DTI-2)

| Action                                      | Allowed? |
| ------------------------------------------- | -------- |
| Change DTI-1's Student Number               | ❌ No    |
| Change DTI-1's Name (e.g. → `DTI-1_Bayu`)   | ✅ Yes   |
| Change a regular student's Student Number   | ✅ Yes   |
| Change a regular student's Name             | ✅ Yes   |

The codename set is `{"DTI-1", "DTI-2"}`. To add more codenames,
edit `_CODENAME_IDS` in `src/utils/database_manager.py`.

---

## 6. Schedule Page

Configure class sessions. The system uses a 12-hour cycle with
**6 AM** and **6 PM** boundaries (configurable in
`config/config.yaml` under `session_rotation`).

At each boundary, the supervisor gracefully restarts both `main.py`
and `dashboard.py` to release accumulated memory (ONNX arenas, CUDA
allocator fragmentation, Streamlit session state).

You don't need to do anything here for normal operation — it's for
viewing / adjusting the schedule and slot student lists.

---

## 7. Students Page

Browse the student roster used for class scheduling. This is separate
from the face database — it's the list of students expected in each
class slot, used to compute attendance percentages.

Student lists per slot live in `storage/student_lists/slot_1.txt`,
`slot_2.txt`, `slot_3.txt`.

---

## 8. Live Attendance Page

Real-time class roster with auto-refresh (every 10s). Shows each
expected student and whether they've been recognized in the current
session window.

This page has its **own** auto-refresh so it doesn't fight with the
main 2s cycle.

---

## 9. Reports Page

Export attendance reports for a session.

### PDF Report
- Includes: session metadata, attendance summary, per-student status,
  stranger captures (with thumbnails if available).
- Click **Export PDF**. The file downloads to your browser's default
  download location.

### XLSX Report
- **Two sheets**:
  1. **Summary** — aggregate counts (present / absent / late), with
     live `COUNTIF` formulas so the totals update if you edit status
     cells.
  2. **Attendance** — one row per student with color-coded status
     (green = present, yellow = late, red = absent).
- **Strangers are EXCLUDED** from the XLSX (kept simple for the
  lecturer).
- Features: freeze panes, data validation dropdowns on the status
  column, color-coded rows.
- Click **Export XLSX**.

> If `openpyxl` is not installed, the XLSX button is disabled with a
> hint: `pip install openpyxl`.

---

## 10. Event Log Page

Timestamped system events: enrollments, restarts, stranger alerts,
schedule transitions. Useful for post-session review.

---

## 11. Stranger Gallery Page

Every unrecognized face captured during the session, grouped by track
ID. Each entry shows:

- Thumbnail
- Track ID
- First-seen / last-seen timestamps
- Best clearshot (if multiple captures exist for the same track)

Use this to identify repeat strangers (same track across the session)
or one-off walk-bys.

---

## 12. Supervisor Behavior

### Automatic restarts

| Trigger                          | What restarts        | Delay  |
| -------------------------------- | -------------------- | ------ |
| New student enrolled             | `main.py` only       | 15s    |
| Photos added to existing student | `main.py` only       | 15s    |
| Student info renamed (this page) | `main.py` only       | 15s    |
| 6 AM / 6 PM session boundary     | Both `main` + `dashboard` | immediate (graceful) |
| Child crash (non-zero exit)      | The crashed child    | 15s cooldown |

The dashboard is **never** restarted by enrollment/rename operations —
only `main.py` is. This is intentional so you see the success toast.

### Restart flag file

When the dashboard writes `data/.restart_main_requested` (JSON
payload with `requested_at`, `delay_s`, `reason`), the supervisor:

1. Polls every cycle (~5s).
2. Once `delay_s` elapses since `requested_at`, gracefully stops
   `main.py` (CTRL_BREAK_EVENT on Windows, SIGINT on Linux).
3. Waits for VRAM/camera release.
4. Respawns `main.py` — it re-reads `data/student_db.pickle` on
   startup.
5. Deletes the flag file.

You can manually trigger a main-only restart by creating the file:

```bash
echo '{"requested_at": 0, "delay_s": 0, "reason": "manual"}' > data/.restart_main_requested
```

(Use `requested_at: 0` to fire immediately on the next poll.)

---

## 13. File Layout

```
SORT-tendance/
├── start_sortendance.py        # Supervisor (launches main + dashboard)
├── main.py                     # Camera + AI pipeline
├── config/
│   └── config.yaml             # Single source of truth for all params
├── ui/
│   ├── dashboard.py            # Streamlit web UI
│   └── scheduling_pages.py     # Schedule / Live Attendance / Reports
├── src/
│   ├── core/
│   │   ├── attendance_engine.py
│   │   ├── identity_matcher.py
│   │   ├── tracking_engine.py
│   │   └── ...
│   └── utils/
│       ├── database_manager.py # EnrollmentService (enroll + rename)
│       ├── pdf_generator.py
│       ├── xlsx_generator.py
│       └── ...
├── scripts/
│   ├── enroll.py               # CLI: --single-student, --add-to
│   └── list_students.py        # CLI: list enrolled students
├── data/
│   ├── student_db.pickle       # THE face-embedding database
│   └── student_faces/          # Raw source photos (id_name/ folders)
│       ├── 2210503008_Budi/
│       ├── DTI-1_Operator/
│       └── ...
├── storage/
│   ├── student_lists/          # slot_1.txt, slot_2.txt, slot_3.txt
│   └── snap_strangers/         # Captured stranger photos (by date/session)
└── logs/                       # Supervisor + child logs (by date)
```

---

## 14. CLI Tools

### List enrolled students

```bash
python scripts/list_students.py            # human-readable table
python scripts/list_students.py --json     # machine-readable JSON
python scripts/list_students.py --verbose  # include embedding counts + timestamps
```

### Enroll via CLI (headless)

```bash
# New student
python scripts/enroll.py --single-student \
  --id 2210503008 \
  --name "Budi Santoso" \
  --photo photo1.jpg \
  --photo photo2.jpg

# Add photos to existing student
python scripts/enroll.py --add-to 2210503008 \
  --photo new1.jpg \
  --photo new2.jpg
```

Both honor the 60s rate limit and write the restart flag on success.

---

## 15. Troubleshooting

### "Edit Student Info panel failed: 'XYZ'"

This was a stale-cache bug fixed in the latest patch. If you still
see it, refresh the page (F5). If it persists, clear Streamlit cache:
`Ctrl+Shift+R` in the browser.

### `main.py` won't restart after enroll/rename

Check:

1. Does `data/.restart_main_requested` exist? If yes, the supervisor
   hasn't picked it up yet (wait 5s) or the `delay_s` hasn't elapsed.
2. Is the supervisor actually running? Check the supervisor console
   for `[supervisor]` log lines.
3. Manually delete the flag file and restart the supervisor:
   ```bash
   del data\.restart_main_requested
   python start_sortendance.py
   ```

### Dashboard shows "Performance column failed: ..."

Usually means `main.py` isn't running or isn't sending UDP telemetry.
Check the `SORT-tendance :: MAIN` console window for errors. Common
cause: camera in use by another app.

### Stranger photos show "(no photo)" in PDF

This was fixed in HOTFIX-3 (multi-root search). If it recurs, check
that `storage/snap_strangers/<date>/<session>/` exists and contains
the expected PNG files.

### BSOD (nvlddmkm.sys)

This is an **NVIDIA driver** issue, not a SORT-tendance bug. Fix
priority:

1. DDU (Display Driver Uninstaller) in safe mode → clean reinstall
   latest Studio driver.
2. Set `TdrDelay` to 8 (seconds) in registry:
   `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`
3. Check GPU thermals (<85°C under load).
4. Disable Hardware-Accelerated GPU Scheduling (HAGS) in Windows
   Graphics settings.
5. Update motherboard BIOS ( chipset / PCIe stability).

### XLSX export button disabled

Install openpyxl:

```bash
pip install openpyxl
```

Then refresh the dashboard page.

### VRAM OOM during enrollment

The registration engine uses `det_size=640×640` (higher quality than
the live engine's 320×320). If you OOM:

1. Make sure no other GPU-heavy app is running.
2. Wait for any pending `main.py` restart to complete before
   enrolling again.
3. The 60s rate limit exists precisely to prevent this — don't bypass
   it.

---

## 16. Config Quick Reference

All parameters live in `config/config.yaml`. Key sections:

| Section        | Purpose                                             |
| -------------- | --------------------------------------------------- |
| `hardware.gpu` | VRAM fraction (0.22 = ~1.76GB on 8GB card), device  |
| `enrollment`   | `student_faces_dir`, `db_pickle_path`, det_size, threshold |
| `dashboard`    | Host, port, refresh interval, stranger cap          |
| `main`         | AI queue size, render FPS, bbox throttle            |
| `session_rotation` | 6 AM / 6 PM restart hours                       |

Never edit values in Python source files — always change the YAML.
The supervisor and all children re-read config on startup.

---

## 17. Daily Workflow Cheat Sheet

```
START OF DAY
  1. python start_sortendance.py
  2. Open http://127.0.0.1:8501
  3. Verify Main page shows live camera + telemetry

DURING CLASS
  4. Monitor Live Attendance page
  5. Check Stranger Gallery for unexpected visitors
  6. If a known student isn't recognized:
       a. Face Database → Add Photos (1-5 new angles)
       b. Wait 15s for main.py restart
       c. Verify recognition improves

NEW STUDENT JOINS
  7. Enroll Student → fill ID + name + 2 photos
  8. Wait 15s for main.py restart
  9. Verify they appear in Live Attendance

TYPO IN STUDENT NAME / ID
  10. Face Database → Edit Student Info
  11. Edit the cell(s)
  12. Review Changes → Confirm & Save
  13. Wait 15s for main.py restart

END OF CLASS
  14. Reports → Export PDF (full report with strangers)
  15. Reports → Export XLSX (lecturer-friendly, no strangers)

END OF DAY
  16. Ctrl+C in supervisor console
  17. (Optional) Review logs/ for any anomalies
```

---

## 18. Getting Help

- **Logs**: `logs/supervisor_main_YYYY-MM-DD.log` and
  `logs/supervisor_dashboard_YYYY-MM-DD.log`
- **Backup**: `_pre_audit_backup_1784257447/` contains the
  pre-audit snapshot of all critical files.
- **Verify install**: `python scripts/smoke_test_imports.py` checks
  all module imports resolve.

---

*SORT-tendance Engineering — graduation build, July 2026.*
