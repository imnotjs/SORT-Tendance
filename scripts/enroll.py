"""
SORT-tendance :: scripts/enroll.py

Standalone CLI entry point for the "24 + 1" enrollment pipeline.

Usage (from project root, e.g. C:\\Skripsi\\SORT-Tendance>):

    python scripts/enroll.py

Or, as a one-liner:

    python scripts/enroll.py --dry-run

This script wires up the full enrollment stack in the correct order:

    1. gpu_linker.register_dlls()           -- Windows DLL search path fix
    2. ConfigRegistry.load()                -- YAML config bootstrap
    3. _LightFaceEngine.initialize().warmup() -- InsightFace CUDA warmup
    4. ArcFaceAligner()                     -- 112x112 alignment template
    5. EnrollmentClusterer(engine, aligner) -- 24+1 clustering driver
    6. clusterer.enroll_all()               -- walk data/student_faces/

The output registry is serialized to data/student_db.pickle by the
clusterer itself; this script just prints the EnrollmentStats summary.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable when running `python scripts/enroll.py`
# from any working directory.  This guarantees that `src.*` imports resolve
# regardless of where the user invokes the script from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Windows DLL registration MUST happen before any ML framework is imported.
# We import gpu_linker first; it is a no-op on Linux/macOS. The public
# entry point in gpu_linker.py is `link_dlls()` (returns a GPULinker instance).
# ---------------------------------------------------------------------------
try:
    from src.utils.gpu_linker import link_dlls as _link_dlls
    _link_dlls()
except Exception as _exc:  # pragma: no cover - diagnostic only
    print(f"[WARN] gpu_linker.link_dlls() failed: {_exc}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Now import the rest of the stack.
# ---------------------------------------------------------------------------
from src.utils.database_manager import (
    ConfigRegistry,
    _LightFaceEngine,
    ArcFaceAligner,
    EnrollmentClusterer,
    EnrollmentStats,
)


# ============================================================================
# Logging
# ============================================================================
logger = logging.getLogger("sortendance.enroll_cli")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)


# ============================================================================
# CLI
# ============================================================================
# ============================================================================
# Patch: Single-Student Enrollment Mode + Rate-Limit + Supervisor Signal
# ----------------------------------------------------------------------------
# This block is INSERTED into enroll.py just before the "if __name__"
# block. It adds:
#   * _check_rate_limit()        -- 60s global gate via data/.enroll_last_run
#   * _mark_rate_limit()         -- writes the timestamp file
#   * _request_supervisor_restart()  -- writes data/.restart_main_requested
#   * _read_image_bgr()          -- cv2.imread with friendly error
#   * _single_student_main()     -- the new --single-student flow
#   * extended parse_args()      -- adds the new CLI flags
#   * extended main()            -- routes to _single_student_main when
#                                   --single-student is set
# ============================================================================

# ---------------------------------------------------------------------------
# Constants for the single-student flow.
# ---------------------------------------------------------------------------
_ENROLL_RATE_LIMIT_S: float = 60.0   # Global: >=60s between any two enrollments.
_ENROLL_RATE_LIMIT_FILE: str = "data/.enroll_last_run"
_RESTART_REQUEST_FILE: str = "data/.restart_main_requested"
_RESTART_DELAY_S: float = 15.0       # Supervisor waits this long after
                                      # the flag is written before
                                      # restarting main.py.

# Exit codes for single-student mode (kept distinct for the UI to surface).
RC_OK: int = 0
RC_RATE_LIMITED: int = 10
RC_DUPLICATE_ID: int = 20
RC_DUPLICATE_FACE: int = 21
RC_BAD_INPUT: int = 30
RC_ENGINE_ERROR: int = 40


def _check_rate_limit() -> Tuple[bool, float]:
    """
    Return (allowed, seconds_remaining).

    `allowed` is True if at least _ENROLL_RATE_LIMIT_S seconds have
    elapsed since the last successful enrollment (or if no prior
    enrollment has ever been recorded).

    `seconds_remaining` is 0.0 when allowed, else the wait time left.
    """
    p = Path(_ENROLL_RATE_LIMIT_FILE)
    if not p.is_file():
        return True, 0.0
    try:
        raw = p.read_text(encoding="utf-8").strip()
        last_ts = float(raw)
    except (ValueError, OSError) as exc:
        logger.warning(
            "Rate-limit file unreadable (%s); allowing enrollment.", exc,
        )
        return True, 0.0
    elapsed = time.time() - last_ts
    if elapsed >= _ENROLL_RATE_LIMIT_S:
        return True, 0.0
    return False, (_ENROLL_RATE_LIMIT_S - elapsed)


def _mark_rate_limit() -> None:
    """Write the current epoch to the rate-limit file (atomic)."""
    os.makedirs(os.path.dirname(_ENROLL_RATE_LIMIT_FILE) or ".", exist_ok=True)
    tmp = _ENROLL_RATE_LIMIT_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(f"{time.time():.6f}")
        os.replace(tmp, _ENROLL_RATE_LIMIT_FILE)
    except OSError as exc:
        logger.warning("Failed to write rate-limit file: %s", exc)


def _request_supervisor_restart() -> None:
    """
    Write a flag file that the start_sortendance.py supervisor polls.
    Once 15s have elapsed since the timestamp written here, the
    supervisor will gracefully restart ONLY the `main` child so the
    freshly-expanded student_db.pickle is reloaded into RAM.
    """
    os.makedirs(os.path.dirname(_RESTART_REQUEST_FILE) or ".", exist_ok=True)
    payload = json.dumps({
        "requested_at": time.time(),
        "delay_s": _RESTART_DELAY_S,
        "reason": "new_student_enrolled",
    })
    tmp = _RESTART_REQUEST_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, _RESTART_REQUEST_FILE)
        logger.info(
            "Supervisor restart flag written -> %s | restart in %.0fs",
            _RESTART_REQUEST_FILE, _RESTART_DELAY_S,
        )
    except OSError as exc:
        logger.warning("Failed to write restart-request flag: %s", exc)


def _read_image_bgr(path: str) -> np.ndarray:
    """Read an image file as BGR uint8 HWC. Raises on failure."""
    if not path:
        raise ValueError("Empty image path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"cv2.imread failed (corrupt or unsupported): {path}")
    return img


def _single_student_main(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Interactive single-student enrollment flow.

    Stages:
      1. Rate-limit gate (60s global).
      2. Read + validate the two photos.
      3. Stage-1 dedup: check_existing_by_id().
      4. Stage-2 dedup: check_existing_by_face() on photo #1 (flat).
      5. enroll_student() -> append to pickle atomically.
      6. Mark rate-limit + write supervisor-restart flag.

    Returns one of the RC_* exit codes above. The Streamlit UI maps
    these to user-facing notifications.
    """
    # ------------------------------------------------------------------
    # 1. Rate-limit gate.
    # ------------------------------------------------------------------
    allowed, wait_s = _check_rate_limit()
    if not allowed:
        logger.warning(
            "Rate-limited: last enrollment was %.1fs ago, need %.0fs. "
            "Try again in %.1fs.",
            _ENROLL_RATE_LIMIT_S - wait_s, _ENROLL_RATE_LIMIT_S, wait_s,
        )
        # Emit a machine-readable last line that the UI can parse.
        print(f"RATE_LIMITED wait_s={wait_s:.1f}")
        return RC_RATE_LIMITED

    # ------------------------------------------------------------------
    # 2. Validate inputs.
    # ------------------------------------------------------------------
    student_id = (args.student_id or "").strip()
    student_name = (args.student_name or "").strip()
    if not student_id:
        logger.error("--student-id is required for --single-student mode.")
        print("BAD_INPUT reason=missing_student_id")
        return RC_BAD_INPUT
    if not args.photo1 or not args.photo2:
        logger.error("Both --photo1 (flat) and --photo2 (subtle) are required.")
        print("BAD_INPUT reason=missing_photos")
        return RC_BAD_INPUT

    try:
        photo1 = _read_image_bgr(args.photo1)   # flat / neutral
        photo2 = _read_image_bgr(args.photo2)   # subtle expression
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Photo read failed: %s", exc)
        print(f"BAD_INPUT reason=read_error msg={exc}")
        return RC_BAD_INPUT

    # ------------------------------------------------------------------
    # 3. Construct EnrollmentService (lazy engine init).
    # ------------------------------------------------------------------
    from src.utils.database_manager import EnrollmentService
    svc = EnrollmentService(config=config)
    try:
        # ------------------------------------------------------------------
        # 4. Stage-1 dedup: by student_id.
        # ------------------------------------------------------------------
        existing = svc.check_existing_by_id(student_id)
        if existing is not None:
            existing_name = existing.get("student_name", student_id)
            existing_ts = existing.get("enrollment_timestamp", "?")
            logger.warning(
                "DUPLICATE_ID: student_id=%s already enrolled as '%s' at %s",
                student_id, existing_name, existing_ts,
            )
            print(
                f"DUPLICATE_ID student_id={student_id} "
                f"existing_name={existing_name} enrolled_at={existing_ts}"
            )
            return RC_DUPLICATE_ID

        # ------------------------------------------------------------------
        # 5. Stage-2 dedup: by face cosine similarity (photo #1).
        # ------------------------------------------------------------------
        face_match = svc.check_existing_by_face(photo1)
        if face_match is not None:
            match_id, match_name, match_score = face_match
            logger.warning(
                "DUPLICATE_FACE: face in photo1 matches student_id=%s "
                "(%s) cosine=%.4f >= threshold %.2f",
                match_id, match_name, match_score, svc.cosine_threshold,
            )
            print(
                f"DUPLICATE_FACE matched_id={match_id} "
                f"matched_name={match_name} cosine={match_score:.4f}"
            )
            return RC_DUPLICATE_FACE

        # ------------------------------------------------------------------
        # 6. Enroll.
        # ------------------------------------------------------------------
        profile = svc.enroll_student(
            student_id=student_id,
            student_name=student_name,
            photos=[photo1, photo2],
        )
        logger.info(
            "ENROLLED: student_id=%s name=%s embeddings=%d",
            profile.student_id, profile.student_name,
            profile.face_embeddings.shape[0],
        )
        print(
            f"ENROLLED student_id={profile.student_id} "
            f"student_name={profile.student_name} "
            f"embeddings={profile.face_embeddings.shape[0]}"
        )

        # ------------------------------------------------------------------
        # 7. Mark rate-limit + signal supervisor to restart main.
        # ------------------------------------------------------------------
        _mark_rate_limit()
        _request_supervisor_restart()
        return RC_OK
    finally:
        try:
            svc.close()
        except Exception as exc:
            logger.warning("EnrollmentService.close() failed: %s", exc)


# ============================================================================
# CLI (extended for --single-student mode)
# ============================================================================
# ---------------------------------------------------------------------------
# Patch: Add-Embeddings mode (--add-to).
# ----------------------------------------------------------------------------
# Appends 1..N new face embeddings to an EXISTING student's profile.
# Skips the two-stage dedup (the student already exists; we are
# augmenting, not re-enrolling). Honors profile_capacity (default 25):
# if the student is already at capacity, returns RC_BAD_INPUT.
#
# Shares the 60s global rate-limit + supervisor-restart-flag pattern
# with --single-student mode.
# ============================================================================
def _add_embeddings_main(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Add more face photos to an EXISTING student's profile.

    Stages:
      1. Rate-limit gate (60s global -- shared with --single-student).
      2. Validate student_id + at least 1 --photo.
      3. Pre-check existence + capacity (early, friendly error).
      4. Read all photos.
      5. EnrollmentService.add_embeddings_to_student().
      6. Mark rate-limit + write supervisor-restart flag.

    Returns one of the RC_* exit codes. The Streamlit UI maps these
    to user-facing notifications.
    """
    # 1. Rate-limit gate.
    allowed, wait_s = _check_rate_limit()
    if not allowed:
        logger.warning(
            "Rate-limited: last enrollment was %.1fs ago, need %.0fs. "
            "Try again in %.1fs.",
            _ENROLL_RATE_LIMIT_S - wait_s, _ENROLL_RATE_LIMIT_S, wait_s,
        )
        print(f"RATE_LIMITED wait_s={wait_s:.1f}")
        return RC_RATE_LIMITED

    # 2. Validate inputs.
    student_id = (args.add_to or "").strip()
    if not student_id:
        logger.error("--add-to requires a student ID.")
        print("BAD_INPUT reason=missing_student_id")
        return RC_BAD_INPUT
    if not args.photo:
        logger.error("--add-to requires at least one --photo.")
        print("BAD_INPUT reason=missing_photos")
        return RC_BAD_INPUT

    # 3. Pre-check existence + capacity for early, friendly error.
    from src.utils.database_manager import EnrollmentService
    svc = EnrollmentService(config=config)
    try:
        existing = svc.check_existing_by_id(student_id)
        if existing is None:
            logger.error("ADD_TO: student_id=%s not found.", student_id)
            print(
                f"BAD_INPUT reason=student_not_found student_id={student_id}"
            )
            return RC_BAD_INPUT
        try:
            cur_count = int(existing.get("face_embeddings").shape[0])
        except Exception:
            cur_count = 0
        capacity = int(existing.get("profile_capacity", 25))
        if cur_count >= capacity:
            logger.error(
                "ADD_TO: student_id=%s already at capacity %d/%d.",
                student_id, cur_count, capacity,
            )
            print(
                f"BAD_INPUT reason=at_capacity student_id={student_id} "
                f"current={cur_count} capacity={capacity}"
            )
            return RC_BAD_INPUT

        # 4. Read photos.
        photos = []
        for p in args.photo:
            try:
                photos.append(_read_image_bgr(p))
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Photo read failed (%s): %s", p, exc)
                print(f"BAD_INPUT reason=read_error path={p} msg={exc}")
                return RC_BAD_INPUT

        # 5. Append embeddings.
        try:
            profile = svc.add_embeddings_to_student(
                student_id=student_id,
                photos=photos,
            )
        except KeyError as exc:
            logger.error("ADD_TO: student not found: %s", exc)
            print(f"BAD_INPUT reason=student_not_found msg={exc}")
            return RC_BAD_INPUT
        except ValueError as exc:
            logger.error("ADD_TO: %s", exc)
            print(f"BAD_INPUT reason=value_error msg={exc}")
            return RC_BAD_INPUT

        logger.info(
            "ADDED_EMBEDDINGS: student_id=%s name=%s total_embeddings=%d/%d",
            profile.student_id, profile.student_name,
            profile.face_embeddings.shape[0], profile.profile_capacity,
        )
        print(
            f"ADDED_EMBEDDINGS student_id={profile.student_id} "
            f"student_name={profile.student_name} "
            f"total_embeddings={profile.face_embeddings.shape[0]} "
            f"capacity={profile.profile_capacity}"
        )

        # 6. Mark rate-limit + signal supervisor to restart main.
        _mark_rate_limit()
        _request_supervisor_restart()
        return RC_OK
    finally:
        try:
            svc.close()
        except Exception as exc:
            logger.warning("EnrollmentService.close() failed: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enroll",
        description="SORT-tendance Enrollment CLI (bulk 24+1 OR single-student)",
    )

    # --- Single-student mode flags ---
    parser.add_argument(
        "--single-student",
        action="store_true",
        help="Enroll ONE student from 2 supplied photos instead of "
             "running the bulk 24+1 directory walk. Requires "
             "--student-id, --photo1 (flat), --photo2 (subtle).",
    )
    parser.add_argument("--student-id", type=str, default=None,
                        help="Student number / NRP (e.g. 221050).")
    parser.add_argument("--student-name", type=str, default=None,
                        help="Human-readable name (optional; defaults to ID).")
    parser.add_argument("--photo1", type=str, default=None,
                        help="Path to photo #1 (flat / neutral expression).")
    parser.add_argument("--photo2", type=str, default=None,
                        help="Path to photo #2 (subtle expression).")

    # --- Add-embeddings mode flags (--add-to) ---
    parser.add_argument(
        "--add-to",
        type=str,
        default=None,
        metavar="STUDENT_ID",
        help="Add more face photos to an EXISTING student. Use with one "
             "or more --photo <path>. Skips dedup checks (student already "
             "exists). Honors profile_capacity (default 25).",
    )
    parser.add_argument(
        "--photo",
        action="append",
        default=None,
        metavar="PATH",
        help="Path to a photo. May be specified MULTIPLE times. Used "
             "with --add-to (any count >= 1).",
    )

    # --- Bulk-mode flags (unchanged) ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize the engine + aligner + clusterer but do NOT write "
             "the pickle. Useful for smoke-testing the InsightFace stack.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml. Defaults to config/config.yaml under the "
             "project root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()



def _bootstrap_config(cli_config: str | None) -> Dict[str, Any]:
    """Load config.yaml via ConfigRegistry, honoring an optional override.

    ConfigRegistry.load() accepts an absolute or relative path. When no
    override is supplied we pass the project-relative default
    "config/config.yaml" and let the caller's CWD resolve it (the user
    is expected to run this script from the project root).
    """
    if cli_config:
        cfg_path = Path(cli_config).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        logger.info("Config override: %s", cfg_path)
        return ConfigRegistry.load(config_path=str(cfg_path))

    # Default: resolve relative to project root so the script works from
    # any CWD (e.g. when invoked as `python scripts/enroll.py`).
    default_path = _PROJECT_ROOT / "config" / "config.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(
            f"Default config not found at: {default_path}"
        )
    return ConfigRegistry.load(config_path=str(default_path))


def _print_stats(stats: EnrollmentStats) -> None:
    """Pretty-print the EnrollmentStats summary to stdout."""
    print("\n" + "=" * 60)
    print("ENROLLMENT SUMMARY")
    print("=" * 60)
    print(f"  Directories scanned : {stats.total_directories_scanned}")
    print(f"  Profiles built      : {stats.total_profiles_built}")
    failed = stats.failed_directories or []
    print(f"  Failed directories  : {len(failed)}")
    if failed:
        for name in failed:
            print(f"      - {name}")
    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Patch: Single-student route. If --single-student is set, dispatch
    # to _single_student_main() and skip the bulk 24+1 enrollment path.
    # ------------------------------------------------------------------
    # Patch: Add-embeddings route. If --add-to is set, dispatch to
    # _add_embeddings_main() and skip all other paths.
    # ------------------------------------------------------------------
    if getattr(args, "add_to", None):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        logger.info("Add-embeddings mode: loading config ...")
        try:
            config: Dict[str, Any] = _bootstrap_config(args.config)
        except Exception:
            logger.error(
                "Config bootstrap failed:\n%s", traceback.format_exc(),
            )
            return 2
        return _add_embeddings_main(args, config)

    # ------------------------------------------------------------------
    # Patch: Single-student route. If --single-student is set, dispatch
    # to _single_student_main() and skip the bulk 24+1 enrollment path.
    # ------------------------------------------------------------------
    if getattr(args, "single_student", False):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        logger.info("Single-student mode: loading config ...")
        try:
            config: Dict[str, Any] = _bootstrap_config(args.config)
        except Exception:
            logger.error(
                "Config bootstrap failed:\n%s", traceback.format_exc(),
            )
            return 2
        return _single_student_main(args, config)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Config bootstrap.
    # ------------------------------------------------------------------
    logger.info("Loading config.yaml ...")
    try:
        config: Dict[str, Any] = _bootstrap_config(args.config)
    except Exception:
        logger.error("Config bootstrap failed:\n%s", traceback.format_exc())
        return 2
    logger.info("Config loaded. Project root = %s", _PROJECT_ROOT)

    # ------------------------------------------------------------------
    # 2. InsightFace engine construction + warmup.
    #
    # Patch 12: Construct the REGISTRATION engine at det_size=(640,640)
    # via the det_size_override parameter. The live engine in main.py
    # stays at the config default (320,320) for speed; registration
    # favors embedding quality per pose to improve side-view matching.
    # The (640,640) engine is closed at the end of this script to free
    # VRAM before the live orchestrator runs.
    # ------------------------------------------------------------------
    enr_cfg = config.get("enrollment", {})
    reg_det_size = tuple(enr_cfg.get("registration_det_size", [640, 640]))
    logger.info(
        "Constructing _LightFaceEngine (registration) | det_size=%s ...",
        reg_det_size,
    )
    engine = _LightFaceEngine(config=config, det_size_override=reg_det_size)
    try:
        logger.info("Initializing InsightFace (det_10g + w600k_r50) ...")
        engine.initialize()
        logger.info("Running double-pass CUDA warmup ...")
        engine.warmup()
    except Exception:
        logger.error(
            "Engine init/warmup failed:\n%s", traceback.format_exc()
        )
        return 3
    logger.info("Registration engine ready | det_size=%s", reg_det_size)

    # ------------------------------------------------------------------
    # 3. ArcFace aligner.
    # ------------------------------------------------------------------
    logger.info("Constructing ArcFaceAligner ...")
    aligner = ArcFaceAligner(config=config)

    # ------------------------------------------------------------------
    # 4. EnrollmentClusterer.
    # ------------------------------------------------------------------
    logger.info("Constructing EnrollmentClusterer ...")
    clusterer = EnrollmentClusterer(
        engine=engine,
        aligner=aligner,
        config=config,
    )

    # ------------------------------------------------------------------
    # 5. Dry-run short-circuit.
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("--dry-run specified; skipping enroll_all().")
        logger.info(
            "Smoke test PASSED: engine + aligner + clusterer all constructed."
        )
        return 0

    # ------------------------------------------------------------------
    # 6. Run the full enrollment pass.
    # ------------------------------------------------------------------
    logger.info("Starting enroll_all() ...")
    t0 = time.perf_counter()
    try:
        _registry, stats = clusterer.enroll_all()
    except Exception:
        logger.error("enroll_all() crashed:\n%s", traceback.format_exc())
        return 4
    elapsed = time.perf_counter() - t0
    logger.info("enroll_all() completed in %.2fs", elapsed)

    _print_stats(stats)

    # ------------------------------------------------------------------
    # 7. Close the registration engine to release VRAM before the live
    #    orchestrator (main.py) starts. The live engine will construct
    #    its own _LightFaceEngine at the config default (320,320).
    # ------------------------------------------------------------------
    try:
        engine.close()
        logger.info("Registration engine closed; VRAM released.")
    except Exception as exc:
        logger.warning("Registration engine close failed: %s", exc)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        sys.exit(130)
