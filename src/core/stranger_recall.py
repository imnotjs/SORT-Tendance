"""
SORT-tendance :: stranger_recall.py
====================================
Patch 67 :: OSNet stranger memory recall from disk clearshots.

PURPOSE
-------
When main.py restarts (every 12 hours at 6AM/6PM via the supervisor,
or mid-session crash recovery), the in-memory OSNet stranger cache
(``DualIndexManager._dynamic_strangers``) is wiped. This module
rebuilds that cache from the clearshot PNGs saved on disk, so the
tracker "remembers" strangers across restarts within the same session.

FLOW
----
1. Compute the current session key (date + 6AM/6PM label, with
   overnight overhang: 00:00-05:59 -> yesterday's 6PM session).
2. If current session is 6PM, ALSO scan today's 6AM session (carries
   morning strangers into the evening). At 6AM (new day), do NOT
   scan yesterday -- fresh start.
3. For each session, scan the ``clearshots/`` subdirectory for files
   matching the pattern:
     {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY:02d}.png
4. Group clearshots by stranger label (e.g. "Stranger_07").
5. For each stranger:
   a. Load each clearshot PNG + sidecar .json (bbox coords).
   b. Crop the person region using the stored bbox + margin.
      (If .json is missing, fall back to full frame -- degraded.)
   c. Compute OSNet 512-dim embedding for each crop.
   d. Consensus selection (medoid): the embedding with the most
      cosine-similarity "pairs" above ``consensus_threshold`` wins.
      Ties broken by highest mean similarity.
6. Return ``{stranger_label: consensus_embedding}`` dict.

The caller (``IdentityMatcher.recall_strangers_from_disk()``) then
loads this dict into the dynamic stranger cache via
``DualIndexManager.load_recall_vectors()``.

NO PERSISTENCE
--------------
The consensus embeddings live ONLY in RAM inside
``DualIndexManager._dynamic_strangers``. They are NOT saved to
.pickle/.pkl. On every restart, they are rebuilt from the PNG
clearshots on disk. At midnight (start of new day's 6AM session),
yesterday's clearshots are not scanned -- they remain on disk as
forensic archive but are not loaded into memory.

Author: SORT-tendance Engineering (Patch 67)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sortendance.stranger_recall")

# ---------------------------------------------------------------------------
# Session key computation (mirrors snap_strangers.py compute_session_key).
# ---------------------------------------------------------------------------
_SESSION_AM_START_HOUR = 6
_SESSION_PM_START_HOUR = 18
_SESSION_DIR_AM = "6AM Session"
_SESSION_DIR_PM = "6PM Session"

# Regex to parse clearshot filenames:
#   {ts_ms}_track{tid}_STRANGER_{label}_CLEARSHOT_{YY}.png
# Group 1: ts_ms (int), Group 2: track_id (int),
# Group 3: stranger_label (e.g. "Stranger_07"), Group 4: clearshot_idx (int)
_CLEARSHOT_RE = re.compile(
    r"^(\d+)_track(\d+)_STRANGER_(.+?)_CLEARSHOT_(\d+)\.png$"
)


def _compute_session_key(now: Optional[_dt.datetime] = None) -> Tuple[str, str, str]:
    """Return (date_str, session_label, session_dir) for the current time.

    Mirrors snap_strangers.compute_session_key() with the overnight
    overhang rule: 00:00-05:59 -> yesterday's 6PM session.
    """
    if now is None:
        now = _dt.datetime.now()
    hour = now.hour
    if hour < _SESSION_AM_START_HOUR:
        # Pre-dawn: still in yesterday's 6PM session.
        session_date = (now - _dt.timedelta(days=1)).date()
        session_label = "06PM"
        session_dir = _SESSION_DIR_PM
    elif hour < _SESSION_PM_START_HOUR:
        # Daytime: 6AM session.
        session_date = now.date()
        session_label = "06AM"
        session_dir = _SESSION_DIR_AM
    else:
        # Evening: 6PM session.
        session_date = now.date()
        session_label = "06PM"
        session_dir = _SESSION_DIR_PM
    return (session_date.strftime("%Y-%m-%d"), session_label, session_dir)


def _scan_session_clearshots(
    clearshot_root: str,
    session_date: str,
    session_dir: str,
) -> Dict[str, List[Tuple[int, Path, Path]]]:
    """Scan one session's clearshot directory.

    Returns ``{stranger_label: [(ts_ms, png_path, json_path), ...]}``.
    The json_path may not exist (old clearshots without sidecar .json);
    callers handle that gracefully.
    """
    session_path = Path(clearshot_root) / session_date / session_dir / "clearshots"
    if not session_path.is_dir():
        logger.debug(
            "stranger_recall: session dir does not exist: %s", session_path
        )
        return {}

    result: Dict[str, List[Tuple[int, Path, Path]]] = {}
    for entry in session_path.iterdir():
        if not entry.is_file() or not entry.name.endswith(".png"):
            continue
        m = _CLEARSHOT_RE.match(entry.name)
        if not m:
            continue
        ts_ms = int(m.group(1))
        # track_id = int(m.group(2))  # not needed for recall
        stranger_label = m.group(3)
        # clearshot_idx = int(m.group(4))  # not needed for recall
        json_path = entry.with_suffix(".json")
        result.setdefault(stranger_label, []).append((ts_ms, entry, json_path))

    return result


def _crop_person(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_px: int,
) -> np.ndarray:
    """Crop the person region from a full frame using bbox + margin.

    Clips to frame boundaries so the margin doesn't go out of bounds.
    Returns a BGR uint8 crop (HxWx3).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - margin_px)
    y1 = max(0, int(y1) - margin_px)
    x2 = min(w, int(x2) + margin_px)
    y2 = min(h, int(y2) + margin_px)
    if x2 <= x1 or y2 <= y1:
        # Degenerate crop -- return the full frame as fallback.
        return frame
    return frame[y1:y2, x1:x2]


def _load_clearshot_crop(
    png_path: Path,
    json_path: Path,
    margin_px: int,
) -> Optional[np.ndarray]:
    """Load a clearshot PNG and crop to the person region.

    Reads bbox from the sidecar .json. If .json is missing or invalid,
    falls back to the full frame (degraded -- OSNet will still produce
    an embedding, but background noise degrades quality).

    Returns a BGR uint8 crop, or None if the PNG can't be loaded.
    """
    try:
        import cv2
    except ImportError:
        logger.error("stranger_recall: cv2 not available -- cannot load clearshots.")
        return None

    frame = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if frame is None:
        logger.warning("stranger_recall: could not read PNG: %s", png_path)
        return None

    # Try to read bbox from sidecar .json.
    bbox: Optional[Tuple[int, int, int, int]] = None
    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
            b = meta.get("bbox")
            if b and len(b) == 4:
                bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        except (OSError, ValueError, TypeError) as exc:
            logger.debug(
                "stranger_recall: sidecar .json read failed for %s: %s",
                json_path, exc,
            )

    if bbox is None:
        # Fallback: use the full frame. This is degraded but functional.
        logger.debug(
            "stranger_recall: no sidecar .json for %s -- using full frame.",
            png_path.name,
        )
        return frame

    return _crop_person(frame, bbox, margin_px)


def _consensus_embedding(
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Pick the consensus (medoid) embedding from a set of N embeddings.

    The "winner" is the embedding with the most cosine-similarity pairs
    above ``threshold``. Ties are broken by highest mean similarity.

    Args:
        embeddings: (N, 512) float32, L2-normalized.
        threshold: cosine similarity threshold for pair counting.

    Returns:
        (512,) float32, L2-normalized -- the consensus embedding.
    """
    n = embeddings.shape[0]
    if n == 0:
        raise ValueError("Cannot compute consensus of empty embeddings.")
    if n == 1:
        return embeddings[0]

    # Cosine similarity = dot product (vectors are L2-normalized).
    sim_matrix = embeddings @ embeddings.T  # (N, N), values in [-1, 1]

    # For each embedding i, count pairs (j != i) where sim >= threshold.
    # Zero out the diagonal so self-similarity doesn't count.
    np.fill_diagonal(sim_matrix, -1.0)
    pair_counts = np.sum(sim_matrix >= threshold, axis=1)  # (N,)

    max_count = int(np.max(pair_counts))
    candidates = np.where(pair_counts == max_count)[0]

    if len(candidates) == 1:
        winner = int(candidates[0])
    else:
        # Tie-break: highest mean similarity to all others.
        np.fill_diagonal(sim_matrix, 0.0)  # exclude self from mean
        mean_sims = np.mean(sim_matrix, axis=1)
        winner = int(candidates[np.argmax(mean_sims[candidates])])

    return embeddings[winner]


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def recall_strangers(
    config: Dict[str, Any],
    osnet_model: Any,
    yolo_model: Any = None,
) -> Dict[str, np.ndarray]:
    """Rebuild the stranger OSNet memory vector from disk clearshots.

    Args:
        config: Full config dict (reads ``stranger_recall`` block).
        osnet_model: ``OSNetReID`` instance (must be initialized).
        yolo_model: Unused (kept for API compatibility). Person crops
                    are read from the sidecar .json bbox, so YOLO
                    re-detection is not needed.

    Returns:
        ``{stranger_label: consensus_embedding}`` where each embedding
        is a (512,) float32 L2-normalized numpy array. Empty dict if
        no clearshots found or recall is disabled.
    """
    rcfg = config.get("stranger_recall", {})
    if not rcfg.get("enabled", True):
        logger.info("stranger_recall: disabled by config. Skipping.")
        return {}

    clearshot_root = str(rcfg.get("clearshot_dir", "storage/snap_strangers"))
    consensus_threshold = float(rcfg.get("consensus_threshold", 0.70))
    max_per_stranger = int(rcfg.get("max_clearshots_per_stranger", 20))
    margin_px = int(rcfg.get("crop_margin_px", 10))

    now = _dt.datetime.now()
    date_str, session_label, session_dir = _compute_session_key(now)
    logger.info(
        "stranger_recall: current session = %s / %s (%s)",
        date_str, session_label, session_dir,
    )

    # Scan the current session's clearshots.
    all_clearshots: Dict[str, List[Tuple[int, Path, Path]]] = {}
    all_clearshots.update(
        _scan_session_clearshots(clearshot_root, date_str, session_dir)
    )

    # If current session is 6PM, ALSO scan today's 6AM session to carry
    # morning strangers into the evening. At 6AM (new day), do NOT scan
    # yesterday -- fresh start.
    if session_label == "06PM":
        am_clearshots = _scan_session_clearshots(
            clearshot_root, date_str, _SESSION_DIR_AM
        )
        for label, shots in am_clearshots.items():
            if label in all_clearshots:
                all_clearshots[label].extend(shots)
            else:
                all_clearshots[label] = list(shots)

    if not all_clearshots:
        logger.info(
            "stranger_recall: no clearshots found for session %s/%s. "
            "Stranger memory will be empty (fresh session).",
            date_str, session_dir,
        )
        return {}

    logger.info(
        "stranger_recall: found %d stranger labels across scanned sessions.",
        len(all_clearshots),
    )

    # For each stranger: load crops, compute embeddings, pick consensus.
    recall_map: Dict[str, np.ndarray] = {}
    t_start = time.time()

    for label, shots in all_clearshots.items():
        # Sort by ts_ms descending, take the most recent max_per_stranger.
        shots_sorted = sorted(shots, key=lambda s: s[0], reverse=True)
        if len(shots_sorted) > max_per_stranger:
            shots_sorted = shots_sorted[:max_per_stranger]

        # Load + crop each clearshot.
        crops: List[np.ndarray] = []
        for ts_ms, png_path, json_path in shots_sorted:
            crop = _load_clearshot_crop(png_path, json_path, margin_px)
            if crop is not None and crop.size > 0:
                crops.append(crop)

        if not crops:
            logger.warning(
                "stranger_recall: no loadable crops for %s -- skipping.",
                label,
            )
            continue

        # Batch OSNet embedding extraction.
        try:
            embeddings = osnet_model.extract_features(crops)  # (N, 512)
        except Exception as exc:
            logger.error(
                "stranger_recall: OSNet extraction failed for %s (%d crops): %s",
                label, len(crops), exc,
            )
            continue

        if embeddings.shape[0] == 0:
            continue

        # Consensus selection.
        try:
            consensus = _consensus_embedding(embeddings, consensus_threshold)
        except ValueError as exc:
            logger.warning(
                "stranger_recall: consensus failed for %s: %s", label, exc,
            )
            continue

        recall_map[label] = consensus
        logger.info(
            "stranger_recall: %s | crops=%d | consensus_dim=%d | "
            "threshold=%.2f",
            label, embeddings.shape[0], consensus.shape[0],
            consensus_threshold,
        )

    elapsed = time.time() - t_start
    logger.info(
        "stranger_recall: complete | strangers_recalled=%d | "
        "total_clearshots_processed=%d | elapsed=%.2fs",
        len(recall_map),
        sum(len(s) for s in all_clearshots.values()),
        elapsed,
    )
    return recall_map


# ---------------------------------------------------------------------------
# Manual test mode: `python stranger_recall.py` prints scan results.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print()
    print("=" * 70)
    print("SORT-tendance stranger_recall.py :: scan preview (no OSNet)")
    print("=" * 70)

    # Load config.
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Could not load config.yaml: {exc}")
        config = {}

    rcfg = config.get("stranger_recall", {})
    clearshot_root = str(rcfg.get("clearshot_dir", "storage/snap_strangers"))
    print(f"Clearshot root: {clearshot_root}")

    now = _dt.datetime.now()
    date_str, session_label, session_dir = _compute_session_key(now)
    print(f"Current session: {date_str} / {session_dir} ({session_label})")
    print()

    # Scan current session.
    current = _scan_session_clearshots(clearshot_root, date_str, session_dir)
    print(f"Current session clearshots ({session_dir}):")
    if current:
        for label, shots in sorted(current.items()):
            print(f"  {label}: {len(shots)} clearshots")
    else:
        print("  (none found)")

    # If 6PM, also scan today's 6AM.
    if session_label == "06PM":
        am = _scan_session_clearshots(clearshot_root, date_str, _SESSION_DIR_AM)
        print(f"\nToday's 6AM session clearshots (carry-over):")
        if am:
            for label, shots in sorted(am.items()):
                print(f"  {label}: {len(shots)} clearshots")
        else:
            print("  (none found)")
    else:
        print("\n(6AM session: no carry-over from previous session -- fresh day)")

    print()
    print("NOTE: This preview only scans filenames. To compute OSNet")
    print("embeddings and build the recall map, run main.py (which calls")
    print("IdentityMatcher.recall_strangers_from_disk() at startup).")
    print("=" * 70)
