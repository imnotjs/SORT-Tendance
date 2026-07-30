"""
SORT-tendance :: scripts/list_students.py

Quick CLI to list every student enrolled in data/student_db.pickle.

Usage (from project root):
    python scripts/list_students.py
    python scripts/list_students.py --json     # machine-readable
    python scripts/list_students.py --verbose  # include anchor hash + mean norm

Author: SORT-tendance Engineering
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Resolve project root the same way enroll.py does.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DB_PATH = _PROJECT_ROOT / "data" / "student_db.pickle"


def _load_registry() -> dict:
    if not _DB_PATH.is_file():
        return {}
    try:
        with open(_DB_PATH, "rb") as fh:
            return pickle.load(fh) or {}
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        print(f"[ERROR] Failed to read {_DB_PATH}: {exc}", file=sys.stderr)
        return {}


def _format_table(rows: list, verbose: bool = False) -> str:
    if not rows:
        return "(no students enrolled)"
    if verbose:
        header = f"{'#':<4} {'ID':<12} {'Name':<25} {'Embeds':<8} {'Anchor':<14} {'MeanNorm':<10} {'EnrolledAt':<26}"
        sep = "-" * len(header)
    else:
        header = f"{'#':<4} {'ID':<12} {'Name':<25} {'Embeds':<8} {'EnrolledAt':<26}"
        sep = "-" * len(header)
    lines = [header, sep]
    for i, r in enumerate(rows, start=1):
        sid = str(r.get("student_id", "?"))
        name = str(r.get("student_name", sid))
        n_emb = r.get("face_embeddings")
        n_emb = int(n_emb.shape[0]) if hasattr(n_emb, "shape") else 0
        ts = str(r.get("enrollment_timestamp", "?"))[:25]
        if verbose:
            anchor = str(r.get("anchor_image_hash", "?"))[:12]
            mean = r.get("mean_embedding")
            try:
                import numpy as np
                mean_norm = float(np.linalg.norm(mean)) if mean is not None else 0.0
            except Exception:
                mean_norm = 0.0
            lines.append(
                f"{i:<4} {sid:<12} {name:<25} {n_emb:<8} {anchor:<14} {mean_norm:<10.4f} {ts:<26}"
            )
        else:
            lines.append(f"{i:<4} {sid:<12} {name:<25} {n_emb:<8} {ts:<26}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="List enrolled students.")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    ap.add_argument("--verbose", action="store_true",
                    help="Include anchor hash + mean embedding norm.")
    args = ap.parse_args()

    registry = _load_registry()

    # Build a deterministic order (by student_id).
    rows = []
    for sid in sorted(registry.keys()):
        prof = registry[sid]
        rows.append({
            "student_id": str(prof.get("student_id", sid)),
            "student_name": str(prof.get("student_name", sid)),
            "nrp": str(prof.get("nrp", "")),
            "num_embeddings": int(prof["face_embeddings"].shape[0])
                              if hasattr(prof.get("face_embeddings"), "shape") else 0,
            "enrollment_timestamp": str(prof.get("enrollment_timestamp", "")),
            "anchor_image_hash": str(prof.get("anchor_image_hash", "")),
        })

    if args.json:
        print(json.dumps({
            "pickle_path": str(_DB_PATH),
            "pickle_exists": _DB_PATH.is_file(),
            "pickle_size_bytes": _DB_PATH.stat().st_size if _DB_PATH.is_file() else 0,
            "total_students": len(rows),
            "students": rows,
        }, indent=2))
        return 0

    print(f"Pickle: {_DB_PATH}")
    print(f"Exists: {_DB_PATH.is_file()}")
    if _DB_PATH.is_file():
        print(f"Size:   {_DB_PATH.stat().st_size / 1024.0:.2f} KB")
    print(f"Total students enrolled: {len(rows)}")
    print()
    print(_format_table(rows, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
