"""
SORT-tendance :: src/core/res_opt_engine.py

Production-grade Resource Optimization Engine.

Responsibilities:
  1. Inference Throttle Controller -- manages two execution modes:
       * IDLE  : triggered when scenes are empty or populated
                 exclusively by low-priority verified tracks. Enforces
                 a strict 40 ms cycle sleep (~25 FPS) to suppress GPU
                 duty cycle to ~25%, preventing thermal throttling.
       * BURST : activated the instant an unverified high-priority
                 target is registered. Strips all cycle sleep to
                 minimize Time-To-First-Match (TTFM).
  2. Adaptive TTFM Tracker -- monitors recent match-confidence outputs
     per track. If 3 of the last 6 hits clear >= 0.60, relax the
     baseline verification threshold from 0.65 to 0.60 for a 30-frame
     evaluation window.
  3. Pose-Weighted EMA Accumulator -- computes a frontal pose weight
     from the 5-point landmark yaw, scaling the EMA learning rate
     (alpha_effective = alpha * pose_weight, base alpha = 0.4) so
     frontal frames drive verification while profile views preserve
     the historical feature array. A match is declared when the
     accumulated score crosses >= 0.62 over a minimum cluster of 4
     frames.
  4. Velocity-Aware Crop Boost -- monitors lateral centroid velocity
     (v_x px/frame). If a track moves toward the scene center at >3.0
     px/frame while its face width is < 40 px, bypass the standard
     skip-frame logic and trigger a 4x bicubic upscaling crop to
     accelerate verification data accumulation.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import gc
import time
import logging
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

# Local config registry import.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.database_manager import ConfigRegistry
except ImportError:                       # pragma: no cover
    ConfigRegistry = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.res_opt_engine")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Enums
# ============================================================================
class ThrottleMode(str, Enum):
    """Execution mode for the Inference Throttle Controller."""
    IDLE = "IDLE"
    BURST = "BURST"


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class TTFMState:
    """Per-track Adaptive TTFM accumulator state."""
    track_id: int
    # Rolling window of recent raw face-similarity hits (size = window_size).
    recent_hits: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    # Active relaxation window (frames remaining at the relaxed threshold).
    relaxation_frames_remaining: int = 0
    # Effective threshold currently in force for this track.
    effective_threshold: float = 0.65

    def register_hit(self, similarity: float) -> None:
        self.recent_hits.append(float(similarity))


@dataclass
class PoseEMAState:
    """Per-track Pose-Weighted EMA Accumulator state."""
    track_id: int
    ema_score: float = 0.0
    cluster_size: int = 0
    # History of (similarity, pose_weight) tuples for diagnostics.
    history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )

    def reset(self) -> None:
        self.ema_score = 0.0
        self.cluster_size = 0
        self.history.clear()


@dataclass
class VelocityState:
    """Per-track centroid velocity tracking state."""
    track_id: int
    last_centroid: Optional[Tuple[float, float]] = None
    last_frame_index: Optional[int] = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # Rolling velocity history for smoothing.
    velocity_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5)
    )


@dataclass
class CropBoostDecision:
    """Output of the Velocity-Aware Crop Boost evaluation."""
    should_boost: bool
    upscaled_crop: Optional[np.ndarray] = None
    scale_factor: int = 1
    reason: str = ""


# ============================================================================
# Resource Optimization Engine
# ============================================================================
class ResourceOptEngine:
    """
    Top-level Resource Optimization Engine.

    Owns:
      * Inference Throttle Controller (IDLE / BURST modes).
      * Adaptive TTFM Tracker (per-track relaxation windows).
      * Pose-Weighted EMA Accumulator.
      * Velocity-Aware Crop Boost.

    The orchestrator queries this engine each frame for:
      - the current ThrottleMode (and sleep_ms to apply).
      - the effective similarity threshold per track.
      - the accumulated EMA score per track.
      - whether to bicubic-upscale a small face crop for early verification.
    """

    # ------------------------------------------------------------------
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        rcfg = self.config["res_opt"]
        gcfg = self.config["gating"]
        self.idle_sleep_ms: float = float(rcfg["idle_mode"]["sleep_ms"])
        self.burst_sleep_ms: float = float(rcfg["burst_mode"]["sleep_ms"])
        self.idle_enabled: bool = bool(rcfg["idle_mode"]["enable"])
        self.burst_enabled: bool = bool(rcfg["burst_mode"]["enable"])

        # TTFM config.
        ttfm_cfg = gcfg["verified_student"]["ttfm"]
        self.ttfm_window: int = int(ttfm_cfg["window_size"])
        self.ttfm_hits_required: int = int(ttfm_cfg["hits_required"])
        self.ttfm_hit_similarity: float = float(ttfm_cfg["hit_similarity"])
        self.relaxation_window_frames: int = int(
            gcfg["verified_student"]["relaxation_window_frames"]
        )
        self.sim_baseline: float = float(
            gcfg["verified_student"]["similarity_baseline"]
        )
        self.sim_dynamic_floor: float = float(
            gcfg["verified_student"]["similarity_dynamic_floor"]
        )
        self.relaxation_enabled: bool = bool(
            gcfg["verified_student"]["enable_dynamic_relaxation"]
        )

        # Pose-Weighted EMA config.
        ema_cfg = gcfg["verified_student"]["pose_ema"]
        self.ema_base_alpha: float = float(ema_cfg["base_alpha"])
        self.pose_weight_frontal: float = float(ema_cfg["pose_weight_frontal"])
        self.pose_weight_floor: float = float(ema_cfg["pose_weight_profile_floor"])
        self.ema_threshold: float = float(ema_cfg["accumulation_threshold"])
        self.ema_min_cluster: int = int(ema_cfg["min_cluster_size"])

        # Velocity-Aware Crop Boost config.
        vcfg = rcfg["velocity_aware_crop_boost"]
        self.velocity_boost_enabled: bool = bool(vcfg["enable"])
        self.min_velocity_px_per_frame: float = float(
            vcfg["min_lateral_velocity_px_per_frame"]
        )
        self.small_face_width_px: int = int(
            vcfg["small_face_width_threshold_px"]
        )
        self.upscale_factor: int = int(vcfg["upscale_factor"])

        # Patch 13: Adaptive YOLO interval config. When the engine is
        # in IDLE mode (all tracks are VERIFIED or STRANGER-locked),
        # skip the YOLO+BoTSORT inference call on N-1 of every N frames.
        # BoTSORT track_buffer=30 keeps lost tracks alive for 30 frames,
        # so a 3-frame cycle is well within safety.
        ayi = rcfg.get("adaptive_yolo_interval", {})
        self.adaptive_yolo_enabled: bool = bool(ayi.get("enable", True))
        self.adaptive_yolo_interval: int = max(1, int(ayi.get("interval", 3)))
        self._yolo_skip_counter: int = 0
        self._yolo_skip_stats: Dict[str, int] = {
            "skipped": 0,
            "executed": 0,
            "burst_overrides": 0,
        }

        # Per-track state registries.
        self._ttfm_states: Dict[int, TTFMState] = {}
        self._ema_states: Dict[int, PoseEMAState] = {}
        self._velocity_states: Dict[int, VelocityState] = {}

        # Force-BURST override flag (set by the gating engine when an
        # ANOMALY state is detected).
        self._force_burst: bool = False
        self._force_burst_until_frame: int = -1

        # Patch 7: Predictive BURST on new track_ids.
        # Maintains the set of track_ids we've already seen at least once.
        # Any track_id NOT in this set on the current frame is treated as
        # a "birth frame" and forces BURST mode for that one frame --
        # regardless of resolved_state. This shaves one IDLE sleep cycle
        # (~40ms) off every new track's TTFM race.
        # The set is pruned against active_tracks so re-entries (BoTSort
        # assigns a new track_id when a person leaves and re-enters) get
        # the BURST treatment again on re-entry.
        self._seen_track_ids_for_burst: set = set()

        # Current mode + last cycle time (for telemetry).
        self.current_mode: ThrottleMode = ThrottleMode.IDLE
        self._last_cycle_start: float = 0.0

    # ==================================================================
    # 1. Inference Throttle Controller.
    # ==================================================================
    def decide_mode(
        self,
        active_tracks: List[Dict[str, Any]],
        current_frame_index: int,
    ) -> Tuple[ThrottleMode, float]:
        """
        Decide the throttle mode for the current frame.

        Patch 7: PREDICTIVE BURST on new track_ids.
        Before falling through to the reactive logic, we check whether
        any track_id in active_tracks is new (not in
        _seen_track_ids_for_burst). If so, we force BURST for this
        frame -- regardless of resolved_state. This shaves ~40ms (one
        IDLE sleep cycle) off every new track's TTFM race because we
        don't wait for the track to enter PENDING state before
        switching from IDLE to BURST.

        Args:
            active_tracks: List of per-track dicts with at least the keys:
                - track_id : int
                - resolved_state : str ("PENDING" | "VERIFIED_STUDENT"
                                        | "STRANGER" | "ANOMALY")
                - face_scanning_active : bool
            current_frame_index: Monotonic frame counter.

        Returns:
            (ThrottleMode, sleep_ms) -- the mode to apply and the
            wall-clock sleep duration in milliseconds.
        """
        # Honor the forced-BURST override window.
        if self._force_burst and current_frame_index < self._force_burst_until_frame:
            self.current_mode = ThrottleMode.BURST
            # Still update the seen-track-id set so we don't lose the
            # birth-frame signal when the override expires.
            self._update_seen_track_ids(active_tracks)
            return ThrottleMode.BURST, self.burst_sleep_ms
        if self._force_burst and current_frame_index >= self._force_burst_until_frame:
            # Override window expired.
            self._force_burst = False

        # --- Patch 7: Predictive BURST on new track_ids ---
        # Detect new track_ids BEFORE the reactive PENDING-state check.
        # A new track_id forces BURST for this frame even if the track
        # hasn't been classified as PENDING yet (the gating engine may
        # still be initializing its state for this track).
        active_tids = {
            int(t["track_id"]) for t in active_tracks if "track_id" in t
        }
        new_tids = active_tids - self._seen_track_ids_for_burst
        if new_tids and self.burst_enabled:
            # Birth frame for at least one track -- force BURST.
            self._seen_track_ids_for_burst |= new_tids
            # Prune disappeared tracks so re-entries get BURST again.
            self._prune_seen_track_ids(active_tids)
            self.current_mode = ThrottleMode.BURST
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "Patch 7: predictive BURST on new track_ids %s "
                    "(frame=%d)",
                    sorted(new_tids), current_frame_index,
                )
            return ThrottleMode.BURST, self.burst_sleep_ms

        # Always update + prune the seen set even on non-birth frames
        # so disappeared tracks are forgotten.
        self._update_seen_track_ids(active_tracks)
        self._prune_seen_track_ids(active_tids)

        # If there are any unresolved tracks with active face scanning,
        # we are in BURST mode.
        if not self.burst_enabled:
            self.current_mode = ThrottleMode.IDLE
            return ThrottleMode.IDLE, self.idle_sleep_ms

        for t in active_tracks:
            state = t.get("resolved_state", "PENDING")
            scanning = t.get("face_scanning_active", True)
            if state in ("PENDING", "ANOMALY") and scanning:
                self.current_mode = ThrottleMode.BURST
                return ThrottleMode.BURST, self.burst_sleep_ms

        # Otherwise -- empty scene or all verified / locked strangers.
        if not self.idle_enabled:
            self.current_mode = ThrottleMode.BURST
            return ThrottleMode.BURST, self.burst_sleep_ms

        self.current_mode = ThrottleMode.IDLE
        return ThrottleMode.IDLE, self.idle_sleep_ms

    # ------------------------------------------------------------------
    def _update_seen_track_ids(
        self, active_tracks: List[Dict[str, Any]]
    ) -> None:
        """Patch 7 helper: add all current track_ids to the seen set."""
        for t in active_tracks:
            tid = t.get("track_id")
            if tid is not None:
                try:
                    self._seen_track_ids_for_burst.add(int(tid))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    def _prune_seen_track_ids(self, active_tids: set) -> None:
        """
        Patch 7 helper: forget track_ids that are no longer active so a
        person who leaves and re-enters the frame (BoTSort assigns a
        new track_id) gets the BURST treatment again on re-entry.
        """
        if not active_tids:
            # Empty scene -- clear everything so the next entry gets BURST.
            if self._seen_track_ids_for_burst:
                self._seen_track_ids_for_burst.clear()
            return
        stale = self._seen_track_ids_for_burst - active_tids
        if stale:
            self._seen_track_ids_for_burst -= stale

    # ------------------------------------------------------------------
    def apply_sleep(self, sleep_ms: float) -> None:
        """Apply the throttled cycle sleep (no-op for BURST mode)."""
        if sleep_ms <= 0.0:
            return
        try:
            time.sleep(sleep_ms / 1000.0)
        except Exception as exc:                # pragma: no cover
            logger.warning("Throttle sleep interrupted: %s", exc)

    # ==================================================================
    # Patch 13: Adaptive YOLO interval.
    # ==================================================================
    def should_skip_yolo(self, current_frame_index: int) -> bool:
        """
        Decide whether the AI thread should skip the YOLO+BoTSORT
        inference call on this frame and reuse the cached track boxes.

        Returns True when ALL of the following hold:
          * adaptive_yolo_interval is enabled in config
          * current_mode == IDLE (all tracks resolved, set by the
            previous frame's decide_mode call)
          * the skip counter has not yet reached the configured interval

        When True is returned, the caller MUST:
          1. Skip self._tracking.process(frame)
          2. Reuse the cached tracks from the last YOLO frame
          3. Still run face detection / gating / rendering as usual

        When the counter reaches `interval`, we return False so this
        frame runs a real YOLO inference (which refreshes the cache).

        New track births force BURST mode (via Patch 7) which resets
        the counter and re-engages YOLO immediately -- so a new person
        entering the scene is never missed due to skip-frame cycling.

        Args:
            current_frame_index: Monotonic frame counter (unused in the
                current implementation but kept for telemetry hooks).

        Returns:
            True if YOLO should be skipped on this frame.
        """
        if not self.adaptive_yolo_enabled:
            return False

        # If we're in BURST mode (new track, anomaly, or PENDING track),
        # always run YOLO. Reset the skip counter so the next IDLE
        # entry starts a fresh cycle.
        if self.current_mode != ThrottleMode.IDLE:
            if self._yolo_skip_counter != 0:
                self._yolo_skip_stats["burst_overrides"] += 1
            self._yolo_skip_counter = 0
            self._yolo_skip_stats["executed"] += 1
            return False

        # IDLE mode. Advance the counter.
        self._yolo_skip_counter += 1

        # Every `interval`-th frame, run YOLO to refresh the cache and
        # let BoTSORT re-anchor its Kalman state.
        if self._yolo_skip_counter >= self.adaptive_yolo_interval:
            self._yolo_skip_counter = 0
            self._yolo_skip_stats["executed"] += 1
            return False

        # Skip YOLO this frame.
        self._yolo_skip_stats["skipped"] += 1
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(
                "Patch 13: skipping YOLO on frame %d (counter=%d/%d)",
                current_frame_index,
                self._yolo_skip_counter,
                self.adaptive_yolo_interval,
            )
        return True

    def yolo_skip_telemetry(self) -> Dict[str, Any]:
        """Return cumulative skip/execute counters for telemetry."""
        s = self._yolo_skip_stats
        total = s["skipped"] + s["executed"]
        return {
            "yolo_skipped_frames": int(s["skipped"]),
            "yolo_executed_frames": int(s["executed"]),
            "yolo_burst_overrides": int(s["burst_overrides"]),
            "yolo_skip_rate": (
                float(s["skipped"]) / float(total) if total > 0 else 0.0
            ),
            "adaptive_yolo_enabled": bool(self.adaptive_yolo_enabled),
            "adaptive_yolo_interval": int(self.adaptive_yolo_interval),
        }

    # ------------------------------------------------------------------
    def force_burst(self, frame_window: int, current_frame_index: int) -> None:
        """
        Force the controller into BURST mode for `frame_window` frames.
        Used by the gating engine when an ANOMALY state is detected to
        prevent data loss.
        """
        self._force_burst = True
        self._force_burst_until_frame = current_frame_index + max(1, frame_window)
        logger.info(
            "Force-BURST override armed until frame %d (window=%d)",
            self._force_burst_until_frame, frame_window,
        )

    # ==================================================================
    # 2. Adaptive TTFM Tracker.
    # ==================================================================
    def register_face_similarity(
        self,
        track_id: int,
        raw_similarity: float,
    ) -> float:
        """
        Register a raw face-similarity hit for a track and return the
        effective threshold to apply for this frame.

        The Adaptive TTFM relaxation logic:
          * Maintain a rolling window of the last 6 raw similarities.
          * If 3 of those 6 hits are >= 0.60 AND relaxation is enabled,
            arm a 30-frame relaxation window during which the effective
            threshold is lowered from 0.65 -> 0.60.
        """
        state = self._ttfm_states.get(track_id)
        if state is None:
            state = TTFMState(track_id=track_id)
            # Ensure the deque is sized to the configured window.
            state.recent_hits = deque(maxlen=self.ttfm_window)
            state.effective_threshold = self.sim_baseline
            self._ttfm_states[track_id] = state

        state.register_hit(raw_similarity)

        # Decrement any active relaxation window.
        if state.relaxation_frames_remaining > 0:
            state.relaxation_frames_remaining -= 1
            if state.relaxation_frames_remaining == 0:
                state.effective_threshold = self.sim_baseline
                logger.debug(
                    "TTFM relaxation expired for track %d -> threshold=%.2f",
                    track_id, state.effective_threshold,
                )
            else:
                state.effective_threshold = self.sim_dynamic_floor
                return state.effective_threshold

        # If relaxation is not active, evaluate whether to arm it.
        if (
            self.relaxation_enabled
            and len(state.recent_hits) >= self.ttfm_window
        ):
            hits_above = sum(
                1 for s in state.recent_hits
                if s >= self.ttfm_hit_similarity
            )
            if hits_above >= self.ttfm_hits_required:
                state.relaxation_frames_remaining = self.relaxation_window_frames
                state.effective_threshold = self.sim_dynamic_floor
                logger.info(
                    "TTFM relaxation armed for track %d | hits=%d/%d | "
                    "threshold=%.2f for %d frames",
                    track_id, hits_above, self.ttfm_window,
                    state.effective_threshold, self.relaxation_window_frames,
                )

        return state.effective_threshold

    # ------------------------------------------------------------------
    def get_effective_threshold(self, track_id: int) -> float:
        """Return the current effective threshold for a track."""
        state = self._ttfm_states.get(track_id)
        if state is None:
            return self.sim_baseline
        return state.effective_threshold

    # ==================================================================
    # 3. Pose-Weighted EMA Accumulator.
    # ==================================================================
    @staticmethod
    def compute_pose_weight(
        landmarks: np.ndarray,
        frontal_value: float = 1.0,
        floor_value: float = 0.25,
    ) -> float:
        """
        Compute a frontal pose weight from the 5-point landmark matrix.

        Approach:
          * The horizontal distance between the two eyes is a proxy for
            how frontal the face is.
          * The horizontal offset of the nose from the eye-pair midpoint
            is a proxy for yaw.
          * yaw_ratio = |nose_x - eye_mid_x| / eye_distance  (0 = frontal,
            larger = profile).
          * Map yaw_ratio in [0, 1.0] to a weight in [frontal_value, floor_value]
            via linear interpolation. yaw_ratio >= 1.0 saturates at the floor.

        Returns a float in [floor_value, frontal_value].
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return floor_value

        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
        except IndexError:
            return floor_value

        eye_dx = float(right_eye[0] - left_eye[0])
        eye_dy = float(right_eye[1] - left_eye[1])
        eye_distance = float(np.hypot(eye_dx, eye_dy))
        if eye_distance < 1e-3:
            return floor_value

        eye_mid_x = (left_eye[0] + right_eye[0]) * 0.5
        nose_offset_x = float(abs(nose[0] - eye_mid_x))
        yaw_ratio = nose_offset_x / eye_distance

        # Linear interpolation: yaw_ratio 0 -> frontal_value,
        #                       yaw_ratio >= 1.0 -> floor_value.
        if yaw_ratio >= 1.0:
            return floor_value
        t = max(0.0, min(1.0, yaw_ratio))
        weight = frontal_value + (floor_value - frontal_value) * t
        return float(weight)

    # ------------------------------------------------------------------
    def update_ema(
        self,
        track_id: int,
        raw_similarity: float,
        landmarks: Optional[np.ndarray],
    ) -> Tuple[float, bool]:
        """
        Update the Pose-Weighted EMA accumulator for a track.

        Args:
            track_id:      Active track ID.
            raw_similarity: Raw face-similarity score for this frame.
            landmarks:     5-point landmark matrix (5, 2) or None.

        Returns:
            (ema_score, is_match) where is_match is True iff:
              ema_score >= self.ema_threshold (0.62)
              AND cluster_size >= self.ema_min_cluster (4)
        """
        state = self._ema_states.get(track_id)
        if state is None:
            state = PoseEMAState(track_id=track_id)
            self._ema_states[track_id] = state

        # Compute the pose weight (floor if landmarks are missing).
        if landmarks is not None and _CV2_AVAILABLE:
            pose_weight = self.compute_pose_weight(
                landmarks,
                frontal_value=self.pose_weight_frontal,
                floor_value=self.pose_weight_floor,
            )
        else:
            pose_weight = self.pose_weight_floor

        # alpha_effective = alpha * pose_weight.
        alpha_eff = self.ema_base_alpha * pose_weight

        # EMA update: ema = (1 - alpha) * ema + alpha * observation.
        state.ema_score = (
            (1.0 - alpha_eff) * state.ema_score + alpha_eff * float(raw_similarity)
        )
        state.cluster_size += 1
        state.history.append((float(raw_similarity), float(pose_weight)))

        is_match = (
            state.ema_score >= self.ema_threshold
            and state.cluster_size >= self.ema_min_cluster
        )
        return state.ema_score, is_match

    # ------------------------------------------------------------------
    def get_ema_state(self, track_id: int) -> Tuple[float, int]:
        state = self._ema_states.get(track_id)
        if state is None:
            return 0.0, 0
        return state.ema_score, state.cluster_size

    # ------------------------------------------------------------------
    def reset_ema(self, track_id: int) -> None:
        state = self._ema_states.get(track_id)
        if state is not None:
            state.reset()

    # ==================================================================
    # 4. Velocity-Aware Crop Boost.
    # ==================================================================
    def update_velocity(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frame_index: int,
        frame_width: int,
    ) -> float:
        """
        Update the per-track velocity state and return the smoothed
        lateral velocity (v_x in px/frame).
        """
        state = self._velocity_states.get(track_id)
        if state is None:
            state = VelocityState(track_id=track_id)
            self._velocity_states[track_id] = state

        if state.last_centroid is None or state.last_frame_index is None:
            state.last_centroid = centroid
            state.last_frame_index = frame_index
            return 0.0

        dt = frame_index - state.last_frame_index
        if dt <= 0:
            return state.velocity_x

        raw_vx = (centroid[0] - state.last_centroid[0]) / float(dt)
        raw_vy = (centroid[1] - state.last_centroid[1]) / float(dt)

        # Compute the direction sign toward the scene center.
        scene_center_x = float(frame_width) * 0.5
        moving_toward_center = (
            (centroid[0] < scene_center_x and raw_vx > 0.0)
            or (centroid[0] >= scene_center_x and raw_vx < 0.0)
        )
        # If not moving toward the center, the effective velocity toward
        # the center is 0 (we only boost on center-seeking motion).
        signed_vx = abs(raw_vx) if moving_toward_center else 0.0

        state.velocity_history.append(signed_vx)
        # Smoothed velocity = mean of the rolling history.
        smoothed_vx = float(np.mean(state.velocity_history)) if state.velocity_history else 0.0
        state.velocity_x = smoothed_vx
        state.velocity_y = float(raw_vy)
        state.last_centroid = centroid
        state.last_frame_index = frame_index
        return smoothed_vx

    # ------------------------------------------------------------------
    def evaluate_crop_boost(
        self,
        track_id: int,
        face_crop_bgr: Optional[np.ndarray],
        face_bbox_width: int,
    ) -> CropBoostDecision:
        """
        Decide whether to apply the 4x bicubic crop boost for a track.

        Conditions (all must be true):
          * velocity_boost_enabled == True
          * smoothed |v_x| >= min_velocity_px_per_frame (3.0 px/frame)
          * face_bbox_width < small_face_width_px (40 px)
          * face_crop_bgr is a valid non-empty image
        """
        if not self.velocity_boost_enabled:
            return CropBoostDecision(should_boost=False, reason="disabled")

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return CropBoostDecision(should_boost=False, reason="empty_crop")

        state = self._velocity_states.get(track_id)
        if state is None:
            return CropBoostDecision(should_boost=False, reason="no_velocity_state")

        if state.velocity_x < self.min_velocity_px_per_frame:
            return CropBoostDecision(
                should_boost=False,
                reason=f"velocity={state.velocity_x:.2f}<{self.min_velocity_px_per_frame}",
            )

        if face_bbox_width >= self.small_face_width_px:
            return CropBoostDecision(
                should_boost=False,
                reason=f"width={face_bbox_width}>={self.small_face_width_px}",
            )

        # All conditions met -- apply the 4x bicubic upscaling.
        if not _CV2_AVAILABLE:
            return CropBoostDecision(should_boost=False, reason="cv2_unavailable")

        try:
            h, w = face_crop_bgr.shape[:2]
            target_w = max(1, int(w * self.upscale_factor))
            target_h = max(1, int(h * self.upscale_factor))
            upscaled = cv2.resize(
                face_crop_bgr,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )
            logger.debug(
                "Crop boost applied | track=%d | (%d,%d) -> (%d,%d) | v_x=%.2f",
                track_id, w, h, target_w, target_h, state.velocity_x,
            )
            return CropBoostDecision(
                should_boost=True,
                upscaled_crop=upscaled,
                scale_factor=self.upscale_factor,
                reason="boost_applied",
            )
        except Exception as exc:
            logger.warning(
                "Crop boost failed for track %d: %s\n%s",
                track_id, exc, traceback.format_exc(),
            )
            return CropBoostDecision(should_boost=False, reason=f"error:{exc}")

    # ==================================================================
    # Lifecycle management.
    # ==================================================================
    def drop_track(self, track_id: int) -> None:
        """Drop all per-track state for a track that has exited the frame."""
        self._ttfm_states.pop(track_id, None)
        self._ema_states.pop(track_id, None)
        self._velocity_states.pop(track_id, None)

    # ------------------------------------------------------------------
    def all_track_ids(self) -> List[int]:
        """Return the union of all tracked IDs across TTFM/EMA/Velocity."""
        ids: set = set()
        ids.update(self._ttfm_states.keys())
        ids.update(self._ema_states.keys())
        ids.update(self._velocity_states.keys())
        return sorted(ids)

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        """Return a snapshot of the engine state for the dashboard."""
        return {
            "current_mode": self.current_mode.value,
            "idle_sleep_ms": self.idle_sleep_ms,
            "burst_sleep_ms": self.burst_sleep_ms,
            "force_burst_active": self._force_burst,
            "ttfm_track_count": len(self._ttfm_states),
            "ema_track_count": len(self._ema_states),
            "velocity_track_count": len(self._velocity_states),
            "sim_baseline": self.sim_baseline,
            "sim_dynamic_floor": self.sim_dynamic_floor,
            "ema_threshold": self.ema_threshold,
        }

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Release all per-track state."""
        self._ttfm_states.clear()
        self._ema_states.clear()
        self._velocity_states.clear()
        self._force_burst = False
        gc.collect()
        logger.info("ResourceOptEngine shut down; per-track state cleared.")


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance res_opt_engine self-test ===")

    cfg = ConfigRegistry.load("config/config.yaml") if ConfigRegistry else {}
    engine = ResourceOptEngine(cfg)

    # --- Throttle mode test ---
    mode, sleep = engine.decide_mode([], current_frame_index=0)
    logger.info("Empty scene -> mode=%s sleep=%.1fms", mode.value, sleep)

    mode, sleep = engine.decide_mode(
        [{"track_id": 1, "resolved_state": "PENDING",
          "face_scanning_active": True}],
        current_frame_index=0,
    )
    logger.info("Pending track -> mode=%s sleep=%.1fms", mode.value, sleep)

    mode, sleep = engine.decide_mode(
        [{"track_id": 1, "resolved_state": "VERIFIED_STUDENT",
          "face_scanning_active": False}],
        current_frame_index=0,
    )
    logger.info("Verified only -> mode=%s sleep=%.1fms", mode.value, sleep)

    # --- TTFM test ---
    for i in range(6):
        thr = engine.register_face_similarity(10, 0.61 if i < 3 else 0.50)
    logger.info("TTFM after 6 hits (3 above 0.60) -> thr=%.2f", thr)

    # --- Pose EMA test ---
    rng = np.random.default_rng(0)
    kps = np.array([
        [30, 50], [80, 50], [55, 80], [40, 100], [70, 100],
    ], dtype=np.float32)
    for _ in range(6):
        ema, is_match = engine.update_ema(11, 0.70, kps)
    logger.info("EMA after 6 frontal hits -> ema=%.3f match=%s", ema, is_match)

    # --- Velocity crop boost test ---
    engine.update_velocity(12, (100.0, 200.0), 0, 1280)
    engine.update_velocity(12, (110.0, 200.0), 1, 1280)
    if _CV2_AVAILABLE:
        crop = np.zeros((30, 30, 3), dtype=np.uint8)
        decision = engine.evaluate_crop_boost(12, crop, face_bbox_width=30)
        logger.info(
            "Crop boost decision -> boost=%s scale=%d reason=%s",
            decision.should_boost, decision.scale_factor, decision.reason,
        )

    engine.shutdown()
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()