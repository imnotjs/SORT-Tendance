"""
SORT-tendance :: src/core/tracking_engine.py

Production-grade Object Tracking Core.

Responsibilities:
  1. Unified YOLOv8 + BoTSORT pipeline (Ultralytics native).
  2. Cold-start pre-warmup using blank matrices + multi-dimensional
     randomized noise arrays to pre-compile CUDA NMS execution paths
     and instantiate internal Kalman filter state structures.
  3. Overlapping Mitigation Strategy: ongoing IoU matrix computation
     for person-class boxes; if IoU > 0.7, flag the affected tracks
     with `identity_locked = True` to suspend state mutations.
  4. Clean Display Rendering: internal BoTSORT track IDs, spatial
     hashes, and Kalman parameters are kept purely backend. The
     exposed BBox payload contains ONLY high-level clean state tags
     resolved by the gating state machine.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import gc
import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guards. The module imports cleanly even when the
# heavyweight ML stack is absent so that the orchestrator can surface a
# proper error rather than crashing on import.
# ---------------------------------------------------------------------------
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _ULTRALYTICS_AVAILABLE = False
    YOLO = None  # type: ignore

# Local config registry -- reuses the singleton from database_manager so
# the entire pipeline shares one cached config dictionary.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.database_manager import ConfigRegistry
except ImportError:                       # pragma: no cover
    ConfigRegistry = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.tracking_engine")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class CleanBBox:
    """
    Clean bounding box payload exposed to the rendering layer.

    Strict invariant: this structure MUST NOT carry internal tracking IDs,
    spatial hashes, Kalman matrices, or BoTSORT track numbers. It exposes
    only the resolved high-level state tag assigned by the gating engine.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    # Resolved label is one of:
    #   "[NRP / Student Name]"  (verified student)
    #   "[Stranger_XX]"         (locked stranger)
    #   "[ANOMALY]"             (face without body)
    #   "[PENDING]"             (track awaiting state resolution)
    resolved_label: str = "[PENDING]"
    # Confidence surface for the detection (NOT exposed to UI overlays
    # but kept here for downstream logging).
    det_conf: float = 0.0
    # Internal handle to the backend track (opaque to the renderer).
    _backend_track_id: Optional[int] = None
    # Lock flag -- mirrors the backend identity_locked state so the
    # rendering loop can skip re-evaluating frozen tracks.
    identity_locked: bool = False

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "resolved_label": self.resolved_label,
            "det_conf": self.det_conf,
            "identity_locked": self.identity_locked,
        }


@dataclass
class InternalTrack:
    """
    Backend track representation. This object lives purely in RAM and
    is NEVER rendered. The gating engine reads/writes its lock state and
    resolved label, then projects them onto a CleanBBox for the UI.
    """
    track_id: int
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    det_conf: float
    # Identity lock -- when True, the tracker must NOT mutate the track's
    # box coordinates, ID association, or Kalman state. Set when another
    # person-class box overlaps this one with IoU > 0.7.
    identity_locked: bool = False
    # Resolved label injected by the gating engine.
    resolved_label: str = "[PENDING]"
    # Last centroid (px) -- cached for IoU and velocity computations.
    centroid: Tuple[float, float] = (0.0, 0.0)
    # Optional face bbox association (set by the orchestrator each frame).
    face_bbox: Optional[Tuple[int, int, int, int]] = None

    def as_clean_bbox(self) -> CleanBBox:
        return CleanBBox(
            x1=int(self.x1), y1=int(self.y1),
            x2=int(self.x2), y2=int(self.y2),
            resolved_label=self.resolved_label,
            det_conf=self.det_conf,
            _backend_track_id=self.track_id,
            identity_locked=self.identity_locked,
        )


# ============================================================================
# Geometry Utilities
# ============================================================================
def _iou(a: InternalTrack, b: InternalTrack) -> float:
    """Standard IoU between two InternalTrack boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _iou_matrix(tracks: List[InternalTrack]) -> np.ndarray:
    """
    Vectorized NxN IoU matrix for the supplied track list.
    Returns a float32 array of shape (N, N) with 0 on the diagonal.
    """
    n = len(tracks)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    boxes = np.array(
        [[t.x1, t.y1, t.x2, t.y2] for t in tracks],
        dtype=np.float32,
    )
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    inter = iw * ih

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - inter
    iou = np.where(union > 0.0, inter / np.where(union == 0.0, 1.0, union), 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou.astype(np.float32)


# ============================================================================
# Tracking Engine
# ============================================================================
class TrackingEngine:
    """
    Unified YOLOv8 detector + BoTSORT tracker with overlapping mitigation.

    Lifecycle:
        engine = TrackingEngine(config)
        engine.initialize()           # builds YOLO + BoTSORT, locks CPU affinity
        engine.warmup()               # cold-start pre-warm (blank + noise passes)
        tracks = engine.process(frame)  # per-frame detection + tracking + IoU lock
        engine.shutdown()
    """

    PERSON_CLASS_ID: int = 0  # COCO 'person'

    # ------------------------------------------------------------------
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required for TrackingEngine. "
                "Install with: pip install ultralytics"
            )
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrackingEngine.")
        if not _CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for TrackingEngine.")

        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        # ------------------------------------------------------------------
        # P2-M3 fix :: defensive config access.
        #
        # Previously: 12+ raw `self.config["tracking"]["yolo"]["device"]`
        # style accesses. If any key was missing from config.yaml, this
        # raised a cryptic KeyError with a deep stack trace and no hint
        # about WHICH key was missing -- operators saw a stack trace
        # instead of a config error message.
        #
        # Fix: chain .get() calls with sensible defaults so a missing
        # key produces a clear log warning + a usable default, rather
        # than crashing the engine. Critical fields (weights) are still
        # validated explicitly below.
        # ------------------------------------------------------------------
        tcfg = self.config.get("tracking", {})
        self.yolo_cfg: Dict[str, Any] = tcfg.get("yolo", {})
        self.botsort_cfg: Dict[str, Any] = tcfg.get("botsort", {})
        self.warmup_cfg: Dict[str, Any] = tcfg.get("warmup", {})

        # Track which config keys fell back to defaults so we can warn once.
        _defaults_used: List[str] = []

        self.iou_lock_threshold: float = float(
            self.botsort_cfg.get("iou_lock_threshold", 0.7)
        )
        if "iou_lock_threshold" not in self.botsort_cfg:
            _defaults_used.append("tracking.botsort.iou_lock_threshold=0.7")

        self.device: str = str(self.yolo_cfg.get("device", "cuda:0"))
        if "device" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.device=cuda:0")

        self.imgsz: int = int(self.yolo_cfg.get("imgsz", 640))
        if "imgsz" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.imgsz=640")

        self.conf_threshold: float = float(self.yolo_cfg.get("conf_threshold", 0.25))
        if "conf_threshold" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.conf_threshold=0.25")

        self.iou_threshold: float = float(self.yolo_cfg.get("iou_threshold", 0.45))
        if "iou_threshold" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.iou_threshold=0.45")

        self.target_classes: List[int] = list(self.yolo_cfg.get("target_classes", [0]))
        if "target_classes" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.target_classes=[0]")

        self.half_precision: bool = bool(self.yolo_cfg.get("half_precision", False))
        if "half_precision" not in self.yolo_cfg:
            _defaults_used.append("tracking.yolo.half_precision=False")

        self.weights_path: str = str(self.yolo_cfg.get("weights", ""))

        # Critical field -- fail with a CLEAR error rather than a stack trace.
        if not self.weights_path:
            raise ValueError(
                "TrackingEngine: required config key 'tracking.yolo.weights' "
                "is missing or empty. Specify the YOLOv8 weights path in "
                "config.yaml (e.g. 'weights: ./weights/yolov8n.pt')."
            )

        if _defaults_used:
            logger.warning(
                "TrackingEngine: %d config key(s) missing -- using defaults: %s",
                len(_defaults_used), ", ".join(_defaults_used),
            )

        # YOLO model handle (initialized lazily).
        self.model: Optional[Any] = None
        # Path to the custom BoTSORT YAML config (Ultralytics 8.4.x API).
        # Written by `_write_botsort_yaml()` during initialize().
        self._tracker_yaml: Optional[str] = None
        self._tracker_initialized: bool = False
        self._warmed: bool = False

        # Rolling cache of InternalTrack objects per frame (keyed by track_id).
        self._track_cache: Dict[int, InternalTrack] = {}

    # ------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Load the YOLOv8 weights, configure the BoTSORT tracker, and bind
        the CUDA device.
        """
        if self.model is not None:
            logger.warning("TrackingEngine already initialized; skipping.")
            return

        # P0-C5 fix: do NOT pre-check weights file existence. Ultralytics'
        # YOLO() constructor auto-downloads missing weights from the
        # official repo (https://github.com/ultralytics/assets/releases),
        # which is more robust than our pre-check that hard-fails before
        # that auto-download path runs. Log the intent so operators can
        # see what's happening if the network is slow.
        if not os.path.isfile(self.weights_path):
            logger.warning(
                "YOLO weights not found at %s -- Ultralytics will attempt "
                "auto-download. If the network is unreachable, place the "
                "weights file there manually.",
                self.weights_path,
            )

        logger.info(
            "Loading YOLOv8 | weights=%s | device=%s | imgsz=%d | half=%s",
            self.weights_path, self.device, self.imgsz, self.half_precision,
        )
        try:
            self.model = YOLO(self.weights_path)
            # Warm-up the underlying nn.Module on the target device so the
            # first real frame does not pay the lazy-transfer cost.
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception as exc:
            logger.error("YOLO load failed: %s\n%s", exc, traceback.format_exc())
            raise

        # Write the BoTSORT YAML config (Ultralytics 8.4.x YAML-based tracker API).
        self._tracker_yaml = self._write_botsort_yaml()
        self._tracker_initialized = True
        logger.info(
            "TrackingEngine initialized | tracker=BoTSORT (YAML) | iou_lock=%.2f | yaml=%s",
            self.iou_lock_threshold, self._tracker_yaml,
        )

    # ------------------------------------------------------------------
    # Cold-Start Pre-Warmup.
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """
        Execute the multi-pass tracking pre-warm sequence:
          1. Blank matrix passes (np.zeros) -- forces cuDNN/cuBLAS handle
             initialization and locks the first NMS execution path.
          2. Multi-dimensional randomized noise passes (np.random) --
             forces the runtime to compile NMS + Kalman update paths
             across a range of input distributions.

        This eliminates the first-frame stall that occurs when the
        detector lazily selects CUDA kernels on the first real image.
        """
        if not self._tracker_initialized:
            raise RuntimeError(
                "TrackingEngine must be initialized before warmup."
            )

        blank_passes: int = int(self.warmup_cfg["blank_passes"])
        noise_passes: int = int(self.warmup_cfg["noise_passes"])

        logger.info(
            "TrackingEngine warmup starting | blank=%d noise=%d imgsz=%d",
            blank_passes, noise_passes, self.imgsz,
        )

        # ---- Pass 1: blank matrices -------------------------------------
        for i in range(blank_passes):
            blank = np.zeros(
                (self.imgsz, self.imgsz, 3), dtype=np.uint8
            )
            try:
                self._run_inference_once(blank, persist=False)
                logger.debug("Warmup blank pass %d/%d OK", i + 1, blank_passes)
            except Exception as exc:
                logger.warning("Warmup blank pass %d failed: %s", i + 1, exc)

        # ---- Pass 2: randomized noise matrices --------------------------
        # Vary structural dimensions to exercise multiple NMS branches.
        noise_shapes: List[Tuple[int, int]] = [
            (self.imgsz, self.imgsz),
            (480, 640),
            (720, 1280),
            (320, 320),
        ]
        rng = np.random.default_rng(seed=2024)
        for i in range(noise_passes):
            h, w = noise_shapes[i % len(noise_shapes)]
            # Alternate uniform / normal noise to vary operator paths.
            if i % 2 == 0:
                noise = rng.uniform(0, 255, size=(h, w, 3)).astype(np.uint8)
            else:
                noise = rng.normal(128, 50, size=(h, w, 3))
                noise = np.clip(noise, 0, 255).astype(np.uint8)
            try:
                self._run_inference_once(noise, persist=False)
                logger.debug(
                    "Warmup noise pass %d/%d shape=(%d,%d) OK",
                    i + 1, noise_passes, h, w,
                )
            except Exception as exc:
                logger.warning("Warmup noise pass %d failed: %s", i + 1, exc)

        # Force a CUDA synchronize so any queued kernel compilations finish
        # before we return control to the orchestrator.
        if _TORCH_AVAILABLE and self.device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                logger.warning("torch.cuda.synchronize() during warmup: %s", exc)

        self._warmed = True
        gc.collect()
        logger.info("TrackingEngine warmup complete.")

    # ------------------------------------------------------------------
    def _write_botsort_yaml(self) -> str:
        """
        Write the BoTSORT tracker configuration to a YAML file that can be
        passed to `model.track(tracker=..., persist=True)`.

        Strategy: load ultralytics's shipped default `botsort.yaml` as the
        base (so we inherit ALL required keys for the running version,
        including any renames like `cmc_method` -> `gmc_method` in 8.4.x),
        then apply our config overrides on top.

        Falls back to a manual dict if the default file can't be located.
        """
        import yaml  # PyYAML, transitive dep of ultralytics

        # ----------------------------------------------------------------
        # Step 1: locate ultralytics's shipped default botsort.yaml
        # ----------------------------------------------------------------
        cfg_dict: Dict[str, Any] = {}
        try:
            import ultralytics
            ultralytics_root = os.path.dirname(ultralytics.__file__)
            default_yaml_path = os.path.join(
                ultralytics_root, "cfg", "trackers", "botsort.yaml"
            )
            if os.path.isfile(default_yaml_path):
                with open(default_yaml_path, "r") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                logger.info(
                    "Loaded ultralytics default botsort.yaml | source=%s",
                    default_yaml_path,
                )
            else:
                logger.warning(
                    "Could not locate ultralytics default botsort.yaml at %s; "
                    "falling back to manual cfg dict.",
                    default_yaml_path,
                )
        except Exception as exc:
            logger.warning(
                "Failed to import ultralytics package for default YAML path "
                "resolution: %s -- falling back to manual cfg dict.", exc,
            )

        # ----------------------------------------------------------------
        # Step 2: fallback manual dict (covers all known 8.4.x keys).
        # Only used if step 1 failed.
        # ----------------------------------------------------------------
        if not cfg_dict:
            cfg_dict = {
                "tracker_type": "botsort",
                "track_high_thresh": 0.5,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_high_confidence_scores": True,
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.25,
                "with_reid": False,
                "model": "auto",
                "gmc_method": "sparseOptFlow",
                "verbose": False,
            }

        # ----------------------------------------------------------------
        # Step 3: apply our config overrides.
        # ----------------------------------------------------------------
        cfg_dict["tracker_type"] = "botsort"
        cfg_dict["track_high_thresh"] = float(
            self.botsort_cfg["new_track_thresh"]
        )
        cfg_dict["track_low_thresh"] = 0.1
        cfg_dict["new_track_thresh"] = float(
            self.botsort_cfg["new_track_thresh"]
        )
        cfg_dict["track_buffer"] = int(self.botsort_cfg["track_buffer"])
        cfg_dict["match_thresh"] = float(self.botsort_cfg["match_thresh"])
        cfg_dict["fuse_high_confidence_scores"] = True
        cfg_dict["proximity_thresh"] = float(
            self.botsort_cfg["proximity_thresh"]
        )
        cfg_dict["appearance_thresh"] = float(
            self.botsort_cfg["appearance_thresh"]
        )
        # ReID integration -- we have our own OSNet AIN pipeline, so disable
        # ultralytics's built-in ReID path.
        cfg_dict["with_reid"] = False
        cfg_dict["model"] = "auto"
        # `gmc_method` was renamed from `cmc_method` in ultralytics 8.4.x.
        # Read either key from our config for backward compatibility.
        # Patch 64 (hotfix M) :: Default changed from "sparseOptFlow" to
        # "none". The sparseOptFlow path (cv2.calcOpticalFlowPyrLK) was
        # the documented root cause of intermittent 0xC0000005 access
        # violations in ultralytics/trackers/utils/gmc.py line 311. GMC
        # is unnecessary for a static camera (the only kind this system
        # supports), so we default to the no-op identity path. If a
        # future moving-camera use case ever needs GMC, set
        # cmc_method/gmc_method explicitly in config.yaml.
        _gmc_default = "none"
        _gmc_value = str(
            self.botsort_cfg.get(
                "gmc_method",
                self.botsort_cfg.get("cmc_method", _gmc_default),
            )
        )
        # Safety net: if config explicitly requests "sparseOptFlow",
        # log a warning so the operator knows the crash risk they are
        # opting into. We do NOT silently override -- if they asked for
        # it explicitly, respect their choice.
        if _gmc_value.lower() in ("sparseoptflow", "opticalflow"):
            logger.warning(
                "TrackingEngine: GMC method '%s' is enabled. This path "
                "(cv2.calcOpticalFlowPyrLK) has caused intermittent "
                "0xC0000005 access violations in main.py. For a static "
                "camera, set cmc_method='none' in config.yaml to "
                "disable GMC and avoid the crash.",
                _gmc_value,
            )
        cfg_dict["gmc_method"] = _gmc_value
        # P2-M2 fix: purge legacy cmc_method key after merge so old
        # Ultralytics builds cannot accidentally reactivate sparseOptFlow.
        cfg_dict.pop("cmc_method", None)
        cfg_dict["verbose"] = False
        # P2-M1 fix: pass frame_rate to BoTSORT so track_buffer (in frames)
        # translates to the correct real-time horizon.
        try:
            cfg_dict["frame_rate"] = int(
                self.config.get("camera", {}).get("target_fps", 30)
            )
        except (TypeError, ValueError):
            cfg_dict["frame_rate"] = 30

        # ----------------------------------------------------------------
        # Step 4: write the merged YAML to storage/.
        # ----------------------------------------------------------------
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        yaml_path = os.path.join(project_root, "storage", "botsort_custom.yaml")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        logger.info(
            "BoTSORT YAML written | buffer=%d | match=%.2f | gmc=%s | path=%s",
            cfg_dict["track_buffer"], cfg_dict["match_thresh"],
            cfg_dict.get("gmc_method"), yaml_path,
        )
        return yaml_path

    # ------------------------------------------------------------------
    # Inference helpers.
    # ------------------------------------------------------------------
    def _run_inference_once(
        self, frame_bgr: np.ndarray, persist: bool = False
    ) -> List[InternalTrack]:
        """
        Run YOLO + (optionally) BoTSORT tracking on a single BGR frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR image.
            persist:   If True, use `model.track()` which maintains BoTSORT
                    Kalman state across calls and returns persistent track
                    IDs. If False, run detection-only warmup pass.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        common_kwargs = dict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device=self.device,
            # P0-C4 fix: `quantize` is NOT a valid Ultralytics inference kwarg
            # (it only applies to model.export). The correct kwarg is `half`.
            # The previous `quantize='fp16'` raised TypeError on every call,
            # caught by the try/except below, silently returning [] (zero
            # detections). System appeared to run but saw nothing.
            half=self.half_precision,
            verbose=False,
            save=False,
        )

        try:
            if persist:
                # Ultralytics 8.4.x: model.track() handles BoTSORT internally,
                # configured via the YAML file written by _write_botsort_yaml().
                # `persist=True` keeps Kalman state alive across calls.
                results = self.model.track(
                    tracker=self._tracker_yaml,
                    persist=True,
                    **common_kwargs,
                )
            else:
                results = self.model.predict(**common_kwargs)
        except Exception as exc:
            logger.error(
                "YOLO %s failed: %s\n%s",
                "track" if persist else "predict", exc, traceback.format_exc(),
            )
            return []

        if not results:
            return []

        res = results[0]
        try:
            if res.boxes is None or len(res.boxes) == 0:
                return []
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls = res.boxes.cls.cpu().numpy().astype(np.float32)
            # `boxes.id` is None when no tracks have been assigned yet (cold
            # start) or when running detection-only (predict). It's a tensor
            # of int64 track IDs otherwise.
            if persist and res.boxes.id is not None:
                tids = res.boxes.id.cpu().numpy().astype(np.int64)
            else:
                tids = None
        except Exception as exc:
            logger.error("Detection tensor extraction failed: %s", exc)
            return []

        tracks: List[InternalTrack] = []
        for i in range(len(xyxy)):
            track_id = int(tids[i]) if tids is not None else -1
            # Carry forward cached state for persistent tracks.
            cached = self._track_cache.get(track_id) if track_id > 0 else None
            resolved_label = cached.resolved_label if cached else "[PENDING]"
            identity_locked = cached.identity_locked if cached else False
            face_bbox = cached.face_bbox if cached else None

            tracks.append(InternalTrack(
                track_id=track_id,
                cls=int(cls[i]),
                x1=float(xyxy[i, 0]), y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]), y2=float(xyxy[i, 3]),
                det_conf=float(conf[i]),
                identity_locked=identity_locked,
                resolved_label=resolved_label,
                centroid=(
                    float(xyxy[i, 0] + xyxy[i, 2]) * 0.5,
                    float(xyxy[i, 1] + xyxy[i, 3]) * 0.5,
                ),
                face_bbox=face_bbox,
            ))

        # Refresh rolling cache for persistent tracks.
        # P1-H4 fix: TTL eviction instead of hard rebuild. Survives
        # single-frame occlusion gaps without losing the gating-engine-
        # injected resolved_label / face_bbox on the cached track.
        if persist:
            new_cache = {t.track_id: t for t in tracks if t.track_id > 0}
            _ttl = max(1, int(self.botsort_cfg.get("track_buffer", 30)))
            for tid, old_t in list(self._track_cache.items()):
                if tid not in new_cache:
                    miss = int(getattr(old_t, "_miss_count", 0)) + 1
                    if miss < _ttl:
                        # Preserve entry; bump miss counter.
                        try:
                            object.__setattr__(old_t, "_miss_count", miss)
                        except (AttributeError, TypeError):
                            pass  # frozen dataclass; skip
                        new_cache[tid] = old_t
            # Reset miss counter for tracks that did appear this frame.
            for tid, t in new_cache.items():
                try:
                    object.__setattr__(t, "_miss_count", 0)
                except (AttributeError, TypeError):
                    pass
            self._track_cache = new_cache

        return tracks

    # ------------------------------------------------------------------
    # Overlapping Mitigation Strategy.
    # ------------------------------------------------------------------
    def _apply_iou_locks(self, tracks: List[InternalTrack]) -> None:
        """
        Compute the NxN IoU matrix for person-class tracks and lock any
        track whose IoU with another person exceeds the threshold (0.7).

        P1-H7 fix (docstring correction): this method only sets the
        Python-level `identity_locked` boolean on InternalTrack objects
        in this engine's _track_cache. It does NOT suspend BoTSORT's
        internal Kalman updates -- BoTSORT runs entirely inside
        `model.track()` (Ultralytics internals) and this engine has no
        hook to suspend per-track state mutation there.

        The lock is ADVISORY ONLY -- consumed by the renderer for visual
        treatment (e.g. distinct box color for locked tracks) and by the
        gating engine to suppress identity re-evaluation. Two overlapping
        persons whose IoU exceeds the threshold will still have their
        BoTSORT track IDs swapped at the Kalman level; downstream code
        must re-apply resolved labels from its own _resolved_track_ids map.

        To actually prevent box swaps, the orchestrator must override the
        returned box with the cached box in _run_inference_once when
        `identity_locked=True` (not currently implemented).
        """
        person_tracks = [t for t in tracks if t.cls == self.PERSON_CLASS_ID]
        if len(person_tracks) < 2:
            # No overlaps possible -- clear any stale locks.
            for t in tracks:
                t.identity_locked = False
            return

        iou = _iou_matrix(person_tracks)
        # A track is locked if ANY other person exceeds the threshold.
        exceed = (iou > self.iou_lock_threshold)
        lock_flags = exceed.any(axis=1)

        # Map the lock flags back onto the original track list (preserving
        # order) and update the cache.
        person_idx = 0
        for t in tracks:
            if t.cls == self.PERSON_CLASS_ID:
                t.identity_locked = bool(lock_flags[person_idx])
                person_idx += 1
            else:
                t.identity_locked = False
            # Refresh cache with the new lock state.
            # P1-H4 fix: skip cache pollution from track_id == -1 (cold-start).
            if t.track_id > 0:
                self._track_cache[t.track_id] = t

        if lock_flags.any():
            logger.debug(
                "IoU lock applied to %d/%d person tracks (threshold=%.2f)",
                int(lock_flags.sum()), len(person_tracks), self.iou_lock_threshold,
            )

    # ------------------------------------------------------------------
    # Public per-frame entry point.
    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> List[InternalTrack]:
        """
        Run the full Detect -> Track -> IoU-Lock pipeline on one frame.

        Returns:
            List[InternalTrack] -- the active backend tracks for this
            frame. The orchestrator/gating engine may then project these
            to CleanBBox objects for the rendering loop.
        """
        if not self._tracker_initialized:
            raise RuntimeError("TrackingEngine is not initialized.")
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # 1) Detection + persistent tracking.
        tracks = self._run_inference_once(frame_bgr, persist=True)
        if not tracks:
            return []

        # 2) IoU-based overlapping mitigation on person-class tracks.
        self._apply_iou_locks(tracks)

        return tracks

    # ------------------------------------------------------------------
    # Clean projection (renderer-facing).
    # ------------------------------------------------------------------
    def to_clean_bboxes(self, tracks: List[InternalTrack]) -> List[CleanBBox]:
        """
        Project backend tracks to clean renderer-facing bounding boxes.

        Strict invariant: this is the ONLY public method that returns
        structures intended for the rendering loop. It strips all
        internal tracking IDs, spatial hashes, and Kalman parameters
        from the visible payload -- the renderer sees only the resolved
        high-level state label and the box coordinates.
        """
        return [t.as_clean_bbox() for t in tracks]

    # ------------------------------------------------------------------
    # Gating engine hooks.
    # ------------------------------------------------------------------
    def set_resolved_label(self, track_id: int, label: str) -> bool:
        """
        Inject a resolved state label ([NRP / Name], [Stranger_XX], or
        [ANOMALY]) onto a cached backend track. Returns True if the
        track was found and updated.
        """
        track = self._track_cache.get(track_id)
        if track is None:
            return False
        track.resolved_label = label
        return True

    def set_face_bbox(
        self, track_id: int, face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> bool:
        track = self._track_cache.get(track_id)
        if track is None:
            return False
        track.face_bbox = face_bbox
        return True

    def get_track(self, track_id: int) -> Optional[InternalTrack]:
        return self._track_cache.get(track_id)

    def all_tracks(self) -> List[InternalTrack]:
        return list(self._track_cache.values())

    def is_warmed(self) -> bool:
        return self._warmed

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Release YOLO model + tracker and free GPU memory."""
        try:
            if self.model is not None and hasattr(self.model, "model"):
                # P2-M4 fix: was `del self.model` + silent `except: pass`.
                # Just unbind the reference; Ultralytics internals may still
                # hold a ref but Python GC will release when refcount hits 0.
                # Log failures so operators notice silent leaks.
                self.model = None
        except Exception as exc:
            logger.warning("TrackingEngine.shutdown: model release failed: %s", exc)
        self.model = None
        self._tracker_yaml = None
        self._tracker_initialized = False
        self._warmed = False
        self._track_cache.clear()
        if _TORCH_AVAILABLE and self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                # P2-M4 fix: was silent -- log so CUDA cleanup failures surface.
                logger.warning("TrackingEngine.shutdown: cuda.empty_cache failed: %s", exc)
        gc.collect()
        logger.info("TrackingEngine shut down; GPU caches cleared.")


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness (requires ultralytics + torch)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance tracking_engine self-test ===")

    cfg = ConfigRegistry.load("config/config.yaml") if ConfigRegistry else {}
    engine = TrackingEngine(cfg)
    engine.initialize()
    engine.warmup()

    # Synthetic frame -> ensure pipeline does not crash.
    rng = np.random.default_rng(0)
    test_frame = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    tracks = engine.process(test_frame)
    clean = engine.to_clean_bboxes(tracks)
    logger.info(
        "Self-test OK | tracks=%d | clean_bboxes=%d",
        len(tracks), len(clean),
    )

    engine.shutdown()
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()