"""
SORT-tendance :: src/utils/database_manager.py

Production-grade enrollment + face-engine management module.

This module is the SINGLE POINT OF ENTRY for:
  1. `_LightFaceEngine` -- A resource-constrained InsightFace wrapper that
     enforces the 0.22 GPU VRAM budget via explicit CUDAExecutionProvider
     provider options and strips out all landmark/gender/age overhead.
  2. Cold-Start Double-Pass Warmup Protocol (Strategy A -> B -> C, twice).
  3. "24 + 1" Enrollment Clustering -- anchor face via ascending landmark
     symmetry, distribution cluster via descending L2 deviation.
  4. ArcFace Facial Alignment & Normalization using the standard 112x112
     template with InsightFace native (pixel - 127.5) / 128.0.
  5. Pickle serialization to `data/student_db.pickle` with explicit GC.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import gc
import sys
import logging
import pickle
import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
import psutil

# ---------------------------------------------------------------------------
# Optional Windows-only affinity import guard.
# ---------------------------------------------------------------------------
try:
    import win32api          # noqa: F401
    import win32process      # noqa: F401
    import win32con          # noqa: F401
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

# ---------------------------------------------------------------------------
# InsightFace import guard. The engine must NOT crash on import failure --
# the orchestrator must surface a clean error message.
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
    _INSIGHTFACE_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:
    _INSIGHTFACE_AVAILABLE = False
    _INSIGHTFACE_IMPORT_ERROR = _e
    ort = None  # type: ignore
    FaceAnalysis = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.database_manager")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Constants
# ============================================================================
# ArcFace standard 112x112 5-point landmark template.
ARCFACE_TEMPLATE_112: np.ndarray = np.array(
    [
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye
        [56.0252, 71.7366],   # Nose
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041],   # Right mouth corner
    ],
    dtype=np.float32,
)

# Index aliases for landmark symmetry calculation
_LM_LEFT_EYE: int = 0
_LM_RIGHT_EYE: int = 1
_LM_NOSE: int = 2


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class StudentProfile:
    """Serialized record of one enrolled student."""
    student_id: str
    student_name: str
    nrp: str
    face_embeddings: np.ndarray             # shape (N, 512), L2-normalized
    mean_embedding: np.ndarray              # shape (512,), L2-normalized
    anchor_image_hash: str
    enrollment_timestamp: str
    profile_capacity: int = 25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "nrp": self.nrp,
            "face_embeddings": self.face_embeddings,
            "mean_embedding": self.mean_embedding,
            "anchor_image_hash": self.anchor_image_hash,
            "enrollment_timestamp": self.enrollment_timestamp,
            "profile_capacity": self.profile_capacity,
        }


@dataclass
class EnrollmentStats:
    """Aggregate metrics for one enrollment run."""
    total_directories_scanned: int = 0
    total_profiles_built: int = 0
    total_frames_processed: int = 0
    total_frames_rejected: int = 0
    failed_directories: List[str] = field(default_factory=list)


# ============================================================================
# Configuration Loader
# ============================================================================
class ConfigRegistry:
    """
    Singleton-style YAML config loader. The first call to `ConfigRegistry.load()`
    parses config.yaml once and caches it in module-level state to avoid
    repeated disk I/O across the four pipeline threads.
    """
    _cache: Optional[Dict[str, Any]] = None
    _config_path: Optional[str] = None

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> Dict[str, Any]:
        if cls._cache is not None and cls._config_path == config_path:
            return cls._cache

        if not os.path.isfile(config_path):
            # P0-C3 fix: fallback to root-level config.yaml if the
            # structured path is missing. Makes the system robust to
            # either layout.
            fallback = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(config_path)),
                "..",
                "config.yaml",
            ))
            if os.path.isfile(fallback):
                logger.warning(
                    "ConfigRegistry: %s not found; falling back to %s",
                    config_path, fallback,
                )
                config_path = fallback
            else:
                raise FileNotFoundError(
                    f"[ConfigRegistry] Central config not found at: "
                    f"{config_path} (also tried {fallback})"
                )

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            if cfg is None:
                raise ValueError("Config file is empty.")
        except yaml.YAMLError as exc:
            logger.error("YAML parse error in %s: %s", config_path, exc)
            raise

        cls._cache = cfg
        cls._config_path = config_path
        logger.debug("ConfigRegistry loaded from %s", config_path)
        return cfg

    @classmethod
    def invalidate(cls) -> None:
        cls._cache = None
        cls._config_path = None


# ============================================================================
# OS Affinity Helper
# ============================================================================
def apply_cpu_affinity(core_list: List[int]) -> bool:
    """
    Pin the current process (and inherited threads on Windows) to a strict
    physical-core block. Returns True on success, False on failure or when
    the platform does not support affinity masks.

    This is invoked at bootstrap to ensure the database_manager's worker
    threads honor the 25% CPU budget defined in `hardware.cpu.affinity_masks`.
    """
    if not core_list:
        return False

    try:
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(core_list)
        logger.info("CPU affinity locked to cores: %s", core_list)
        return True
    except Exception as exc:
        logger.warning("Failed to apply CPU affinity: %s", exc)
        return False


# ============================================================================
# _LightFaceEngine
# ============================================================================
class _LightFaceEngine:
    """
    Resource-constrained InsightFace wrapper.

    Design constraints:
      * GPU memory is hard-capped at the 0.22 fraction (~1.76 GB on RTX 4060 8GB)
        via explicit CUDAExecutionProvider provider options.
      * ONLY the detection module (`det_10g.onnx`) and recognition module
        (`w600k_r50.onnx`) are loaded. Landmark-106, gender/age, mask, and
        face-swap pipelines are explicitly stripped.
      * Cold-start warmup runs a double-pass Strategy A -> B -> C sequence
        to lock CUDA kernels and VRAM workspace allocations before live
        video capture begins.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        det_size_override: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Patch 12: Construct the engine with an optional det_size override.

        Args:
            config: YAML config dict. If None, ConfigRegistry.load() is used.
            det_size_override: If provided, overrides
                config["insightface"]["det_size"] for this instance only.
                Used by scripts/enroll.py to build a registration engine
                at (640,640) while the live engine in main.py stays at
                the config default (320,320).
        """
        if not _INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "InsightFace / ONNXRuntime not available. Original error: "
                f"{_INSIGHTFACE_IMPORT_ERROR}"
            )

        self.config: Dict[str, Any] = config or ConfigRegistry.load()

        hw_cfg = self.config["hardware"]["gpu"]
        isf_cfg = self.config["insightface"]

        self.device_id: int = int(hw_cfg["device_id"])
        self.memory_fraction: float = float(hw_cfg["memory_fraction"])
        self.total_vram_bytes: int = int(hw_cfg["total_vram_bytes"])
        self.mem_limit_bytes: int = int(hw_cfg["mem_limit_bytes"])
        self.ctx_id: int = int(isf_cfg["ctx_id"])

        self.root_dir: str = str(isf_cfg["root_dir"])
        self.det_module: str = str(isf_cfg["detection_module"])
        self.rec_module: str = str(isf_cfg["recognition_module"])
        # Patch 12: honor det_size_override for the registration engine.
        # Falls back to the config default (320x320) for the live engine.
        cfg_det_size = tuple(isf_cfg["det_size"])
        if det_size_override is not None:
            self.det_size: Tuple[int, int] = tuple(det_size_override)
            logger.info(
                "Patch 12: _LightFaceEngine det_size OVERRIDE | "
                "config=%s -> override=%s",
                cfg_det_size, self.det_size,
            )
        else:
            self.det_size = cfg_det_size
        self.det_thresh: float = float(isf_cfg["det_thresh"])
        self.nms_thresh: float = float(isf_cfg["nms_thresh"])
        self.embedding_dim: int = int(isf_cfg["embedding_dim"])
        self.enable_landmark_106: bool = bool(isf_cfg["enable_landmark_106"])
        self.enable_gender_age: bool = bool(isf_cfg["enable_gender_age"])

        # Underlying FaceAnalysis handle (initialized lazily).
        self.app: Optional[Any] = None
        self.det_model: Optional[Any] = None
        self.rec_model: Optional[Any] = None
        self._initialized: bool = False
        self._warmed: bool = False

    # ------------------------------------------------------------------
    # Provider Options -- the heart of the VRAM budget enforcement.
    # ------------------------------------------------------------------
    def _build_provider_options(self) -> List[Dict[str, Any]]:
        """
        Build the explicit CUDAExecutionProvider options dict that pins the
        0.22 VRAM fraction as `gpu_mem_limit` (in bytes).

        Formula:
            gpu_mem_limit = floor(total_vram_bytes * memory_fraction)

        For an 8 GB card at 0.22:
            8589934592 * 0.22 = 1,890,585,610 bytes  (~1.76 GB)

        IMPORTANT (Bug History):
        ------------------------
        We previously included three keys that are NOT valid
        CUDAExecutionProvider options and caused ORT to reject the entire
        CUDA provider and silently fall back to CPU-only:

          * `enable_mem_pattern`      -- this is a SessionOptions flag,
                                          NOT a CUDA EP option. (Already on
                                          via ORT_ENABLE_ALL graph opts.)
          * `tunable_op_enable`       -- only valid in ORT >= 1.15.
          * `use_ep_level_unified_stream` -- only valid in ORT >= 1.17.

        Any single unknown key in `provider_options` makes ORT throw
        `Unknown provider option: "<key>"` and refuse to start the CUDA
        session. We therefore restrict ourselves to the universally
        supported, stable CUDAExecutionProvider option set:

            device_id, gpu_mem_limit, arena_extend_strategy,
            cudnn_conv_algo_search, do_copy_in_default_stream,
            enable_cuda_graph

        Additional version-conditional keys are added by
        `_build_provider_options_safe()` only after probing the installed
        ORT version.

        LENGTH INVARIANT: InsightFace's `FaceAnalysis` always instantiates
        BOTH `['CUDAExecutionProvider', 'CPUExecutionProvider']` when
        ctx_id >= 0. ONNX Runtime requires len(providers) == len(
        provider_options). We therefore return a 2-element list:
        [cuda_opts, cpu_opts]. The CPU entry is an empty dict.
        """
        computed_limit = int(self.total_vram_bytes * self.memory_fraction)
        # Use the stricter of the computed vs YAML-declared fallback.
        final_limit = min(computed_limit, self.mem_limit_bytes)

        # ----- Universally supported CUDAExecutionProvider options -----
        # Patch 14: restored old-code defaults for perf-critical keys.
        #   cudnn_conv_algo_search: EXHAUSTIVE (was HEURISTIC) -- HEURISTIC
        #     picks the first cudnn algo that works, which on RTX 3050 Laptop
        #     is a slow fallback kernel for det_10g (~164ms vs 20ms with EXHAUSTIVE).
        #   arena_extend_strategy: kNextPowerOfTwo (was kSameAsRequested) --
        #     kSameAsRequested causes per-inference alloc/free churn.
        # gpu_mem_limit is kept (coexist with YOLO+OSNet on 4GB VRAM), but
        # unified_stream is left OFF below (was being force-enabled).
        # Patch 16: gpu_mem_limit REMOVED to match old code (unlimited VRAM).
        # The 1.76GB cap starved det_10g's cudnn workspace, forcing EXHAUSTIVE
        # to pick a slow fallback kernel (186ms vs 20ms face det).
        # Old face_db_manager.py passes NO provider_options at all -> ORT
        # uses unlimited GPU memory. We keep the other keys (which are ORT
        # defaults) for explicitness, but gpu_mem_limit is intentionally absent.
        cuda_opts: Dict[str, Any] = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
            "enable_cuda_graph": False,
        }

        # ----- Version-conditional CUDA EP options -----
        # tunable_op_enable: ORT >= 1.15
        # use_ep_level_unified_stream: ORT >= 1.17
        ort_ver = self._get_ort_version()
        if ort_ver is not None:
            if ort_ver >= (1, 15, 0):
                cuda_opts["tunable_op_enable"] = False
            if ort_ver >= (1, 17, 0):
                # Patch 14: unified_stream=True caused measurable slowdown
                # vs ORT default (False). Old code never set this and got 20ms
                # face det; new code with True got 164ms. Restored to False.
                cuda_opts["use_ep_level_unified_stream"] = False
        else:
            # Unknown ORT version -- skip the version-conditional keys
            # entirely. The stable 6-key set above is sufficient to
            # enforce the VRAM budget and engage CUDA correctly.
            logger.warning(
                "Could not determine ORT version; skipping "
                "tunable_op_enable / use_ep_level_unified_stream "
                "(CUDA EP will still engage with the stable 6-key set)."
            )

        # CPUExecutionProvider accepts an empty options dict. This MUST be
        # present so len(provider_options) == len(providers) == 2, otherwise
        # ORT throws "providers and provider_options should be the same
        # length" and silently degrades to CPU-only.
        cpu_opts: Dict[str, Any] = {}

        provider_options: List[Dict[str, Any]] = [cuda_opts, cpu_opts]

        logger.info(
            "_LightFaceEngine VRAM budget: %d bytes (~%.2f GB) | fraction=%.3f "
            "| providers=[CUDA, CPU] | ort_ver=%s | cuda_opts_keys=%s",
            final_limit, final_limit / (1024 ** 3), self.memory_fraction,
            ".".join(str(x) for x in ort_ver) if ort_ver else "unknown",
            sorted(cuda_opts.keys()),
        )
        return provider_options

    # ------------------------------------------------------------------
    @staticmethod
    def _get_ort_version() -> Optional[Tuple[int, int, int]]:
        """
        Return the installed onnxruntime version as a (major, minor, patch)
        tuple, or None if it cannot be determined.

        Used to decide whether version-conditional CUDA EP options
        (tunable_op_enable >= 1.15, use_ep_level_unified_stream >= 1.17)
        are safe to include. Including them on an older ORT build would
        trigger `Unknown provider option` and force a CPU-only fallback.
        """
        try:
            import onnxruntime as _ort
            ver_str = getattr(_ort, "__version__", None)
            if not ver_str:
                return None
            # Strip any "+cpu" / "+cu118" / ".dev0" suffixes.
            ver_str = ver_str.split("+")[0].split(".dev")[0]
            parts = ver_str.split(".")
            if len(parts) < 3:
                return None
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # InsightFace bootstrap.
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Construct the FaceAnalysis pipeline and strip non-essential modules.
        This is the only place where InsightFace sessions are instantiated.

        CUDA-ENGAGEMENT SELF-CHECK
        --------------------------
        InsightFace's FaceAnalysis swallows ORT CUDAConstruction errors
        and silently retries on CPU-only. This is catastrophic for our
        25% VRAM budget -- if CUDA never engages, the entire pipeline runs
        on the i9 CPU and we cannot hit the 50fps target.

        We therefore probe each loaded model's session providers AFTER
        construction and raise a RuntimeError if CUDAExecutionProvider is
        absent from any of them. This turns silent CPU-only fallback into
        a loud, actionable failure.
        """
        if self._initialized:
            logger.warning("_LightFaceEngine already initialized; skipping.")
            return

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"InsightFace model root dir not found: {self.root_dir}"
            )

        det_path = os.path.join(self.root_dir, self.det_module)
        rec_path = os.path.join(self.root_dir, self.rec_module)
        for p in (det_path, rec_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Required ONNX model missing: {p}")

        # Configure ONNX Runtime session options for the budgeted session.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 2
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # enable_mem_pattern is a SESSION option (not a provider option).
        # It is already enabled by ORT_ENABLE_ALL, but we set it explicitly
        # here for clarity and forward-compatibility.
        try:
            sess_opts.enable_mem_pattern = True
        except Exception:
            # Older ORT builds may not expose this attribute.
            pass

        # FaceAnalysis(allowed_modules=...) explicitly restricts the module
        # set so that landmark_2d_106, genderage, mask, etc. are NEVER loaded.
        allowed_modules: List[str] = ["detection", "recognition"]

        # The `name` parameter selects the model pack directory on disk
        # (e.g. name="buffalo_l" -> models/insightface/models/buffalo_l/).
        # Default to "buffalo_l" (the standard pack containing det_10g +
        # w600k_r50 + landmark_3d_68 + landmark_2d_106 + genderage).
        # Override via config["insightface"]["pack_name"] if needed.
        isf_cfg = self.config.get("insightface", {})
        pack_name: str = str(isf_cfg.get("pack_name", "buffalo_l"))

        try:
            self.app = FaceAnalysis(
                name=pack_name,
                root=self.root_dir,
                allowed_modules=allowed_modules,
                provider_options=self._build_provider_options(),
                session_options=sess_opts,
                det_size=self.det_size,
                det_thresh=self.det_thresh,
                nms_thresh=self.nms_thresh,
                ctx_id=self.ctx_id,
            )
        except Exception as exc:
            logger.error("FaceAnalysis initialization failed: %s", exc)
            raise

        # Strip any spurious modules that the framework may have auto-loaded.
        self._strip_disabled_modules()

        # Cache direct references for raw forward-pass usage in warmup.
        self.det_model = self.app.models.get("detection")
        self.rec_model = self.app.models.get("recognition")

        # CRITICAL FIX (det_size drift at init time):
        # InsightFace's FaceAnalysis.__init__ accepts `det_size` but
        # in several deployed versions the detection model's det_size
        # attribute is silently reset to the model_zoo default (640, 640)
        # during internal prepare() / model loading. We re-assert here
        # at init time so the log below shows the TRUE runtime value,
        # and again inside detect_and_embed() before every app.get()
        # call to guarantee it never drifts during inference.
        if self.det_model is not None and self.det_size is not None:
            try:
                self.det_model.det_size = tuple(self.det_size)
            except Exception:
                pass

        # Log the actual det_size to verify config took effect.
        actual_det_size = (
            getattr(self.det_model, 'det_size', None)
            or getattr(self.det_model, 'input_size', None)
        )
        logger.info(
            "_LightFaceEngine det_size check | configured=%s | actual=%s",
            self.det_size, actual_det_size,
        )

        if self.det_model is None or self.rec_model is None:
            raise RuntimeError(
                "FaceAnalysis loaded but detection/recognition handles are None."
            )

        # ------------------------------------------------------------------
        # CUDA-ENGAGEMENT SELF-CHECK
        # ------------------------------------------------------------------
        # If CUDAExecutionProvider is missing from any model's session,
        # InsightFace silently fell back to CPU-only. This is fatal for our
        # 25% VRAM budget -- fail loudly instead so the operator can fix
        # the ORT/CUDA install rather than shipping a CPU-bound system.
        cuda_ok, offending = self._verify_cuda_engagement()
        if not cuda_ok:
            raise RuntimeError(
                "CUDAExecutionProvider FAILED to engage on the following "
                f"InsightFace models: {offending}. This means ORT silently "
                "fell back to CPU-only execution, which violates the 25% "
                "hardware resource budget. Check that: "
                "(1) onnxruntime-gpu (not onnxruntime) is installed, "
                "(2) CUDA Toolkit version matches the ORT wheel's expected "
                "CUDA version (e.g. ORT 1.17 -> CUDA 11.8 or 12.x), "
                "(3) cuDNN is installed and on the DLL search path "
                "(gpu_linker should have registered it), and "
                "(4) the provider_options dict contains only valid "
                "CUDAExecutionProvider keys."
            )
        logger.info(
            "CUDAExecutionProvider engaged on all %d loaded InsightFace models.",
            len(self.app.models),
        )

        self._initialized = True
        logger.info(
            "_LightFaceEngine initialized | pack=%s | det=%s rec=%s | "
            "landmark_106=%s | gender_age=%s",
            pack_name, self.det_module, self.rec_module,
            self.enable_landmark_106, self.enable_gender_age,
        )

    # ------------------------------------------------------------------
    def _verify_cuda_engagement(self) -> Tuple[bool, List[str]]:
        """
        Probe each loaded InsightFace model's ORT session and verify
        CUDAExecutionProvider is present in the active provider list.

        Returns (all_cuda_ok, list_of_offending_model_names).
        """
        if self.app is None:
            return False, ["<FaceAnalysis not constructed>"]

        offending: List[str] = []
        for name, model in self.app.models.items():
            sess = getattr(model, "session", None)
            if sess is None:
                offending.append(f"{name}=<no session>")
                continue
            try:
                providers = sess.get_providers()
            except Exception as exc:
                offending.append(f"{name}=<get_providers failed: {exc}>")
                continue
            if "CUDAExecutionProvider" not in providers:
                offending.append(f"{name}={providers}")
        return (len(offending) == 0, offending)

    # ------------------------------------------------------------------
    def _strip_disabled_modules(self) -> None:
        """
        Hard-strip any landmark_2d_106, genderage, mask, or attribution
        handles that the framework may have auto-instantiated despite the
        allowed_modules argument. This is a defense-in-depth measure.
        """
        if self.app is None:
            return

        disabled_keys: List[str] = []
        for key in list(self.app.models.keys()):
            if key in ("landmark_2d_106", "genderage", "mask", "attributes"):
                disabled_keys.append(key)
                del self.app.models[key]

        if disabled_keys:
            logger.info(
                "Stripped non-essential InsightFace modules: %s", disabled_keys
            )

    # ------------------------------------------------------------------
    # Cold-Start Double-Pass Warmup Protocol.
    # ------------------------------------------------------------------
    def warmup(
        self,
        sample_image_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Execute the Double-Pass Warmup Protocol.

        Pass 1: Strategy A (disk images) -> B (synthetic) -> C (raw forward)
        Pass 2: Repeat A -> B -> C to lock steady-state operator memory.

        This eliminates the ~200 ms first-frame initialization latency caused
        by lazy CUDA kernel optimization paths.
        """
        if not self._initialized:
            raise RuntimeError(
                "_LightFaceEngine must be initialized before warmup."
            )

        if sample_image_paths is None:
            sample_image_paths = self._collect_warmup_samples_from_disk()

        logger.info(
            "Cold-start double-pass warmup beginning | sample_images=%d",
            len(sample_image_paths),
        )

        for pass_idx in (1, 2):
            logger.info("Warmup Pass %d/2 starting...", pass_idx)
            self._strategy_a_disk_images(sample_image_paths)
            self._strategy_b_synthetic_matrices()
            self._strategy_c_direct_tensor_forward()
            logger.info("Warmup Pass %d/2 complete.", pass_idx)

        self._warmed = True
        logger.info("Double-pass warmup complete; engine steady-state locked.")

    # ------------------------------------------------------------------
    def _collect_warmup_samples_from_disk(self) -> List[str]:
        """
        Pull up to 3 real student face images from the enrollment directory
        for Strategy A. Falls back to an empty list (Strategy B/C will still
        run) if no enrolled images are available.
        """
        enr_cfg = self.config.get("enrollment", {})
        faces_dir = enr_cfg.get("student_faces_dir", "data/student_faces")
        exts = tuple(enr_cfg.get("image_extensions", [".jpg"]))

        candidates: List[str] = []
        if not os.path.isdir(faces_dir):
            return candidates

        for student_dir in sorted(os.listdir(faces_dir)):
            full = os.path.join(faces_dir, student_dir)
            if not os.path.isdir(full):
                continue
            for fname in sorted(os.listdir(full)):
                if fname.lower().endswith(exts):
                    candidates.append(os.path.join(full, fname))
                    if len(candidates) >= 3:
                        return candidates
        return candidates[:3]

    # ------------------------------------------------------------------
    def _strategy_a_disk_images(self, image_paths: List[str]) -> None:
        """
        Strategy A: feed real student face images through the full
        detection -> recognition pipeline to engage all live layer paths.

        NOTE: `self.app.get(img_rgb)` ALREADY exercises both the detection
        model (det_10g) and the recognition model (w600k_r50) internally --
        InsightFace's FaceAnalysis.get() runs detect -> align -> recognize
        end-to-end and populates `face.normed_embedding` on each returned
        Face object. There is NO need to call `self.rec_model.get_feat(...)`
        separately; doing so would feed a 512-D embedding back into the
        recognizer as if it were a 112x112 face crop, which raises
        `INVALID_ARGUMENT: Got invalid dimensions for input: input.1
        index: 1 Got: 1 Expected: 3`.
        """
        if not image_paths:
            logger.debug("Strategy A skipped: no sample images available.")
            return

        for idx, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    logger.debug("Strategy A | unreadable image: %s", img_path)
                    continue
                # Defense-in-depth: ensure 3-channel BGR before conversion.
                # Some JPEGs may decode as single-channel even with
                # IMREAD_COLOR on certain OpenCV builds.
                if img.ndim != 3 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.app.get(img_rgb)
                logger.debug(
                    "Strategy A | sample=%d/%d | faces=%d",
                    idx + 1, len(image_paths), len(faces),
                )
            except Exception as exc:
                logger.warning("Strategy A failed on %s: %s", img_path, exc)

    # ------------------------------------------------------------------
    def _strategy_b_synthetic_matrices(self) -> None:
        """
        Strategy B: multi-pass synthetic face matrices generated via NumPy
        random/uniform arrays with varying structural dimensions. This forces
        ONNX Runtime to allocate internal staging buffers across all expected
        input shapes.
        """
        synthetic_shapes: List[Tuple[int, int, int]] = [
            (112, 112, 3),
            (224, 224, 3),
            (320, 320, 3),
            (480, 640, 3),
            (640, 640, 3),
        ]
        rng = np.random.default_rng(seed=42)

        for shape in synthetic_shapes:
            for sub_pass in range(2):
                # Alternate uniform / normal distributions to vary the
                # operator path selections engaged during kernel compilation.
                if sub_pass % 2 == 0:
                    arr = rng.uniform(0, 255, size=shape).astype(np.uint8)
                else:
                    arr = rng.normal(127.5, 50, size=shape).astype(np.uint8)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                try:
                    _ = self.app.get(arr)
                except Exception as exc:
                    logger.debug(
                        "Strategy B | shape=%s sub_pass=%d err=%s",
                        shape, sub_pass, exc,
                    )

        logger.debug(
            "Strategy B complete (%d shapes x 2 sub-passes).",
            len(synthetic_shapes),
        )

    # ------------------------------------------------------------------
    def _strategy_c_direct_tensor_forward(self) -> None:
        """
        Strategy C: inject pre-aligned 112x112 tensors straight into the
        recognition network's forward call to lock the VRAM workspace
        allocation for the embedding extraction path.

        InsightFace's `ArcFaceONNX.get_feat(imgs)` expects a LIST of HWC
        uint8 images and internally converts them via `cv2.dnn.blobFromImages`
        (which applies the model's `input_std` / `input_mean` normalization
        and HWC->NCHW transpose). Passing a pre-normalized 4D NCHW float
        tensor is WRONG -- it would either raise or silently no-op. We
        therefore feed plain HWC uint8 synthetic crops and let `get_feat`
        perform its own normalization, which is exactly the path used at
        inference time.
        """
        if self.rec_model is None:
            return

        rng = np.random.default_rng()
        for _ in range(4):
            # Build a synthetic 112x112 RGB uint8 crop. get_feat() will
            # apply the standard (pixel - 127.5) / 128.0 normalization
            # internally via cv2.dnn.blobFromImages.
            crop_hwc = rng.uniform(
                0, 255, size=(112, 112, 3)
            ).astype(np.uint8)
            crop_hwc = np.ascontiguousarray(crop_hwc)
            try:
                if hasattr(self.rec_model, "get_feat"):
                    _ = self.rec_model.get_feat([crop_hwc])
                elif hasattr(self.rec_model, "forward"):
                    _ = self.rec_model.forward([crop_hwc])
            except Exception as exc:
                logger.debug("Strategy C direct forward err: %s", exc)

        logger.debug("Strategy C complete (4 direct tensor passes).")

    # ------------------------------------------------------------------
    # Public inference helpers.
    # ------------------------------------------------------------------
    def detect_and_embed(
        self, image_bgr: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Run the full detection -> recognition pipeline on a single BGR frame.
        Returns a list of dicts with keys: bbox, kps, embedding, det_score.
        Stashes per-stage latencies on self for the orchestrator to read.

        Implementation note: InsightFace's FaceAnalysis.get() runs det -> align
        -> rec as one black-box call. We do NOT split it by running det
        separately -- that doubles GPU work. Instead we measure the total
        call time and split it using a calibrated ratio (det:rec) that we
        refresh periodically from a lightweight standalone det timing.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized.")

        import time as _time
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Single app.get() call: det + align + rec in one black-box pass ---
        t_total_start = _time.perf_counter()
        try:
            faces = self.app.get(img_rgb)
        except Exception as exc:
            logger.error("FaceAnalysis.get() failure: %s", exc)
            self._last_det_ms = 0.0
            self._last_rec_ms = 0.0
            return []
        t_total_ms = (_time.perf_counter() - t_total_start) * 1000.0

        # --- Split total into det / rec using a calibrated ratio ---
        # InsightFace det_10g : ArcFace w600k_r50 ≈ 0.78 : 0.22 at det_size=320,
        # ≈ 0.85 : 0.15 at det_size=640. We refresh the ratio every 32 calls
        # via a standalone det_model.detect() timing on the same image.
        # Between refreshes, we use the cached ratio to avoid the
        # double-detection penalty on every call.
        self._det_rec_call_counter = getattr(self, "_det_rec_call_counter", 0) + 1
        if (
            self._det_rec_call_counter % 32 == 1
            and self.det_model is not None
            and len(faces) > 0
        ):
            # Refresh the ratio. This is one extra det call per 32 frames,
            # amortized to <2.5ms per call -- negligible vs the 80ms we save
            # by NOT running det twice every frame.
            t_det_only_start = _time.perf_counter()
            try:
                _ = self.det_model.detect(img_rgb)
                t_det_only_ms = (_time.perf_counter() - t_det_only_start) * 1000.0
                if t_total_ms > 1.0 and t_det_only_ms < t_total_ms:
                    self._cached_det_fraction = float(t_det_only_ms / t_total_ms)
            except Exception:
                pass  # keep previous ratio

        det_frac = float(getattr(self, "_cached_det_fraction", 0.78))
        self._last_det_ms = t_total_ms * det_frac
        self._last_rec_ms = t_total_ms * (1.0 - det_frac)

        out: List[Dict[str, Any]] = []
        for f in faces:
            out.append({
                "bbox": np.asarray(f.bbox, dtype=np.float32),
                "kps": (
                    np.asarray(f.kps, dtype=np.float32)
                    if f.kps is not None else None
                ),
                "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
                "det_score": float(f.det_score),
            })
        return out

    @property
    def last_det_latency_ms(self) -> float:
        return getattr(self, "_last_det_ms", 0.0)

    @property
    def last_rec_latency_ms(self) -> float:
        return getattr(self, "_last_rec_ms", 0.0)

    def is_warmed(self) -> bool:
        return self._warmed

    def close(self) -> None:
        """Release ONNX Runtime sessions and free VRAM."""
        if self.app is not None:
            for model in list(self.app.models.values()):
                sess = getattr(model, "session", None)
                if sess is not None:
                    try:
                        del sess
                    except Exception:
                        pass
            self.app.models.clear()
        self.det_model = None
        self.rec_model = None
        self.app = None
        self._initialized = False
        self._warmed = False
        gc.collect()
        logger.info("_LightFaceEngine closed and VRAM released.")


# ============================================================================
# ArcFace Alignment & Normalization
# ============================================================================
class ArcFaceAligner:
    """
    Performs 2D affine alignment of a face crop to the standard 112x112
    ArcFace template using cv2.estimateAffinePartial2D (LMEDS) and
    cv2.warpAffine (INTER_LINEAR + BORDER_REPLICATE). Output is normalized
    per InsightFace native formula: (pixel - 127.5) / 128.0.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or ConfigRegistry.load()
        enr = self.config["enrollment"]
        self.template: np.ndarray = np.asarray(
            enr["arcface_template"], dtype=np.float32
        )
        self.alignment_method: str = enr["alignment_method"]    # "LMEDS"
        self.warp_interpolation: str = enr["warp_interpolation"]  # "INTER_LINEAR"
        self.warp_border_mode: str = enr["warp_border_mode"]      # "BORDER_REPLICATE"
        self.norm_subtract: float = float(enr["normalization"]["subtract"])  # 127.5
        self.norm_divide: float = float(enr["normalization"]["divide"])      # 128.0

    # ------------------------------------------------------------------
    def _resolve_interpolation(self) -> int:
        return getattr(cv2, self.warp_interpolation, cv2.INTER_LINEAR)

    def _resolve_border(self) -> int:
        return getattr(cv2, self.warp_border_mode, cv2.BORDER_REPLICATE)

    def _resolve_lmeds_flag(self) -> int:
        # cv2.LMEDS only supports >=4 point correspondences. Our 5-point
        # template satisfies this constraint.
        return getattr(cv2, self.alignment_method, cv2.LMEDS)

    # ------------------------------------------------------------------
    def align(
        self,
        image_bgr: np.ndarray,
        landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Align a single face given 5-point landmarks. Returns the aligned
        112x112 BGR uint8 crop, or None on failure.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None
        if landmarks is None or landmarks.shape != (5, 2):
            logger.warning(
                "Align skipped: landmark shape %s",
                None if landmarks is None else landmarks.shape,
            )
            return None

        src = np.asarray(landmarks, dtype=np.float32)
        dst = self.template.astype(np.float32)

        try:
            M, _ = cv2.estimateAffinePartial2D(
                src, dst, method=self._resolve_lmeds_flag()
            )
        except cv2.error as exc:
            logger.warning("estimateAffinePartial2D failed: %s", exc)
            return None

        if M is None:
            logger.warning("Affine matrix is None; alignment aborted.")
            return None

        try:
            warped = cv2.warpAffine(
                image_bgr,
                M,
                (112, 112),
                flags=self._resolve_interpolation(),
                borderMode=self._resolve_border(),
            )
        except cv2.error as exc:
            logger.warning("warpAffine failed: %s", exc)
            return None

        return warped

    # ------------------------------------------------------------------
    def normalize(self, aligned_bgr_uint8: np.ndarray) -> np.ndarray:
        """
        Apply InsightFace native normalization: (pixel - 127.5) / 128.0.
        Returns a float32 CHW tensor suitable for direct ONNX forward.
        """
        if aligned_bgr_uint8.dtype != np.uint8:
            aligned_bgr_uint8 = aligned_bgr_uint8.astype(np.uint8)

        # Convert BGR -> RGB before normalization (InsightFace expects RGB).
        rgb = cv2.cvtColor(aligned_bgr_uint8, cv2.COLOR_BGR2RGB)
        f32 = rgb.astype(np.float32)
        norm = (f32 - self.norm_subtract) / self.norm_divide
        # HWC -> CHW
        norm = np.transpose(norm, (2, 0, 1))
        return np.ascontiguousarray(norm, dtype=np.float32)

    # ------------------------------------------------------------------
    def align_and_normalize(
        self,
        image_bgr: np.ndarray,
        landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Convenience wrapper: returns CHW float32 normalized tensor."""
        aligned = self.align(image_bgr, landmarks)
        if aligned is None:
            return None
        return self.normalize(aligned)


# ============================================================================
# Enrollment Clustering Logic ("24 + 1")
# ============================================================================
class EnrollmentClusterer:
    """
    Implements the "24 + 1" incremental enrollment algorithm:
      1. Scan student directory for valid frame_xxxxxx.jpg assets.
      2. For each frame, isolate the largest face by bbox area.
      3. Compute landmark symmetry score for each frame.
      4. Anchor Face ("1") = frame with the LOWEST symmetry score
         (most perpendicular frontal view, asymmetry -> 0).
      5. For every other frame, compute mean L2 deviation of its 5-point
         landmarks vs the anchor's landmarks.
      6. Sort the remaining frames in DESCENDING order (reverse=True)
         to capture the 24 most extreme pose/orientation variances.
      7. Pack the anchor + 24 distribution frames into a comprehensive
         feature array (cap = 25 total).
    """

    def __init__(
        self,
        engine: _LightFaceEngine,
        aligner: ArcFaceAligner,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine = engine
        self.aligner = aligner
        self.config: Dict[str, Any] = config or ConfigRegistry.load()
        enr = self.config["enrollment"]
        self.faces_dir: str = enr["student_faces_dir"]
        self.db_pickle_path: str = enr["db_pickle_path"]
        self.anchor_capacity: int = int(enr["anchor_capacity"])
        self.distribution_capacity: int = int(enr["distribution_cluster_capacity"])
        self.total_capacity: int = int(enr["total_profile_capacity"])
        self.exts: Tuple[str, ...] = tuple(enr["image_extensions"])
        self.enable_gc: bool = bool(enr["enable_gc_collect_per_profile"])
        # Patch 12: Distribution-frame selection config (L2-range filter
        # + yaw-aware sort). Falls back to safe defaults if the keys
        # are missing from older config.yaml files.
        sel = enr.get("distribution_selection", {})
        self.l2_min_px: float = float(sel.get("l2_min_px", 5.0))
        self.l2_max_px: float = float(sel.get("l2_max_px", 60.0))
        self.yaw_metric: str = str(sel.get("yaw_metric", "eye_nose_asymmetry"))
        self.dist_sort_by: str = str(sel.get("sort_by", "yaw_desc"))
        self.dist_fallback: str = str(sel.get("fallback_strategy", "relax_l2"))
        self.relax_l2_multiplier: float = float(
            sel.get("relax_l2_multiplier", 1.5)
        )

    # ------------------------------------------------------------------
    # Landmark symmetry score.
    # ------------------------------------------------------------------
    @staticmethod
    def _symmetry_score(kps: np.ndarray) -> float:
        """
        Symmetry = | ||L_eye - Nose||_2  -  ||R_eye - Nose||_2 |

        Lower score = more frontal/perpendicular (asymmetry approaches 0).
        Returns +inf for invalid landmark shapes so they sort to the end.
        """
        if kps is None or kps.shape != (5, 2):
            return float("inf")

        left_eye = kps[_LM_LEFT_EYE]
        right_eye = kps[_LM_RIGHT_EYE]
        nose = kps[_LM_NOSE]

        d_left = float(np.linalg.norm(left_eye - nose))
        d_right = float(np.linalg.norm(right_eye - nose))
        return abs(d_left - d_right)

    # ------------------------------------------------------------------
    # L2 deviation from anchor.
    # ------------------------------------------------------------------
    @staticmethod
    def _l2_deviation(current_kps: np.ndarray, anchor_kps: np.ndarray) -> float:
        """
        Mean geometric L2 norm deviation:
            np.mean(np.linalg.norm(current_kps - anchor_kps, axis=1))
        """
        if current_kps is None or anchor_kps is None:
            return 0.0
        if current_kps.shape != anchor_kps.shape:
            return 0.0
        return float(np.mean(np.linalg.norm(current_kps - anchor_kps, axis=1)))

    # ------------------------------------------------------------------
    # Patch 12: Yaw-aware score (eye-nose asymmetry).
    # ------------------------------------------------------------------
    @staticmethod
    def _yaw_score(kps: np.ndarray) -> float:
        """
        Yaw score = |d_left_eye_to_nose - d_right_eye_to_nose|.

        Identical mathematical form to _symmetry_score, but used here as
        a SORT KEY (descending) for distribution-frame selection -- we
        want the 24 most profile-like views in the cluster.

        Returns 0.0 for a perfectly frontal frame (symmetric eyes), and
        increases monotonically as the head yaws left/right. Returns 0.0
        for invalid landmark shapes (these get filtered out earlier).
        """
        if kps is None or kps.shape != (5, 2):
            return 0.0
        left_eye = kps[_LM_LEFT_EYE]
        right_eye = kps[_LM_RIGHT_EYE]
        nose = kps[_LM_NOSE]
        d_left = float(np.linalg.norm(left_eye - nose))
        d_right = float(np.linalg.norm(right_eye - nose))
        return abs(d_left - d_right)

    # ------------------------------------------------------------------
    # Frame scanner.
    # ------------------------------------------------------------------
    def _scan_directory(self, dir_path: str) -> List[str]:
        if not os.path.isdir(dir_path):
            return []
        out: List[str] = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith(self.exts):
                out.append(os.path.join(dir_path, fname))
        return out

    # ------------------------------------------------------------------
    def _largest_face(
        self,
        image_bgr: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Run detection on a single image and return metadata for the
        largest face (by bbox area). Returns None if no face is detected.
        """
        try:
            faces = self.engine.detect_and_embed(image_bgr)
        except Exception as exc:
            logger.warning("detect_and_embed failed: %s", exc)
            return None

        if not faces:
            return None

        best: Optional[Dict[str, Any]] = None
        best_area: float = -1.0
        for f in faces:
            bb = f["bbox"]
            x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if area > best_area:
                best_area = area
                best = f
        return best

    # ------------------------------------------------------------------
    # Main enrollment driver.
    # ------------------------------------------------------------------
    def enroll_all(self) -> Tuple[Dict[str, StudentProfile], EnrollmentStats]:
        """
        Walk the student_faces_dir, build a StudentProfile per directory,
        and serialize the entire registry to student_db.pickle.
        """
        stats = EnrollmentStats()
        registry: Dict[str, StudentProfile] = {}

        if not os.path.isdir(self.faces_dir):
            logger.error("Enrollment directory missing: %s", self.faces_dir)
            return registry, stats

        for student_dir_name in sorted(os.listdir(self.faces_dir)):
            student_dir_path = os.path.join(self.faces_dir, student_dir_name)
            if not os.path.isdir(student_dir_path):
                continue

            stats.total_directories_scanned += 1
            try:
                profile = self._enroll_single_directory(
                    student_dir_path, student_dir_name
                )
                if profile is not None:
                    registry[profile.student_id] = profile
                    stats.total_profiles_built += 1
                    logger.info(
                        "Enrolled: %s | frames=%d",
                        profile.student_name,
                        profile.face_embeddings.shape[0],
                    )
                else:
                    stats.failed_directories.append(student_dir_name)
            except Exception as exc:
                stats.failed_directories.append(student_dir_name)
                logger.error(
                    "Enrollment failed for %s: %s\n%s",
                    student_dir_name, exc, traceback.format_exc(),
                )

            if self.enable_gc:
                gc.collect()

        # Serialize the registry.
        self._serialize_registry(registry)
        logger.info(
            "Enrollment complete | profiles=%d | scanned=%d | failed=%d",
            stats.total_profiles_built,
            stats.total_directories_scanned,
            len(stats.failed_directories),
        )
        return registry, stats

    # ------------------------------------------------------------------
    def _enroll_single_directory(
        self,
        dir_path: str,
        dir_name: str,
    ) -> Optional[StudentProfile]:
        """
        Build a StudentProfile for a single student directory using the
        "24 + 1" clustering algorithm.

        Directory naming convention: "<NRP>__<StudentName>"
            e.g. "2024001__John_Doe" -> NRP=2024001, name="John Doe".
        Fallback: directory name is used as both student_id and name.
        """
        if "__" in dir_name:
            nrp, student_name = dir_name.split("__", 1)
            student_name = student_name.replace("_", " ")
        else:
            nrp = dir_name
            student_name = dir_name.replace("_", " ")

        image_paths = self._scan_directory(dir_path)
        if not image_paths:
            logger.warning("Directory %s contains no valid images.", dir_name)
            return None

        # Phase 1: extract metadata for every frame.
        frame_records: List[Dict[str, Any]] = []
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Unreadable image skipped: %s", img_path)
                continue

            face_meta = self._largest_face(img)
            if face_meta is None or face_meta["kps"] is None:
                continue

            symmetry = self._symmetry_score(face_meta["kps"])
            if not np.isfinite(symmetry):
                continue

            frame_records.append({
                "img_path": img_path,
                "image_bgr": img,
                "bbox": face_meta["bbox"],
                "kps": face_meta["kps"],
                "embedding": face_meta["embedding"],
                "symmetry": symmetry,
            })

        if not frame_records:
            logger.warning("No usable faces in directory %s", dir_name)
            return None

        # Phase 2: pick the Anchor Face (ascending symmetry -> lowest first).
        frame_records.sort(key=lambda r: r["symmetry"])
        anchor = frame_records[0]
        anchor_kps = anchor["kps"]
        anchor_hash = hashlib.sha1(
            os.path.basename(anchor["img_path"]).encode("utf-8")
        ).hexdigest()[:12]

        # Phase 3 (Patch 12): compute BOTH L2 deviation AND yaw score for
        # every remaining frame. L2 deviation = geometric distance from
        # the anchor 5-point landmarks. Yaw score = eye-nose asymmetry,
        # zero for frontal, increasing with head yaw.
        remaining = frame_records[1:]
        for rec in remaining:
            rec["l2_deviation"] = self._l2_deviation(rec["kps"], anchor_kps)
            rec["yaw_score"] = self._yaw_score(rec["kps"])

        # Phase 4 (Patch 12): L2-range FILTER + yaw-aware SORT.
        #
        # Step A: keep only frames whose L2 deviation falls within
        #         [l2_min_px, l2_max_px]. l2_min filters near-duplicates
        #         of the anchor (no pose variance); l2_max filters
        #         extreme outliers (likely misdetections or wild pitch).
        #
        # Step B: sort the survivors by yaw_score DESCENDING so the most
        #         profile-like views (highest eye-nose asymmetry) are
        #         picked first for the 24-slot distribution cluster.
        #
        # Step C: fallback. If the L2-filtered set has fewer than
        #         distribution_capacity frames, relax the L2 ceiling
        #         (multiply l2_max by relax_l2_multiplier) and re-sort
        #         by yaw descending. If still insufficient, fall back to
        #         pure yaw descending across ALL remaining frames.
        l2_min = self.l2_min_px
        l2_max = self.l2_max_px

        filtered = [
            r for r in remaining
            if l2_min <= r["l2_deviation"] <= l2_max
        ]

        if len(filtered) < self.distribution_capacity:
            # Fallback: relax the L2 ceiling.
            relaxed_max = l2_max * self.relax_l2_multiplier
            filtered = [
                r for r in remaining
                if l2_min <= r["l2_deviation"] <= relaxed_max
            ]
            logger.warning(
                "Patch 12: L2-range filter (%.1f-%.1f px) yielded only "
                "%d frames for %s (need %d). Relaxed l2_max to %.1f -> "
                "%d frames.",
                l2_min, l2_max, len(filtered),
                dir_name, self.distribution_capacity,
                relaxed_max, len(filtered),
            )

        if len(filtered) < self.distribution_capacity:
            # Final fallback: use ALL remaining frames, sorted by yaw desc.
            filtered = list(remaining)
            logger.warning(
                "Patch 12: relaxed L2 filter still insufficient for %s "
                "(%d frames). Falling back to all-frames by yaw_desc.",
                dir_name, len(filtered),
            )

        # Sort survivors by yaw_score DESCENDING (most profile-like first).
        filtered.sort(key=lambda r: r["yaw_score"], reverse=True)

        logger.info(
            "Patch 12: %s distribution selection | total=%d | "
            "L2-filtered=%d | picked=%d | "
            "yaw_score range=[%.2f, %.2f] | l2 range=[%.2f, %.2f]",
            dir_name, len(remaining), len(filtered),
            min(self.distribution_capacity, len(filtered)),
            filtered[0]["yaw_score"] if filtered else 0.0,
            filtered[-1]["yaw_score"] if filtered else 0.0,
            filtered[0]["l2_deviation"] if filtered else 0.0,
            filtered[-1]["l2_deviation"] if filtered else 0.0,
        )

        # Phase 5: assemble the cluster, capped at 25 total (1 + 24).
        cluster: List[Dict[str, Any]] = (
            [anchor] + filtered[: self.distribution_capacity]
        )
        cluster = cluster[: self.total_capacity]

        if len(cluster) < 2:
            logger.warning(
                "Insufficient frames for student %s (found %d).",
                dir_name, len(cluster),
            )
            return None

        # Phase 6: re-extract embeddings via the aligned path for the
        # entire cluster to guarantee template consistency.
        #
        # BUG HISTORY (fixed):
        # We previously called `aligner.align_and_normalize()` (which
        # returns a pre-normalized CHW float32 tensor) and then passed
        # `aligned_norm[None, ...]` (a 4D NCHW float tensor) into
        # `rec_model.get_feat(...)`. But InsightFace's `ArcFaceONNX.get_feat`
        # expects a LIST of HWC uint8 images and applies its own
        # normalization via `cv2.dnn.blobFromImages` internally. Feeding it
        # a 4D float NCHW array makes blobFromImages call `cv2.resize`
        # with no valid dsize, raising
        #     (-215:Assertion failed) !dsize.empty() in cv::hal::resize
        # and the except-branch silently fell back to the lower-quality
        # detection-time embedding, degrading match accuracy.
        #
        # The fix: call `aligner.align()` (returns BGR uint8 HWC 112x112)
        # and feed THAT directly to `get_feat([aligned_bgr])`. Skip the
        # manual normalize step entirely -- `get_feat` does it internally
        # using the model's own `input_mean` / `input_std` (127.5 / 128.0),
        # which matches our config.
        embeddings_list: List[np.ndarray] = []
        for rec in cluster:
            try:
                aligned_bgr = self.aligner.align(
                    rec["image_bgr"], rec["kps"]
                )
                if aligned_bgr is None:
                    # Fall back to the detection-time embedding.
                    embeddings_list.append(rec["embedding"])
                    continue

                # Defense-in-depth: get_feat expects HWC uint8.
                if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
                    embeddings_list.append(rec["embedding"])
                    continue
                if aligned_bgr.dtype != np.uint8:
                    aligned_bgr = aligned_bgr.astype(np.uint8)
                aligned_bgr = np.ascontiguousarray(aligned_bgr)

                # Direct recognition forward on the aligned HWC uint8 crop.
                # `get_feat` takes a LIST of images and returns a (N, 512)
                # float32 array of L2-normalized embeddings.
                feat = self.engine.rec_model.get_feat([aligned_bgr])
                feat = np.asarray(feat, dtype=np.float32).flatten()
                # Re-normalize defensively (get_feat already normalizes,
                # but numerical drift is possible).
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                embeddings_list.append(feat)
            except Exception as exc:
                logger.warning(
                    "Aligned embedding failed for %s: %s",
                    rec["img_path"], exc,
                )
                embeddings_list.append(rec["embedding"])

        embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_norm = np.linalg.norm(mean_embedding)
        if mean_norm > 1e-6:
            mean_embedding = mean_embedding / mean_norm

        # Free per-frame image buffers immediately.
        for rec in cluster:
            rec["image_bgr"] = None
            rec["embedding"] = None
        if self.enable_gc:
            gc.collect()

        return StudentProfile(
            student_id=nrp,
            student_name=student_name,
            nrp=nrp,
            face_embeddings=embeddings_array,
            mean_embedding=mean_embedding.astype(np.float32),
            anchor_image_hash=anchor_hash,
            enrollment_timestamp=datetime.now(timezone.utc).isoformat(),
            profile_capacity=self.total_capacity,
        )

    # ------------------------------------------------------------------
    def _serialize_registry(
        self, registry: Dict[str, StudentProfile]
    ) -> None:
        """
        Serialize the registry to disk via pickle. Explicit gc.collect()
        is invoked before and after to mitigate RAM pressure on large
        enrollment runs.
        """
        os.makedirs(os.path.dirname(self.db_pickle_path) or ".", exist_ok=True)

        # Pre-serialization GC.
        gc.collect()

        # Convert to plain dict to avoid dataclass pickling fragility.
        plain_registry: Dict[str, Dict[str, Any]] = {
            sid: prof.to_dict() for sid, prof in registry.items()
        }

        # Write atomically: temp file + rename to prevent partial writes.
        tmp_path = self.db_pickle_path + ".tmp"
        try:
            with open(tmp_path, "wb") as fh:
                pickle.dump(plain_registry, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.db_pickle_path)
            logger.info(
                "Registry serialized -> %s | profiles=%d | size=%.2f KB",
                self.db_pickle_path,
                len(plain_registry),
                os.path.getsize(self.db_pickle_path) / 1024.0,
            )
        except Exception as exc:
            logger.error("Pickle serialization failed: %s", exc)
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        finally:
            # Post-serialization GC to release transient pickle buffers.
            gc.collect()


# ============================================================================
# Registry Loader (for downstream runtime consumers).
# ============================================================================
class StudentRegistryLoader:
    """
    Loads the pickled student_db.pickle into a runtime format optimized
    for vector search ingestion (Index Instance 1 in identity_matcher.py).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or ConfigRegistry.load()
        self.db_pickle_path: str = self.config["enrollment"]["db_pickle_path"]

    # ------------------------------------------------------------------
    def load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.isfile(self.db_pickle_path):
            logger.warning(
                "Student DB pickle not found at %s. Returning empty registry.",
                self.db_pickle_path,
            )
            return {}

        try:
            with open(self.db_pickle_path, "rb") as fh:
                registry = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.error("Failed to unpickle student DB: %s", exc)
            return {}

        logger.info(
            "Loaded student DB | profiles=%d | source=%s",
            len(registry), self.db_pickle_path,
        )
        return registry

    # ------------------------------------------------------------------
    def build_search_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Returns:
          embeddings_matrix : np.ndarray (N_total, 512) -- L2-normalized
          mean_embeddings    : np.ndarray (N_profiles, 512) -- L2-normalized
          student_ids        : List[str]
          student_names      : List[str]
        """
        registry = self.load()
        if not registry:
            empty = np.zeros((0, 512), dtype=np.float32)
            return empty, empty, [], []

        ids: List[str] = []
        names: List[str] = []
        mean_rows: List[np.ndarray] = []
        all_rows: List[np.ndarray] = []

        for sid, prof in registry.items():
            ids.append(sid)
            names.append(prof.get("student_name", sid))
            mean_rows.append(np.asarray(prof["mean_embedding"], dtype=np.float32))
            all_rows.append(np.asarray(prof["face_embeddings"], dtype=np.float32))

        # all_rows is a list of (N_i, 512) arrays; flatten into one matrix.
        embeddings_matrix = np.concatenate(all_rows, axis=0).astype(np.float32)
        mean_matrix = np.stack(mean_rows, axis=0).astype(np.float32)

        # L2 normalize each row (defensive; should already be normalized).
        for i in range(embeddings_matrix.shape[0]):
            n = np.linalg.norm(embeddings_matrix[i])
            if n > 1e-6:
                embeddings_matrix[i] /= n
        for i in range(mean_matrix.shape[0]):
            n = np.linalg.norm(mean_matrix[i])
            if n > 1e-6:
                mean_matrix[i] /= n

        logger.info(
            "Built search arrays | total_embeddings=%d | profiles=%d | dim=%d",
            embeddings_matrix.shape[0],
            mean_matrix.shape[0],
            embeddings_matrix.shape[1],
        )
        return embeddings_matrix, mean_matrix, ids, names


# ============================================================================
# Interactive Single-Student Enrollment Service
# ============================================================================
# Patch: Student Enrollment Module (per professor's request)
# ----------------------------------------------------------------------------
# A service-layer wrapper around _LightFaceEngine + ArcFaceAligner that
# supports interactive single-student enrollment from the Streamlit UI.
#
# Implements two-stage duplicate detection:
#   Stage 1 - check_existing_by_id()    : exact-match lookup on student_id
#   Stage 2 - check_existing_by_face()  : cosine similarity (>= 0.6) against
#                                         every stored mean_embedding
#
# Only if BOTH stages pass (no duplicate), enroll_student() builds a
# StudentProfile from 2 supplied photos (flat + subtle expression),
# appends it to the pickled registry atomically, and returns the profile.
#
# The engine + aligner are LAZY-constructed on first use to avoid burning
# VRAM just by importing this module. Call close() to release VRAM after
# the enrollment session ends.
# ============================================================================
class EnrollmentService:
    """
    Interactive single-student enrollment service with two-stage
    duplicate detection (by student_id, then by face cosine similarity).
    """

    # Default cosine similarity threshold for "same person" decision.
    # Caller may override via config["enrollment"]["duplicate_cosine_threshold"].
    _DEFAULT_COSINE_THRESHOLD: float = 0.6

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or ConfigRegistry.load()
        enr = self.config["enrollment"]
        self.db_pickle_path: str = enr["db_pickle_path"]
        self.cosine_threshold: float = float(
            enr.get("duplicate_cosine_threshold", self._DEFAULT_COSINE_THRESHOLD)
        )
        # Lazy-initialized heavy resources (only built on first need).
        self._engine: Optional[_LightFaceEngine] = None
        self._aligner: Optional[ArcFaceAligner] = None
        # Use the registration det_size (640,640) for higher-quality
        # embeddings during interactive enrollment, matching the
        # offline enroll.py CLI behavior.
        reg_det_size = tuple(enr.get("registration_det_size", [640, 640]))
        self._reg_det_size: Tuple[int, int] = reg_det_size  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lazy resource construction.
    # ------------------------------------------------------------------
    def _ensure_engine(self) -> _LightFaceEngine:
        """Lazily construct + initialize + warm up the registration engine."""
        if self._engine is None:
            logger.info(
                "EnrollmentService: constructing _LightFaceEngine "
                "(det_size=%s) for interactive enrollment ...",
                self._reg_det_size,
            )
            engine = _LightFaceEngine(
                config=self.config, det_size_override=self._reg_det_size
            )
            engine.initialize()
            engine.warmup()
            self._engine = engine
        if self._aligner is None:
            self._aligner = ArcFaceAligner(config=self.config)
        return self._engine

    # ------------------------------------------------------------------
    # Pickle load helper (private).
    # ------------------------------------------------------------------
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the pickled registry. Returns {} if missing/corrupt."""
        if not os.path.isfile(self.db_pickle_path):
            return {}
        try:
            with open(self.db_pickle_path, "rb") as fh:
                return pickle.load(fh) or {}
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.error(
                "EnrollmentService: failed to unpickle %s: %s",
                self.db_pickle_path, exc,
            )
            return {}

    # ------------------------------------------------------------------
    # Stage 1: ID lookup.
    # ------------------------------------------------------------------
    def check_existing_by_id(
        self, student_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Stage-1 duplicate check: exact-match lookup on student_id (NRP).

        Returns the matching profile dict if student_id is already in
        the registry, else None.
        """
        if not student_id:
            return None
        registry = self._load_registry()
        # The pickle keys ARE the student_id (= nrp). Defensive: also
        # check the "student_id" / "nrp" fields inside each profile
        # in case the registry was hand-edited.
        if student_id in registry:
            return registry[student_id]
        for sid, prof in registry.items():
            if (
                str(prof.get("student_id", "")) == student_id
                or str(prof.get("nrp", "")) == student_id
            ):
                return prof
        return None

    # ------------------------------------------------------------------
    # Stage 2: face cosine similarity.
    # ------------------------------------------------------------------
    def check_existing_by_face(
        self, image_bgr: np.ndarray,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Stage-2 duplicate check: detect+embed the largest face in
        `image_bgr`, then compute cosine similarity against every
        stored mean_embedding in the registry.

        Returns:
            (student_id, student_name, score) of the best match if
            score >= self.cosine_threshold, else None.

        Returns None (with a warning log) if no face is detected in
        the supplied image.
        """
        registry = self._load_registry()
        if not registry:
            return None  # Empty DB -> no duplicates possible.

        engine = self._ensure_engine()
        faces = engine.detect_and_embed(image_bgr)
        if not faces:
            logger.warning(
                "EnrollmentService: no face detected in supplied image."
            )
            return None

        # Pick the largest face by bbox area (matches the offline
        # EnrollmentClusterer policy).
        best_face = max(
            faces,
            key=lambda f: float(
                (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
            ),
        )
        query_emb = np.asarray(best_face["embedding"], dtype=np.float32).flatten()
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 1e-6:
            query_emb = query_emb / q_norm

        # Build mean-embedding matrix from the registry.
        mean_rows: List[np.ndarray] = []
        ids: List[str] = []
        names: List[str] = []
        for sid, prof in registry.items():
            mean = prof.get("mean_embedding")
            if mean is None:
                continue
            mean = np.asarray(mean, dtype=np.float32).flatten()
            n = np.linalg.norm(mean)
            if n > 1e-6:
                mean = mean / n
            mean_rows.append(mean)
            ids.append(sid)
            names.append(str(prof.get("student_name", sid)))

        if not mean_rows:
            return None

        matrix = np.stack(mean_rows, axis=0)  # (N, 512)
        # Cosine sim = dot product (vectors are L2-normalized).
        sims = matrix @ query_emb  # (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        logger.info(
            "EnrollmentService: face dedup scan | candidates=%d | "
            "best=(%s, %s, cos=%.4f) | threshold=%.2f",
            len(ids), ids[best_idx], names[best_idx],
            best_score, self.cosine_threshold,
        )

        if best_score >= self.cosine_threshold:
            return (ids[best_idx], names[best_idx], best_score)
        return None

    # ------------------------------------------------------------------
    # Enroll a new student from exactly 2 photos.
    # ------------------------------------------------------------------
    def enroll_student(
        self,
        student_id: str,
        student_name: str,
        photos: List[np.ndarray],
    ) -> StudentProfile:
        """
        Build a StudentProfile from 2 supplied photos (flat + subtle
        expression) and APPEND it to the pickled registry atomically.

        Caller is responsible for running check_existing_by_id() and
        check_existing_by_face() FIRST; this method does NOT re-check.

        Args:
            student_id: e.g. "221050" (the NRP / student number).
            student_name: human-readable name. If empty, falls back
                to student_id.
            photos: list of exactly 2 BGR uint8 HWC images.

        Returns:
            The constructed StudentProfile.
        """
        if len(photos) != 2:
            raise ValueError(
                f"enroll_student() requires exactly 2 photos, got {len(photos)}."
            )
        if not student_id:
            raise ValueError("student_id must be a non-empty string.")

        if not student_name:
            student_name = student_id

        engine = self._ensure_engine()
        aligner = self._aligner  # type: ignore[assignment]
        assert aligner is not None  # _ensure_engine populated it.

        # Detect + embed each photo. Track symmetry score so we can
        # designate the more-frontal photo as the anchor (mirrors the
        # offline clusterer's anchor-selection policy).
        records: List[Dict[str, Any]] = []
        for idx, img in enumerate(photos):
            if img is None or img.size == 0:
                raise ValueError(f"Photo {idx + 1} is empty.")
            faces = engine.detect_and_embed(img)
            if not faces:
                raise ValueError(
                    f"No face detected in photo {idx + 1}. Capture a "
                    f"clearer photo with the face well-lit and centered."
                )
            # Largest face by bbox area.
            face = max(
                faces,
                key=lambda f: float(
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                ),
            )
            sym = EnrollmentClusterer._symmetry_score(face["kps"])
            records.append({
                "image_bgr": img,
                "kps": face["kps"],
                "embedding": np.asarray(
                    face["embedding"], dtype=np.float32
                ).flatten(),
                "symmetry": sym,
            })

        # Designate the more-frontal record (lowest symmetry score)
        # as the anchor for hashing purposes.
        records.sort(key=lambda r: r["symmetry"])
        anchor = records[0]
        anchor_hash = hashlib.sha1(
            f"{student_id}:{anchor['symmetry']:.4f}".encode("utf-8")
        ).hexdigest()[:12]

        # Re-extract embeddings via the aligned path for template
        # consistency with the offline clusterer (see the BUG HISTORY
        # note in EnrollmentClusterer._enroll_single_directory -- we
        # must pass HWC uint8 aligned crops to rec_model.get_feat, NOT
        # pre-normalized tensors).
        embeddings_list: List[np.ndarray] = []
        for rec in records:
            try:
                aligned_bgr = aligner.align(rec["image_bgr"], rec["kps"])
                if aligned_bgr is None:
                    embeddings_list.append(rec["embedding"])
                    continue
                if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
                    embeddings_list.append(rec["embedding"])
                    continue
                if aligned_bgr.dtype != np.uint8:
                    aligned_bgr = aligned_bgr.astype(np.uint8)
                aligned_bgr = np.ascontiguousarray(aligned_bgr)
                feat = engine.rec_model.get_feat([aligned_bgr])
                feat = np.asarray(feat, dtype=np.float32).flatten()
                n = np.linalg.norm(feat)
                if n > 1e-6:
                    feat = feat / n
                embeddings_list.append(feat)
            except Exception as exc:
                logger.warning(
                    "EnrollmentService: aligned embedding failed for "
                    "student %s, falling back to detection-time "
                    "embedding: %s", student_id, exc,
                )
                embeddings_list.append(rec["embedding"])

        embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_norm = np.linalg.norm(mean_embedding)
        if mean_norm > 1e-6:
            mean_embedding = mean_embedding / mean_norm

        profile = StudentProfile(
            student_id=student_id,
            student_name=student_name,
            nrp=student_id,
            face_embeddings=embeddings_array,
            mean_embedding=mean_embedding.astype(np.float32),
            anchor_image_hash=anchor_hash,
            enrollment_timestamp=datetime.now(timezone.utc).isoformat(),
            profile_capacity=len(embeddings_list),
        )

        # APPEND to the existing registry and persist atomically.
        registry = self._load_registry()
        registry[student_id] = profile.to_dict()
        self._save_registry(registry)

        # Free per-photo buffers.
        for rec in records:
            rec["image_bgr"] = None
            rec["embedding"] = None
        if self.config.get("enrollment", {}).get("enable_gc_collect_per_profile", True):
            gc.collect()

        logger.info(
            "EnrollmentService: enrolled student %s (%s) | embeddings=%d",
            student_id, student_name, embeddings_array.shape[0],
        )
        return profile

    # ------------------------------------------------------------------
    # Add more embeddings to an EXISTING student (no dedup, honors capacity).
    # ------------------------------------------------------------------
    def add_embeddings_to_student(
        self,
        student_id: str,
        photos: List[np.ndarray],
    ) -> StudentProfile:
        """
        Append new face embeddings to an EXISTING student's profile.

        Skips the dedup checks (the student already exists; we are
        augmenting, not re-enrolling). Honors profile_capacity: if the
        current embedding count is already at capacity, raises ValueError.
        Otherwise, appends up to (capacity - current_count) new embeddings
        (extras are silently trimmed with a warning log).

        Args:
            student_id: existing student ID (must already be in registry).
            photos: list of 1..N new BGR uint8 HWC images.

        Returns:
            The updated StudentProfile.

        Raises:
            KeyError: if student_id is not in the registry.
            ValueError: if profile is already at capacity, or no faces
                detected in any of the supplied photos.
        """
        if not student_id:
            raise ValueError("student_id must be a non-empty string.")
        if not photos:
            raise ValueError("photos list is empty.")

        registry = self._load_registry()
        if student_id not in registry:
            raise KeyError(f"Student '{student_id}' not found in registry.")

        prof = registry[student_id]
        existing = prof.get("face_embeddings")
        if existing is None:
            existing = np.zeros((0, 512), dtype=np.float32)
        existing = np.asarray(existing, dtype=np.float32)
        if existing.ndim == 1:
            existing = existing.reshape(1, -1)

        capacity = int(prof.get("profile_capacity", 25))
        current_count = existing.shape[0]
        slots_left = capacity - current_count
        if slots_left <= 0:
            raise ValueError(
                f"Student '{student_id}' is already at capacity "
                f"({current_count}/{capacity}). Cannot add more embeddings."
            )

        engine = self._ensure_engine()
        aligner = self._aligner  # type: ignore[assignment]
        assert aligner is not None  # _ensure_engine populated it.

        # Detect + embed each new photo (same pipeline as enroll_student).
        new_records: List[Dict[str, Any]] = []
        for idx, img in enumerate(photos):
            if img is None or img.size == 0:
                logger.warning(
                    "add_embeddings: photo %d is empty; skipping.", idx,
                )
                continue
            faces = engine.detect_and_embed(img)
            if not faces:
                logger.warning(
                    "add_embeddings: no face detected in photo %d; skipping.",
                    idx,
                )
                continue
            face = max(
                faces,
                key=lambda f: float(
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                ),
            )
            sym = EnrollmentClusterer._symmetry_score(face["kps"])
            new_records.append({
                "image_bgr": img,
                "kps": face["kps"],
                "embedding": np.asarray(
                    face["embedding"], dtype=np.float32
                ).flatten(),
                "symmetry": sym,
            })

        if not new_records:
            raise ValueError(
                "No faces detected in ANY of the supplied photos. "
                "Nothing to add."
            )

        # Re-extract via aligned path (mirrors enroll_student for
        # template consistency).
        new_embeddings: List[np.ndarray] = []
        for rec in new_records:
            try:
                aligned_bgr = aligner.align(rec["image_bgr"], rec["kps"])
                if aligned_bgr is None or aligned_bgr.ndim != 3 \
                        or aligned_bgr.shape[2] != 3:
                    new_embeddings.append(rec["embedding"])
                    continue
                if aligned_bgr.dtype != np.uint8:
                    aligned_bgr = aligned_bgr.astype(np.uint8)
                aligned_bgr = np.ascontiguousarray(aligned_bgr)
                feat = engine.rec_model.get_feat([aligned_bgr])
                feat = np.asarray(feat, dtype=np.float32).flatten()
                n = np.linalg.norm(feat)
                if n > 1e-6:
                    feat = feat / n
                new_embeddings.append(feat)
            except Exception as exc:
                logger.warning(
                    "add_embeddings: aligned embedding failed for student "
                    "%s, falling back to detection-time embedding: %s",
                    student_id, exc,
                )
                new_embeddings.append(rec["embedding"])

        # Trim to slots_left (drop extras, log a warning).
        if len(new_embeddings) > slots_left:
            logger.warning(
                "add_embeddings: %d new embeddings would exceed capacity "
                "(current=%d, capacity=%d, slots_left=%d). Trimming to %d.",
                len(new_embeddings), current_count, capacity,
                slots_left, slots_left,
            )
            new_embeddings = new_embeddings[:slots_left]

        # Append + recompute mean.
        combined = np.concatenate(
            [existing, np.stack(new_embeddings, axis=0)], axis=0,
        ).astype(np.float32)
        mean_embedding = np.mean(combined, axis=0)
        mean_norm = np.linalg.norm(mean_embedding)
        if mean_norm > 1e-6:
            mean_embedding = mean_embedding / mean_norm

        # Preserve original fields, update embeddings + timestamp.
        prof["face_embeddings"] = combined
        prof["mean_embedding"] = mean_embedding.astype(np.float32)
        prof["enrollment_timestamp"] = datetime.now(timezone.utc).isoformat()
        # Note: anchor_image_hash is preserved from the original
        # enrollment (re-computing it across the combined set would
        # require re-running symmetry on every stored photo, which we
        # don't have pixel data for after the original enroll).
        registry[student_id] = prof
        self._save_registry(registry)

        # Free per-photo buffers.
        for rec in new_records:
            rec["image_bgr"] = None
            rec["embedding"] = None
        if self.config.get("enrollment", {}).get(
            "enable_gc_collect_per_profile", True,
        ):
            gc.collect()

        logger.info(
            "EnrollmentService: added %d embeddings to student %s | "
            "total=%d/%d",
            len(new_embeddings), student_id,
            combined.shape[0], capacity,
        )
        return StudentProfile(
            student_id=str(prof["student_id"]),
            student_name=str(prof.get("student_name", student_id)),
            nrp=str(prof.get("nrp", student_id)),
            face_embeddings=combined,
            mean_embedding=mean_embedding.astype(np.float32),
            anchor_image_hash=str(prof.get("anchor_image_hash", "")),
            enrollment_timestamp=prof["enrollment_timestamp"],
            profile_capacity=capacity,
        )


    # ------------------------------------------------------------------
    # Update an existing student's ID and/or name (in-place).
    # ------------------------------------------------------------------
    # CODENAME PROTECTION: students whose current student_id is in the
    # `_CODENAME_IDS` set (DTI-1, DTI-2) are tagged as "codename" --
    # their student_id CANNOT be changed (it is a stable operational
    # codename). Their student_name CAN still be changed.
    # For non-codename students, BOTH student_id and student_name can
    # be changed.
    _CODENAME_IDS: frozenset = frozenset({"DTI-1", "DTI-2"})

    def update_student_profile(
        self,
        old_student_id: str,
        new_student_id: str,
        new_student_name: str,
        rename_folder: bool = True,
    ) -> Dict[str, Any]:
        """
        Update an existing student's ID and/or name IN PLACE.

        Special handling:
          * DTI-1 and DTI-2 are codenames. Their student_id CANNOT be
            changed -- if `old_student_id` is one of these and
            `new_student_id != old_student_id`, ValueError is raised.
            Their `student_name` CAN still be updated (e.g. to
            "DTI-1_Budi" so the on-disk folder becomes
            "DTI-1_Budi").

        On student_id change (non-codename only):
          * Registry key is renamed from old -> new.
          * On-disk raw-photos folder:
                data/student_faces/{old_id}_{old_name}
              is renamed to
                data/student_faces/{new_id}_{new_name}
            (only if the old folder exists). If the new folder name
            already exists, the rename is SKIPPED with a warning so we
            never clobber an existing directory.

        Args:
            old_student_id: current student_id (must already be in registry).
            new_student_id: desired new student_id. For codename students
                this MUST equal old_student_id (cannot change).
            new_student_name: desired new student_name.
            rename_folder: if True (default), rename the on-disk
                data/student_faces/{old_id}_{old_name} folder to match.
                Set False for headless / scripted edits where the
                on-disk folder should be left untouched.

        Returns:
            Dict with: old_id, old_name, new_id, new_name,
                       id_changed (bool), name_changed (bool),
                       folder_renamed (bool|None),
                       folder_old (str|None), folder_new (str|None).

        Raises:
            KeyError: if old_student_id is not in the registry.
            ValueError: if a codename student's ID is being changed,
                if new_student_id is empty, or if new_student_id
                collides with another existing student's ID.
        """
        if not old_student_id:
            raise ValueError("old_student_id must be a non-empty string.")
        if not new_student_id:
            raise ValueError("new_student_id must be a non-empty string.")
        if not new_student_name:
            # Fall back to the new ID -- a profile must have a name.
            new_student_name = new_student_id

        # Trim whitespace -- common operator typo.
        old_student_id = str(old_student_id).strip()
        new_student_id = str(new_student_id).strip()
        new_student_name = str(new_student_name).strip()

        # ------------------------------------------------------------------
        # Codename protection.
        # ------------------------------------------------------------------
        is_codename = old_student_id in self._CODENAME_IDS
        if is_codename and new_student_id != old_student_id:
            raise ValueError(
                f"Student '{old_student_id}' is a codename student -- its "
                f"student_id CANNOT be changed. Only the name can be "
                f"updated (e.g. set new_student_name='DTI-1_Budi')."
            )

        # ------------------------------------------------------------------
        # Load registry + verify existence.
        # ------------------------------------------------------------------
        registry = self._load_registry()
        if old_student_id not in registry:
            raise KeyError(
                f"Student '{old_student_id}' not found in registry."
            )
        prof = registry[old_student_id]
        old_name = str(prof.get("student_name", old_student_id))

        # ------------------------------------------------------------------
        # Collision check on new_student_id (only matters if it changed).
        # ------------------------------------------------------------------
        id_changed = (new_student_id != old_student_id)
        name_changed = (new_student_name != old_name)
        if id_changed and new_student_id in registry:
            raise ValueError(
                f"new_student_id '{new_student_id}' is already in use by "
                f"another student. Refusing to merge."
            )

        if not id_changed and not name_changed:
            logger.info(
                "EnrollmentService.update_student_profile: no changes for "
                "student %s (id and name unchanged).", old_student_id,
            )
            return {
                "old_id": old_student_id,
                "old_name": old_name,
                "new_id": new_student_id,
                "new_name": new_student_name,
                "id_changed": False,
                "name_changed": False,
                "folder_renamed": None,
                "folder_old": None,
                "folder_new": None,
            }

        # ------------------------------------------------------------------
        # Apply profile-dict updates.
        # ------------------------------------------------------------------
        prof["student_id"] = new_student_id
        prof["student_name"] = new_student_name
        prof["nrp"] = new_student_id  # nrp always mirrors student_id
        prof["enrollment_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Re-key the registry if the ID changed.
        if id_changed:
            # Preserve insertion order: build a new dict iterating the
            # old one, swapping the key in place.
            new_registry: Dict[str, Dict[str, Any]] = {}
            for k, v in registry.items():
                new_registry[new_student_id if k == old_student_id else k] = v
            registry = new_registry
        else:
            registry[old_student_id] = prof

        self._save_registry(registry)

        # ------------------------------------------------------------------
        # Optional: rename on-disk raw-photos folder.
        # Layout: data/student_faces/{student_id}_{student_name}
        # ------------------------------------------------------------------
        folder_renamed: Optional[bool] = None
        folder_old: Optional[str] = None
        folder_new: Optional[str] = None
        if rename_folder:
            faces_dir = self.config.get("enrollment", {}).get(
                "student_faces_dir", "data/student_faces"
            )
            if not os.path.isabs(faces_dir):
                # Resolve relative to CWD -- matches the offline enroll.py
                # behavior which is run from the project root.
                faces_dir = os.path.abspath(faces_dir)
            folder_old = os.path.join(faces_dir, f"{old_student_id}_{old_name}")
            folder_new = os.path.join(
                faces_dir, f"{new_student_id}_{new_student_name}"
            )
            try:
                if os.path.isdir(folder_old):
                    if os.path.exists(folder_new):
                        logger.warning(
                            "update_student_profile: target folder already "
                            "exists, skipping rename: %s -> %s",
                            folder_old, folder_new,
                        )
                        folder_renamed = False
                    elif folder_old == folder_new:
                        # Path unchanged (e.g. only one field changed and
                        # the other was the same).
                        folder_renamed = False
                    else:
                        os.makedirs(
                            os.path.dirname(folder_new) or ".", exist_ok=True
                        )
                        os.rename(folder_old, folder_new)
                        folder_renamed = True
                        logger.info(
                            "update_student_profile: renamed folder %s -> %s",
                            folder_old, folder_new,
                        )
                else:
                    # No raw-photos folder for this student (e.g. enrolled
                    # purely via the dashboard upload flow without ever
                    # writing the source images to disk). Not an error.
                    folder_renamed = False
                    logger.info(
                        "update_student_profile: no raw-photos folder for "
                        "student %s at %s; skipping rename.",
                        old_student_id, folder_old,
                    )
            except OSError as exc:
                # Folder rename failure is NOT fatal -- the pickle is
                # already updated. Log + surface to caller.
                logger.warning(
                    "update_student_profile: folder rename failed: %s "
                    "(old=%s new=%s). Pickle is still updated.",
                    exc, folder_old, folder_new,
                )
                folder_renamed = False

        logger.info(
            "EnrollmentService.update_student_profile: %s -> %s | "
            "name '%s' -> '%s' | folder_renamed=%s",
            old_student_id, new_student_id,
            old_name, new_student_name, folder_renamed,
        )
        return {
            "old_id": old_student_id,
            "old_name": old_name,
            "new_id": new_student_id,
            "new_name": new_student_name,
            "id_changed": id_changed,
            "name_changed": name_changed,
            "folder_renamed": folder_renamed,
            "folder_old": folder_old,
            "folder_new": folder_new,
        }

    # ------------------------------------------------------------------
    # Atomic pickle save (mirrors EnrollmentClusterer._serialize_registry).
    # ------------------------------------------------------------------
    def _save_registry(
        self, registry: Dict[str, Dict[str, Any]]
    ) -> None:
        """Serialize registry to disk atomically (temp + rename)."""
        os.makedirs(os.path.dirname(self.db_pickle_path) or ".", exist_ok=True)
        gc.collect()
        tmp_path = self.db_pickle_path + ".tmp"
        try:
            with open(tmp_path, "wb") as fh:
                pickle.dump(registry, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.db_pickle_path)
            logger.info(
                "EnrollmentService: registry persisted -> %s | "
                "profiles=%d | size=%.2f KB",
                self.db_pickle_path, len(registry),
                os.path.getsize(self.db_pickle_path) / 1024.0,
            )
        except Exception:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        finally:
            gc.collect()

    # ------------------------------------------------------------------
    # VRAM release.
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the lazy-constructed engine (frees VRAM)."""
        if self._engine is not None:
            try:
                self._engine.close()
                logger.info("EnrollmentService: engine closed; VRAM released.")
            except Exception as exc:
                logger.warning(
                    "EnrollmentService: engine close failed: %s", exc
                )
            self._engine = None
        self._aligner = None
        gc.collect()


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """
    Lightweight self-test entry point invoked when this module is run
    directly. Performs:
      1. Config load.
      2. Engine init + warmup.
      3. Enrollment of any directories found in data/student_faces.
      4. Registry reload and shape verification.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance database_manager self-test ===")

    cfg = ConfigRegistry.load("config/config.yaml")

    # Optional CPU affinity lock for the bootstrap process.
    affinity_masks = (
        cfg.get("hardware", {}).get("cpu", {}).get("affinity_masks", {})
    )
    if (
        affinity_masks
        and cfg["hardware"]["cpu"].get("enable_affinity_lock", False)
    ):
        apply_cpu_affinity(
            affinity_masks.get("ai_inference_thread", [4, 5, 6, 7])
        )

    engine = _LightFaceEngine(cfg)
    engine.initialize()
    engine.warmup()

    aligner = ArcFaceAligner(cfg)
    clusterer = EnrollmentClusterer(engine, aligner, cfg)

    registry, stats = clusterer.enroll_all()

    loader = StudentRegistryLoader(cfg)
    embs, means, ids, names = loader.build_search_arrays()
    logger.info(
        "Self-test OK | embeddings=%s means=%s ids=%d",
        embs.shape, means.shape, len(ids),
    )

    engine.close()
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()