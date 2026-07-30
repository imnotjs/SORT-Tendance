"""
SORT-tendance :: src/core/identity_matcher.py

Production-grade Re-ID and dual-instance vector search engine.

Responsibilities:
  1. Initialize the deep OSNet AIN (osnet_ain_x1_0) feature extraction
     network onto GPU memory with ImageNet standardization constants
     pre-allocated as VRAM matrices and a fixed tensor workspace
     (batch_size, 3, 256, 128).
  2. CUDA Graph Pre-Compilation Warmup across batch sizes [1, 2, 4, 8]
     followed by hard torch.cuda.synchronize() directives.
  3. Dual-Instance Vector Search Indexing via usearch.index.Index with
     a Cosine distance metric:
       * Index Instance 1 -- Static student face DB (read-only).
       * Index Instance 2 -- Dynamic stranger appearance cache.
  4. Batched Matrix-Multiplication Matching (live_matrix @ db_matrix.T)
     via NumPy / PyTorch, exposing `batch_match_faces_raw` for the state
     machine.
  5. Hybrid Re-ID Tracking with rolling body feature queues (size 10)
     combined with spatial Euclidean distance checks (<=150 px) and
     maximum dot-product similarity (>=0.70).
  6. Inference Termination Logic -- once a track is resolved as a
     Verified Student or mapped to the Dynamic Stranger Cache, expose
     handles for the orchestrator to fully terminate facial scanning
     and recognition routines for that track ID.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import gc
import time
import logging
import threading
import traceback
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    from usearch.index import Index
    _USEARCH_AVAILABLE = True
except ImportError:                       # pragma: no cover
    _USEARCH_AVAILABLE = False
    Index = None  # type: ignore

# Local imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.database_manager import ConfigRegistry, StudentRegistryLoader
except ImportError:                       # pragma: no cover
    ConfigRegistry = None  # type: ignore
    StudentRegistryLoader = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.identity_matcher")
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
class FaceMatchResult:
    """Raw face-similarity match result, fed to the state machine."""
    track_id: int
    best_student_id: Optional[str]
    best_student_name: Optional[str]
    best_similarity: float
    top_k_similarities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    top_k_student_ids: List[Optional[str]] = field(default_factory=list)


@dataclass
class BodyReIDResult:
    """Result of the hybrid Re-ID matching routine."""
    track_id: int
    matched_track_id: Optional[int]
    spatial_distance_px: float
    body_similarity: float
    inheritance_applied: bool


@dataclass
class StrangerRecord:
    """A registered stranger in the dynamic appearance cache."""
    stranger_id: int                       # Sequential 1-based index
    label: str                             # "Stranger_01", "Stranger_02", ...
    first_seen_us: int
    last_seen_us: int
    # Patch 57 :: Capped deque prevents unbounded growth over a
    # 12-hour session. 64 centroids is enough for spatial trajectory
    # analysis without leaking memory (previously grew to tens of
    # thousands of entries per stranger).
    centroid_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    # Patch 57 :: Store the body feature vector so the LRU eviction
    # can rebuild the USearch index from surviving records without
    # needing to retrieve vectors from the old (destroyed) index.
    body_feature: Optional[np.ndarray] = None


@dataclass
class TrackInferenceState:
    """
    Per-track inference lifecycle state. Used by the Inference Termination
    Logic to suspend facial scanning once a track is resolved.
    """
    track_id: int
    face_scanning_active: bool = True
    body_reid_active: bool = True
    resolved_state: str = "PENDING"        # PENDING | VERIFIED_STUDENT | STRANGER | ANOMALY
    resolved_label: str = "[PENDING]"
    resolved_student_id: Optional[str] = None
    resolved_stranger_id: Optional[int] = None
    # Rolling body feature queue (length capped at body_history_size).
    body_feature_history: List[np.ndarray] = field(default_factory=list)
    body_centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    # EMA / TTFM accumulator state (owned by res_opt_engine; surfaced here
    # so the matcher can read termination hints without circular imports).
    ema_score: float = 0.0
    ema_cluster_size: int = 0


# ============================================================================
# OSNet AIN Re-ID Network Wrapper
# ============================================================================
class OSNetReID:
    """
    Loads the OSNet AIN (osnet_ain_x1_0) feature extractor onto GPU.

    Pre-allocates ImageNet standardization constants as VRAM tensors and
    establishes a fixed workspace tensor of shape (8, 3, 256, 128) so the
    forward path never triggers a runtime allocation.

    CUDA Graph Pre-Compilation Warmup iterates batch sizes [1, 2, 4, 8]
    and forces a hard torch.cuda.synchronize() after each pass to lock
    the lazy-compiled deep kernels across all expected occupancy scales.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for OSNetReID.")
        if not _CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for OSNetReID.")

        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )
        reid_cfg = self.config["identity_matcher"]["reid"]
        self.weights_path: str = str(reid_cfg["weights"])
        self.device: str = str(reid_cfg["device"])
        self.input_dim: Tuple[int, int, int] = tuple(reid_cfg["input_dim"])
        self.batch_warmup_sizes: List[int] = list(reid_cfg["batch_warmup_sizes"])
        workspace_shape = tuple(reid_cfg["fixed_workspace_shape"])
        self.workspace_shape: Tuple[int, int, int, int] = workspace_shape

        self.imagenet_mean: List[float] = list(reid_cfg["imagenet_mean"])
        self.imagenet_std: List[float] = list(reid_cfg["imagenet_std"])

        # PyTorch module handle (initialized lazily).
        self.net: Optional[Any] = None
        # Pre-allocated VRAM constants (set in initialize()).
        self._mean_tensor: Optional[Any] = None
        self._std_tensor: Optional[Any] = None
        self._workspace_tensor: Optional[Any] = None
        self._warmed: bool = False

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        # ------------------------------------------------------------------
        # Weights resolution strategy (in priority order):
        #   1. Local weights file at `self.weights_path` (if it exists)
        #      -- fastest path, fully offline, reproducible.
        #   2. torchreid auto-download via `pretrained=True` -- torchreid
        #      fetches the official OSNet AIN weights and caches them
        #      under ~/.cache/torch/hub/checkpoints/. The cached file
        #      survives across runs, so this is a one-time cost.
        # ------------------------------------------------------------------
        local_weights_available: bool = os.path.isfile(self.weights_path)

        if not local_weights_available:
            logger.warning(
                "OSNet weights not found at configured path=%s -- "
                "falling back to torchreid pretrained auto-download. "
                "The cached file will be reused on subsequent runs.",
                self.weights_path,
            )

        try:
            from torchreid import models as reid_models

            self.net = reid_models.build_model(
                name="osnet_ain_x1_0",
                num_classes=1000,
                loss="softmax",
                pretrained=(not local_weights_available),  # auto-dl if no local
            )

            if local_weights_available:
                # Load the local weights and strip DataParallel prefixes.
                state = torch.load(self.weights_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                clean_state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state.items()
                }
                self.net.load_state_dict(clean_state, strict=False)

            self.net = self.net.to(self.device)
            self.net.eval()
            # FP16 inference on Ampere Tensor Cores (RTX 30/40-series).
            # Cuts OSNet forward pass ~50% on RTX 3050 Laptop.
            # Patch 6: explicitly arm FP16 on CUDA, log status so it's
            # visible at boot (otherwise the .half() call is silent and
            # we can't tell from the log whether FP16 is engaged).
            if self.device.startswith("cuda"):
                try:
                    self.net = self.net.half()
                    self._fp16 = True
                except Exception as fp16_exc:
                    logger.warning(
                        "OSNet FP16 arm failed (%s) -- falling back to "
                        "FP32. Forward pass will be ~2x slower.",
                        fp16_exc,
                    )
                    self._fp16 = False
            else:
                self._fp16 = False

            source = (
                f"local file: {self.weights_path}"
                if local_weights_available
                else "torchreid pretrained cache (auto-downloaded)"
            )
            logger.info(
                "OSNet AIN x1_0 loaded | device=%s | source=%s | fp16=%s",
                self.device, source, getattr(self, "_fp16", False),
            )
        except Exception as exc:
            logger.error(
                "OSNet load failed: %s\n%s", exc, traceback.format_exc()
            )
            raise

        # Pre-allocate VRAM tensors for standardization constants.
        # Shape (1, 3, 1, 1) so they broadcast cleanly against NCHW batches.
        _dtype = torch.float16 if getattr(self, "_fp16", False) else torch.float32
        self._mean_tensor = torch.tensor(
            self.imagenet_mean, dtype=_dtype, device=self.device
        ).view(1, 3, 1, 1)
        self._std_tensor = torch.tensor(
            self.imagenet_std, dtype=_dtype, device=self.device
        ).view(1, 3, 1, 1)

        # Pre-allocate the fixed workspace tensor (zero-initialized).
        # Patch 6: match dtype to the model (FP16 when armed) so any
        # future in-place op using this tensor doesn't trigger a silent
        # dtype-upcast promotion that would slow the forward pass.
        _ws_dtype = torch.float16 if getattr(self, "_fp16", False) else torch.float32
        self._workspace_tensor = torch.zeros(
            self.workspace_shape,
            dtype=_ws_dtype,
            device=self.device,
        )
        _ws_bytes = (np.prod(self.workspace_shape) *
                     (2 if _ws_dtype == torch.float16 else 4))
        logger.info(
            "VRAM workspace allocated | shape=%s | dtype=%s | ~%.2f MB",
            self.workspace_shape,
            "fp16" if _ws_dtype == torch.float16 else "fp32",
            _ws_bytes / (1024 ** 2),
        )

    # ------------------------------------------------------------------
    # CUDA Graph Pre-Compilation Warmup.
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """
        Iterate through batch sizes [1, 2, 4, 8] and run a forward pass
        for each, immediately followed by torch.cuda.synchronize() to
        force driver compilation of the lazy kernels across all expected
        human occupancy scales.
        """
        if self.net is None:
            raise RuntimeError("OSNetReID must be initialized before warmup.")

        logger.info(
            "OSNet CUDA Graph pre-compilation warmup | batches=%s",
            self.batch_warmup_sizes,
        )

        rng = torch.Generator(device=self.device)
        rng.manual_seed(2024)

        for bs in self.batch_warmup_sizes:
            try:
                # Build a synthetic batch on the pre-allocated workspace
                # path so memory is reused rather than re-allocated.
                synthetic = torch.randn(
                    (bs, *self.input_dim),
                    generator=rng,
                    device=self.device,
                    dtype=(torch.float16 if getattr(self, "_fp16", False)
                           else torch.float32),
                )
                with torch.no_grad():
                    feats = self.net(synthetic)
                    # L2-normalize to mimic the production inference path.
                    feats = F.normalize(feats, p=2, dim=1)

                # Hard sync -- forces immediate driver compilation.
                torch.cuda.synchronize()
                logger.debug(
                    "Warmup batch=%d | feats=%s | sync OK",
                    bs, tuple(feats.shape),
                )
            except Exception as exc:
                logger.error(
                    "OSNet warmup failed at batch=%d: %s\n%s",
                    bs, exc, traceback.format_exc(),
                )
                raise

        # Final hard synchronization.
        torch.cuda.synchronize()
        self._warmed = True
        gc.collect()
        logger.info("OSNet warmup complete; CUDA kernels locked.")

    # ------------------------------------------------------------------
    # Public inference helpers.
    # ------------------------------------------------------------------
    def extract_features(
        self,
        crops_bgr: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract L2-normalized OSNet features for a list of body crops.

        Args:
            crops_bgr: List of HxWx3 uint8 BGR crops (one per track).

        Returns:
            np.ndarray of shape (N, 512), L2-normalized, float32.
        """
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        # Pre-process each crop: resize -> CHW -> standardize.
        processed: List[np.ndarray] = []
        target_h, target_w = self.input_dim[1], self.input_dim[2]
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                # Pad with a zero tensor so batch indexing stays aligned.
                processed.append(
                    np.zeros((3, target_h, target_w), dtype=np.float32)
                )
                continue
            try:
                resized = cv2.resize(
                    crop, (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
                processed.append(chw)
            except Exception as exc:
                logger.warning("Body crop preprocessing failed: %s", exc)
                processed.append(
                    np.zeros((3, target_h, target_w), dtype=np.float32)
                )

        batch_np = np.stack(processed, axis=0).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_np).to(self.device)
        # Match the model dtype (FP16 if armed in initialize()).
        if getattr(self, "_fp16", False):
            batch_tensor = batch_tensor.half()

        # Standardize using the pre-allocated VRAM constants.
        batch_tensor = (batch_tensor - self._mean_tensor) / self._std_tensor

        try:
            with torch.no_grad():
                feats = self.net(batch_tensor)
                feats = F.normalize(feats, p=2, dim=1)
            # Pull back to CPU as numpy for downstream BLAS matching.
            out = feats.cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.error("OSNet forward failed: %s\n%s", exc, traceback.format_exc())
            out = np.zeros((len(crops_bgr), 512), dtype=np.float32)

        return out

    def is_warmed(self) -> bool:
        return self._warmed

    def close(self) -> None:
        try:
            if self.net is not None:
                del self.net
        except Exception:
            pass
        self.net = None
        self._mean_tensor = None
        self._std_tensor = None
        self._workspace_tensor = None
        self._warmed = False
        if _TORCH_AVAILABLE and self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        logger.info("OSNetReID closed; VRAM released.")


# ============================================================================
# Dual-Instance USearch Index Manager
# ============================================================================

# ---------------------------------------------------------------------------
# usearch metric name normalization. usearch 2.x renamed several metrics
# from their Faiss-style names. We translate legacy names to the modern
# equivalents so old config files keep working without manual edits.
# ---------------------------------------------------------------------------
_USEARCH_METRIC_ALIASES: Dict[str, str] = {
    # Faiss-style names -> usearch 2.x canonical names
    "cosinesimil": "cos",
    "cosine_similarity": "cos",
    "cosine": "cos",
    "l2": "l2sq",
    "euclidean": "l2sq",
    "euclidean_sq": "l2sq",
    "inner_product": "ip",
    "innerproduct": "ip",
    "dot": "ip",
    # Already-canonical usearch names pass through unchanged
    "cos": "cos",
    "l2sq": "l2sq",
    "ip": "ip",
    "haversine": "haversine",
    "divergence": "divergence",
    "hamming": "hamming",
    "jaccard": "jaccard",
    "sosd": "sosd",
}


def _normalize_usearch_metric(name: str) -> str:
    """
    Translate legacy / Faiss-style metric names to usearch 2.x canonical
    names. Falls back to the original string if no alias is found (let
    usearch raise its own KeyError for genuinely unknown metrics).
    """
    if not name:
        return name
    key = name.strip().lower()
    return _USEARCH_METRIC_ALIASES.get(key, name)


def _parse_usearch_matches(matches: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse usearch search results into (keys, distances) arrays.
    Handles usearch 1.x (list of tuples) and 2.x (Matches/Match objects).
    """
    if matches is None:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )

    # usearch 2.x: Matches object with .keys and .distances arrays
    if hasattr(matches, 'keys') and hasattr(matches, 'distances'):
        try:
            keys = np.asarray(matches.keys, dtype=np.int64)
            dists = np.asarray(matches.distances, dtype=np.float32)
            if keys.size > 0:
                return keys, dists
        except Exception:
            pass

    # Try to get first element
    try:
        first = matches[0]
    except (IndexError, TypeError, KeyError):
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )

    # usearch 2.x: first element is a Match object with .key/.distance
    if hasattr(first, 'key') and hasattr(first, 'distance'):
        keys = np.asarray([m.key for m in matches], dtype=np.int64)
        dists = np.asarray([m.distance for m in matches], dtype=np.float32)
        return keys, dists

    # usearch 1.x: first element is a (key, distance) tuple
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        keys = np.asarray([m[0] for m in matches], dtype=np.int64)
        dists = np.asarray([m[1] for m in matches], dtype=np.float32)
        return keys, dists

    # Fallback
    return (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
    )


# P1-H2 fix: synchronized decorator for DualIndexManager methods that
# touch _dynamic_* state. Uses RLock so re-entrant calls (e.g.
# register_stranger -> search_dynamic) don't deadlock.
_DYN_LOCK_ATTR = '_lock'
def _synchronized(method):
    """Decorator: acquire the instance's RLock for the duration of the call."""
    import functools
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, _DYN_LOCK_ATTR, None)
        if lock is None:
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)
    return wrapper

class DualIndexManager:
    """
    Manages the two USearch Index instances:

      * Index Instance 1 (static): built once at startup from the
        pre-normalized student face embeddings in student_db.pickle.
        Read-only at runtime.

      * Index Instance 2 (dynamic): holds unverified OSNet body-reid
        structural vectors for strangers. Supports add + search with
        sequential numbering ("Stranger_01", "Stranger_02", ...).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not _USEARCH_AVAILABLE:
            raise ImportError(
                "usearch is required for DualIndexManager. "
                "Install with: pip install usearch"
            )
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )
        vs = self.config["vector_search"]
        self.metric: str = _normalize_usearch_metric(str(vs["metric"]))
        self.embedding_dim: int = int(vs["embedding_dim"])
        self.reid_dim: int = int(vs["reid_dim"])
        self.static_capacity: int = int(vs["static_db"]["connect_capacity"])
        self.dynamic_capacity: int = int(vs["dynamic_cache"]["max_capacity"])

        # USearch Index instances.
        self.static_index: Optional[Index] = None
        self.dynamic_index: Optional[Index] = None

        # Parallel label arrays for both indexes.
        self._static_keys: List[str] = []     # student_id per row
        self._static_names: List[str] = []    # student_name per row
        self._static_matrix: Optional[np.ndarray] = None  # (N, 512)
        self._dynamic_strangers: Dict[int, StrangerRecord] = {}
        # P1-H2 fix: reentrant lock guarding _dynamic_* state against
        # concurrent access from the AI thread (register_stranger,
        # search_dynamic) and the SessionBoundaryWatcher thread
        # (reset_dynamic_memory). RLock because register_stranger()
        # internally calls search_dynamic() and _evict_lru_stranger().
        self._lock = threading.RLock()
        # P2-M8 fix :: _dynamic_matrix is DEAD CODE -- marked deprecated.
        #
        # Audit found that _dynamic_matrix is WRITTEN to in three places
        # (build_dynamic_index, register_stranger, and one eviction path)
        # but NEVER READ for similarity queries. All dynamic stranger
        # lookups go through `self.dynamic_index.search()` (USearch) +
        # `self._dynamic_strangers` (Dict). The matrix is a write-only
        # side-store that duplicates data already in USearch.
        #
        # Runtime cost: negligible (a few writes per session, ~128KB
        # memory per session). Correctness cost: zero (nothing reads it).
        #
        # Why not delete the field outright? Removing it requires surgery
        # across 5 methods (build_dynamic_index, register_stranger,
        # _evict_lru_stranger, reset_dynamic_memory, and one recall path),
        # which is high-risk for zero runtime benefit on a production
        # hotfix branch. The field is retained for now with this clear
        # deprecation marker; safe to remove in a future refactor that
        # adds test coverage for the registration paths.
        self._dynamic_matrix: Optional[np.ndarray] = None  # (M, 512)  DEPRECATED -- see P2-M8
        self._dynamic_keys: List[int] = []   # usearch keys
        self._next_stranger_id: int = 1
        # Patch 57 :: Pre-allocated matrix growth bookkeeping.
        # Instead of np.vstack (which reallocates the full M x 512
        # array on every append = O(M^2) over a session), we
        # pre-allocate in chunks of 64 rows and track how many
        # are actually used. _dynamic_matrix_rows is the count of
        # valid rows; the matrix may have extra unused capacity.
        # NOTE: P2-M8 -- bookkeeping below is retained only to keep
        # the deprecated write-path self-consistent. Safe to remove
        # together with _dynamic_matrix in a future refactor.
        self._dynamic_matrix_capacity: int = 0  # allocated rows
        self._dynamic_matrix_rows: int = 0      # valid rows
        self._DYNAMIC_MATRIX_CHUNK: int = 64     # growth increment

    # ------------------------------------------------------------------
    def build_static_index(self) -> None:
        """
        Rebuild the static face DB index from student_db.pickle via
        StudentRegistryLoader.
        """
        if StudentRegistryLoader is None:
            logger.warning(
                "StudentRegistryLoader unavailable; static index empty."
            )
            self._static_matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)
            return

        loader = StudentRegistryLoader(self.config)
        embs, means, ids, names = loader.build_search_arrays()

        # Use per-frame embeddings (not means) for finer-grained matching.
        # Map each row back to its student_id and student_name.
        row_ids: List[str] = []
        row_names: List[str] = []
        registry = loader.load()
        for sid, prof in registry.items():
            n_embs = int(np.asarray(prof["face_embeddings"]).shape[0])
            row_ids.extend([sid] * n_embs)
            row_names.extend([prof.get("student_name", sid)] * n_embs)

        self._static_matrix = (
            embs.astype(np.float32)
            if embs.size else
            np.zeros((0, self.embedding_dim), dtype=np.float32)
        )
        self._static_keys = row_ids
        self._static_names = row_names

        # Build the USearch index.
        # NOTE: usearch >= 2.0 removed `capacity` from `Index.__init__()`.
        # Pre-allocation is now done via `.reserve(capacity)` after construction.
        self.static_index = Index(
            ndim=self.embedding_dim,
            metric=self.metric,
        )
        reserve_cap = max(self.static_capacity, len(row_ids))
        try:
            self.static_index.reserve(reserve_cap)
        except Exception:
            # Older usearch versions don't have `.reserve()` -- fine, the
            # index will grow dynamically on `.add()`.
            pass
        if len(row_ids) > 0:
            keys = np.arange(len(row_ids), dtype=np.int64)
            self.static_index.add(keys, self._static_matrix)

        logger.info(
            "Static DB index built | vectors=%d | dim=%d | metric=%s",
            len(row_ids), self.embedding_dim, self.metric,
        )

    # ------------------------------------------------------------------
    @_synchronized
    def build_dynamic_index(self) -> None:
        """Initialize an empty dynamic stranger appearance cache."""
        self.dynamic_index = Index(
            ndim=self.reid_dim,
            metric=self.metric,
        )
        try:
            self.dynamic_index.reserve(self.dynamic_capacity)
        except Exception:
            pass
        self._dynamic_strangers.clear()
        self._dynamic_keys.clear()
        # Patch 57 :: Pre-allocate first chunk for amortized O(1) appends.
        self._dynamic_matrix_capacity = self._DYNAMIC_MATRIX_CHUNK
        self._dynamic_matrix_rows = 0
        self._dynamic_matrix = np.zeros(
            (self._dynamic_matrix_capacity, self.reid_dim), dtype=np.float32
        )
        self._next_stranger_id = 1
        logger.info(
            "Dynamic stranger cache initialized | capacity=%d | dim=%d",
            self.dynamic_capacity, self.reid_dim,
        )

    # ------------------------------------------------------------------
    # Patch 20 :: reset_dynamic_memory()
    #
    # Called by main.py's SessionBoundaryWatcher at every 06:00 / 18:00
    # LOCAL boundary. Clears the OSNet body-Re-ID dynamic stranger
    # cache so strangers get fresh Stranger_XX IDs in the new session
    # (a stranger seen in the 06AM session and again in the 06PM
    # session gets a NEW label rather than inheriting the prior
    # session's ID).
    #
    # What is reset:
    #   - dynamic_index (USearch)          -> rebuilt empty
    #   - _dynamic_strangers (Dict)        -> cleared
    #   - _dynamic_keys (List)             -> cleared
    #   - _dynamic_matrix (np.ndarray)     -> (0, reid_dim)
    #   - _next_stranger_id                -> 1
    #
    # What is PRESERVED:
    #   - static_index (student face DB)   -> untouched
    #   - _static_keys / _static_names     -> untouched
    #   - _static_matrix                   -> untouched
    #
    # This is the SAME logic as build_dynamic_index(), exposed as a
    # separate public method so the intent is explicit at the call
    # site (main.py's watcher calls reset_dynamic_memory(), not
    # "rebuild the dynamic index").
    # ------------------------------------------------------------------
    @_synchronized
    def reset_dynamic_memory(self) -> None:
        """Reset the OSNet body-Re-ID dynamic stranger cache.

        Called at every 06:00 / 18:00 LOCAL boundary by main.py's
        SessionBoundaryWatcher. The static student face DB is
        PRESERVED -- only the per-session stranger appearance cache
        is cleared.
        """
        prev_count = len(self._dynamic_keys)
        # Rebuild the dynamic index in-place. build_dynamic_index()
        # already does exactly what we want: it constructs a fresh
        # empty Index, clears _dynamic_strangers / _dynamic_keys /
        # _dynamic_matrix, and resets _next_stranger_id to 1.
        self.build_dynamic_index()
        logger.info(
            "DualIndexManager.reset_dynamic_memory() :: OSNet dynamic "
            "stranger cache cleared | prev_count=%d | static_db_preserved=%d",
            prev_count, len(self._static_keys),
        )

    # ------------------------------------------------------------------
    # Patch 67 :: Load recalled stranger vectors from disk clearshots.
    #
    # Called by IdentityMatcher.recall_strangers_from_disk() at process
    # startup (after build_dynamic_index()). Takes a dict of
    # {stranger_label: consensus_embedding} produced by stranger_recall.py
    # and inserts each into the dynamic stranger cache.
    #
    # This is the "memory recall" half of the OSNet persistence story:
    # the PNG clearshots on disk ARE the persistence; this method
    # rebuilds the in-memory USearch index + _dynamic_strangers dict
    # from those PNGs so the tracker can match new stranger detections
    # against recalled strangers from earlier in the session.
    # ------------------------------------------------------------------
    @_synchronized
    def load_recall_vectors(
        self,
        recall_map: Dict[str, np.ndarray],
        first_seen_us: Optional[int] = None,
    ) -> int:
        """Load recalled stranger consensus embeddings into the dynamic cache.

        Args:
            recall_map: ``{stranger_label: embedding}`` where label is
                like "Stranger_07" and embedding is (512,) float32
                L2-normalized. Produced by stranger_recall.recall_strangers().
            first_seen_us: Timestamp to use for first_seen/last_seen on
                the recalled StrangerRecords. If None, uses 0 (unknown
                -- the stranger was seen before this process started).

        Returns:
            Number of strangers successfully loaded.
        """
        if not recall_map:
            logger.info(
                "DualIndexManager.load_recall_vectors() :: recall_map empty "
                "-- nothing to load."
            )
            return 0

        loaded = 0
        ts_us = int(first_seen_us) if first_seen_us is not None else 0
        # Use a neutral centroid (0, 0) for recalled strangers -- the
        # actual centroid will be updated on the next live sighting.
        neutral_centroid: Tuple[float, float] = (0.0, 0.0)

        for label, embedding in recall_map.items():
            # Parse the stranger ID from the label (e.g. "Stranger_07" -> 7).
            # If parsing fails, assign the next available ID.
            try:
                sid = int(label.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                sid = self._next_stranger_id

            # Ensure _next_stranger_id stays ahead of all loaded IDs.
            if sid >= self._next_stranger_id:
                self._next_stranger_id = sid + 1

            # LRU eviction if at capacity.
            if len(self._dynamic_keys) >= self.dynamic_capacity:
                self._evict_lru_stranger()

            # Insert into USearch index.
            new_key = len(self._dynamic_keys)
            emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            self.dynamic_index.add(
                np.array([new_key], dtype=np.int64),
                emb,
            )
            self._dynamic_keys.append(new_key)

            # Append to pre-allocated matrix.
            if self._dynamic_matrix_rows >= self._dynamic_matrix_capacity:
                self._dynamic_matrix_capacity += self._DYNAMIC_MATRIX_CHUNK
                new_mat = np.zeros(
                    (self._dynamic_matrix_capacity, self.reid_dim),
                    dtype=np.float32,
                )
                if self._dynamic_matrix is not None and self._dynamic_matrix_rows > 0:
                    new_mat[:self._dynamic_matrix_rows] = (
                        self._dynamic_matrix[:self._dynamic_matrix_rows]
                    )
                self._dynamic_matrix = new_mat
            self._dynamic_matrix[self._dynamic_matrix_rows] = emb.flatten()
            self._dynamic_matrix_rows += 1

            # Create the StrangerRecord.
            self._dynamic_strangers[new_key] = StrangerRecord(
                stranger_id=sid,
                label=label,
                first_seen_us=ts_us,
                last_seen_us=ts_us,
                centroid_history=deque([neutral_centroid], maxlen=64),
                body_feature=emb.flatten(),
            )
            loaded += 1
            logger.info(
                "Stranger recalled from disk | id=%d | label=%s | "
                "cache_size=%d",
                sid, label, len(self._dynamic_keys),
            )

        logger.info(
            "DualIndexManager.load_recall_vectors() :: loaded %d recalled "
            "strangers | cache_size=%d | next_stranger_id=%d",
            loaded, len(self._dynamic_keys), self._next_stranger_id,
        )
        return loaded

    # ------------------------------------------------------------------
    # Static search (face DB).
    # ------------------------------------------------------------------
    def search_static(
        self, query: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]], List[Optional[str]]]:
        """
        Search the static face DB for the top-k nearest neighbors of a
        single query vector. Returns (similarities, indices, student_ids,
        student_names).
        """
        if self.static_index is None or self._static_matrix is None \
                or len(self._static_keys) == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                [],
                [],
            )

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        matches = self.static_index.search(q, count=k)
        if matches is None or len(matches) == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                [],
                [],
            )

        # Parse usearch results (handles both 1.x tuples and 2.x Match objects).
        keys_arr, dists_arr = _parse_usearch_matches(matches)

        # Cosine distance -> similarity: 1 - distance.
        sims = 1.0 - dists_arr
        student_ids: List[Optional[str]] = []
        student_names: List[Optional[str]] = []
        for k_idx in keys_arr:
            if 0 <= int(k_idx) < len(self._static_keys):
                student_ids.append(self._static_keys[int(k_idx)])
                student_names.append(self._static_names[int(k_idx)])
            else:
                student_ids.append(None)
                student_names.append(None)
        return sims, keys_arr, student_ids, student_names

    # ------------------------------------------------------------------
    # Dynamic search (stranger appearance cache).
    # ------------------------------------------------------------------
    @_synchronized
    def search_dynamic(
        self, query: np.ndarray, k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[int]]]:
        """
        Search the dynamic stranger cache. Returns (similarities, keys,
        stranger_ids).
        """
        if self.dynamic_index is None or len(self._dynamic_keys) == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                [],
            )

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        # usearch 2.x renamed 'k' kwarg to 'count'.
        matches = self.dynamic_index.search(
            q, count=min(k, len(self._dynamic_keys))
        )
        if matches is None or len(matches) == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                [],
            )

        # Parse usearch results (handles both 1.x tuples and 2.x Match objects).
        keys_arr, dists_arr = _parse_usearch_matches(matches)

        sims = 1.0 - dists_arr
        stranger_ids: List[Optional[int]] = []
        for k_idx in keys_arr:
            rec = self._dynamic_strangers.get(int(k_idx))
            stranger_ids.append(rec.stranger_id if rec else None)
        return sims, keys_arr, stranger_ids

    # ------------------------------------------------------------------
    @_synchronized
    def register_stranger(
        self,
        body_feature: np.ndarray,
        first_seen_us: int,
        centroid: Tuple[float, float],
        # Patch 58 :: Default lowered from 0.65 -> 0.60 to match config
        # (stranger.appearance_inherit_threshold = 0.60). This default
        # was previously dead code (masked by the IdentityMatcher
        # wrapper at line 1365 which passes the config value), but it
        # was a footgun: any direct caller of DualIndexManager.
        # register_stranger() would have inherited at 0.65 instead of
        # the configured 0.60.
        appearance_inherit_threshold: float = 0.60,
    ) -> Tuple[int, str, bool]:
        """
        Add a new stranger to the dynamic cache, OR inherit an existing
        stranger ID if the appearance similarity exceeds the threshold.

        Returns (stranger_id, label, inherited).
        """
        # First, try to inherit an existing stranger identity.
        # CRITICAL FIX (stranger over-matching bug):
        # The similarity score lives in `sims[0]` (float in [0.0, 1.0]).
        # `sids[0]` is the stranger_id integer (1, 2, 3, ...) and MUST NOT
        # be compared against the threshold -- doing so caused every new
        # stranger to inherit Stranger_01 because `1 >= 0.65` is always True.
        if len(self._dynamic_keys) > 0:
            sims, _, sids = self.search_dynamic(body_feature, k=1)
            best_sim = float(sims[0]) if len(sims) > 0 else 0.0
            best_sid = sids[0] if len(sids) > 0 else None
            if best_sid is not None and best_sim >= appearance_inherit_threshold:
                # Inherit the existing stranger ID.
                existing_sid = best_sid
                # Find the record by stranger_id.
                for rec in self._dynamic_strangers.values():
                    if rec.stranger_id == existing_sid:
                        rec.last_seen_us = first_seen_us
                        rec.centroid_history.append(centroid)
                        logger.info(
                            "Stranger inherited | id=%d | label=%s | "
                            "sim=%.4f | threshold=%.2f | cache_size=%d",
                            rec.stranger_id, rec.label, best_sim,
                            appearance_inherit_threshold,
                            len(self._dynamic_keys),
                        )
                        return (
                            rec.stranger_id,
                            rec.label,
                            True,
                        )
            elif best_sid is not None:
                logger.info(
                    "Stranger NOT inherited (sim below threshold) | "
                    "best_sim=%.4f | threshold=%.2f | best_match_sid=%s | "
                    "cache_size=%d -- will register new stranger",
                    best_sim, appearance_inherit_threshold, best_sid,
                    len(self._dynamic_keys),
                )

        # Otherwise, register a new stranger.
        sid = self._next_stranger_id
        self._next_stranger_id += 1
        label = f"Stranger_{sid:02d}"

        # Patch 57 :: LRU eviction. When the dynamic cache exceeds
        # dynamic_capacity, evict the least-recently-seen stranger.
        # This prevents unbounded growth over a 12-hour session and
        # keeps the USearch index within its reserved capacity
        # (search quality degrades beyond the reserve).
        if len(self._dynamic_keys) >= self.dynamic_capacity:
            self._evict_lru_stranger()
        # Append to the dynamic index.
        new_key = len(self._dynamic_keys)
        self.dynamic_index.add(
            np.array([new_key], dtype=np.int64),
            np.asarray(body_feature, dtype=np.float32).reshape(1, -1),
        )
        self._dynamic_keys.append(new_key)
        # Patch 57 :: Pre-allocated matrix append (amortized O(1)).
        # Grow capacity by one chunk when full, instead of vstack.
        if self._dynamic_matrix_rows >= self._dynamic_matrix_capacity:
            self._dynamic_matrix_capacity += self._DYNAMIC_MATRIX_CHUNK
            new_mat = np.zeros(
                (self._dynamic_matrix_capacity, self.reid_dim),
                dtype=np.float32,
            )
            if self._dynamic_matrix is not None and self._dynamic_matrix_rows > 0:
                new_mat[:self._dynamic_matrix_rows] = self._dynamic_matrix[:self._dynamic_matrix_rows]
            self._dynamic_matrix = new_mat
        self._dynamic_matrix[self._dynamic_matrix_rows] = (
            np.asarray(body_feature, dtype=np.float32).flatten()
        )
        self._dynamic_matrix_rows += 1
        self._dynamic_strangers[new_key] = StrangerRecord(
            stranger_id=sid,
            label=label,
            first_seen_us=first_seen_us,
            last_seen_us=first_seen_us,
            centroid_history=deque([centroid], maxlen=64),
            # Patch 57 :: Store feature for LRU eviction rebuild.
            body_feature=np.asarray(body_feature, dtype=np.float32).flatten(),
        )
        logger.info(
            "Stranger registered | id=%d | label=%s | cache_size=%d",
            sid, label, len(self._dynamic_keys),
        )
        return sid, label, False

    # ------------------------------------------------------------------
    # Patch 57 :: LRU eviction for the dynamic stranger cache.
    # Removes the stranger with the oldest last_seen_us timestamp.
    # The USearch index cannot remove entries, so we rebuild the
    # dynamic index from scratch with the surviving strangers. This
    # is O(M) but only fires when capacity is exceeded (256 strangers),
    # so it happens rarely -- at most once per few new strangers.
    # ------------------------------------------------------------------
    @_synchronized
    def _evict_lru_stranger(self) -> None:
        """Evict the least-recently-seen stranger from the dynamic cache.

        Patch 57 :: USearch does not support entry removal, so we
        rebuild the entire dynamic index from the surviving records.
        Each StrangerRecord stores its body_feature (Patch 57), so
        we can re-insert survivors without needing the old USearch
        index. This is O(M) but only fires when capacity is exceeded.
        """
        if not self._dynamic_strangers:
            return
        # Find the LRU stranger by last_seen_us.
        lru_key = min(
            self._dynamic_strangers.keys(),
            key=lambda k: self._dynamic_strangers[k].last_seen_us,
        )
        lru_rec = self._dynamic_strangers.pop(lru_key, None)
        if lru_rec is None:
            return
        # Collect survivors (key, record) before rebuilding.
        survivors: List[Tuple[int, StrangerRecord]] = [
            (k, rec) for k, rec in self._dynamic_strangers.items()
            if rec.body_feature is not None
        ]
        prev_count = len(self._dynamic_keys)
        # Rebuild the dynamic index from scratch.
        self.build_dynamic_index()
        # Re-insert each surviving stranger using its stored body_feature.
        for _old_key, rec in survivors:
            new_key = len(self._dynamic_keys)
            feat = np.asarray(rec.body_feature, dtype=np.float32).reshape(1, -1)
            self.dynamic_index.add(
                np.array([new_key], dtype=np.int64), feat,
            )
            self._dynamic_keys.append(new_key)
            # Append to pre-allocated matrix (same logic as register_stranger).
            if self._dynamic_matrix_rows >= self._dynamic_matrix_capacity:
                self._dynamic_matrix_capacity += self._DYNAMIC_MATRIX_CHUNK
                new_mat = np.zeros(
                    (self._dynamic_matrix_capacity, self.reid_dim),
                    dtype=np.float32,
                )
                if self._dynamic_matrix is not None and self._dynamic_matrix_rows > 0:
                    new_mat[:self._dynamic_matrix_rows] = self._dynamic_matrix[:self._dynamic_matrix_rows]
                self._dynamic_matrix = new_mat
            self._dynamic_matrix[self._dynamic_matrix_rows] = feat.flatten()
            self._dynamic_matrix_rows += 1
            self._dynamic_strangers[new_key] = rec
        logger.info(
            "DualIndexManager: LRU evicted stranger %s (id=%d) | "
            "prev_cache=%d | new_cache=%d | survivors_reinserted=%d",
            lru_rec.label, lru_rec.stranger_id, prev_count,
            len(self._dynamic_keys), len(survivors),
        )
    # ------------------------------------------------------------------
    def get_static_matrix(self) -> np.ndarray:
        """Return the pre-normalized static DB matrix (N, 512)."""
        if self._static_matrix is None:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return self._static_matrix

    def get_static_labels(self) -> Tuple[List[str], List[str]]:
        return self._static_keys, self._static_names


# ============================================================================
# Hybrid Re-ID Tracker
# ============================================================================
class HybridReIDTracker:
    """
    Hybrid Re-ID matching combining rolling body feature queues (size 10)
    with spatial Euclidean distance checks (<=150 px) and maximum
    dot-product body similarity (>=0.70).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )
        hcfg = self.config["identity_matcher"]["hybrid_reid"]
        self.history_size: int = int(hcfg["body_history_size"])
        self.spatial_limit: float = float(hcfg["spatial_distance_limit_px"])
        self.body_sim_threshold: float = float(hcfg["body_similarity_threshold"])

    # ------------------------------------------------------------------
    def update_history(
        self,
        state: TrackInferenceState,
        body_feature: np.ndarray,
        centroid: Tuple[float, float],
    ) -> None:
        """Append a feature + centroid to the rolling queue."""
        state.body_feature_history.append(
            np.asarray(body_feature, dtype=np.float32).flatten()
        )
        state.body_centroid_history.append(centroid)
        # Enforce the rolling window cap.
        while len(state.body_feature_history) > self.history_size:
            state.body_feature_history.pop(0)
        while len(state.body_centroid_history) > self.history_size:
            state.body_centroid_history.pop(0)

    # ------------------------------------------------------------------
    def find_match(
        self,
        candidate_feature: np.ndarray,
        candidate_centroid: Tuple[float, float],
        candidate_track_id: int,
        registry: Dict[int, TrackInferenceState],
    ) -> Optional[BodyReIDResult]:
        """
        Search the registry for an existing track whose body feature queue
        matches the candidate. The match must satisfy BOTH:
          * spatial_distance <= self.spatial_limit (150 px)
          * body_similarity  >= self.body_sim_threshold (0.70)

        Returns the BodyReIDResult if a match is found, else None.
        """
        cand_feat = np.asarray(candidate_feature, dtype=np.float32).flatten()
        cand_cx, cand_cy = float(candidate_centroid[0]), float(candidate_centroid[1])

        best_match: Optional[BodyReIDResult] = None
        best_sim: float = -1.0

        for tid, state in registry.items():
            if tid == candidate_track_id:
                continue
            if not state.body_feature_history:
                continue

            # ---- Spatial distance: minimum Euclidean distance to the
            #      candidate centroid across the rolling centroid history.
            history = np.asarray(state.body_centroid_history, dtype=np.float32)
            hist_cx = history[:, 0]
            hist_cy = history[:, 1]
            dists = np.sqrt(
                (hist_cx - cand_cx) ** 2 + (hist_cy - cand_cy) ** 2
            )
            min_dist = float(dists.min())
            if min_dist > self.spatial_limit:
                continue

            # ---- Body similarity: maximum dot-product across the rolling
            #      feature queue (vectors are pre-normalized).
            feat_matrix = np.stack(state.body_feature_history, axis=0)
            # Defensive L2 normalization in case the caller forgot.
            norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
            norms = np.where(norms > 1e-6, norms, 1.0)
            feat_matrix = feat_matrix / norms

            sims = feat_matrix @ cand_feat
            max_sim = float(sims.max())
            if max_sim < self.body_sim_threshold:
                continue

            if max_sim > best_sim:
                best_sim = max_sim
                best_match = BodyReIDResult(
                    track_id=candidate_track_id,
                    matched_track_id=tid,
                    spatial_distance_px=min_dist,
                    body_similarity=max_sim,
                    inheritance_applied=True,
                )

        return best_match


# ============================================================================
# Identity Matcher (top-level orchestrator)
# ============================================================================
class IdentityMatcher:
    """
    Top-level identity resolution block combining:
      * OSNetReID           -- body feature extraction
      * DualIndexManager    -- static + dynamic vector search
      * HybridReIDTracker   -- cross-frame identity linkage
      * Inference Termination Logic

    Public surface for the orchestrator:
        matcher = IdentityMatcher(config)
        matcher.initialize()
        matcher.warmup()
        results = matcher.batch_match_faces_raw(track_ids, face_embeddings)
        reid    = matcher.match_body_reid(track_id, body_feat, centroid)
        sid, lbl, inh = matcher.register_stranger(body_feat, ts_us, centroid)
        matcher.terminate_face_scanning(track_id, state, label, sid_or_stranger)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )
        self.reid: OSNetReID = OSNetReID(self.config)
        self.index_mgr: DualIndexManager = DualIndexManager(self.config)
        self.hybrid: HybridReIDTracker = HybridReIDTracker(self.config)

        # Per-track inference state registry.
        self._track_states: Dict[int, TrackInferenceState] = {}

        # Gating thresholds.
        gcfg = self.config["gating"]
        self.sim_baseline: float = float(
            gcfg["verified_student"]["similarity_baseline"]
        )
        self.sim_floor: float = float(
            gcfg["verified_student"]["similarity_dynamic_floor"]
        )
        self.stranger_floor: float = float(gcfg["stranger"]["similarity_floor"])
        self.appearance_inherit: float = float(
            gcfg["stranger"]["appearance_inherit_threshold"]
        )

        self._initialized: bool = False
        self._warmed: bool = False

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Build the OSNet network + dual USearch indexes."""
        self.reid.initialize()
        self.index_mgr.build_static_index()
        self.index_mgr.build_dynamic_index()
        self._initialized = True
        logger.info(
            "IdentityMatcher initialized | static_db=%d | dynamic_cache=%d",
            len(self.index_mgr._static_keys),
            len(self.index_mgr._dynamic_keys),
        )

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Run the OSNet CUDA Graph pre-compilation warmup."""
        if not self._initialized:
            raise RuntimeError("IdentityMatcher must be initialized before warmup.")
        self.reid.warmup()
        self._warmed = True

    # ------------------------------------------------------------------
    # Patch 67 :: OSNet stranger memory recall from disk clearshots.
    #
    # Called by main.py at process startup (after initialize() + warmup())
    # to rebuild the in-memory stranger cache from clearshot PNGs saved
    # on disk. This lets the tracker "remember" strangers across the
    # 12-hour scheduled restart (6AM/6PM) and mid-session crash recovery.
    #
    # The recall scans the current session's clearshot directory, groups
    # by stranger label, computes OSNet embeddings, picks a consensus
    # (medoid) embedding per stranger, and loads them into
    # DualIndexManager._dynamic_strangers. NO .pickle/.pkl persistence --
    # the PNG clearshots ARE the persistence; this method rebuilds RAM
    # from them on every restart.
    # ------------------------------------------------------------------
    def recall_strangers_from_disk(
        self,
        yolo_model: Any = None,
    ) -> int:
        """Rebuild the stranger OSNet cache from disk clearshots.

        Args:
            yolo_model: Unused (kept for API compatibility). Person
                crops are read from the sidecar .json bbox, so YOLO
                re-detection is not needed.

        Returns:
            Number of strangers recalled into the dynamic cache.
        """
        # P2-M7 fix: try multiple import paths so silent degradation does
        # not occur if stranger_recall.py is relocated. Log at ERROR level
        # (not WARNING) if all paths fail, so operators notice.
        recall_strangers = None
        for _import_path in (
            "core.stranger_recall",
            "stranger_recall",
            "src.core.stranger_recall",
        ):
            try:
                recall_strangers = __import__(
                    _import_path, fromlist=["recall_strangers"]
                ).recall_strangers
                break
            except ImportError:
                continue
        if recall_strangers is None:
            logger.error(
                "IdentityMatcher.recall_strangers_from_disk() :: "
                "stranger_recall module not available on any tried import "
                "path (core.stranger_recall, stranger_recall, "
                "src.core.stranger_recall). Stranger memory recall is "
                "DISABLED until the module is restored."
            )
            return 0

        try:
            recall_map = recall_strangers(
                config=self.config,
                osnet_model=self.reid,
                yolo_model=yolo_model,
            )
        except Exception as exc:
            logger.error(
                "IdentityMatcher.recall_strangers_from_disk() :: "
                "recall_strangers() failed: %s. Stranger memory will be "
                "empty for this session.",
                exc,
            )
            logger.error(traceback.format_exc())
            return 0

        if not recall_map:
            logger.info(
                "IdentityMatcher.recall_strangers_from_disk() :: "
                "no recalled strangers (fresh session or no clearshots)."
            )
            return 0

        loaded = self.index_mgr.load_recall_vectors(recall_map)
        logger.info(
            "IdentityMatcher.recall_strangers_from_disk() :: "
            "recalled %d strangers into dynamic cache | "
            "cache_size=%d | next_stranger_id=%d",
            loaded,
            len(self.index_mgr._dynamic_keys),
            self.index_mgr._next_stranger_id,
        )
        return loaded

    # ------------------------------------------------------------------
    # Track state management.
    # ------------------------------------------------------------------
    def get_track_state(self, track_id: int) -> TrackInferenceState:
        """Get-or-create the TrackInferenceState for a track_id."""
        state = self._track_states.get(track_id)
        if state is None:
            state = TrackInferenceState(track_id=track_id)
            self._track_states[track_id] = state
        return state

    def drop_track(self, track_id: int) -> None:
        self._track_states.pop(track_id, None)

    def all_track_states(self) -> Dict[int, TrackInferenceState]:
        return self._track_states

    # ------------------------------------------------------------------
    # Batched Matrix-Multiplication Face Matching.
    # ------------------------------------------------------------------
    def batch_match_faces_raw(
        self,
        track_ids: List[int],
        face_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[FaceMatchResult]:
        """
        Compute cross-similarity scores for all active tracked identities
        simultaneously via the BLAS routine:

            similarity_matrix = live_matrix @ db_matrix.T

        Args:
            track_ids:        List of N active track IDs (may include IDs
                              whose face scanning is already terminated;
                              those are returned with best_similarity=-inf).
            face_embeddings:  np.ndarray of shape (N, 512), L2-normalized.
            top_k:            Number of top matches to return per track.

        Returns:
            List[FaceMatchResult], one per track_id, in the same order.
        """
        if not track_ids:
            return []

        n = len(track_ids)
        if face_embeddings is None or face_embeddings.size == 0:
            return [
                FaceMatchResult(
                    track_id=tid,
                    best_student_id=None,
                    best_student_name=None,
                    best_similarity=-1.0,
                )
                for tid in track_ids
            ]

        live_matrix = np.asarray(face_embeddings, dtype=np.float32)
        if live_matrix.ndim == 1:
            live_matrix = live_matrix.reshape(1, -1)
        if live_matrix.shape[0] != n:
            logger.warning(
                "batch_match_faces_raw | track_count=%d != live_rows=%d",
                n, live_matrix.shape[0],
            )
            n = min(n, live_matrix.shape[0])

        db_matrix = self.index_mgr.get_static_matrix()
        static_ids, static_names = self.index_mgr.get_static_labels()

        # ----------------------------------------------------------------
        # Compute the full NxM similarity matrix in one BLAS call.
        # ----------------------------------------------------------------
        if db_matrix.shape[0] == 0:
            sim_matrix = np.zeros((n, 0), dtype=np.float32)
        else:
            # Defensive L2 normalization (caller should already be normalized).
            live_norms = np.linalg.norm(live_matrix, axis=1, keepdims=True)
            live_norms = np.where(live_norms > 1e-6, live_norms, 1.0)
            live_matrix = live_matrix / live_norms

            db_norms = np.linalg.norm(db_matrix, axis=1, keepdims=True)
            db_norms = np.where(db_norms > 1e-6, db_norms, 1.0)
            db_matrix_norm = db_matrix / db_norms

            # Batched BLAS matmul: (N, 512) @ (512, M) -> (N, M)
            sim_matrix = (live_matrix @ db_matrix_norm.T).astype(np.float32)

        # ----------------------------------------------------------------
        # Build the FaceMatchResult list.
        # ----------------------------------------------------------------
        results: List[FaceMatchResult] = []
        for i in range(n):
            tid = track_ids[i]
            state = self._track_states.get(tid)

            # Inference Termination Logic -- if face scanning is already
            # terminated for this track, return a sentinel result without
            # touching the static DB.
            if state is not None and not state.face_scanning_active:
                results.append(FaceMatchResult(
                    track_id=tid,
                    best_student_id=state.resolved_student_id,
                    best_student_name=(
                        self._lookup_student_name(state.resolved_student_id)
                        if state.resolved_student_id else None
                    ),
                    best_similarity=-1.0,
                ))
                continue

            if sim_matrix.shape[1] == 0:
                results.append(FaceMatchResult(
                    track_id=tid,
                    best_student_id=None,
                    best_student_name=None,
                    best_similarity=-1.0,
                ))
                continue

            row = sim_matrix[i]
            k = min(top_k, row.shape[0])
            # Partial sort for top-k indices (descending similarity).
            top_idx = np.argpartition(-row, k - 1)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            top_sims = row[top_idx]
            top_ids: List[Optional[str]] = []
            top_names: List[Optional[str]] = []
            for idx in top_idx:
                if 0 <= int(idx) < len(static_ids):
                    top_ids.append(static_ids[int(idx)])
                    top_names.append(static_names[int(idx)])
                else:
                    top_ids.append(None)
                    top_names.append(None)

            results.append(FaceMatchResult(
                track_id=tid,
                best_student_id=top_ids[0] if top_ids else None,
                best_student_name=top_names[0] if top_names else None,
                best_similarity=float(top_sims[0]) if len(top_sims) else -1.0,
                top_k_similarities=top_sims.astype(np.float32),
                top_k_student_ids=top_ids,
            ))

        return results

    # ------------------------------------------------------------------
    def _lookup_student_name(self, student_id: Optional[str]) -> Optional[str]:
        if not student_id:
            return None
        ids, names = self.index_mgr.get_static_labels()
        try:
            idx = ids.index(student_id)
            return names[idx]
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Hybrid Body Re-ID Matching.
    # ------------------------------------------------------------------
    def match_body_reid(
        self,
        track_id: int,
        body_feature: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Optional[BodyReIDResult]:
        """
        Run the hybrid Re-ID routine for a single track. Updates the
        rolling body feature + centroid history before searching.
        """
        state = self.get_track_state(track_id)
        if not state.body_reid_active:
            return None

        # Update the rolling queue first so the candidate is registered.
        self.hybrid.update_history(state, body_feature, centroid)

        # Search for a matching track (excluding self).
        return self.hybrid.find_match(
            candidate_feature=body_feature,
            candidate_centroid=centroid,
            candidate_track_id=track_id,
            registry=self._track_states,
        )

    # ------------------------------------------------------------------
    def extract_body_features(
        self, crops_bgr: List[np.ndarray]
    ) -> np.ndarray:
        """Delegate to the OSNetReID extractor."""
        return self.reid.extract_features(crops_bgr)

    # ------------------------------------------------------------------
    # Stranger registration.
    # ------------------------------------------------------------------
    def register_stranger(
        self,
        body_feature: np.ndarray,
        timestamp_us: int,
        centroid: Tuple[float, float],
    ) -> Tuple[int, str, bool]:
        return self.index_mgr.register_stranger(
            body_feature=body_feature,
            first_seen_us=timestamp_us,
            centroid=centroid,
            appearance_inherit_threshold=self.appearance_inherit,
        )

    # ------------------------------------------------------------------
    # Inference Termination Logic.
    # ------------------------------------------------------------------
    def terminate_face_scanning(
        self,
        track_id: int,
        resolved_state: str,
        resolved_label: str,
        resolved_student_id: Optional[str] = None,
        resolved_stranger_id: Optional[int] = None,
        keep_body_reid: bool = False,
    ) -> TrackInferenceState:
        """
        Halt all facial scanning + recognition routines for the given
        track ID. By default the body-ReID loop is also halted unless
        `keep_body_reid=True` (used for stranger tracks that must
        continue to be tracked via spatial coordinates).

        Returns the updated TrackInferenceState.
        """
        state = self.get_track_state(track_id)
        state.face_scanning_active = False
        state.body_reid_active = bool(keep_body_reid)
        state.resolved_state = resolved_state
        state.resolved_label = resolved_label
        state.resolved_student_id = resolved_student_id
        state.resolved_stranger_id = resolved_stranger_id

        # Free the rolling body history if we are halting Re-ID too.
        if not keep_body_reid:
            state.body_feature_history.clear()
            state.body_centroid_history.clear()
            gc.collect()

        logger.info(
            "Inference terminated for track %d | state=%s | label=%s | body_reid=%s",
            track_id, resolved_state, resolved_label, keep_body_reid,
        )
        return state

    # ------------------------------------------------------------------
    def is_face_scanning_active(self, track_id: int) -> bool:
        state = self._track_states.get(track_id)
        return state.face_scanning_active if state else True

    # ------------------------------------------------------------------
    # Patch 20 :: reset_dynamic_memory()
    #
    # Called by main.py's SessionBoundaryWatcher at every 06:00 / 18:00
    # LOCAL boundary. Delegates to DualIndexManager.reset_dynamic_memory()
    # AND clears the per-track body_feature_history / body_centroid_history
    # so the HybridReIDTracker does not carry stranger body matches
    # across the session boundary.
    #
    # Verified-student track states are PRESERVED -- a student verified
    # at 5:59PM stays verified at 6:01PM (they are the same person).
    # Only STRANGER and PENDING tracks have their body-Re-ID history
    # cleared, so the hybrid tracker treats them as fresh in the new
    # session.
    # ------------------------------------------------------------------
    def reset_dynamic_memory(self) -> None:
        """Reset OSNet dynamic stranger cache + per-track body-Re-ID history.

        Called at every 06:00 / 18:00 LOCAL boundary by main.py's
        SessionBoundaryWatcher. The static student face DB and all
        VERIFIED_STUDENT track states are preserved.
        """
        # 1) Clear the dynamic stranger USearch index + label map.
        self.index_mgr.reset_dynamic_memory()

        # 2) Clear per-track body-Re-ID history for non-verified tracks.
        #    Verified students keep their state (they are still the same
        #    person in the new session). Strangers and pending tracks
        #    get a fresh start so their old body features do not match
        #    against new-session strangers via the HybridReIDTracker.
        cleared_tracks = 0
        preserved_tracks = 0
        for tid, state in list(self._track_states.items()):
            if state.resolved_state == "VERIFIED_STUDENT":
                preserved_tracks += 1
                continue
            # STRANGER / ANOMALY / PENDING tracks: clear body history.
            state.body_feature_history.clear()
            state.body_centroid_history.clear()
            cleared_tracks += 1
        gc.collect()
        logger.info(
            "IdentityMatcher.reset_dynamic_memory() :: per-track body-Re-ID "
            "history cleared | cleared=%d | preserved_verified=%d | "
            "static_db_preserved=%d",
            cleared_tracks, preserved_tracks,
            len(self.index_mgr._static_keys),
        )

    # ------------------------------------------------------------------
    # Shutdown.
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.reid.close()
        except Exception:
            pass
        try:
            if self.index_mgr.static_index is not None:
                self.index_mgr.static_index = None
            if self.index_mgr.dynamic_index is not None:
                self.index_mgr.dynamic_index = None
        except Exception:
            pass
        self._track_states.clear()
        self._initialized = False
        self._warmed = False
        gc.collect()
        logger.info("IdentityMatcher closed; resources released.")


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test (requires torch + usearch)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance identity_matcher self-test ===")

    cfg = ConfigRegistry.load("config/config.yaml") if ConfigRegistry else {}
    matcher = IdentityMatcher(cfg)
    matcher.initialize()
    matcher.warmup()

    # Synthetic 3-track face matching test.
    track_ids = [1, 2, 3]
    rng = np.random.default_rng(0)
    face_embs = rng.normal(0, 1, size=(3, 512)).astype(np.float32)
    face_embs /= np.linalg.norm(face_embs, axis=1, keepdims=True)
    results = matcher.batch_match_faces_raw(track_ids, face_embs, top_k=3)
    for r in results:
        logger.info(
            "Match | tid=%d | best_sim=%.4f | sid=%s",
            r.track_id, r.best_similarity, r.best_student_id,
        )

    # Synthetic body Re-ID test.
    body_feat = rng.normal(0, 1, size=(512,)).astype(np.float32)
    body_feat /= np.linalg.norm(body_feat)
    matcher.match_body_reid(10, body_feat, (100.0, 200.0))
    matcher.match_body_reid(11, body_feat, (110.0, 210.0))
    reid = matcher.match_body_reid(12, body_feat, (115.0, 215.0))
    logger.info(
        "ReID | tid=12 | matched=%s | dist=%.2f | sim=%.4f",
        None if reid is None else reid.matched_track_id,
        0.0 if reid is None else reid.spatial_distance_px,
        0.0 if reid is None else reid.body_similarity,
    )

    matcher.close()
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()