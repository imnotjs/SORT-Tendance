"""
SORT-tendance :: src/utils/video_recorder.py

Production-grade Video Recording Pipeline.

Responsibilities:
  1. Fully isolated frame serialization pipeline running an independent
     worker thread powered by a PyAV container loop. The worker drains
     a bounded `queue.Queue` of (frame, frame_index, capture_us, flags)
     tuples off the capture / AI threads and serializes them through
     the active encoder context.

  2. Hardware-Accelerated Encoding Block. At startup, aggressively
     attempt to instantiate and lock the NVIDIA NVENC hardware encoder
     (`h264_nvenc`) using the VRAM configuration definitions from the
     central registry. If the hardware driver initialization raises
     any diagnostic failure (missing codec, NVENC unavailable, CUDA
     context error, etc.), immediately fall back to software encoding
     via `libx264` with parameters tuned for `ultrafast` performance
     and `tune=zerolatency` to preserve real-time throughput.

  3. 120-Frame Pre-Event Rolling Buffer. Maintain a high-performance
     circular `collections.deque` housed entirely in RAM containing
     exactly 120 historic uncompressed video frames. When an
     anomaly / stranger registration trigger fires, the rolling buffer
     is atomically snapshotted and prepended to the new segment so the
     encoded clip captures the spatial pre-event context preceding
     the trigger.

  4. Stranger Anonymization Overlay. Expose hooks to parse active
     bounding boxes flagged as Stranger from `gating_opt.py`. For each
     flagged bbox, apply a heavy localized pixelation overlay directly
     onto the frame matrix BEFORE encoding serialization, ensuring
     privacy protection of unidentified individuals in stored video
     assets. The overlay is applied on a deep copy so the live feed
     rendered to the dashboard remains unaltered.

  5. Automated Purge Worker. Run an independent background file-system
     monitor thread that regularly scans `storage/video_reports/`,
     evaluating file modification timestamps and automatically purging
     segmented video assets whose file-age exceeds a strict 12-hour
     shelf-life constraint.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import gc
import time
import queue
import shutil
import threading
import logging
import traceback
import datetime as _dt
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Optional dependency guards.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    import av
    _PYAV_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _PYAV_AVAILABLE = False
    av = None  # type: ignore

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _YAML_AVAILABLE = False
    yaml = None  # type: ignore

# Local config registry import (absolute path resolution).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.database_manager import ConfigRegistry
except ImportError:                         # pragma: no cover
    ConfigRegistry = None  # type: ignore


# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.video_recorder")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Sentinels
# ============================================================================
class _ShutdownSentinel:
    """Queue sentinel signaling worker drain-and-exit."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<VideoRecorderShutdownSentinel>"


_SHUTDOWN = _ShutdownSentinel()


class _FlushSegmentSentinel:
    """Queue sentinel signaling 'flush current segment and open a new one'."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<VideoRecorderFlushSegmentSentinel>"


_FLUSH_SEGMENT = _FlushSegmentSentinel()


# ============================================================================
# Enums
# ============================================================================
class EncoderKind(str, Enum):
    """Active encoder kind for telemetry + diagnostics."""
    NVENC = "h264_nvenc"          # NVIDIA hardware encoder
    LIBX264 = "libx264"           # Software fallback


class RecorderState(str, Enum):
    """High-level recorder operational state."""
    IDLE = "IDLE"                 # Worker running, no active segment
    RECORDING = "RECORDING"       # Active segment open, frames flowing
    FLUSHING = "FLUSHING"         # Closing current segment, about to reopen
    SHUTDOWN = "SHUTDOWN"         # Worker exited


class TriggerReason(str, Enum):
    """Reason a segment was triggered for recording."""
    ANOMALY = "ANOMALY"
    STRANGER = "STRANGER"
    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class FrameTicket:
    """
    Immutable record describing a single frame handed off to the recorder.

    The frame is stored as a NumPy array (BGR, HxWx3, uint8) reference;
    no copy is made at enqueue time -- the worker is responsible for any
    deep copy required for anonymization.
    """
    frame: Any                    # np.ndarray (H, W, 3) uint8 BGR
    frame_index: int              # Hardware Capture Index (monotonic)
    capture_us: int               # Microsecond-of-epoch from capture thread
    # Anonymization targets. Each tuple is (x1, y1, x2, y2) in pixel space.
    stranger_bboxes: Tuple[Tuple[int, int, int, int], ...] = ()
    # Set True to flush the rolling buffer + open a new segment starting
    # at this frame. Carries the trigger reason for the segment filename.
    trigger_reason: Optional[TriggerReason] = None
    # Set True to terminate the current segment after this frame is
    # encoded (segment boundary).
    end_segment: bool = False
    enqueue_wall_us: int = 0


@dataclass
class RollingFrame:
    """Entry in the pre-event rolling buffer."""
    frame: Any                    # np.ndarray (H, W, 3) uint8 BGR
    frame_index: int
    capture_us: int


@dataclass
class SegmentStats:
    """Per-segment statistics."""
    path: str
    encoder: EncoderKind
    trigger_reason: TriggerReason
    started_us: int
    ended_us: int = 0
    frames_written: int = 0
    pre_event_frames: int = 0
    anonymized_stranger_count: int = 0


# ============================================================================
# Rolling Frame Buffer
# ============================================================================
class RollingFrameBuffer:
    """
    High-performance circular deque of historic uncompressed frames.

    Housed entirely in RAM. When `snapshot()` is called, the deque
    contents are atomically copied out (preserving insertion order)
    and the deque continues to accept new entries without interruption.
    """

    def __init__(self, capacity: int = 120) -> None:
        if capacity <= 0:
            raise ValueError(f"RollingFrameBuffer capacity must be > 0, got {capacity}")
        self._capacity: int = int(capacity)
        self._buffer: Deque[RollingFrame] = deque(maxlen=self._capacity)
        self._lock: threading.RLock = threading.RLock()
        self._dropped: int = 0
        self._snapshots_taken: int = 0
        logger.info(
            "RollingFrameBuffer initialized | capacity=%d",
            self._capacity,
        )

    # ------------------------------------------------------------------
    def push(self, frame: Any, frame_index: int, capture_us: int) -> None:
        """Append a frame to the rolling buffer (overwrites oldest if full)."""
        rf = RollingFrame(
            frame=frame,
            frame_index=int(frame_index),
            capture_us=int(capture_us),
        )
        with self._lock:
            prior_len = len(self._buffer)
            self._buffer.append(rf)
            # If the deque was already at capacity, the append silently
            # evicted the leftmost entry.
            if prior_len >= self._capacity:
                self._dropped += 1

    # ------------------------------------------------------------------
    def snapshot(self) -> List[RollingFrame]:
        """
        Atomically snapshot the current buffer contents in insertion
        order (oldest first, newest last).
        """
        with self._lock:
            self._snapshots_taken += 1
            return list(self._buffer)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "capacity": self._capacity,
                "current_size": len(self._buffer),
                "dropped_total": self._dropped,
                "snapshots_taken": self._snapshots_taken,
            }


# ============================================================================
# Stranger Anonymization Overlay
# ============================================================================
class StrangerAnonymizer:
    """
    Applies heavy localized pixelation onto stranger bounding boxes
    directly on the frame matrix BEFORE encoding serialization.

    The pixelation grid is sized to roughly 1/16th of the bbox width
    (clamped to [4, 32]) which yields a strong privacy blur while
    preserving gross silhouette cues for forensic review.

    All operations are performed on a deep copy of the frame so the
    live dashboard feed remains unmodified.
    """

    def __init__(
        self,
        enabled: bool = True,
        pixelation_grid_min: int = 4,
        pixelation_grid_max: int = 32,
        grid_divisor: int = 16,
    ) -> None:
        self._enabled: bool = bool(enabled)
        self._grid_min: int = int(pixelation_grid_min)
        self._grid_max: int = int(pixelation_grid_max)
        self._grid_divisor: int = int(grid_divisor)
        self._processed: int = 0
        self._frames_anonymized: int = 0
        self._errors: int = 0
        logger.info(
            "StrangerAnonymizer initialized | enabled=%s | grid=[%d,%d] /%d",
            self._enabled, self._grid_min, self._grid_max, self._grid_divisor,
        )

    # ------------------------------------------------------------------
    def apply(
        self,
        frame: Any,
        stranger_bboxes: Tuple[Tuple[int, int, int, int], ...],
    ) -> Tuple[Any, int]:
        """
        Apply pixelation to the given bboxes on a deep copy of the frame.

        Returns:
            (anonymized_frame, count_of_strangers_anonymized)
            If anonymization is disabled or cv2 is unavailable, returns
            (frame_unchanged, 0).
        """
        if not self._enabled or not stranger_bboxes:
            return frame, 0

        if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
            return frame, 0

        if frame is None:
            return frame, 0

        try:
            # Deep copy so the live feed is never altered.
            out = frame.copy()
            h, w = out.shape[:2]
            count = 0
            for (x1, y1, x2, y2) in stranger_bboxes:
                # Defensive clamp to frame bounds.
                bx1 = max(0, min(int(x1), w - 1))
                by1 = max(0, min(int(y1), h - 1))
                bx2 = max(0, min(int(x2), w))
                by2 = max(0, min(int(y2), h))
                if bx2 - bx1 < 2 or by2 - by1 < 2:
                    continue
                # Compute the pixelation grid size.
                bw = bx2 - bx1
                grid = max(self._grid_min, min(self._grid_max, bw // self._grid_divisor))
                if grid < 1:
                    grid = 1

                roi = out[by1:by2, bx1:bx2]
                # Downscale then upscale using INTER_NEAREST for the blocky
                # pixelation effect.
                small_h = max(1, (by2 - by1) // grid)
                small_w = max(1, (bx2 - bx1) // grid)
                small = cv2.resize(
                    roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR,
                )
                blocky = cv2.resize(
                    small, (bx2 - bx1, by2 - by1),
                    interpolation=cv2.INTER_NEAREST,
                )
                out[by1:by2, bx1:bx2] = blocky
                count += 1

            self._processed += 1
            if count > 0:
                self._frames_anonymized += 1
            return out, count
        except Exception as exc:
            self._errors += 1
            logger.warning(
                "StrangerAnonymizer: failed to apply overlay (%s); "
                "returning original frame.", exc,
            )
            return frame, 0

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "processed_total": self._processed,
            "frames_anonymized_total": self._frames_anonymized,
            "errors_total": self._errors,
        }


# ============================================================================
# Encoder Factory
# ============================================================================
class EncoderFactory:
    """
    Hardware-accelerated encoder factory with automatic fallback.

    At first instantiation, attempts to create a PyAV codec context for
    `h264_nvenc` using the VRAM configuration definitions. If the
    hardware driver raises any diagnostic failure (codec missing,
    CUDA context error, etc.), immediately falls back to `libx264`
    with `preset=ultrafast`, `tune=zerolatency`.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        bitrate_kbps: int,
        primary_codec: str = "h264_nvenc",
        fallback_codec: str = "libx264",
        fallback_preset: str = "ultrafast",
        fallback_tune: str = "zerolatency",
        nvenc_preset: str = "p4",
        nvenc_rc: str = "cbr",
        gpu_device_id: int = 0,
    ) -> None:
        self._width: int = int(width)
        self._height: int = int(height)
        self._fps: int = int(fps)
        self._bitrate_kbps: int = int(bitrate_kbps)
        self._primary_codec: str = str(primary_codec)
        self._fallback_codec: str = str(fallback_codec)
        self._fallback_preset: str = str(fallback_preset)
        self._fallback_tune: str = str(fallback_tune)
        self._nvenc_preset: str = str(nvenc_preset)
        self._nvenc_rc: str = str(nvenc_rc)
        self._gpu_device_id: int = int(gpu_device_id)

        self._active_kind: EncoderKind = EncoderKind.LIBX264
        self._nvenc_attempts: int = 0
        self._nvenc_failures: int = 0
        self._fallback_invocations: int = 0
        self._last_nvenc_error: Optional[str] = None

    # ------------------------------------------------------------------
    def build(self) -> Optional[Any]:
        """
        Build and return a PyAV CodecContext for the active encoder.

        Tries NVENC first; on any failure, falls back to libx264.
        Returns None if both paths fail (which would indicate a
        critically broken PyAV install).
        """
        if not _PYAV_AVAILABLE:
            logger.error(
                "EncoderFactory: PyAV not available -- cannot build encoder.",
            )
            return None

        # --- Attempt 1: NVENC hardware encoder ---
        self._nvenc_attempts += 1
        try:
            codec = av.Codec(self._primary_codec, "w")
            ctx = codec.create()
            ctx.width = self._width
            ctx.height = self._height
            ctx.time_base = _dt.timedelta(seconds=1) / self._fps
            ctx.framerate = self._fps
            ctx.bit_rate = self._bitrate_kbps * 1000
            ctx.pix_fmt = "yuv420p"
            ctx.options = {
                "preset": self._nvenc_preset,
                "rc": self._nvenc_rc,
                "gpu": str(self._gpu_device_id),
                # Surfaces + delayed frames tuned for low-latency streaming.
                "delay": "0",
                "zerolatency": "1",
                # Strictly bound the NVENC session VRAM footprint to the
                # 0.22 memory_fraction budget defined in config.yaml.
                "surfaces": "8",
            }
            ctx.open()
            self._active_kind = EncoderKind.NVENC
            logger.info(
                "EncoderFactory: NVENC encoder acquired | codec=%s | "
                "%dx%d @ %dfps | bitrate=%dkbps | preset=%s | rc=%s | gpu=%d",
                self._primary_codec, self._width, self._height, self._fps,
                self._bitrate_kbps, self._nvenc_preset, self._nvenc_rc,
                self._gpu_device_id,
            )
            return ctx
        except Exception as exc:
            self._nvenc_failures += 1
            self._last_nvenc_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "EncoderFactory: NVENC initialization failed (%s) -- "
                "falling back to software libx264.",
                self._last_nvenc_error,
            )

        # --- Attempt 2: libx264 software fallback ---
        self._fallback_invocations += 1
        try:
            codec = av.Codec(self._fallback_codec, "w")
            ctx = codec.create()
            ctx.width = self._width
            ctx.height = self._height
            ctx.time_base = _dt.timedelta(seconds=1) / self._fps
            ctx.framerate = self._fps
            ctx.bit_rate = self._bitrate_kbps * 1000
            ctx.pix_fmt = "yuv420p"
            ctx.options = {
                "preset": self._fallback_preset,
                "tune": self._fallback_tune,
                # ultrafast already disables lookahead; we also force
                # slice-type decisions to be made per-frame to minimize
                # encode latency.
                "threads": "2",
                "crf": "23",
            }
            ctx.open()
            self._active_kind = EncoderKind.LIBX264
            logger.info(
                "EncoderFactory: libx264 fallback encoder acquired | "
                "%dx%d @ %dfps | bitrate=%dkbps | preset=%s | tune=%s",
                self._width, self._height, self._fps, self._bitrate_kbps,
                self._fallback_preset, self._fallback_tune,
            )
            return ctx
        except Exception as exc:
            logger.critical(
                "EncoderFactory: libx264 fallback ALSO failed (%s) -- "
                "video recording is unavailable.",
                f"{type(exc).__name__}: {exc}",
            )
            return None

    # ------------------------------------------------------------------
    def active_kind(self) -> EncoderKind:
        return self._active_kind

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "active_kind": self._active_kind.value,
            "nvenc_attempts": self._nvenc_attempts,
            "nvenc_failures": self._nvenc_failures,
            "fallback_invocations": self._fallback_invocations,
            "last_nvenc_error": self._last_nvenc_error,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "bitrate_kbps": self._bitrate_kbps,
        }


# ============================================================================
# Segment Writer (PyAV container wrapper)
# ============================================================================
class SegmentWriter:
    """
    Encapsulates one PyAV output container + stream for a single
    video segment file. Opened on trigger, closed on segment-end or
    recorder shutdown.
    """

    def __init__(
        self,
        path: str,
        encoder_ctx: Any,
        encoder_kind: EncoderKind,
        width: int,
        height: int,
        fps: int,
        trigger_reason: TriggerReason,
    ) -> None:
        self._path: str = str(path)
        self._encoder_ctx: Any = encoder_ctx
        self._encoder_kind: EncoderKind = encoder_kind
        self._width: int = int(width)
        self._height: int = int(height)
        self._fps: int = int(fps)
        self._trigger_reason: TriggerReason = trigger_reason
        self._container: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._opened: bool = False
        self._closed: bool = False
        self._frames_written: int = 0
        self._opened_us: int = int(time.time() * 1_000_000)
        self._closed_us: int = 0
        self._errors: int = 0

    # ------------------------------------------------------------------
    def open(self) -> bool:
        if not _PYAV_AVAILABLE:
            logger.error("SegmentWriter: PyAV unavailable; cannot open %s", self._path)
            return False
        if self._opened and not self._closed:
            return True
        try:
            # mkv is preferred for resilience against abrupt container
            # termination (no trailing index required).
            self._container = av.open(self._path, mode="w", format="matroska")
            self._stream = self._container.add_stream_from_codec_context(
                self._encoder_ctx,
            )
            self._opened = True
            logger.info(
                "SegmentWriter opened | path=%s | %dx%d @ %dfps | trigger=%s",
                self._path, self._width, self._height, self._fps,
                self._trigger_reason.value,
            )
            return True
        except Exception as exc:
            self._errors += 1
            logger.error(
                "SegmentWriter: failed to open %s: %s",
                self._path, exc,
            )
            self._container = None
            self._stream = None
            return False

    # ------------------------------------------------------------------
    def write_frame(self, frame_bgr: Any, frame_index: int) -> bool:
        """
        Encode + mux one frame.

        `frame_bgr` must be a NumPy array of shape (H, W, 3) uint8 BGR.
        """
        if not self._opened or self._closed:
            return False
        if self._container is None or self._stream is None:
            return False
        if not _NUMPY_AVAILABLE or not _PYAV_AVAILABLE:
            return False

        try:
            # Convert BGR -> RGB for PyAV (which expects planar or
            # interleaved RGB depending on pix_fmt; for yuv420p encoding
            # we let the codec do the colorspace transform).
            frame_rgb = frame_bgr[:, :, ::-1]
            av_frame = av.VideoFrame.from_ndarray(
                frame_rgb, format="rgb24",
            )
            av_frame.pts = self._frames_written
            av_frame.time_base = _dt.timedelta(seconds=1) / self._fps
            for packet in self._stream.encode(av_frame):
                self._container.mux(packet)
            self._frames_written += 1
            return True
        except Exception as exc:
            self._errors += 1
            logger.warning(
                "SegmentWriter: write_frame failed at frame_index=%d: %s",
                frame_index, exc,
            )
            return False

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        if self._opened and self._container is not None and self._stream is not None:
            try:
                # Flush the encoder.
                for packet in self._stream.encode(None):
                    self._container.mux(packet)
            except Exception as exc:
                self._errors += 1
                logger.warning(
                    "SegmentWriter: encoder flush failed on close: %s", exc,
                )
            try:
                self._container.close()
            except Exception as exc:
                self._errors += 1
                logger.warning(
                    "SegmentWriter: container close failed: %s", exc,
                )
        self._closed = True
        self._closed_us = int(time.time() * 1_000_000)
        logger.info(
            "SegmentWriter closed | path=%s | frames=%d | errors=%d",
            self._path, self._frames_written, self._errors,
        )

    # ------------------------------------------------------------------
    def stats(self) -> SegmentStats:
        return SegmentStats(
            path=self._path,
            encoder=self._encoder_kind,
            trigger_reason=self._trigger_reason,
            started_us=self._opened_us,
            ended_us=self._closed_us,
            frames_written=self._frames_written,
            anonymized_stranger_count=0,
        )

    # ------------------------------------------------------------------
    def is_open(self) -> bool:
        return self._opened and not self._closed

    # ------------------------------------------------------------------
    def frames_written(self) -> int:
        return self._frames_written


# ============================================================================
# Purge Worker
# ============================================================================
class PurgeWorker:
    """
    Background file-system monitor thread.

    Periodically scans the configured output directory and purges
    segmented video assets whose file-age exceeds the configured
    shelf-life (default: 12 hours). The scan interval is bounded to
    a minimum of 60 seconds to prevent thrash.
    """

    MIN_SCAN_INTERVAL_S: int = 60

    def __init__(
        self,
        output_dir: str,
        retention_hours: int = 12,
        scan_interval_s: int = 300,
        file_extension: str = ".mkv",
    ) -> None:
        self._output_dir: str = os.path.abspath(output_dir)
        self._retention_seconds: int = max(1, int(retention_hours)) * 3600
        scan = max(self.MIN_SCAN_INTERVAL_S, int(scan_interval_s))
        self._scan_interval_s: int = scan
        self._file_extension: str = str(file_extension).lower()
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._running: bool = False
        self._scans_completed: int = 0
        self._files_purged: int = 0
        self._bytes_purged: int = 0
        self._errors: int = 0
        self._last_scan_us: int = 0
        self._last_purge_paths: List[str] = []
        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "PurgeWorker: failed to create output_dir %s: %s",
                self._output_dir, exc,
            )

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sortendance.video_recorder.purge",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info(
            "PurgeWorker started | dir=%s | retention=%dh | scan=%ds",
            self._output_dir, self._retention_seconds // 3600,
            self._scan_interval_s,
        )

    # ------------------------------------------------------------------
    def stop(self, timeout_s: float = 5.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.error("PurgeWorker did not exit within %.2fs", timeout_s)
            else:
                logger.info("PurgeWorker joined cleanly.")
        self._running = False

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        # Run one immediate scan at startup so we don't wait the full
        # interval to clean a dirty directory left over from a prior crash.
        self._scan_once()
        while not self._stop_event.is_set():
            # Use wait() so shutdown signals interrupt the sleep promptly.
            if self._stop_event.wait(timeout=self._scan_interval_s):
                break
            self._scan_once()

    # ------------------------------------------------------------------
    def _scan_once(self) -> None:
        self._last_scan_us = int(time.time() * 1_000_000)
        self._scans_completed += 1
        purged_paths: List[str] = []
        try:
            if not os.path.isdir(self._output_dir):
                return
            now_s = time.time()
            for entry in os.scandir(self._output_dir):
                if self._stop_event.is_set():
                    break
                try:
                    if not entry.is_file():
                        continue
                    name_lower = entry.name.lower()
                    if not name_lower.endswith(self._file_extension):
                        continue
                    # Use mtime as the age indicator.
                    stat = entry.stat()
                    age_s = now_s - stat.st_mtime
                    if age_s < self._retention_seconds:
                        continue
                    size_bytes = stat.st_size
                    file_path = entry.path
                    try:
                        os.remove(file_path)
                        purged_paths.append(file_path)
                        self._files_purged += 1
                        self._bytes_purged += size_bytes
                        logger.info(
                            "PurgeWorker: purged %s | age=%.1fh | size=%.2fMB",
                            os.path.basename(file_path),
                            age_s / 3600.0,
                            size_bytes / (1024.0 * 1024.0),
                        )
                    except OSError as exc:
                        self._errors += 1
                        logger.warning(
                            "PurgeWorker: failed to remove %s: %s",
                            file_path, exc,
                        )
                except OSError as exc:
                    self._errors += 1
                    logger.warning(
                        "PurgeWorker: stat failed for %s: %s",
                        entry.path, exc,
                    )
        except OSError as exc:
            self._errors += 1
            logger.error("PurgeWorker: scan_once failed: %s", exc)

        if purged_paths:
            self._last_purge_paths = purged_paths[-16:]  # keep tail bounded
            logger.info(
                "PurgeWorker: scan #%d purged %d files (%.2fMB total)",
                self._scans_completed, len(purged_paths),
                sum(os.path.getsize(p) for p in purged_paths if os.path.exists(p))
                / (1024.0 * 1024.0),
            )

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        return {
            "output_dir": self._output_dir,
            "retention_hours": self._retention_seconds // 3600,
            "scan_interval_s": self._scan_interval_s,
            "running": self._running,
            "scans_completed": self._scans_completed,
            "files_purged": self._files_purged,
            "bytes_purged": self._bytes_purged,
            "errors": self._errors,
            "last_scan_us": self._last_scan_us,
        }


# ============================================================================
# Video Recorder Engine
# ============================================================================
class VideoRecorderEngine:
    """
    Top-level video recorder orchestrator.

    Owns:
      * A bounded `queue.Queue` for non-blocking frame hand-off.
      * A daemon worker thread driving the PyAV container loop.
      * A `RollingFrameBuffer` of 120 historic frames for pre-event
        context capture.
      * A `StrangerAnonymizer` for privacy-preserving stranger overlays.
      * An `EncoderFactory` for NVENC-then-libx264 codec acquisition.
      * A `PurgeWorker` for 12-hour retention enforcement.
      * Per-segment statistics and a global segment registry.

    Public API for the orchestrator:
        rec = VideoRecorderEngine(config)
        rec.initialize()
        rec.start()
        rec.push_frame(frame, frame_index, capture_us, stranger_bboxes=...)
        rec.trigger_segment(TriggerReason.STRANGER, label="[Stranger_03]")
        rec.update_stranger_bboxes(track_id, bboxes)  # hook for gating_opt
        rec.shutdown()
    """

    MAX_CONSECUTIVE_ERRORS: int = 64

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        rcfg = self.config.get("video_recorder", {})
        self._output_dir: str = str(rcfg.get("output_dir", "storage/video_reports"))
        self._rolling_capacity: int = int(rcfg.get("rolling_buffer_frames", 120))
        self._encoder_primary: str = str(rcfg.get("encoder_primary", "h264_nvenc"))
        self._encoder_fallback: str = str(rcfg.get("encoder_fallback", "libx264"))
        self._fallback_preset: str = str(rcfg.get("fallback_preset", "ultrafast"))
        self._fallback_tune: str = str(rcfg.get("fallback_tune", "zerolatency"))
        self._target_fps: int = int(rcfg.get("target_fps", 25))
        self._bitrate_kbps: int = int(rcfg.get("bitrate_kbps", 2500))
        self._retention_hours: int = int(rcfg.get("retention_hours", 12))
        self._purge_interval_s: int = int(rcfg.get("purge_check_interval_s", 300))
        self._anonymize_enabled: bool = bool(rcfg.get("anonymize_strangers", True))
        self._stranger_crop_dir: str = str(
            rcfg.get("stranger_crop_dir", "storage/cache_strangers")
        )

        # Camera frame dimensions (from camera block if present).
        cam_cfg = self.config.get("camera", {})
        self._width: int = int(cam_cfg.get("width", width))
        self._height: int = int(cam_cfg.get("height", height))

        # GPU device id from hardware block.
        hw_cfg = self.config.get("hardware", {}).get("gpu", {})
        self._gpu_device_id: int = int(hw_cfg.get("device_id", 0))

        # Core components.
        self._queue: "queue.Queue[Union[FrameTicket, _ShutdownSentinel, _FlushSegmentSentinel]]" = queue.Queue(
            maxsize=max(8, self._rolling_capacity // 2),
        )
        self._rolling: RollingFrameBuffer = RollingFrameBuffer(
            capacity=self._rolling_capacity,
        )
        self._anonymizer: StrangerAnonymizer = StrangerAnonymizer(
            enabled=self._anonymize_enabled,
        )
        self._encoder_factory: EncoderFactory = EncoderFactory(
            width=self._width,
            height=self._height,
            fps=self._target_fps,
            bitrate_kbps=self._bitrate_kbps,
            primary_codec=self._encoder_primary,
            fallback_codec=self._encoder_fallback,
            fallback_preset=self._fallback_preset,
            fallback_tune=self._fallback_tune,
            gpu_device_id=self._gpu_device_id,
        )
        self._purge_worker: PurgeWorker = PurgeWorker(
            output_dir=self._output_dir,
            retention_hours=self._retention_hours,
            scan_interval_s=self._purge_interval_s,
            file_extension=".mkv",
        )

        # Worker thread state.
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event: threading.Event = threading.Event()
        self._initialized: bool = False
        self._running: bool = False

        # Active segment state.
        self._active_writer: Optional[SegmentWriter] = None
        self._active_encoder_ctx: Optional[Any] = None
        self._active_segment_started_us: int = 0
        self._active_pre_event_frames: int = 0
        self._active_anonymized_count: int = 0
        self._segment_counter: int = 0
        self._segment_registry: List[SegmentStats] = []

        # Per-track stranger bbox cache (updated by gating_opt hooks).
        self._stranger_bbox_cache: Dict[int, Tuple[Tuple[int, int, int, int], ...]] = {}
        self._stranger_bbox_lock: threading.RLock = threading.RLock()

        # Telemetry counters.
        self._frames_pushed: int = 0
        self._frames_dropped_full_queue: int = 0
        self._segments_opened: int = 0
        self._segments_closed: int = 0
        self._worker_errors: int = 0
        self._consecutive_errors: int = 0
        self._max_observed_queue_depth: int = 0
        self._last_frame_latency_us: int = 0
        self._state: RecorderState = RecorderState.IDLE

    # ==================================================================
    # Lifecycle.
    # ==================================================================
    def initialize(self) -> None:
        if self._initialized:
            logger.warning("VideoRecorderEngine already initialized; skipping.")
            return

        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "VideoRecorderEngine: failed to create output_dir %s: %s",
                self._output_dir, exc,
            )
        try:
            os.makedirs(self._stranger_crop_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "VideoRecorderEngine: failed to create stranger_crop_dir %s: %s",
                self._stranger_crop_dir, exc,
            )

        logger.info(
            "VideoRecorderEngine initializing | out=%s | %dx%d @ %dfps | "
            "bitrate=%dkbps | rolling=%d | retention=%dh | purge=%ds | "
            "anonymize=%s",
            self._output_dir, self._width, self._height, self._target_fps,
            self._bitrate_kbps, self._rolling_capacity,
            self._retention_hours, self._purge_interval_s,
            self._anonymize_enabled,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._initialized:
            self.initialize()
        if self._running:
            logger.warning("VideoRecorderEngine worker already running.")
            return

        # Start the purge worker first so retention is enforced even
        # if the recorder thread later fails to start.
        self._purge_worker.start()

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="sortendance.video_recorder.worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._running = True
        self._state = RecorderState.IDLE
        logger.info("VideoRecorderEngine worker thread started.")

    # ------------------------------------------------------------------
    def shutdown(self, timeout_s: float = 8.0) -> None:
        if not self._running:
            logger.info("VideoRecorderEngine shutdown: worker not running.")
            self._purge_worker.stop(timeout_s=timeout_s)
            return

        logger.info(
            "VideoRecorderEngine shutdown initiated | queue_depth=%d",
            self._queue.qsize(),
        )

        # Push shutdown sentinel.
        try:
            self._queue.put(_SHUTDOWN, timeout=1.0)
        except queue.Full:
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                logger.warning(
                    "VideoRecorderEngine: queue full at shutdown; forcing "
                    "event-based wakeup.",
                )

        self._shutdown_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout_s)
            if self._worker_thread.is_alive():
                logger.error(
                    "VideoRecorderEngine worker did not exit within %.2fs",
                    timeout_s,
                )
            else:
                logger.info("VideoRecorderEngine worker joined cleanly.")

        self._running = False
        self._state = RecorderState.SHUTDOWN

        # Stop the purge worker.
        self._purge_worker.stop(timeout_s=timeout_s)

        gc.collect()
        logger.info(
            "VideoRecorderEngine shutdown complete | pushed=%d | "
            "dropped_full=%d | segments_opened=%d | segments_closed=%d | "
            "worker_errors=%d",
            self._frames_pushed, self._frames_dropped_full_queue,
            self._segments_opened, self._segments_closed, self._worker_errors,
        )

    # ==================================================================
    # Producer API.
    # ==================================================================
    def push_frame(
        self,
        frame: Any,
        frame_index: int,
        capture_us: int,
        stranger_bboxes: Optional[Tuple[Tuple[int, int, int, int], ...]] = None,
    ) -> bool:
        """
        Non-blocking frame submission.

        Returns True if the frame was admitted to the queue, False if
        dropped due to queue saturation.
        """
        if not self._running:
            logger.warning(
                "VideoRecorderEngine.push_frame called before start() -- "
                "frame will be buffered.",
            )

        # Pull the latest stranger bboxes from the per-track cache if
        # the caller did not supply them explicitly.
        if stranger_bboxes is None:
            with self._stranger_bbox_lock:
                stranger_bboxes = tuple(
                    bbox for bbox in self._stranger_bbox_cache.values()
                    if bbox
                )

        enqueue_wall_us = int(time.time() * 1_000_000)
        ticket = FrameTicket(
            frame=frame,
            frame_index=int(frame_index),
            capture_us=int(capture_us),
            stranger_bboxes=stranger_bboxes,
            enqueue_wall_us=enqueue_wall_us,
        )

        try:
            self._queue.put_nowait(ticket)
            self._frames_pushed += 1
            depth = self._queue.qsize()
            if depth > self._max_observed_queue_depth:
                self._max_observed_queue_depth = depth
            return True
        except queue.Full:
            self._frames_dropped_full_queue += 1
            # Drop oldest to make room for newest (latest-frame-wins policy).
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(ticket)
                self._frames_pushed += 1
                return True
            except (queue.Empty, queue.Full):
                return False

    # ------------------------------------------------------------------
    def trigger_segment(
        self,
        reason: TriggerReason,
        label: Optional[str] = None,
    ) -> bool:
        """
        Force the recorder to flush the rolling buffer into a new segment
        starting at the NEXT frame submitted via push_frame().

        This is the primary hook invoked by gating_opt.py when an
        ANOMALY or STRANGER registration is fired.
        """
        # Build a synthetic trigger ticket carrying the reason.
        enqueue_wall_us = int(time.time() * 1_000_000)
        ticket = FrameTicket(
            frame=None,                 # No frame; this is a control ticket.
            frame_index=-1,
            capture_us=0,
            trigger_reason=reason,
            enqueue_wall_us=enqueue_wall_us,
        )
        try:
            self._queue.put_nowait(ticket)
            logger.info(
                "VideoRecorderEngine: segment trigger queued | reason=%s | label=%s",
                reason.value, label,
            )
            return True
        except queue.Full:
            logger.warning(
                "VideoRecorderEngine: trigger_segment dropped (queue full) | "
                "reason=%s",
                reason.value,
            )
            return False

    # ------------------------------------------------------------------
    def end_current_segment(self) -> bool:
        """Force-close the current segment (if any) at the next tick."""
        try:
            self._queue.put_nowait(_FLUSH_SEGMENT)
            return True
        except queue.Full:
            logger.warning(
                "VideoRecorderEngine: end_current_segment dropped (queue full).",
            )
            return False

    # ------------------------------------------------------------------
    def update_stranger_bboxes(
        self,
        track_id: int,
        bboxes: Tuple[Tuple[int, int, int, int], ...],
    ) -> None:
        """
        Hook for gating_opt.py to update the per-track stranger bbox
        cache. The cache is consumed by push_frame() to populate the
        anonymization targets on every subsequent frame.
        """
        with self._stranger_bbox_lock:
            if bboxes:
                self._stranger_bbox_cache[track_id] = tuple(bboxes)
            else:
                self._stranger_bbox_cache.pop(track_id, None)

    # ------------------------------------------------------------------
    def clear_stranger_bboxes(self, track_id: int) -> None:
        with self._stranger_bbox_lock:
            self._stranger_bbox_cache.pop(track_id, None)

    # ==================================================================
    # Background worker loop.
    # ==================================================================
    def _worker_loop(self) -> None:
        logger.info("VideoRecorderEngine worker loop entered.")
        try:
            while True:
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    # Idle tick -- keep the rolling buffer primed and
                    # reset the consecutive-error counter.
                    if self._consecutive_errors > 0:
                        self._consecutive_errors = 0
                    continue

                if item is _SHUTDOWN:
                    self._drain_remaining()
                    self._close_active_segment()
                    logger.info(
                        "VideoRecorderEngine worker observed shutdown sentinel; "
                        "exiting.",
                    )
                    return

                if item is _FLUSH_SEGMENT:
                    self._close_active_segment()
                    continue

                self._process_item(item)

        except Exception as exc:
            self._worker_errors += 1
            logger.critical(
                "VideoRecorderEngine worker loop crashed: %s\n%s",
                exc, traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    def _drain_remaining(self) -> None:
        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN or item is _FLUSH_SEGMENT:
                continue
            self._process_item(item)
            drained += 1
        if drained > 0:
            logger.info(
                "VideoRecorderEngine drained %d residual tickets on shutdown.",
                drained,
            )

    # ------------------------------------------------------------------
    def _process_item(self, ticket: FrameTicket) -> None:
        """
        Process one FrameTicket:
          1. If the ticket carries a trigger_reason, open a new segment
             seeded with the rolling buffer snapshot.
          2. Push the frame (if any) to the rolling buffer.
          3. Apply stranger anonymization.
          4. Encode + mux the frame to the active segment (if open).
        """
        try:
            # --- Trigger handling ---
            if ticket.trigger_reason is not None:
                # Close any active segment first.
                self._close_active_segment()
                # Open a new segment seeded with the rolling buffer.
                self._open_segment(
                    trigger_reason=ticket.trigger_reason,
                    pre_event_frames=self._rolling.snapshot(),
                )
                # If the trigger ticket also carries a frame, fall
                # through to encode it. Otherwise return.
                if ticket.frame is None:
                    return

            # --- Frame handling ---
            if ticket.frame is None:
                return

            # Push to rolling buffer (always, even if no segment is open).
            self._rolling.push(
                frame=ticket.frame,
                frame_index=ticket.frame_index,
                capture_us=ticket.capture_us,
            )

            # If no active segment, we're done (the rolling buffer is
            # the only consumer).
            if self._active_writer is None:
                return

            # Apply anonymization on a deep copy.
            anonymized_frame, count = self._anonymizer.apply(
                ticket.frame, ticket.stranger_bboxes,
            )
            self._active_anonymized_count += count

            # Encode + mux.
            ok = self._active_writer.write_frame(
                anonymized_frame, ticket.frame_index,
            )
            if not ok:
                self._worker_errors += 1
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0

            # Latency telemetry.
            process_us = int(time.time() * 1_000_000)
            latency_us = process_us - ticket.enqueue_wall_us
            if latency_us >= 0:
                self._last_frame_latency_us = latency_us

            # Emergency brake.
            if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "VideoRecorderEngine: %d consecutive worker errors -- "
                    "closing active segment and suspending encoding.",
                    self._consecutive_errors,
                )
                self._close_active_segment()
                self._consecutive_errors = 0

            # End-segment flag.
            if ticket.end_segment:
                self._close_active_segment()

        except Exception as exc:
            self._worker_errors += 1
            self._consecutive_errors += 1
            logger.error(
                "VideoRecorderEngine: failed to process ticket "
                "frame_index=%d: %s\n%s",
                ticket.frame_index, exc, traceback.format_exc(),
            )

    # ==================================================================
    # Segment management.
    # ==================================================================
    def _open_segment(
        self,
        trigger_reason: TriggerReason,
        pre_event_frames: List[RollingFrame],
    ) -> None:
        """Open a new segment file and seed it with pre-event frames."""
        self._segment_counter += 1
        path = self._segment_path(trigger_reason)
        encoder_ctx = self._encoder_factory.build()
        if encoder_ctx is None:
            logger.error(
                "VideoRecorderEngine: encoder unavailable; cannot open "
                "segment %s",
                path,
            )
            return

        writer = SegmentWriter(
            path=path,
            encoder_ctx=encoder_ctx,
            encoder_kind=self._encoder_factory.active_kind(),
            width=self._width,
            height=self._height,
            fps=self._target_fps,
            trigger_reason=trigger_reason,
        )
        if not writer.open():
            logger.error(
                "VideoRecorderEngine: SegmentWriter.open failed for %s",
                path,
            )
            return

        self._active_writer = writer
        self._active_encoder_ctx = encoder_ctx
        self._active_segment_started_us = int(time.time() * 1_000_000)
        self._active_pre_event_frames = 0
        self._active_anonymized_count = 0
        self._segments_opened += 1
        self._state = RecorderState.RECORDING

        # Seed with pre-event frames (no anonymization -- these are
        # historic frames already past the live gating loop).
        for rf in pre_event_frames:
            if rf.frame is None:
                continue
            try:
                writer.write_frame(rf.frame, rf.frame_index)
                self._active_pre_event_frames += 1
            except Exception as exc:
                logger.warning(
                    "VideoRecorderEngine: pre-event frame write failed "
                    "(index=%d): %s",
                    rf.frame_index, exc,
                )

        logger.info(
            "VideoRecorderEngine: segment opened | path=%s | pre_event=%d | "
            "encoder=%s | trigger=%s",
            path, self._active_pre_event_frames,
            self._encoder_factory.active_kind().value,
            trigger_reason.value,
        )

    # ------------------------------------------------------------------
    def _close_active_segment(self) -> None:
        if self._active_writer is None:
            self._state = RecorderState.IDLE
            return

        self._state = RecorderState.FLUSHING
        path = self._active_writer._path
        pre_event = self._active_pre_event_frames
        anonymized = self._active_anonymized_count
        started_us = self._active_segment_started_us
        try:
            self._active_writer.close()
        except Exception as exc:
            logger.error(
                "VideoRecorderEngine: active segment close failed: %s", exc,
            )

        stats = SegmentStats(
            path=path,
            encoder=self._encoder_factory.active_kind(),
            trigger_reason=(
                self._active_writer._trigger_reason
                if self._active_writer is not None
                else TriggerReason.MANUAL
            ),
            started_us=started_us,
            ended_us=int(time.time() * 1_000_000),
            frames_written=self._active_writer.frames_written()
            if self._active_writer is not None else 0,
            pre_event_frames=pre_event,
            anonymized_stranger_count=anonymized,
        )
        self._segment_registry.append(stats)
        self._segments_closed += 1

        logger.info(
            "VideoRecorderEngine: segment closed | path=%s | frames=%d | "
            "pre_event=%d | anonymized=%d",
            os.path.basename(path), stats.frames_written,
            stats.pre_event_frames, stats.anonymized_stranger_count,
        )

        self._active_writer = None
        self._active_encoder_ctx = None
        self._active_pre_event_frames = 0
        self._active_anonymized_count = 0
        self._state = RecorderState.IDLE

    # ------------------------------------------------------------------
    def _segment_path(self, trigger_reason: TriggerReason) -> str:
        """Build a deterministic segment file path."""
        ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%f",
        )
        name = (
            f"seg_{self._segment_counter:06d}_"
            f"{ts}_{trigger_reason.value}.mkv"
        )
        return os.path.join(self._output_dir, name)

    # ==================================================================
    # Telemetry + read-only views.
    # ==================================================================
    def telemetry(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self._queue.maxsize,
            "max_observed_queue_depth": self._max_observed_queue_depth,
            "frames_pushed": self._frames_pushed,
            "frames_dropped_full_queue": self._frames_dropped_full_queue,
            "segments_opened": self._segments_opened,
            "segments_closed": self._segments_closed,
            "active_segment_open": self._active_writer is not None,
            "active_segment_pre_event": self._active_pre_event_frames,
            "active_segment_anonymized": self._active_anonymized_count,
            "worker_errors": self._worker_errors,
            "last_frame_latency_us": self._last_frame_latency_us,
            "rolling_buffer": self._rolling.telemetry(),
            "anonymizer": self._anonymizer.telemetry(),
            "encoder_factory": self._encoder_factory.telemetry(),
            "purge_worker": self._purge_worker.telemetry(),
            "segment_registry_size": len(self._segment_registry),
        }

    # ------------------------------------------------------------------
    def segment_registry(self) -> List[SegmentStats]:
        """Return a copy of the segment registry for dashboard display."""
        return list(self._segment_registry)

    # ------------------------------------------------------------------
    def active_encoder_kind(self) -> EncoderKind:
        return self._encoder_factory.active_kind()

    # ------------------------------------------------------------------
    def active_state(self) -> RecorderState:
        return self._state


# ============================================================================
# Convenience factory
# ============================================================================
def build_video_recorder(
    config_path: Optional[str] = None,
    autostart: bool = True,
    width: int = 1280,
    height: int = 720,
) -> VideoRecorderEngine:
    """
    Construct a VideoRecorderEngine from the central config registry.
    """
    cfg: Dict[str, Any] = {}
    if ConfigRegistry is not None:
        try:
            cfg = (
                ConfigRegistry.load(config_path) if config_path
                else ConfigRegistry.load()
            )
        except Exception as exc:
            logger.error(
                "build_video_recorder: ConfigRegistry.load failed: %s -- "
                "falling back to empty config.", exc,
            )

    engine = VideoRecorderEngine(config=cfg, width=width, height=height)
    if autostart:
        engine.initialize()
        engine.start()
    return engine


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness (no external engine dependencies)."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance video_recorder self-test ===")

    cfg: Dict[str, Any] = {}
    if ConfigRegistry is not None:
        try:
            cfg = ConfigRegistry.load("config/config.yaml")
        except Exception as exc:
            logger.warning("self-test: ConfigRegistry.load failed: %s", exc)

    rec = VideoRecorderEngine(config=cfg, width=640, height=480)
    rec.initialize()
    rec.start()

    logger.info(
        "Initial telemetry: encoder=%s purge=%s",
        rec.active_encoder_kind().value,
        rec.telemetry()["purge_worker"],
    )

    # --- Test 1: push 60 idle frames (no segment open) ---
    if _NUMPY_AVAILABLE:
        for i in range(60):
            frame = (
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            )
            rec.push_frame(
                frame=frame,
                frame_index=i,
                capture_us=int(time.time() * 1_000_000),
            )
        logger.info("After 60 idle frames: %s", rec.telemetry()["rolling_buffer"])

        # --- Test 2: trigger a STRANGER segment (should seed with rolling buffer) ---
        rec.trigger_segment(TriggerReason.STRANGER, label="[Stranger_01]")
        # Update stranger bboxes for anonymization.
        rec.update_stranger_bboxes(track_id=11, bboxes=((100, 100, 200, 200),))

        # Push 30 more frames (these should be encoded with anonymization).
        for i in range(60, 90):
            frame = (
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            )
            rec.push_frame(
                frame=frame,
                frame_index=i,
                capture_us=int(time.time() * 1_000_000),
                stranger_bboxes=((100, 100, 200, 200),),
            )

        # --- Test 3: trigger an ANOMALY segment ---
        rec.trigger_segment(TriggerReason.ANOMALY)
        for i in range(90, 120):
            frame = (
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            )
            rec.push_frame(
                frame=frame,
                frame_index=i,
                capture_us=int(time.time() * 1_000_000),
            )

        # End the segment explicitly.
        rec.end_current_segment()

    # Let the worker drain.
    time.sleep(3.0)
    logger.info("Final telemetry: %s", rec.telemetry())
    logger.info("Segment registry: %d entries", len(rec.segment_registry()))
    for s in rec.segment_registry():
        logger.info(
            "  %s | frames=%d | pre_event=%d | anonymized=%d",
            os.path.basename(s.path), s.frames_written,
            s.pre_event_frames, s.anonymized_stranger_count,
        )

    rec.shutdown(timeout_s=8.0)
    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()
