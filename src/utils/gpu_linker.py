"""
SORT-tendance :: src/utils/gpu_linker.py

Absolute Windows DLL Registration Module.

This module MUST be imported at the absolute top of `main.py` BEFORE any
machine learning framework (onnxruntime, torch, torchvision, ultralytics,
insightface, numpy with MKL backend, etc.) is imported. The import order
invariant is critical because:

  * ONNX Runtime's `capi` layer dlopen()'s CUDA / cuBLAS / cuDNN DLLs
    lazily during the first CUDAExecutionProvider session creation.
  * On Windows, the default DLL search path does NOT include the
    Python site-packages `nvidia/*` wheels' `bin/` directories, and
    does NOT include the CUDA Toolkit install path unless it has been
    added to the global `PATH` (which is fragile and not recommended
    for process-isolated deployments).
  * Without explicit registration, the dlopen() raises Win32 Error 126
    ("The specified module could not be found") at the FIRST attempt to
    instantiate a CUDA session, which is far too late to recover from
    cleanly.

The fix: call `os.add_dll_directory(path)` for each candidate binary
directory enumerated in `config.main.dll_directories` BEFORE the ML
frameworks load. `os.add_dll_directory` was added in Python 3.8 and
registers an additional directory in the process-wide DLL search set
used by `LoadLibraryEx(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`.

This module is a NO-OP on non-Windows platforms (the Linux/macOS dynamic
loader uses LD_LIBRARY_PATH and the framework wheels' RPATHs), but the
probe diagnostics are still logged for symmetry.

Author: SORT-tendance Engineering
"""

from __future__ import annotations

import os
import sys
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Logging Configuration
# ============================================================================
logger = logging.getLogger("sortendance.gpu_linker")
if not logger.handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# Constants
# ============================================================================
# File extensions considered as DLL assets on Windows. We accept both the
# canonical `.dll` and the `.pyd` extension used by Python C-extensions
# (which are themselves PE DLLs).
_WIN_DLL_EXTS: Tuple[str, ...] = (".dll", ".pyd")

# Canonical subdirectory names that the NVIDIA wheel ecosystem uses to
# store runtime binaries. We probe these under each candidate root.
_NVIDIA_BIN_SUBDIRS: Tuple[str, ...] = (
    "bin",
    "lib/x64",
    "Lib/x64",
)

# Canonical environment variable names that, if already set by the user,
# are honored instead of (or in addition to) the config-driven paths.
_ENV_PATH_NAMES: Tuple[str, ...] = ("PATH",)


# ============================================================================
# Result Dataclasses
# ============================================================================
class _ProbeResult:
    """Result of probing one candidate directory."""

    __slots__ = ("path", "exists", "is_dir", "added", "dll_count", "error")

    def __init__(
        self,
        path: str,
        exists: bool,
        is_dir: bool,
        added: bool,
        dll_count: int,
        error: Optional[str] = None,
    ) -> None:
        self.path = path
        self.exists = exists
        self.is_dir = is_dir
        self.added = added
        self.dll_count = dll_count
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_dir": self.is_dir,
            "added": self.added,
            "dll_count": self.dll_count,
            "error": self.error,
        }


# ============================================================================
# GPU Linker
# ============================================================================
class GPULinker:
    """
    Process-wide Windows DLL registration manager.

    Responsibilities:
      1. Read the `main.dll_directories` list from the central config
         registry (or accept an explicit list).
      2. Probe each candidate path for existence and DLL contents.
      3. Register valid directories via `os.add_dll_directory(...)` on
         Windows, or log a NO-OP notice on non-Windows platforms.
      4. Append valid directories to the process `PATH` environment
         variable as a defensive fallback (some legacy loaders still
         consult PATH before the add_dll_directory registry).
      5. Return a comprehensive probe report for diagnostic logging.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dll_directories: Optional[List[str]] = None,
    ) -> None:
        # Lazy import to avoid a hard circular dependency at module-load.
        try:
            sys.path.append(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from utils.database_manager import ConfigRegistry
        except ImportError:                     # pragma: no cover
            ConfigRegistry = None  # type: ignore

        self.config: Dict[str, Any] = (
            config or (ConfigRegistry.load() if ConfigRegistry else {})
        )

        # Explicit override takes precedence over config.
        if dll_directories is not None:
            self._dll_directories: List[str] = list(dll_directories)
        else:
            main_cfg = self.config.get("main", {})
            self._dll_directories = list(main_cfg.get("dll_directories", []))

        # Probe results (populated by link()).
        self._probe_results: List[_ProbeResult] = []
        self._linked_paths: List[str] = []
        self._linked: bool = False
        self._platform: str = sys.platform

    # ------------------------------------------------------------------
    def link(self) -> bool:
        """
        Probe and register all configured DLL directories.

        Returns True if at least one directory was successfully registered
        (or if the platform is non-Windows and the call is a NO-OP).
        Returns False if no directory could be registered on Windows.
        """
        if self._linked:
            logger.warning("GPULinker.link() already invoked; skipping re-link.")
            return True

        logger.info(
            "GPULinker: probing %d candidate DLL directories on platform=%s",
            len(self._dll_directories), self._platform,
        )

        if self._platform != "win32":
            # Non-Windows: log a NO-OP notice but do not fail. Linux/macOS
            # use RPATHs and LD_LIBRARY_PATH which are baked into the wheel
            # installs and do not require process-level registration.
            logger.info(
                "GPULinker: non-Windows platform (%s) -- DLL registration is a "
                "NO-OP; relying on RPATH/LD_LIBRARY_PATH.",
                self._platform,
            )
            for path in self._dll_directories:
                exists = os.path.exists(path)
                self._probe_results.append(_ProbeResult(
                    path=path,
                    exists=exists,
                    is_dir=exists and os.path.isdir(path),
                    added=False,
                    dll_count=0,
                    error="non_windows_noop" if exists else "missing",
                ))
            self._linked = True
            return True

        # Windows path.
        return self._link_windows()

    # ------------------------------------------------------------------
    def _link_windows(self) -> bool:
        """
        Windows-specific DLL registration via `os.add_dll_directory`.
        """
        added_any = False

        # Defensive: os.add_dll_directory was added in Python 3.8. If we
        # are running on an older interpreter (which should never happen
        # given the Python 3.10+ baseline), fall back to PATH-only.
        has_add_dll_directory = hasattr(os, "add_dll_directory")
        if not has_add_dll_directory:
            logger.warning(
                "GPULinker: os.add_dll_directory unavailable (Python < 3.8?) -- "
                "falling back to PATH-only registration.",
            )

        for raw_path in self._dll_directories:
            # Expand user / env vars / relative paths to absolute.
            expanded = os.path.expandvars(os.path.expanduser(raw_path))
            abs_path = os.path.abspath(expanded)

            if not os.path.exists(abs_path):
                self._probe_results.append(_ProbeResult(
                    path=abs_path, exists=False, is_dir=False,
                    added=False, dll_count=0, error="missing",
                ))
                logger.warning(
                    "GPULinker: candidate path MISSING -> %s",
                    abs_path,
                )
                continue

            if not os.path.isdir(abs_path):
                self._probe_results.append(_ProbeResult(
                    path=abs_path, exists=True, is_dir=False,
                    added=False, dll_count=0, error="not_a_directory",
                ))
                logger.warning(
                    "GPULinker: candidate path is not a directory -> %s",
                    abs_path,
                )
                continue

            # Count the DLL assets in this directory.
            dll_count = self._count_dlls(abs_path)

            # Register via os.add_dll_directory.
            added = False
            error: Optional[str] = None
            if has_add_dll_directory:
                try:
                    handle = os.add_dll_directory(abs_path)
                    added = True
                    # The returned handle is an opaque structure; we
                    # intentionally do NOT close it -- closing would
                    # remove the directory from the search set.
                    self._linked_paths.append(abs_path)
                    logger.info(
                        "GPULinker: REGISTERED -> %s | dlls=%d",
                        abs_path, dll_count,
                    )
                except OSError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "GPULinker: os.add_dll_directory FAILED -> %s | %s",
                        abs_path, error,
                    )
                except Exception as exc:                     # pragma: no cover
                    error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "GPULinker: unexpected error registering %s: %s\n%s",
                        abs_path, exc, traceback.format_exc(),
                    )

            # Append to PATH as a defensive fallback (some legacy loaders
            # and a few CUDA helper DLLs still consult PATH first).
            self._append_to_path(abs_path)

            self._probe_results.append(_ProbeResult(
                path=abs_path, exists=True, is_dir=True,
                added=added, dll_count=dll_count, error=error,
            ))
            if added:
                added_any = True

        # Probe NVIDIA wheel subdirectories under site-packages as a
        # belt-and-suspenders fallback (catches the case where the user
        # forgot to add them to config.yaml).
        nvidia_extra = self._probe_site_packages_nvidia_wheels()
        if nvidia_extra:
            logger.info(
                "GPULinker: %d additional NVIDIA wheel bin/ directories "
                "discovered and registered.",
                nvidia_extra,
            )
            added_any = True

        self._linked = True

        if not added_any:
            logger.critical(
                "GPULinker: NO DLL directories were registered. ONNX Runtime "
                "CUDA session creation will likely fail with Win32 Error 126.",
            )
        return added_any

    # ------------------------------------------------------------------
    def _count_dlls(self, dir_path: str) -> int:
        """Count the number of DLL/PYD files in a directory (non-recursive)."""
        try:
            count = 0
            for entry in os.scandir(dir_path):
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in _WIN_DLL_EXTS:
                    count += 1
            return count
        except OSError:
            return 0

    # ------------------------------------------------------------------
    def _append_to_path(self, dir_path: str) -> None:
        """Append a directory to the process PATH env var (defensive)."""
        try:
            current_path = os.environ.get("PATH", "")
            # Avoid duplicate entries.
            parts = [p for p in current_path.split(os.pathsep) if p]
            if dir_path in parts:
                return
            parts.append(dir_path)
            os.environ["PATH"] = os.pathsep.join(parts)
        except Exception as exc:                # pragma: no cover
            logger.warning(
                "GPULinker: failed to append %s to PATH: %s",
                dir_path, exc,
            )

    # ------------------------------------------------------------------
    def _probe_site_packages_nvidia_wheels(self) -> int:
        """
        Scan the Python site-packages directory for `nvidia/*` wheel
        installs and register each `bin/` subdirectory found.

        This is a defensive fallback that catches the case where the
        user installed `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.
        via pip but forgot to enumerate them in config.yaml.
        """
        try:
            import site
        except ImportError:                     # pragma: no cover
            return 0

        site_dirs: List[str] = []
        try:
            site_dirs.extend(site.getsitepackages())
        except Exception:                       # pragma: no cover
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                site_dirs.append(user_site)
        except Exception:                       # pragma: no cover
            pass

        added = 0
        for sd in site_dirs:
            nvidia_root = os.path.join(sd, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            try:
                for entry in os.scandir(nvidia_root):
                    if not entry.is_dir():
                        continue
                    # Each wheel installs a subdirectory like
                    # nvidia/cublas, nvidia/cudnn, nvidia/cuda_runtime, etc.
                    for sub in _NVIDIA_BIN_SUBDIRS:
                        bin_path = os.path.join(entry.path, sub)
                        if not os.path.isdir(bin_path):
                            continue
                        # Skip if already registered.
                        if bin_path in self._linked_paths:
                            continue
                        dll_count = self._count_dlls(bin_path)
                        if dll_count == 0:
                            continue
                        try:
                            os.add_dll_directory(bin_path)
                            self._linked_paths.append(bin_path)
                            self._probe_results.append(_ProbeResult(
                                path=bin_path, exists=True, is_dir=True,
                                added=True, dll_count=dll_count, error=None,
                            ))
                            self._append_to_path(bin_path)
                            logger.info(
                                "GPULinker: AUTO-REGISTERED NVIDIA wheel bin -> "
                                "%s | dlls=%d",
                                bin_path, dll_count,
                            )
                            added += 1
                        except OSError as exc:
                            logger.warning(
                                "GPULinker: auto-registration failed for %s: %s",
                                bin_path, exc,
                            )
            except OSError as exc:
                logger.warning(
                    "GPULinker: site-packages NVIDIA scan failed at %s: %s",
                    nvidia_root, exc,
                )
        return added

    # ------------------------------------------------------------------
    def is_linked(self) -> bool:
        return self._linked

    # ------------------------------------------------------------------
    def linked_paths(self) -> List[str]:
        return list(self._linked_paths)

    # ------------------------------------------------------------------
    def probe_report(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._probe_results]

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        total_added = sum(1 for r in self._probe_results if r.added)
        total_dlls = sum(r.dll_count for r in self._probe_results if r.added)
        return {
            "platform": self._platform,
            "linked": self._linked,
            "candidate_count": len(self._dll_directories),
            "registered_count": total_added,
            "total_dlls_registered": total_dlls,
            "linked_paths": list(self._linked_paths),
            "probe_results": self.probe_report(),
        }


# ============================================================================
# Module-level convenience function
# ============================================================================
def link_dlls(
    config: Optional[Dict[str, Any]] = None,
    dll_directories: Optional[List[str]] = None,
) -> GPULinker:
    """
    Construct a GPULinker, invoke link(), and return the instance.

    This is the canonical entry point that `main.py` should call at the
    absolute top of its bootstrap sequence (before importing any ML
    framework).
    """
    linker = GPULinker(config=config, dll_directories=dll_directories)
    try:
        linker.link()
    except Exception as exc:                    # pragma: no cover
        logger.critical(
            "GPULinker.link() raised an unexpected exception: %s\n%s",
            exc, traceback.format_exc(),
        )
    return linker


# ============================================================================
# Module Entry Point
# ============================================================================
def _self_test() -> None:
    """Lightweight self-test harness."""
    logging.basicConfig(level=logging.INFO)
    logger.info("=== SORT-tendance gpu_linker self-test ===")

    linker = GPULinker()
    linker.link()

    report = linker.telemetry()
    logger.info("Telemetry:")
    for k, v in report.items():
        if k == "probe_results":
            logger.info("  probe_results (%d entries):", len(v))
            for r in v:
                logger.info(
                    "    %s | exists=%s added=%s dlls=%d err=%s",
                    r["path"], r["exists"], r["added"],
                    r["dll_count"], r["error"],
                )
        else:
            logger.info("  %s = %s", k, v)

    logger.info("=== self-test complete ===")


if __name__ == "__main__":
    _self_test()
