# R3GAN/kernels/_build.py
"""Shared JIT build + cache machinery for all R3GAN kernel extensions.

Arch-agnostic: every extension in every arch package (sm100/, sm120/, ...)
funnels through :func:`ensure_extension`. Design goals, in order:

1. Build exactly once per (sources, flags, toolchain) fingerprint, then load
   the cached binary on every subsequent launch without invoking nvcc *or*
   ninja (a "fast path" import of the .so, a few ms).
2. Be safe under concurrent launches: any number of processes (multiple
   training runs, DDP ranks, array jobs) may race the same first build. One
   process builds; the rest wait on an fcntl lock and then fast-path import.
   A builder that dies mid-compile (OOM kill, SLURM preemption) releases the
   lock automatically -- no stale-lock hangs.
3. Never mutate global state (no os.environ writes) and never write into the
   source tree.

How it works
------------
Each extension is described by a :class:`KernelSpec`. A SHA-256 key is
computed over the *contents* of all source and header files, any generated
source, all compiler/linker flags, the include paths, a CUTLASS fingerprint,
and the toolchain (python/torch/CUDA versions). The build lives in::

    <cache_root>/<name>/<key16>/
        <generated sources, if any>
        <name>.so            built artifact
        BUILD_OK.json        atomic completion marker (written last)
        .lock                fcntl lock file
        build_failed.log     traceback of the last failed build, if any

Directories are immutable once BUILD_OK.json exists: any change to any input
produces a *different* key, so readers never race writers and stale binaries
are structurally impossible (contrast torch's default cache, which is keyed
by name only and relies on ninja mtime checks every launch).

Cache root resolution: ``$R3GAN_KERNEL_CACHE_DIR`` if set, else
``$XDG_CACHE_HOME/r3gan/kernels``, else ``~/.cache/r3gan/kernels``. On a
cluster with a shared home (e.g. GPFS/NFS), one build serves every node.

Environment variables
---------------------
R3GAN_KERNEL_CACHE_DIR      Override the cache root.
R3GAN_KERNEL_REBUILD        "1": force a rebuild into a throwaway per-process
                            dir (never touches published dirs). Any other
                            non-empty value is used as a deterministic salt,
                            shared across ranks.
R3GAN_KERNEL_BUILD_TIMEOUT_S  Max seconds to wait for a concurrent build
                            (default 0 = wait forever, with heartbeat logs).
R3GAN_KERNEL_SKIP_ARCH_CHECK  "1": skip the GPU compute-capability check.
R3GAN_CCBIN                 Host compiler for nvcc -ccbin (default /usr/bin/g++).
CUTLASS_DIR                 CUTLASS checkout (default ~/cutlass-main).
CUTLASS                     Legacy variable; IGNORED with a warning (footgun
                            29: stale checkouts spill) unless CUTLASS_DIR is
                            also set.

Requires a POSIX platform (fcntl). Linux and WSL2 are supported.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("r3gan.kernels")

_MARKER = "BUILD_OK.json"
_LOCKFILE = ".lock"
_FAILLOG = "build_failed.log"
_HEARTBEAT_S = 30.0

# In-process registry: extension name -> (key, module). The pybind module
# name is baked into the .so at compile time, so one process must never load
# two differently-keyed builds of the same extension.
_loaded: Dict[str, Tuple[str, Any]] = {}
_loaded_mu = threading.Lock()


# -----------------------------------------------------------------------------
# Spec
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelSpec:
    """Complete description of one JIT extension build.

    ``sources`` are compiled; ``hash_files`` are additional files whose
    contents invalidate the cache (project-local headers included by the
    sources -- list them explicitly, the fast path does no dependency
    scanning). ``generated`` maps filename -> source text to materialize
    inside the build dir and append to the compile (e.g. a pybind binding
    produced from a parsed config file).
    """
    name: str
    sources: Tuple[str, ...]
    hash_files: Tuple[str, ...] = ()
    generated: Tuple[Tuple[str, str], ...] = ()          # (filename, content)
    cflags: Tuple[str, ...] = ()
    cuda_cflags: Tuple[str, ...] = ()
    ldflags: Tuple[str, ...] = ()
    include_paths: Tuple[str, ...] = ()
    src_dir: str = ""            # for path normalization in the key only
    extra_key: Tuple[str, ...] = ()   # e.g. the CUTLASS fingerprint
    cc_major: Optional[int] = None    # runtime GPU guard (10 = SM100/B200);
                                      # not part of the key -- the arch flags
                                      # in cuda_cflags already are


# -----------------------------------------------------------------------------
# CUTLASS resolution (single home for footgun 29)
# -----------------------------------------------------------------------------

def resolve_cutlass(tag: str) -> str:
    """CUTLASS_DIR if set, else ~/cutlass-main. Warns (once per process, per
    tag) if only the legacy CUTLASS variable is set: stale checkouts silently
    change the headers the extension builds against (footgun 29)."""
    legacy = os.environ.get("CUTLASS")
    if legacy and not os.environ.get("CUTLASS_DIR"):
        log.warning("[%s] legacy CUTLASS=%s is IGNORED "
                    "(footgun 29: stale checkouts spill; set CUTLASS_DIR to override)",
                    tag, legacy)
    return os.environ.get("CUTLASS_DIR") or os.path.expanduser("~/cutlass-main")


def cutlass_include_dirs(root: str) -> List[str]:
    return [os.path.join(root, "include"),
            os.path.join(root, "tools", "util", "include")]


def require_dirs(paths: Sequence[str], hint: str) -> None:
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"missing include director{'ies' if len(missing) > 1 else 'y'}: "
            f"{', '.join(missing)}  ({hint})")


def cutlass_fingerprint(root: str) -> str:
    """Cheap content fingerprint of a CUTLASS checkout: the version header
    plus the path. Editing a checkout in place at an unchanged version is not
    detected -- use R3GAN_KERNEL_REBUILD=1 in that (rare, dev-only) case."""
    h = hashlib.sha256(root.encode())
    for rel in ("include/cutlass/version.h", "CHANGELOG.md"):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read(65536))
    return h.hexdigest()[:16]


def ccbin() -> str:
    return os.environ.get("R3GAN_CCBIN", "/usr/bin/g++")


# -----------------------------------------------------------------------------
# Key / cache-dir computation
# -----------------------------------------------------------------------------

def cache_root() -> str:
    env = os.environ.get("R3GAN_KERNEL_CACHE_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(xdg, "r3gan", "kernels")


def _toolchain_fingerprint() -> List[str]:
    parts = [f"py{sys.version_info.major}.{sys.version_info.minor}",
             sys.platform]
    try:
        import torch  # noqa: PLC0415
        parts += [f"torch={torch.__version__}",
                  f"cuda={getattr(torch.version, 'cuda', None)}"]
    except ImportError:                     # tests / tooling without torch
        parts += ["torch=absent"]
    return parts


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_path(p: str, src_dir: str) -> str:
    """Absolute paths inside the source tree hash as tree-relative tokens so
    identical checkouts at different locations share one cache entry (their
    file *contents* are hashed separately)."""
    p = os.path.abspath(p)
    if src_dir:
        src = os.path.abspath(src_dir)
        if p == src or p.startswith(src + os.sep):
            return "<src>/" + os.path.relpath(p, src)
    return p


_rebuild_salt: Optional[str] = None


def _rebuild_requested() -> Optional[str]:
    global _rebuild_salt
    v = os.environ.get("R3GAN_KERNEL_REBUILD", "")
    if not v:
        return None
    if v != "1":
        return v                                # deterministic user salt
    if _rebuild_salt is None:
        _rebuild_salt = uuid.uuid4().hex        # per-process throwaway
    return _rebuild_salt


def compute_key(spec: KernelSpec) -> str:
    h = hashlib.sha256()

    def put(*items: str) -> None:
        for it in items:
            h.update(it.encode())
            h.update(b"\0")

    put("r3gan-kernel-cache-v1", spec.name)
    put(*_toolchain_fingerprint())
    for p in sorted(spec.sources) + sorted(spec.hash_files):
        put("file", _norm_path(p, spec.src_dir), _file_sha(p))
    for fname, content in sorted(spec.generated):
        put("gen", fname, hashlib.sha256(content.encode()).hexdigest())
    put("cflags", *spec.cflags)
    put("cuda_cflags", *spec.cuda_cflags)
    put("ldflags", *spec.ldflags)
    put("includes", *(_norm_path(p, spec.src_dir) for p in spec.include_paths))
    put("extra", *spec.extra_key)
    salt = _rebuild_requested()
    if salt:
        put("rebuild-salt", salt)
    return h.hexdigest()[:16]


def build_dir_for(spec: KernelSpec, cache_root_override: Optional[str] = None) -> str:
    root = os.path.abspath(os.path.expanduser(cache_root_override)) \
        if cache_root_override else cache_root()
    return os.path.join(root, spec.name, compute_key(spec))


# -----------------------------------------------------------------------------
# Locking
# -----------------------------------------------------------------------------

class _BuildLock:
    """Blocking-with-heartbeat exclusive flock on <dir>/.lock.

    flock is released by the kernel when the holding process dies, so a
    builder killed mid-compile (OOM, scancel) never wedges other runs --
    unlike create-exclusive lock *files*, which require the dead process to
    have cleaned up after itself.
    """

    def __init__(self, directory: str, name: str):
        self._path = os.path.join(directory, _LOCKFILE)
        self._name = name
        self._fd: Optional[int] = None

    def __enter__(self) -> "_BuildLock":
        timeout = float(os.environ.get("R3GAN_KERNEL_BUILD_TIMEOUT_S", "0") or 0)
        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        t0 = time.monotonic()
        next_beat = t0 + _HEARTBEAT_S
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (BlockingIOError, InterruptedError):
                now = time.monotonic()
                if timeout and now - t0 > timeout:
                    os.close(self._fd)
                    self._fd = None
                    raise TimeoutError(
                        f"timed out after {timeout:.0f}s waiting for a concurrent "
                        f"build of '{self._name}' (lock: {self._path}). If no other "
                        f"process is building, remove the lock file and retry.")
                if now >= next_beat:
                    log.warning("waiting for a concurrent build of '%s' "
                                "(%.0fs elapsed; lock: %s)",
                                self._name, now - t0, self._path)
                    next_beat = now + _HEARTBEAT_S
                time.sleep(0.5)

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


# -----------------------------------------------------------------------------
# Marker + import
# -----------------------------------------------------------------------------

def _so_path(directory: str, name: str) -> str:
    return os.path.join(directory, f"{name}.so")


def _marker_ok(directory: str, name: str) -> bool:
    m = os.path.join(directory, _MARKER)
    if not (os.path.isfile(m) and os.path.isfile(_so_path(directory, name))):
        return False
    try:
        with open(m, "r") as f:
            return json.load(f).get("schema") == 1
    except (OSError, ValueError):
        return False


def _write_marker(directory: str, spec: KernelSpec, key: str) -> None:
    payload = {
        "schema": 1,
        "name": spec.name,
        "key": key,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "toolchain": _toolchain_fingerprint(),
        "sources": {_norm_path(p, spec.src_dir): _file_sha(p)
                    for p in list(spec.sources) + list(spec.hash_files)},
        "cflags": list(spec.cflags),
        "cuda_cflags": list(spec.cuda_cflags),
        "ldflags": list(spec.ldflags),
        "include_paths": list(spec.include_paths),
        "extra_key": list(spec.extra_key),
    }
    tmp = os.path.join(directory, f".{_MARKER}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(directory, _MARKER))


def _import_built(directory: str, name: str) -> Any:
    # libtorch symbols must already be in the process image before the
    # extension .so is dlopened; importing torch first guarantees it.
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:
        pass
    so = _so_path(directory, name)
    spec = importlib.util.spec_from_file_location(name, so)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Default (torch) builder
# -----------------------------------------------------------------------------

def _check_gpu_arch(spec: KernelSpec) -> None:
    if spec.cc_major is None or os.environ.get("R3GAN_KERNEL_SKIP_ARCH_CHECK") == "1":
        return
    import torch  # noqa: PLC0415
    if not torch.cuda.is_available():
        log.info("no CUDA device visible; building anyway (prebuild mode)")
        return
    major, minor = torch.cuda.get_device_capability()
    if major != spec.cc_major:
        raise RuntimeError(
            f"{spec.name} requires compute capability {spec.cc_major}.x; this "
            f"device reports {major}.{minor}. Set R3GAN_KERNEL_SKIP_ARCH_CHECK=1 "
            f"to build anyway (the kernels will not run on this GPU).")


def _torch_builder(spec: KernelSpec, directory: str, verbose: bool) -> Any:
    """Compile with torch's cpp_extension into `directory` and return the
    imported module. Runs only under the build lock."""
    _check_gpu_arch(spec)
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    sources = list(spec.sources) + [os.path.join(directory, fname)
                                    for fname, _ in spec.generated]
    return load(
        name=spec.name,
        sources=sources,
        extra_cflags=list(spec.cflags) or None,
        extra_cuda_cflags=list(spec.cuda_cflags) or None,
        extra_ldflags=list(spec.ldflags) or None,
        extra_include_paths=list(spec.include_paths) or None,
        build_directory=directory,
        verbose=verbose,
        with_cuda=True,
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def ensure_extension(spec: KernelSpec,
                     verbose: bool = False,
                     cache_root_override: Optional[str] = None,
                     builder: Callable[[KernelSpec, str, bool], Any] = _torch_builder,
                     ) -> Any:
    """Return the built extension module for `spec`, compiling at most once
    per content key across all processes sharing the cache root.

    Fast path (marker present): direct import of the cached .so; no ninja, no
    source stat-ing, no lock. Slow path: take the build lock, re-check the
    marker (another process may have finished while we waited), otherwise
    materialize generated sources, build, publish the marker atomically.
    """
    for p in list(spec.sources) + list(spec.hash_files):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[{spec.name}] missing source file: {p}")

    key = compute_key(spec)
    with _loaded_mu:
        prior = _loaded.get(spec.name)
        if prior is not None:
            prior_key, module = prior
            if prior_key != key:
                raise RuntimeError(
                    f"extension '{spec.name}' was already loaded in this process "
                    f"with key {prior_key}, but is now requested with key {key} "
                    f"(conflicting flags/CUTLASS/env within one process).")
            return module

    directory = build_dir_for(spec, cache_root_override)
    os.makedirs(directory, exist_ok=True)

    if _marker_ok(directory, spec.name):
        module = _import_built(directory, spec.name)
        log.debug("[%s] cache hit: %s", spec.name, directory)
    else:
        with _BuildLock(directory, spec.name):
            if _marker_ok(directory, spec.name):        # built while we waited
                module = _import_built(directory, spec.name)
                log.debug("[%s] cache hit after wait: %s", spec.name, directory)
            else:
                for fname, content in spec.generated:
                    path = os.path.join(directory, fname)
                    if not (os.path.isfile(path)
                            and open(path, "r").read() == content):
                        with open(path, "w") as f:
                            f.write(content)
                fail = os.path.join(directory, _FAILLOG)
                if os.path.isfile(fail):
                    os.unlink(fail)
                log.info("[%s] building into %s (first build of this "
                         "configuration; may take several minutes)",
                         spec.name, directory)
                t0 = time.monotonic()
                try:
                    module = builder(spec, directory, verbose)
                except BaseException:
                    with open(fail, "w") as f:
                        f.write(traceback.format_exc())
                    log.error("[%s] build FAILED; traceback saved to %s",
                              spec.name, fail)
                    raise
                _write_marker(directory, spec, key)
                log.info("[%s] build finished in %.1fs", spec.name,
                         time.monotonic() - t0)

    with _loaded_mu:
        _loaded.setdefault(spec.name, (key, module))
    return module


# -----------------------------------------------------------------------------
# Cache management
# -----------------------------------------------------------------------------

def _locked_elsewhere(directory: str) -> bool:
    """True if some process currently holds the build lock for `directory`."""
    path = os.path.join(directory, _LOCKFILE)
    if not os.path.exists(path):
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except (BlockingIOError, InterruptedError):
        return True
    finally:
        os.close(fd)


def clear_kernel_cache(name: Optional[str] = None,
                       cache_root_override: Optional[str] = None) -> None:
    """Delete the cache for one extension (or the whole root). In-flight
    builds are skipped, not interrupted."""
    root = cache_root_override or cache_root()
    base = os.path.join(root, name) if name else root
    if not os.path.isdir(base):
        return
    for entry in os.listdir(base) if name else \
            [os.path.join(n, k) for n in os.listdir(base)
             for k in os.listdir(os.path.join(base, n))
             if os.path.isdir(os.path.join(base, n))]:
        d = os.path.join(base, entry)
        if os.path.isdir(d) and not _locked_elsewhere(d):
            shutil.rmtree(d, ignore_errors=True)


def prune_kernel_cache(max_age_days: float = 30.0, keep_latest: int = 1,
                       cache_root_override: Optional[str] = None,
                       dry_run: bool = False) -> List[str]:
    """Remove cached builds older than `max_age_days`, always keeping the
    `keep_latest` most recent per extension. Returns the pruned paths."""
    root = cache_root_override or cache_root()
    pruned: List[str] = []
    if not os.path.isdir(root):
        return pruned
    cutoff = time.time() - max_age_days * 86400.0
    for name in sorted(os.listdir(root)):
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        variants = []
        for k in os.listdir(base):
            d = os.path.join(base, k)
            m = os.path.join(d, _MARKER)
            if os.path.isdir(d):
                variants.append((os.path.getmtime(m) if os.path.isfile(m)
                                 else os.path.getmtime(d), d))
        variants.sort(reverse=True)
        for mtime, d in variants[keep_latest:]:
            if mtime < cutoff and not _locked_elsewhere(d):
                pruned.append(d)
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
    return pruned
