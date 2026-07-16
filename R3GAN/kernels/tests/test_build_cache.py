# R3GAN/kernels/tests/test_build_cache.py
"""Build-cache behavior tests (no GPU, no torch, no nvcc required).

The production compile step is injected, so these tests exercise the parts
that were previously broken in production -- caching across launches and
correctness under concurrent launches -- with a real gcc-compiled CPython
extension standing in for the CUDA build.

Run:  pytest R3GAN/kernels/tests/ -v
"""
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_PY = os.path.join(_HERE, "..", "_build.py")
_WORKER = os.path.join(_HERE, "_worker.py")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None,
                                reason="gcc required for the stub extension")


def _load_build():
    spec = importlib.util.spec_from_file_location("r3gan_build_under_test", _BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def _prep(tmp_path, name):
    cache_root = str(tmp_path / "cache")
    builds = str(tmp_path / "builds")
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(builds, exist_ok=True)
    with open(os.path.join(cache_root, f"{name}_input.c"), "w") as f:
        f.write("/* hashed input v1 */\n")
    return cache_root, builds


def _run_worker(*args, timeout=90):
    return subprocess.run([sys.executable, _WORKER, *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


def _n_builds(builds_dir):
    return len([f for f in os.listdir(builds_dir) if f.endswith(".built")])


# -----------------------------------------------------------------------------

def test_build_then_fast_path_across_processes(tmp_path):
    """Launch 1 compiles; launch 2 (fresh process) must import the cached .so
    without invoking the builder at all -- the bug being fixed was a
    recompile+relink on every launch."""
    cache_root, builds = _prep(tmp_path, "ext_fastpath")
    r1 = _run_worker("build", _BUILD_PY, cache_root, "ext_fastpath", builds)
    assert r1.returncode == 0 and "IMPORT_OK:42" in r1.stdout, r1.stderr
    assert _n_builds(builds) == 1

    r2 = _run_worker("build", _BUILD_PY, cache_root, "ext_fastpath", builds)
    assert r2.returncode == 0 and "IMPORT_OK:42" in r2.stdout, r2.stderr
    assert _n_builds(builds) == 1, "second launch must not rebuild"


def test_concurrent_launches_build_exactly_once(tmp_path):
    """Eight simultaneous launches racing an uncached build: exactly one
    compile, zero crashes -- the overlapping-runs crash being fixed."""
    cache_root, builds = _prep(tmp_path, "ext_race")
    procs = [subprocess.Popen(
        [sys.executable, _WORKER, "build", _BUILD_PY, cache_root, "ext_race",
         builds, "1.5"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(8)]
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, err
        assert "IMPORT_OK:42" in out
    assert _n_builds(builds) == 1, "concurrent launches must share one build"


def test_dead_builder_does_not_wedge_others(tmp_path):
    """A builder killed mid-compile (OOM kill, scancel) must not block later
    runs: fcntl locks die with their holder, unlike lock *files*."""
    cache_root, builds = _prep(tmp_path, "ext_stale")
    holder = subprocess.Popen(
        [sys.executable, _WORKER, "hold", _BUILD_PY, cache_root, "ext_stale"],
        stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "HOLDING"

    t0 = time.monotonic()
    late = subprocess.Popen(
        [sys.executable, _WORKER, "build", _BUILD_PY, cache_root, "ext_stale",
         builds], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.0)                       # let it block on the held lock
    assert late.poll() is None, "should be waiting on the lock"
    holder.send_signal(signal.SIGKILL)
    holder.wait()
    out, err = late.communicate(timeout=60)
    assert late.returncode == 0 and "IMPORT_OK:42" in out, err
    assert time.monotonic() - t0 < 55, "lock must be released by holder death"


def test_key_sensitivity_and_stability(tmp_path):
    b = _load_build()
    src = str(tmp_path / "a.c")
    open(src, "w").write("v1")
    base = dict(name="k", sources=(src,), cflags=("-O0",),
                generated=(("g.txt", "G1"),), src_dir=str(tmp_path))
    k0 = b.compute_key(b.KernelSpec(**base))
    assert k0 == b.compute_key(b.KernelSpec(**base)), "key must be deterministic"
    assert k0 != b.compute_key(b.KernelSpec(**{**base, "cflags": ("-O2",)}))
    assert k0 != b.compute_key(b.KernelSpec(**{**base, "generated": (("g.txt", "G2"),)}))
    assert k0 != b.compute_key(b.KernelSpec(**{**base, "extra_key": ("cutlass:x",)}))
    open(src, "w").write("v2")
    assert k0 != b.compute_key(b.KernelSpec(**base)), "source content must be hashed"


def test_include_paths_normalized_inside_src_dir(tmp_path):
    """Two checkouts of identical sources at different paths share a key."""
    b = _load_build()
    keys = []
    for co in ("checkout_a", "checkout_b"):
        d = tmp_path / co
        d.mkdir()
        src = str(d / "a.c")
        open(src, "w").write("same-content")
        keys.append(b.compute_key(b.KernelSpec(
            name="k", sources=(src,), include_paths=(str(d),), src_dir=str(d))))
    assert keys[0] == keys[1]


def test_conflicting_key_same_process_raises(tmp_path):
    b = _load_build()
    cache_root, builds = _prep(tmp_path, "ext_conflict")
    sys.path.insert(0, _HERE)
    try:
        import _worker
        spec1 = _worker._spec_for(b, cache_root, "ext_conflict")
        builder = _worker._make_stub_builder(builds, 0.0)
        m1 = b.ensure_extension(spec1, cache_root_override=cache_root, builder=builder)
        assert m1.answer() == 42
        # Same spec again in-process: registry hit, same module, no rebuild.
        assert b.ensure_extension(spec1, cache_root_override=cache_root,
                                  builder=builder) is m1
        assert _n_builds(builds) == 1
        spec2 = b.KernelSpec(name="ext_conflict", sources=spec1.sources,
                             generated=spec1.generated, cflags=("-O3",))
        with pytest.raises(RuntimeError, match="conflicting"):
            b.ensure_extension(spec2, cache_root_override=cache_root, builder=builder)
    finally:
        sys.path.remove(_HERE)


def test_nothing_written_outside_cache_root(tmp_path):
    """Generated sources and build products land in the cache dir only --
    never in the source tree (the old code wrote a shared binding file into
    the package directory)."""
    cache_root, builds = _prep(tmp_path, "ext_clean")
    before = set(os.listdir(_HERE)) | set(os.listdir(os.path.dirname(_BUILD_PY)))
    r = _run_worker("build", _BUILD_PY, cache_root, "ext_clean", builds)
    assert r.returncode == 0, r.stderr
    after = set(os.listdir(_HERE)) | set(os.listdir(os.path.dirname(_BUILD_PY)))
    assert after - before <= {"__pycache__"}
    variant_dirs = os.listdir(os.path.join(cache_root, "ext_clean"))
    assert len(variant_dirs) == 1
    inside = set(os.listdir(os.path.join(cache_root, "ext_clean", variant_dirs[0])))
    assert {"ext_clean.so", "BUILD_OK.json", "gen_note.txt", ".lock"} <= inside


def test_prune_keeps_latest(tmp_path):
    b = _load_build()
    root = str(tmp_path / "cache")
    for i, key in enumerate(("aaaa", "bbbb", "cccc")):
        d = os.path.join(root, "extp", key)
        os.makedirs(d)
        open(os.path.join(d, "BUILD_OK.json"), "w").write('{"schema": 1}')
        old = time.time() - (90 - i * 30) * 86400          # 90/60/30 days old
        os.utime(os.path.join(d, "BUILD_OK.json"), (old, old))
        os.utime(d, (old, old))
    pruned = b.prune_kernel_cache(max_age_days=45, keep_latest=1,
                                  cache_root_override=root)
    left = sorted(os.listdir(os.path.join(root, "extp")))
    assert left == ["cccc"], (pruned, left)
    assert sorted(os.path.basename(p) for p in pruned) == ["aaaa", "bbbb"]
