# R3GAN/kernels/tests/_worker.py
"""Subprocess worker for the build-cache tests.

Loaded by file path (not via the package) so the tests run without torch.

    python _worker.py build <build_py> <cache_root> <name> <builds_log_dir> [sleep_s]
    python _worker.py hold  <build_py> <cache_root> <name>

`build` runs ensure_extension with a stub builder that (optionally sleeps,
then) compiles a real CPython extension with gcc and drops one sentinel file
per actual compile into <builds_log_dir>. Prints IMPORT_OK:<answer()>.

`hold` computes the same cache dir, takes the fcntl build lock, prints
HOLDING, and sleeps forever (the parent SIGKILLs it to simulate a builder
dying mid-compile).
"""
import importlib.util
import os
import subprocess
import sys
import sysconfig
import time
import uuid


def _load_build(build_py):
    spec = importlib.util.spec_from_file_location("r3gan_build_under_test", build_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def _spec_for(b, cache_root, name):
    src = os.path.join(cache_root, f"{name}_input.c")  # content only hashed
    return b.KernelSpec(
        name=name,
        sources=(src,),
        generated=(("gen_note.txt", "generated-content-v1"),),
        cflags=("-O0",),
    )


def _make_stub_builder(builds_log_dir, sleep_s):
    def builder(spec, directory, verbose):
        if sleep_s:
            time.sleep(sleep_s)
        c = os.path.join(directory, "ext.c")
        with open(c, "w") as f:
            f.write(f"""
#include <Python.h>
static PyObject* answer(PyObject* s, PyObject* a) {{ return PyLong_FromLong(42); }}
static PyMethodDef m[] = {{{{"answer", answer, METH_NOARGS, ""}}, {{0}}}};
static struct PyModuleDef d = {{PyModuleDef_HEAD_INIT, "{spec.name}", 0, -1, m}};
PyMODINIT_FUNC PyInit_{spec.name}(void) {{ return PyModule_Create(&d); }}
""")
        inc = sysconfig.get_paths()["include"]
        so = os.path.join(directory, f"{spec.name}.so")
        subprocess.run(["gcc", "-shared", "-fPIC", f"-I{inc}", c, "-o", so],
                       check=True)
        with open(os.path.join(builds_log_dir, f"{uuid.uuid4().hex}.built"), "w"):
            pass
        # Import exactly like the production torch builder does (in-process).
        spec_ = importlib.util.spec_from_file_location(spec.name, so)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        return mod
    return builder


def main():
    cmd, build_py, cache_root, name = sys.argv[1:5]
    b = _load_build(build_py)
    spec = _spec_for(b, cache_root, name)

    if cmd == "hold":
        d = b.build_dir_for(spec, cache_root_override=cache_root)
        os.makedirs(d, exist_ok=True)
        import fcntl
        fd = os.open(os.path.join(d, ".lock"), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        print("HOLDING", flush=True)
        time.sleep(3600)
        return

    builds_log_dir = sys.argv[5]
    sleep_s = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    builder = _make_stub_builder(builds_log_dir, sleep_s)
    mod = b.ensure_extension(spec, cache_root_override=cache_root, builder=builder)
    print(f"IMPORT_OK:{mod.answer()}", flush=True)


if __name__ == "__main__":
    main()
