# R3GAN/kernels/prebuild.py
"""Compile kernel extensions into the shared cache ahead of time.

    python -m R3GAN.kernels.prebuild                 # every arch package
    python -m R3GAN.kernels.prebuild sm100           # specific arch(es)
    python -m R3GAN.kernels.prebuild --list

Typical use on a cluster: run once (login node with the CUDA toolkit loaded,
or a small setup job); every subsequent training launch then loads the cached
binaries in milliseconds, including many launches starting at the same moment.
Requires nvcc + torch; does NOT require a GPU (the target device is only
needed at run time). Safe to run concurrently with itself or with training.

Arch packages (sm100/, sm120/, ...) opt in by defining::

    def prebuild_targets():
        return (("<ext name>", lambda verbose: <build it>), ...)

Cache location and knobs: see R3GAN/kernels/_build.py.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import re
import sys
import time


def _arch_packages():
    pkg = importlib.import_module(__package__)
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                  if m.ispkg and re.fullmatch(r"sm\d+\w*", m.name))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("arch", nargs="*",
                   help="arch package(s) to build (default: all discovered)")
    p.add_argument("--list", action="store_true", help="list arch packages and exit")
    p.add_argument("--verbose", action="store_true", help="show ninja/nvcc output")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    available = _arch_packages()
    if args.list:
        print("\n".join(available) or "(no arch packages found)")
        return 0
    unknown = [a for a in args.arch if a not in available]
    if unknown:
        p.error(f"unknown arch package(s) {unknown}; available: {available}")
    arches = args.arch or available

    from ._build import cache_root  # noqa: PLC0415
    print(f"kernel cache root: {cache_root()}")

    failed = []
    for arch in arches:
        mod = importlib.import_module(f".{arch}", __package__)
        targets = getattr(mod, "prebuild_targets", None)
        if targets is None:
            print(f"[{arch}] no prebuild_targets(); skipping")
            continue
        for name, build in targets():
            t0 = time.monotonic()
            try:
                build(args.verbose)
                print(f"  {name:<24} ready in {time.monotonic() - t0:7.1f}s")
            except Exception as e:  # noqa: BLE001 -- report all, then exit nonzero
                failed.append(name)
                print(f"  {name:<24} FAILED: {e}")
    if failed:
        print(f"{len(failed)} build(s) failed: {failed}")
        return 1
    print("done; training launches will fast-path these binaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
