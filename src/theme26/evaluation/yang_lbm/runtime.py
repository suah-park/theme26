"""Taichi initialization only inside a fresh Python worker, never at import."""
import sys
import importlib.util
import importlib.metadata
import hashlib
import json
from pathlib import Path

UPSTREAM_COMMIT = "fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c"


def core_fingerprint():
    root=Path(__file__).parent
    paths=[root/'_vendor/periodic_core.py',root/'_vendor/lattice.py',
           root/'_vendor/formulation.py',
           *[Path(__file__).with_name(name) for name in
             ('permeability.py','engine.py','connectivity.py','statistics.py')]]
    h=hashlib.sha256()
    for p in paths: h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def doctor():
    packages={}
    for p in ('numpy','scipy','PyYAML','taichi'):
        try: packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p]=None
    return {"python_executable":sys.executable,"python":sys.version,"packages":packages,"taichi_installed":packages['taichi'] is not None,
            "core_sha256":core_fingerprint(),"gpu_runtime_checked":False,
            "note":"CUDA availability is tested inside an actual solver worker; no silent CPU fallback."}


def init_runtime(config):
    import taichi as ti
    requested=ti.cuda if config.backend=='cuda' else ti.cpu
    fp = ti.f64
    ti.init(arch=requested, default_fp=fp, cpu_max_num_threads=config.cpu_threads,
            device_memory_fraction=config.device_memory_fraction, offline_cache=False,
            enable_fallback=False, fast_math=False)
    actual=ti.lang.impl.current_cfg().arch
    if actual!=requested:
        raise RuntimeError(f"Requested {requested}, got {actual}; refusing silent fallback")
    return ti

