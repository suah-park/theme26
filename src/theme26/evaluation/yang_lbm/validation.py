"""Strict canonical input validation: never guess axes or segment grayscale."""
from __future__ import annotations
import hashlib
import json
import math
from numbers import Real
import numpy as np


def validate_pore(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 3:
        raise TypeError("Input must be a 3D NumPy array in XYZ order")
    if value.dtype != np.bool_:
        raise TypeError("Solver requires bool pore=True; convert explicit labels in I/O")
    if min(value.shape) < 3:
        raise ValueError("Each dimension must be at least 3")
    if value.all():
        raise ValueError("All-pore periodic domain has no solid drag; finite K undefined")
    return value  # all-solid is an explicitly non-percolating limit


def validate_spacing(value) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("spacing_um_xyz must contain three numbers")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real)
           or not math.isfinite(float(x)) or x <= 0 for x in value):
        raise ValueError("spacing must be finite, positive, numeric")
    v = tuple(float(x) for x in value)
    if not np.allclose(v, v[0], rtol=1e-12, atol=0):
        raise ValueError("D3Q19 lattice requires isotropic HR spacing; sparse z spacing is invalid")
    return v


def checksum_array(value: np.ndarray) -> str:
    h = hashlib.sha256(json.dumps({"shape": value.shape, "dtype": str(value.dtype),
                                   "axis_order": "XYZ"}, sort_keys=True).encode())
    h.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()
