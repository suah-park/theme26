"""Read-only data-audit primitives for the 11 Sandstones RAW dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHAPE_XYZ = (1000, 1000, 1000)
EXPECTED_DTYPE = np.dtype(np.uint8)
PERCENTILES = (0.5, 1.0, 99.0, 99.5)
QC_CROP_ORIGINS_XYZ = ((0, 0, 0), (372, 372, 372), (744, 744, 744))
MIN_ALIGNMENT_CONTRAST = 1.0


def discover_cores(data_root: str | Path) -> list[Path]:
    """Return immediate core directories in deterministic order."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist or is not a directory: {root}")
    return sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.casefold())


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("cannot calculate a percentile of an empty histogram")
    rank = int(np.ceil((percentile / 100.0) * (total - 1)))
    return int(np.searchsorted(np.cumsum(histogram), rank + 1, side="left"))


def _find_one(core_dir: Path, pattern: str) -> Path:
    matches = sorted(core_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern!r} in {core_dir}, found {len(matches)}")
    return matches[0]


def _validate_raw_file(path: Path, *, shape_xyz: tuple[int, int, int]) -> None:
    expected_bytes = int(np.prod(shape_xyz, dtype=np.int64)) * EXPECTED_DTYPE.itemsize
    if not path.is_file():
        raise FileNotFoundError(f"not a regular file: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path.name}: expected {expected_bytes} bytes, found {path.stat().st_size}")


def _empty_result(core_dir: Path) -> dict[str, Any]:
    return {
        "core_id": core_dir.name,
        "core_dir": str(core_dir),
        "status": "fail",
        "failures": [],
        "source_contract": {
            "shape_xyz": list(EXPECTED_SHAPE_XYZ), "dtype": "uint8",
            # The portal declares little endian. It is immaterial for uint8 but
            # preserved in the audit provenance rather than silently omitted.
            "endianness": "little (not material for uint8)",
        },
    }


def audit_core(
    core_dir: str | Path,
    *,
    shape_xyz: tuple[int, int, int] = EXPECTED_SHAPE_XYZ,
    chunk_x: int = 32,
    qc_crop_origins_xyz: tuple[tuple[int, int, int], ...] | None = None,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    """Audit one core without changing its source files.

    Returns a JSON-ready result and the middle aligned QC crop when validation
    succeeds. Statistics are exact histogram/count reductions, not samples.
    """

    directory = Path(core_dir)
    result = _empty_result(directory)
    result["source_contract"]["shape_xyz"] = list(shape_xyz)
    if chunk_x <= 0:
        raise ValueError("chunk_x must be positive")
    try:
        grayscale_path = _find_one(directory, "*_grayscale_filtered.raw")
        binary_path = _find_one(directory, "*_binary.raw")
        _validate_raw_file(grayscale_path, shape_xyz=shape_xyz)
        _validate_raw_file(binary_path, shape_xyz=shape_xyz)
    except (OSError, ValueError) as error:
        result["failures"].append(str(error))
        return result, None

    result["files"] = {
        "grayscale_filtered": {"name": grayscale_path.name, "bytes": grayscale_path.stat().st_size},
        "binary": {"name": binary_path.name, "bytes": binary_path.stat().st_size},
    }
    grayscale = np.memmap(grayscale_path, dtype=EXPECTED_DTYPE, mode="r", shape=shape_xyz, order="C")
    binary = np.memmap(binary_path, dtype=EXPECTED_DTYPE, mode="r", shape=shape_xyz, order="C")
    gray_histogram = np.zeros(256, dtype=np.int64)
    binary_histogram = np.zeros(256, dtype=np.int64)
    class_sums = np.zeros(2, dtype=np.int64)
    class_counts = np.zeros(2, dtype=np.int64)
    for x0 in range(0, shape_xyz[0], chunk_x):
        x1 = min(x0 + chunk_x, shape_xyz[0])
        gray_chunk = np.asarray(grayscale[x0:x1])
        binary_chunk = np.asarray(binary[x0:x1])
        gray_histogram += np.bincount(gray_chunk.ravel(), minlength=256)
        binary_histogram += np.bincount(binary_chunk.ravel(), minlength=256)
        for label in (0, 1):
            values = gray_chunk[binary_chunk == label]
            class_counts[label] += values.size
            class_sums[label] += int(values.sum(dtype=np.int64))

    labels = np.flatnonzero(binary_histogram)
    if not np.array_equal(labels, np.array([0, 1])):
        result["failures"].append(f"binary labels must be exactly [0, 1], found {labels.tolist()}")
        return result, None
    if np.any(class_counts == 0):
        result["failures"].append("binary labels must both occur")
        return result, None

    class_means = class_sums / class_counts
    source_pore_value = int(np.argmin(class_means))
    source_solid_value = 1 - source_pore_value
    alignment_contrast = float(class_means[source_solid_value] - class_means[source_pore_value])
    origins = qc_crop_origins_xyz or QC_CROP_ORIGINS_XYZ
    crop_polarities: list[int] = []
    crop_contrasts: list[float] = []
    middle_crop: tuple[np.ndarray, np.ndarray] | None = None
    crop_size = min(256, *shape_xyz)
    for origin in origins:
        if any(value < 0 or value + crop_size > limit for value, limit in zip(origin, shape_xyz)):
            result["failures"].append(f"QC crop origin {origin} does not fit shape {shape_xyz}")
            return result, None
        x0, y0, z0 = origin
        gray_crop = np.asarray(grayscale[x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + crop_size])
        binary_crop = np.asarray(binary[x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + crop_size])
        means = [float(gray_crop[binary_crop == label].mean()) for label in (0, 1)]
        crop_polarity = int(np.argmin(means))
        crop_polarities.append(crop_polarity)
        crop_contrasts.append(float(means[1 - crop_polarity] - means[crop_polarity]))
        if middle_crop is None and origin == origins[len(origins) // 2]:
            middle_crop = gray_crop.copy(), binary_crop.copy()

    if any(value != source_pore_value for value in crop_polarities):
        result["failures"].append(f"source pore polarity is inconsistent across QC crops: {crop_polarities}")
    if alignment_contrast < MIN_ALIGNMENT_CONTRAST or any(value < MIN_ALIGNMENT_CONTRAST for value in crop_contrasts):
        result["failures"].append(
            f"grayscale/binary alignment contrast is below {MIN_ALIGNMENT_CONTRAST}: "
            f"global={alignment_contrast:.3f}, crops={crop_contrasts}"
        )
    result.update(
        {
            "binary": {
                "labels": labels.tolist(), "source_pore_value_inferred": source_pore_value,
                "source_solid_value": source_solid_value,
                "pore_fraction": float(class_counts[source_pore_value] / class_counts.sum()),
            },
            "grayscale": {
                "min": int(np.flatnonzero(gray_histogram)[0]), "max": int(np.flatnonzero(gray_histogram)[-1]),
                "mean": float(np.dot(np.arange(256, dtype=np.int64), gray_histogram) / gray_histogram.sum()),
                "percentiles": {str(value): _percentile_from_histogram(gray_histogram, value) for value in PERCENTILES},
            },
            "alignment_qc": {
                "method": "same-coordinate grayscale means conditioned on binary label; lower intensity is inferred pore",
                "global_mean_by_source_label": {"0": float(class_means[0]), "1": float(class_means[1])},
                "global_contrast_solid_minus_pore": alignment_contrast,
                "crop_origins_xyz": [list(origin) for origin in origins],
                "crop_inferred_pore_values": crop_polarities,
                "crop_contrasts_solid_minus_pore": crop_contrasts,
                "threshold": MIN_ALIGNMENT_CONTRAST,
            },
        }
    )
    result["status"] = "pass" if not result["failures"] else "fail"
    return result, middle_crop
