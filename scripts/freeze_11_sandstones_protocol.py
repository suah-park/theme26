"""Freeze requested split, train-only normalization, and fixed evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from theme26.data.audit import _percentile_from_histogram


SPLITS = {
    "train": ["Bandera Brown", "Bandera Gray", "Berea Sister Gray", "Berea Upper Gray", "Buff Berea", "CastleGate", "Kirby", "Leopard"],
    "validation": ["Berea"],
    "test": ["Bentheimer", "Parker"],
}
ORIGINS = [[64, 64, 64], [680, 64, 680], [64, 680, 680], [680, 680, 64]]
SHAPE = [256, 256, 256]
FULL_SHAPE = (1000, 1000, 1000)


def _find_file(core_dir: Path, pattern: str) -> Path:
    paths = sorted(core_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"{core_dir}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def _source_entry(root: Path, core_id: str) -> dict[str, object]:
    directory = root / core_id
    gray, binary = _find_file(directory, "*_grayscale_filtered.raw"), _find_file(directory, "*_binary.raw")
    expected_bytes = int(np.prod(FULL_SHAPE, dtype=np.int64))
    for path in (gray, binary):
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{path}: expected {expected_bytes} uint8 bytes")
    return {"source_core_id": core_id, "grayscale_filtered_filename": gray.name, "binary_filename": binary.name, "source_pore_value": 0}


def _train_histogram(root: Path) -> np.ndarray:
    histogram = np.zeros(256, dtype=np.int64)
    for core_id in SPLITS["train"]:
        path = _find_file(root / core_id, "*_grayscale_filtered.raw")
        data = np.memmap(path, dtype=np.uint8, mode="r", shape=FULL_SHAPE)
        for x0 in range(0, FULL_SHAPE[0], 32):
            histogram += np.bincount(np.asarray(data[x0:x0 + 32]).ravel(), minlength=256)
    return histogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze 11 Sandstones phase-2 data protocol")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    args = parser.parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    all_entries = {split: [_source_entry(args.data_root, core) for core in cores] for split, cores in SPLITS.items()}
    split_manifest = {
        "manifest_version": "11-sandstones-split-v1", "split_rationale": "Frozen parent-core split supplied for the primary protocol; no core appears in more than one split.",
        "axis_order": "XYZ", "source_shape_xyz": FULL_SHAPE, "dtype": "uint8", "endianness": "little (not material for uint8)", "splits": all_entries,
    }
    (args.config_dir / "splits.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    histogram = _train_histogram(args.data_root)
    lower, upper = _percentile_from_histogram(histogram, 0.5), _percentile_from_histogram(histogram, 99.5)
    normalization = {
        "normalization_id": "11-sandstones-train-pooled-p0.5-p99.5-v1", "method": "pooled uint8 histogram over train grayscale_filtered.raw only; clip to [p0.5, p99.5], then (x-lower)/(upper-lower)",
        "lower": lower, "upper": upper, "dtype": "float32", "output_range": [0.0, 1.0], "train_cores": SPLITS["train"],
        "histogram_voxels": int(histogram.sum()), "percentiles": {"0.5": lower, "99.5": upper},
    }
    (args.config_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    crops = []
    for split in ("validation", "test"):
        for core_id in SPLITS[split]:
            for index, origin in enumerate(ORIGINS):
                if any(value < 0 or value + size > limit for value, size, limit in zip(origin, SHAPE, FULL_SHAPE)):
                    raise ValueError(f"invalid crop origin {origin}")
                crops.append({"crop_id": f"{core_id.replace(' ', '_').lower()}_{index:02d}", "source_core_id": core_id, "split": split,
                              "origin_xyz": origin, "shape_xyz": SHAPE, "axis_order": "XYZ", "sparse_axis": "Z", "sampling_factor": 4, "sampling_offset": 0,
                              "voxel_spacing_um_xyz": [2.25, 2.25, 2.25]})
    eval_manifest = {"manifest_version": "11-sandstones-eval-crops-v1", "normalization_id": normalization["normalization_id"], "crops": crops}
    (args.config_dir / "eval_crop_manifest.json").write_text(json.dumps(eval_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen split: train={len(SPLITS['train'])}, validation={len(SPLITS['validation'])}, test={len(SPLITS['test'])}")
    print(f"Frozen normalization: lower={lower}, upper={upper}")
    print(f"Frozen eval crops: {len(crops)}")


if __name__ == "__main__":
    main()
