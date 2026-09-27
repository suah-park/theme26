"""Build the frozen Berea-only spatial split manifest."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from theme26.contracts import CropSpec, SamplingSpec
from theme26.data import CropRecord, validate_spatial_split, write_crop_manifest

MANIFEST_VERSION = "berea_spatial_v1"
VOLUME_SHAPE = (1000, 1000, 1000)
CROP_SHAPE = (256, 256, 256)
BUFFER_VOXELS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "configs/data/berea_spatial_split_v1.csv",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-crops", type=int, default=48)
    return parser


def _record(split: str, index: int, origin: tuple[int, int, int]) -> CropRecord:
    return CropRecord(
        manifest_version=MANIFEST_VERSION,
        sample_id=f"berea_{split}_{index:03d}",
        source_core_id="berea_2d25um",
        split=split,  # type: ignore[arg-type]
        crop=CropSpec(origin_xyz=origin, shape_xyz=CROP_SHAPE),
        sampling=SamplingSpec(factor=4, offset=0),
    )


def build_records(*, seed: int, train_crops: int) -> tuple[CropRecord, ...]:
    if train_crops < 1:
        raise ValueError("train_crops must be positive")
    rng = np.random.default_rng(seed)
    origins: set[tuple[int, int, int]] = set()
    while len(origins) < train_crops:
        origins.add(
            (
                int(rng.integers(0, 145)),
                int(rng.integers(0, 745)),
                int(rng.integers(0, 745)),
            )
        )
    records = [
        _record("train", index, origin)
        for index, origin in enumerate(sorted(origins))
    ]

    grid = (0, 372, 744)
    for split, x_origin in (("validation", 432), ("test", 720)):
        for index, (y_origin, z_origin) in enumerate(product(grid, repeat=2)):
            records.append(_record(split, index, (x_origin, y_origin, z_origin)))
    return tuple(records)


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(seed=args.seed, train_crops=args.train_crops)
    validate_spatial_split(
        records,
        volume_shape_xyz=VOLUME_SHAPE,
        minimum_buffer_voxels=BUFFER_VOXELS,
    )
    write_crop_manifest(args.output, records)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
    }
    print(f"manifest={args.output}")
    print(f"counts={counts}")
    print(f"minimum_cross_split_buffer_voxels={BUFFER_VOXELS}")


if __name__ == "__main__":
    main()
