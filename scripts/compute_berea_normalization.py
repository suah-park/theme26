"""Compute frozen uint8 normalization bounds from the Berea training slab only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import yaml

from theme26.contracts import VolumeMetadata
from theme26.data import read_raw_volume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grayscale", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "configs/data/berea_normalization_v1.yaml",
    )
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--chunk-x", type=int, default=8)
    return parser


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> int:
    threshold = percentile / 100.0 * int(histogram.sum())
    return int(np.searchsorted(np.cumsum(histogram), threshold, side="left"))


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if args.chunk_x < 1:
        raise ValueError("chunk-x must be positive")

    metadata = VolumeMetadata(
        sample_id="berea",
        source_core_id="berea_2d25um",
        shape_xyz=(1000, 1000, 1000),
        voxel_spacing_um_xyz=(2.25, 2.25, 2.25),
        dtype="uint8",
        endianness="not-applicable",
        storage_order="ZYX",
    )
    volume = read_raw_volume(args.grayscale, metadata=metadata)
    histogram = np.zeros(256, dtype=np.int64)
    train_x_start, train_x_stop = 0, 400
    for start in range(train_x_start, train_x_stop, args.chunk_x):
        stop = min(start + args.chunk_x, train_x_stop)
        chunk = np.asarray(volume[start:stop, :, :])
        histogram += np.bincount(chunk.reshape(-1), minlength=256)

    lower = _percentile_from_histogram(histogram, args.lower_percentile)
    upper = _percentile_from_histogram(histogram, args.upper_percentile)
    if lower >= upper:
        raise ValueError(f"invalid normalization bounds from histogram: {lower}, {upper}")
    nonzero = np.flatnonzero(histogram)
    result = {
        "normalization_id": "berea_train_slab_p005_p995_v1",
        "method": "clipped_percentile_minmax",
        "source_split": "train",
        "source_region_xyz": [[0, 400], [0, 1000], [0, 1000]],
        "voxel_count": int(histogram.sum()),
        "lower_percentile": float(args.lower_percentile),
        "upper_percentile": float(args.upper_percentile),
        "lower_uint8": lower,
        "upper_uint8": upper,
        "observed_min_uint8": int(nonzero[0]),
        "observed_max_uint8": int(nonzero[-1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False), end="")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
