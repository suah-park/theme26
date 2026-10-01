"""Create one contract-compliant Berea crop from the raw 11 Sandstones files.

The normalization limits and binary polarity are intentionally required command
line inputs: both must have been frozen from training-core metadata, not derived
from this held-out core.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data.crop_sampler import extract_aligned_crop
from theme26.data.raw_volume import normalize_grayscale, read_raw_volume, to_pore_mask
from theme26.data.sparse_sampling import linear_interpolate_z, make_sparse_sample


SOURCE_SHAPE = (1000, 1000, 1000)
SPACING_UM_XYZ = (2.25, 2.25, 2.25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a 256³ aligned Berea crop")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the two Berea .raw files")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz artifact (not tracked by Git)")
    parser.add_argument("--crop-origin", type=int, nargs=3, metavar=("X0", "Y0", "Z0"), required=True)
    parser.add_argument("--normalization-lower", type=float, required=True)
    parser.add_argument("--normalization-upper", type=float, required=True)
    parser.add_argument("--source-pore-value", type=int, choices=(0, 1), required=True)
    parser.add_argument("--offset", type=int, default=0, help="Frozen x4 z-sampling offset")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grayscale_metadata = VolumeMetadata(
        sample_id="Berea_grayscale_filtered",
        source_core_id="Berea",
        shape_xyz=SOURCE_SHAPE,
        voxel_spacing_um_xyz=SPACING_UM_XYZ,
        dtype="uint8",
        endianness="not-applicable",
    )
    binary_metadata = VolumeMetadata(
        sample_id="Berea_binary",
        source_core_id="Berea",
        shape_xyz=SOURCE_SHAPE,
        voxel_spacing_um_xyz=SPACING_UM_XYZ,
        dtype="uint8",
        endianness="not-applicable",
    )
    grayscale_raw = read_raw_volume(args.data_dir / "Berea_2d25um_grayscale_filtered.raw", metadata=grayscale_metadata)
    binary_raw = read_raw_volume(args.data_dir / "Berea_2d25um_binary.raw", metadata=binary_metadata)
    grayscale = normalize_grayscale(
        grayscale_raw, lower=args.normalization_lower, upper=args.normalization_upper
    )
    pore = to_pore_mask(binary_raw, source_pore_value=args.source_pore_value)
    crop, pore_crop = extract_aligned_crop(grayscale, pore, crop=CropSpec(tuple(args.crop_origin)))
    sample = make_sparse_sample(crop, sampling=SamplingSpec(factor=4, offset=args.offset))
    e0 = linear_interpolate_z(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        grayscale=crop,
        pore_mask=pore_crop,
        sparse_volume=sample.sparse_volume,
        observed_indices=np.asarray(sample.observed_indices, dtype=np.int32),
        observed_mask=sample.observed_mask,
        missing_mask=sample.missing_mask,
        primary_metric_mask=sample.primary_metric_mask,
        e0_linear=e0,
        crop_origin_xyz=np.asarray(args.crop_origin, dtype=np.int32),
        sampling_factor=np.int32(4),
        sampling_offset=np.int32(args.offset),
        normalization_lower=np.float32(args.normalization_lower),
        normalization_upper=np.float32(args.normalization_upper),
        source_pore_value=np.int8(args.source_pore_value),
    )


if __name__ == "__main__":
    main()
