"""Read-only QC for one registered E1 validation reconstruction."""

from __future__ import annotations

import base64
import html
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import yaml

from theme26.data import (compose_full_volume, extract_volume_crop,
                          linear_interpolate_z, make_sparse_sample,
                          normalize_grayscale, read_raw_volume)
from theme26.training.workflow import file_hash, load_experiment_config, read_berea_context


SLICES = (("axial_z125", "Axial: missing Z=125", "X", "Y", lambda a: a[:, :, 125]),
          ("sagittal_x128", "Sagittal: X=128", "Y", "Z", lambda a: a[128, :, :]),
          ("coronal_y128", "Coronal: Y=128", "X", "Z", lambda a: a[:, 128, :]))


def _png_bytes(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if rgb.dtype != np.uint8 or channels != 3:
        raise ValueError("expected uint8 RGB image")
    def chunk(name: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + name + data +
                struct.pack(">I", zlib.crc32(name + data) & 0xffffffff))
    rows = b"".join(b"\0" + rgb[row].tobytes() for row in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _gray(array: np.ndarray) -> np.ndarray:
    values = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.repeat(values[:, :, None], 3, axis=2)


def _error(array: np.ndarray, maximum: float) -> np.ndarray:
    value = np.clip(array / maximum, 0, 1)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([[10, 7, 27], [78, 26, 94], [171, 40, 72],
                       [245, 119, 40], [255, 237, 151]], dtype=np.float32)
    return np.stack([np.interp(value, stops, colors[:, channel])
                     for channel in range(3)], axis=-1).astype(np.uint8)


def _figure(path: Path, *, title: str, x_axis: str, y_axis: str,
            truth: np.ndarray, e0: np.ndarray, e1: np.ndarray,
            error_maximum: float) -> None:
    panels = (("Truth", truth, False), ("E0 linear", e0, False),
              ("E1 SAINT", e1, False), ("|E0 - Truth|", np.abs(e0 - truth), True),
              ("|E1 - Truth|", np.abs(e1 - truth), True))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="1750" height="520" viewBox="0 0 1750 520">',
        '<rect width="1750" height="520" fill="white"/>',
        f'<text x="875" y="37" text-anchor="middle" font-size="25" font-weight="bold">'
        f'{html.escape(title)}</text>',
        '<text x="875" y="64" text-anchor="middle" font-size="16">'
        'Shared grayscale [0,1]; shared E0/E1 absolute-error scale</text>',
    ]
    for index, (label, plane, is_error) in enumerate(panels):
        left = 70 + index * 340
        top = 112
        # Arrays are [horizontal, vertical]; transpose and flip for origin=lower.
        oriented = np.flipud(plane.T)
        rgb = _error(oriented, error_maximum) if is_error else _gray(oriented)
        embedded = base64.b64encode(_png_bytes(rgb)).decode("ascii")
        parts.extend([
            f'<text x="{left+128}" y="98" text-anchor="middle" font-size="18" '
            f'font-weight="bold">{html.escape(label)}</text>',
            f'<image x="{left}" y="{top}" width="256" height="256" '
            f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{embedded}"/>',
            f'<rect x="{left}" y="{top}" width="256" height="256" fill="none" '
            'stroke="#222" stroke-width="1"/>',
            f'<text x="{left+128}" y="405" text-anchor="middle" font-size="16">'
            f'{x_axis} (voxel)</text>',
            f'<text x="{left-35}" y="{top+60}" text-anchor="middle" font-size="15">'
            f'{y_axis}</text>',
            f'<text x="{left-35}" y="{top+78}" text-anchor="middle" font-size="12">'
            '(voxel)</text>',
        ])
        for fraction, tick in ((0, "0"), (0.5, "128"), (1, "255")):
            xx = left + round(256 * fraction)
            yy = top + round(256 * (1 - fraction))
            parts.append(f'<text x="{xx}" y="385" text-anchor="middle" font-size="12">{tick}</text>')
            parts.append(f'<text x="{left-7}" y="{yy+4}" text-anchor="end" font-size="12">{tick}</text>')
    ramp = np.tile(np.linspace(0, 1, 350, dtype=np.float32), (18, 1))
    gray_bar = base64.b64encode(_png_bytes(_gray(ramp))).decode("ascii")
    error_bar = base64.b64encode(_png_bytes(_error(ramp, 1.0))).decode("ascii")
    parts.extend([
        '<text x="175" y="449" font-size="15">Image intensity</text>',
        f'<image x="320" y="433" width="350" height="18" '
        f'xlink:href="data:image/png;base64,{gray_bar}"/>',
        '<rect x="320" y="433" width="350" height="18" fill="none" stroke="#333"/>',
        '<text x="315" y="472" text-anchor="middle" font-size="13">0</text>',
        '<text x="670" y="472" text-anchor="middle" font-size="13">1</text>',
        '<text x="975" y="449" font-size="15">Absolute error</text>',
        f'<image x="1110" y="433" width="350" height="18" '
        f'xlink:href="data:image/png;base64,{error_bar}"/>',
        '<rect x="1110" y="433" width="350" height="18" fill="none" stroke="#333"/>',
        '<text x="1105" y="472" text-anchor="middle" font-size="13">0</text>',
        f'<text x="1460" y="472" text-anchor="middle" font-size="13">'
        f'{error_maximum:.4f}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def run_validation_crop_qc(*, result_dir: Path, raw: Path,
                           output_dir: Path, source_config: Path) -> None:
    """Check one registered E1 crop and save shared-scale comparison figures."""
    result_dir = result_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"QC destination already exists: {output_dir}")
    crop_metadata = yaml.safe_load((result_dir / "metadata.yaml").read_text(encoding="utf-8"))
    data_config = load_experiment_config(source_config.resolve())["data"]
    os.environ[data_config["path_env"]] = str(raw.resolve())
    raw_path, volume_metadata, records, normalization, protocol_sha = read_berea_context(data_config)
    matches = [row for row in records if row.sample_id == crop_metadata["crop_id"]]
    if len(matches) != 1:
        raise ValueError("result crop ID does not identify one manifest record")
    record = matches[0]
    checks = {
        "registered_validation_crop": record.split == crop_metadata["split"] == "validation",
        "crop_coordinates": (list(record.crop.origin_xyz) == crop_metadata["origin_xyz"]
                             and list(record.crop.shape_xyz) == crop_metadata["shape_xyz"]),
        "sampling": (record.sampling.factor == crop_metadata["sampling_factor"]
                     and record.sampling.offset == crop_metadata["sampling_offset"]),
        "axis_spacing_normalization": (crop_metadata["axis_order"] == "XYZ"
            and list(volume_metadata.voxel_spacing_um_xyz) == crop_metadata["voxel_spacing_um_xyz"]
            and volume_metadata.normalization_id == crop_metadata["normalization_id"]),
        "source_raw_sha256": file_hash(raw_path) == crop_metadata["source_raw_sha256"],
        "protocol_sha256": protocol_sha == crop_metadata["protocol_sha256"],
    }
    raw = read_raw_volume(raw_path, metadata=volume_metadata)
    dense = normalize_grayscale(extract_volume_crop(raw, crop=record.crop),
                                lower=float(normalization["lower"]),
                                upper=float(normalization["upper"]))
    sample = make_sparse_sample(dense, sampling=record.sampling)
    prediction = np.load(result_dir / "prediction.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(result_dir / "primary_mask.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = (256, 256, 256)
    checks.update({
        "prediction_shape_dtype": prediction.shape == expected_shape and prediction.dtype == np.float32,
        "mask_shape_dtype": mask.shape == expected_shape and mask.dtype == np.bool_,
    })
    if not checks["prediction_shape_dtype"] or not checks["mask_shape_dtype"]:
        raise ValueError(f"unexpected prediction/mask shape or dtype: {prediction.shape}/{prediction.dtype}, "
                         f"{mask.shape}/{mask.dtype}")
    actual_min = float(prediction.min())
    actual_max = float(prediction.max())
    checks.update({
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "prediction_in_0_1": 0.0 <= actual_min <= actual_max <= 1.0,
        "primary_mask_exact": bool(np.array_equal(mask, sample.primary_metric_mask)),
        "primary_slices_189": (int(np.count_nonzero(mask.any(axis=(0, 1)))) == 189
                               and int(mask.sum()) == 189 * 256 * 256),
        "observed_reinsertion_exact": bool(np.array_equal(
            prediction[..., sample.observed_indices], sample.sparse_volume)),
    })
    if not all(checks.values()):
        raise ValueError(f"QC checks failed: {[name for name, passed in checks.items() if not passed]}")
    e0 = compose_full_volume(np.clip(linear_interpolate_z(sample), 0, 1), sample)
    e1 = compose_full_volume(np.clip(np.asarray(prediction), 0, 1), sample)
    checks["e1_policy_idempotent"] = bool(np.array_equal(e1, prediction))
    if not checks["e1_policy_idempotent"]:
        raise ValueError("E1 final output changed under common clip/reinsertion policy")
    expected_mask = sample.primary_metric_mask
    e0_mae = float(np.abs(e0 - dense)[expected_mask].mean())
    e1_mae = float(np.abs(e1 - dense)[expected_mask].mean())
    checks["primary_mae_matches_metadata"] = abs(e1_mae - crop_metadata["primary_mae"]) <= 1e-8
    checks["primary_voxels_match_metadata"] = int(expected_mask.sum()) == crop_metadata["primary_voxels"]
    if not all(checks.values()):
        raise ValueError(f"QC checks failed: {[name for name, passed in checks.items() if not passed]}")
    figure_data = []
    for name, title, x_axis, y_axis, select in SLICES:
        figure_data.append((name, title, x_axis, y_axis,
                            select(dense), select(e0), select(e1)))
    error_maximum = max(float(np.max(np.abs(a - truth)))
                        for _, _, _, _, truth, e0_plane, e1_plane in figure_data
                        for a in (e0_plane, e1_plane))
    if error_maximum <= 0:
        error_maximum = 1.0
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_paths = []
    for name, title, x_axis, y_axis, truth, e0_plane, e1_plane in figure_data:
        path = output_dir / f"{name}.svg"
        _figure(path, title=title, x_axis=x_axis, y_axis=y_axis,
                truth=truth, e0=e0_plane, e1=e1_plane, error_maximum=error_maximum)
        figure_paths.append(path.name)
    png_paths = []
    if shutil.which("convert"):
        for svg_name in figure_paths:
            png_name = Path(svg_name).with_suffix(".png").name
            subprocess.run(["convert", "-background", "white",
                            str(output_dir / svg_name), str(output_dir / png_name)], check=True)
            png_paths.append(png_name)
    report = {
        "scope": "one registered validation crop; not whole-validation performance",
        "crop_id": record.sample_id,
        "origin_xyz": list(record.crop.origin_xyz),
        "normalization_id": normalization["normalization_id"],
        "normalization_lower_upper": [normalization["lower"], normalization["upper"]],
        "source_raw_sha256": crop_metadata["source_raw_sha256"],
        "prediction_sha256": file_hash(result_dir / "prediction.npy"),
        "primary_mask_sha256": file_hash(result_dir / "primary_mask.npy"),
        "prediction_shape_xyz": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "prediction_actual_min_max": [actual_min, actual_max],
        "observed_slices": len(sample.observed_indices),
        "primary_missing_slices": 189,
        "primary_voxels": int(expected_mask.sum()),
        "checks": checks,
        "metric_region": "same bracketed-missing primary mask for E0 and E1",
        "range_policy": "clip [0,1], then reinsert observed slices for both methods",
        "e0_primary_mae": e0_mae,
        "e1_primary_mae": e1_mae,
        "metadata_primary_mae": float(crop_metadata["primary_mae"]),
        "e1_minus_e0_primary_mae": e1_mae - e0_mae,
        "figure_image_range": [0.0, 1.0],
        "figure_shared_absolute_error_range": [0.0, error_maximum],
        "figures": figure_paths,
        "png_figures": png_paths,
    }
    (output_dir / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False),
                                             encoding="utf-8")
    print(yaml.safe_dump({"qc_dir": str(output_dir), "e0_primary_mae": e0_mae,
                          "e1_primary_mae": e1_mae, "checks": checks}, sort_keys=False))
