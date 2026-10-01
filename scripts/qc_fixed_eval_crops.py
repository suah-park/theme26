"""Generate and verify every frozen validation/test crop without writing crop NPZs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from theme26.contracts import CropSpec, SamplingSpec
from theme26.data.dataset import CoreSource, preprocess_crop


def _save_png(result: dict[str, Any], crop: Any, path: Path) -> None:
    gray = crop.grayscale[:, :, 128].T
    pore = crop.pore_mask[:, :, 128].T
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[0].set_title("Normalized grayscale")
    axes[1].imshow(pore, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("Pore mask (pore=1)")
    axes[2].imshow(gray, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[2].contour(pore.astype(np.uint8), levels=[0.5], colors="#ff2d2d", linewidths=.55, origin="lower")
    axes[2].set_title("Pore boundary over grayscale")
    for axis in axes:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
    figure.suptitle(f"{result['crop_id']} — {'PASS' if result['passed'] else 'FAIL'}")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _validate(crop: Any) -> dict[str, bool]:
    expected = (256, 256, 256)
    sparse = crop.sparse
    observed = np.arange(0, 256, 4)
    means = (float(crop.grayscale[crop.pore_mask].mean()), float(crop.grayscale[~crop.pore_mask].mean()))
    return {
        "shape": crop.grayscale.shape == expected and crop.pore_mask.shape == expected and crop.e0_linear.shape == expected and sparse.sparse_volume.shape == (256, 256, 64),
        "dtype": crop.grayscale.dtype == np.float32 and crop.pore_mask.dtype == np.bool_ and crop.e0_linear.dtype == np.float32,
        "finite": bool(np.isfinite(crop.grayscale).all() and np.isfinite(crop.e0_linear).all()),
        "range_0_1": bool(np.all((crop.grayscale >= 0) & (crop.grayscale <= 1)) and np.all((crop.e0_linear >= 0) & (crop.e0_linear <= 1))),
        "binary_alignment": means[0] < means[1],
        "observed_indices": sparse.observed_indices == tuple(observed.tolist()),
        "observed_data_consistency": bool(np.array_equal(sparse.sparse_volume, crop.grayscale[..., observed]) and np.array_equal(crop.e0_linear[..., observed], crop.grayscale[..., observed])),
        "primary_mask": int(sparse.primary_metric_mask[0, 0].sum()) == 189 and int(sparse.primary_metric_mask.sum()) == 256 * 256 * 189,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QC frozen 11 Sandstones validation/test crops")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    split_config = json.loads((args.config_dir / "splits.json").read_text())
    normalization = json.loads((args.config_dir / "normalization.json").read_text())
    manifest = json.loads((args.config_dir / "eval_crop_manifest.json").read_text())
    source_by_core = {entry["source_core_id"]: (split, entry) for split, entries in split_config["splits"].items() for entry in entries}
    png_dir = args.output_dir / "final_preprocessing_qc"
    png_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for entry in manifest["crops"]:
        split, source_entry = source_by_core[entry["source_core_id"]]
        source = CoreSource(entry["source_core_id"], split, args.data_root / entry["source_core_id"] / source_entry["grayscale_filtered_filename"], args.data_root / entry["source_core_id"] / source_entry["binary_filename"], source_entry["source_pore_value"])
        crop = preprocess_crop(source, crop=CropSpec(tuple(entry["origin_xyz"]), tuple(entry["shape_xyz"])), normalization_lower=normalization["lower"], normalization_upper=normalization["upper"], sampling=SamplingSpec(entry["sampling_factor"], entry["sampling_offset"]))
        checks = _validate(crop)
        result = {"crop_id": entry["crop_id"], "core_id": source.core_id, "split": split, "origin_xyz": entry["origin_xyz"], "checks": checks, "passed": all(checks.values()),
                  "grayscale_min": float(crop.grayscale.min()), "grayscale_max": float(crop.grayscale.max()), "pore_fraction": float(crop.pore_mask.mean())}
        results.append(result)
        _save_png(result, crop, png_dir / f"{entry['crop_id']}.png")
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {entry['crop_id']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_preprocessing_qc.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    rows = [{"crop_id": result["crop_id"], "core_id": result["core_id"], "split": result["split"], "origin_xyz": str(result["origin_xyz"]), "passed": result["passed"], "pore_fraction": result["pore_fraction"], **result["checks"]} for result in results]
    with (args.output_dir / "final_preprocessing_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    passed = sum(item["passed"] for item in results)
    lines = ["# Frozen evaluation crop preprocessing QC", "", f"**Result: {passed}/{len(results)} crops passed.**", "", "| Crop | Core | Split | Result | Pore fraction |", "|---|---|---|---:|---:|"]
    lines += [f"| {item['crop_id']} | {item['core_id']} | {item['split']} | {'PASS' if item['passed'] else 'FAIL'} | {item['pore_fraction']:.6f} |" for item in results]
    lines += ["", "All checks: shape, dtype, finite values, [0,1] range, same-coordinate grayscale/binary relationship, observed indices, observed-slice data consistency, and 189 primary slices.", ""]
    (args.output_dir / "final_preprocessing_qc.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"QC SUMMARY: {passed}/{len(results)} passed; failures: {len(results) - passed}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
