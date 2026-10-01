"""Create visual and numeric QC artifacts for a Berea preprocessing .npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPECTED_SHAPE = (256, 256, 256)
EXPECTED_SPARSE_SHAPE = (256, 256, 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC a contract-compliant Berea preprocessing artifact")
    parser.add_argument("--input", type=Path, required=True, help="Input berea_crop.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for QC PNG, JSON, and Markdown")
    parser.add_argument("--z-index", type=int, default=128, help="Axial Z index to render")
    return parser


def _check(condition: bool, message: str, checks: dict[str, bool]) -> None:
    checks[message] = bool(condition)


def inspect_artifact(path: Path, *, z_index: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}

    required = {
        "grayscale", "pore_mask", "sparse_volume", "observed_indices", "observed_mask",
        "missing_mask", "primary_metric_mask", "e0_linear", "crop_origin_xyz",
        "sampling_factor", "sampling_offset", "normalization_lower", "normalization_upper",
        "source_pore_value",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"artifact is missing required arrays: {missing}")
    if not 0 <= z_index < arrays["grayscale"].shape[2]:
        raise ValueError(f"z-index {z_index} is outside grayscale depth {arrays['grayscale'].shape[2]}")

    gray = arrays["grayscale"]
    pore = arrays["pore_mask"]
    sparse = arrays["sparse_volume"]
    observed_indices = arrays["observed_indices"]
    observed = arrays["observed_mask"]
    missing_mask = arrays["missing_mask"]
    primary = arrays["primary_metric_mask"]
    e0 = arrays["e0_linear"]
    checks: dict[str, bool] = {}

    _check(gray.shape == EXPECTED_SHAPE, "grayscale shape is (256, 256, 256)", checks)
    _check(pore.shape == EXPECTED_SHAPE, "pore mask shape is (256, 256, 256)", checks)
    _check(e0.shape == EXPECTED_SHAPE, "E0 shape is (256, 256, 256)", checks)
    _check(sparse.shape == EXPECTED_SPARSE_SHAPE, "sparse shape is (256, 256, 64)", checks)
    _check(all(mask.shape == EXPECTED_SHAPE for mask in (observed, missing_mask, primary)), "all HR masks have shape (256, 256, 256)", checks)
    _check(gray.dtype == np.float32 and e0.dtype == np.float32 and sparse.dtype == np.float32, "grayscale, sparse, and E0 are float32", checks)
    _check(pore.dtype == np.bool_ and observed.dtype == np.bool_ and missing_mask.dtype == np.bool_ and primary.dtype == np.bool_, "pore and sampling masks are bool", checks)
    _check(bool(np.isfinite(gray).all() and np.isfinite(sparse).all() and np.isfinite(e0).all()), "no NaN or Inf in grayscale, sparse, or E0", checks)
    _check(bool(np.all((gray >= 0.0) & (gray <= 1.0)) and np.all((sparse >= 0.0) & (sparse <= 1.0)) and np.all((e0 >= 0.0) & (e0 <= 1.0))), "grayscale, sparse, and E0 are within [0, 1]", checks)

    expected_indices = np.arange(0, 256, 4, dtype=observed_indices.dtype)
    _check(np.array_equal(observed_indices, expected_indices), "observed indices are z=0,4,...,252", checks)
    _check(np.array_equal(sparse, gray[..., observed_indices]), "sparse observed slices exactly match original grayscale", checks)
    _check(np.array_equal(e0[..., observed_indices], gray[..., observed_indices]), "E0 observed slices exactly match original grayscale", checks)
    expected_observed = np.broadcast_to(
        np.isin(np.arange(256)[None, None, :], observed_indices), EXPECTED_SHAPE
    )
    _check(np.array_equal(observed, expected_observed), "observed mask exactly matches observed indices", checks)
    _check(bool(np.array_equal(missing_mask, ~observed)), "missing mask is the complement of observed mask", checks)
    expected_primary = np.zeros(EXPECTED_SHAPE, dtype=bool)
    expected_primary[..., 1:252] = ~np.isin(np.arange(1, 252), observed_indices)[None, None, :]
    _check(np.array_equal(primary, expected_primary), "primary mask contains only 189 bracketed missing slices", checks)

    report: dict[str, object] = {
        "input": str(path),
        "z_index_rendered": z_index,
        "checks": checks,
        "passed": all(checks.values()),
        "statistics": {
            "grayscale_min": float(gray.min()), "grayscale_max": float(gray.max()),
            "grayscale_mean": float(gray.mean()), "pore_fraction": float(pore.mean()),
            "observed_voxels": int(observed.sum()), "missing_voxels": int(missing_mask.sum()),
            "primary_metric_voxels": int(primary.sum()), "observed_slices": int(observed_indices.size),
            "primary_metric_slices": int(primary[0, 0].sum()),
        },
    }
    return report, arrays


def save_figure(arrays: dict[str, np.ndarray], *, z_index: int, path: Path) -> None:
    grayscale = arrays["grayscale"][:, :, z_index].T
    pore = arrays["pore_mask"][:, :, z_index].T
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
    axes[0].set_title(f"Grayscale (z={z_index})")
    axes[1].imshow(pore, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("Binary pore mask (pore=1)")
    axes[2].imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
    axes[2].contour(pore.astype(np.uint8), levels=[0.5], colors="#ff2d2d", linewidths=0.55, origin="lower")
    axes[2].set_title("Pore boundary over grayscale")
    for axis in axes:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_markdown(report: dict[str, object], path: Path, figure_path: Path) -> None:
    checks = report["checks"]
    statistics = report["statistics"]
    assert isinstance(checks, dict) and isinstance(statistics, dict)
    lines = [
        "# Berea preprocessing QC report", "",
        f"- Input: `{report['input']}`",
        f"- Rendered axial slice: `z={report['z_index_rendered']}`",
        f"- Overall result: **{'PASS' if report['passed'] else 'FAIL'}**", "",
        "## Contract checks", "",
    ]
    lines.extend(f"- [{'PASS' if result else 'FAIL'}] {name}" for name, result in checks.items())
    lines.extend(["", "## Statistics", ""])
    lines.extend(f"- {name}: `{value}`" for name, value in statistics.items())
    lines.extend(["", f"Visualization: `{figure_path.name}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = inspect_artifact(args.input, z_index=args.z_index)
    figure_path = args.output_dir / "berea_preprocessing_qc_z128.png"
    json_path = args.output_dir / "berea_preprocessing_qc.json"
    markdown_path = args.output_dir / "berea_preprocessing_qc.md"
    save_figure(arrays, z_index=args.z_index, path=figure_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_markdown(report, markdown_path, figure_path)
    if not report["passed"]:
        raise SystemExit("QC failed; see report for failed checks")
    print(f"QC passed: {markdown_path}")


if __name__ == "__main__":
    main()
