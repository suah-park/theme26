"""Run the read-only phase-1 audit for all 11 Sandstones cores."""

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

from theme26.data.audit import audit_core, discover_cores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit of 11 Sandstones filtered grayscale/binary RAW pairs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def _save_qc_image(core_id: str, crop: tuple[np.ndarray, np.ndarray] | None, result: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if crop is None:
        for axis in axes:
            axis.axis("off")
        axes[1].text(0.5, 0.5, "AUDIT FAILED\n" + "\n".join(result["failures"]), ha="center", va="center", wrap=True)
    else:
        grayscale, binary = crop
        z = grayscale.shape[2] // 2
        image = grayscale[:, :, z].T
        pore_value = result["binary"]["source_pore_value_inferred"]
        pore = (binary[:, :, z].T == pore_value)
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[0].set_title(f"{core_id}: grayscale z={z}")
        axes[1].imshow(pore, cmap="viridis", vmin=0, vmax=1, origin="lower")
        axes[1].set_title(f"Pore mask (source value={pore_value})")
        axes[2].imshow(image, cmap="gray", vmin=0, vmax=255, origin="lower")
        axes[2].contour(pore.astype(np.uint8), levels=[0.5], colors="#ff2d2d", linewidths=0.55, origin="lower")
        axes[2].set_title("Pore boundary over grayscale")
        for axis in axes:
            axis.set_xlabel("X")
            axis.set_ylabel("Y")
    figure.suptitle(f"{core_id} — {'PASS' if result['status'] == 'pass' else 'FAIL'}")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"core_id": result["core_id"], "status": result["status"], "failures": " | ".join(result["failures"])}
    if result["status"] == "pass":
        row.update({
            "grayscale_file": result["files"]["grayscale_filtered"]["name"],
            "binary_file": result["files"]["binary"]["name"],
            "grayscale_bytes": result["files"]["grayscale_filtered"]["bytes"],
            "binary_bytes": result["files"]["binary"]["bytes"],
            "source_pore_value": result["binary"]["source_pore_value_inferred"],
            "pore_fraction": result["binary"]["pore_fraction"],
            "grayscale_min": result["grayscale"]["min"], "grayscale_max": result["grayscale"]["max"],
            "grayscale_mean": result["grayscale"]["mean"],
            **{f"p{key}": value for key, value in result["grayscale"]["percentiles"].items()},
            "alignment_contrast": result["alignment_qc"]["global_contrast_solid_minus_pore"],
        })
    return row


def _write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    passed = sum(result["status"] == "pass" for result in results)
    lines = [
        "# 11 Sandstones data audit", "",
        "This is a read-only phase-1 audit. No split, normalization constants, or crop artifacts were selected.", "",
        f"**Result: {passed}/{len(results)} cores passed.**", "",
        "| Core | Status | Pore source value | Pore fraction | Gray min/max/mean | p0.5 / p1 / p99 / p99.5 | Alignment contrast |", 
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for result in results:
        if result["status"] == "pass":
            gray, binary, alignment = result["grayscale"], result["binary"], result["alignment_qc"]
            percentiles = gray["percentiles"]
            lines.append(
                f"| {result['core_id']} | PASS | {binary['source_pore_value_inferred']} | {binary['pore_fraction']:.6f} | "
                f"{gray['min']} / {gray['max']} / {gray['mean']:.3f} | {percentiles['0.5']} / {percentiles['1.0']} / {percentiles['99.0']} / {percentiles['99.5']} | "
                f"{alignment['global_contrast_solid_minus_pore']:.3f} |"
            )
        else:
            lines.append(f"| {result['core_id']} | FAIL | — | — | — | — | — |")
            lines.append(f"\nFailure: {'; '.join(result['failures'])}\n")
    lines.extend(["", "Each core's same-coordinate grayscale/mask QC visualization is in `polarity_qc/`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = args.output_dir / "polarity_qc"
    png_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for core in discover_cores(args.data_root):
        result, crop = audit_core(core)
        results.append(result)
        _save_qc_image(core.name, crop, result, png_dir / f"{core.name.replace(' ', '_')}_polarity_qc.png")
        print(f"[{result['status'].upper()}] {core.name}")

    csv_rows = [_csv_row(result) for result in results]
    fields = sorted({key for row in csv_rows for key in row})
    with (args.output_dir / "data_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "data_audit.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_markdown(results, args.output_dir / "data_audit.md")
    passed = sum(result["status"] == "pass" for result in results)
    failures = [result["core_id"] for result in results if result["status"] != "pass"]
    print(f"AUDIT SUMMARY: {passed}/{len(results)} cores passed; failures: {', '.join(failures) if failures else 'none'}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
