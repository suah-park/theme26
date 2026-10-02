"""Command-line checks owned by the data package.

Run with ``python -m theme26.data audit ...`` or
``python -m theme26.data qc ...``.  Training reads RAW files lazily, so no
separate preprocessing export step is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from theme26.contracts import CropSpec, SamplingSpec

from .audit import audit_core
from .dataset import preprocess_crop, source_from_entry
from .protocol import load_protocol, source_directory
from .qc import crop_checks


def _protocol_dir(value: str) -> Path:
    return Path(value)


def _audit(args: argparse.Namespace) -> None:
    config, _, _ = load_protocol(args.config_dir)
    entry = config["splits"]["train"][0]
    result, _ = audit_core(source_directory(args.data_root, entry))
    result["core_id"] = entry["source_core_id"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Berea source audit: {result['status']}; failures={result['failures']}")
    if result["status"] != "pass":
        raise SystemExit(1)


def _qc(args: argparse.Namespace) -> None:
    config, normalization, evaluation = load_protocol(args.config_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in evaluation["crops"]:
        source_entry = next(
            item
            for item in config["splits"][entry["split"]]
            if item["region_id"] == entry["region_id"]
        )
        source = source_from_entry(
            args.data_root,
            config,
            entry["split"],
            source_entry,
            normalization["normalization_id"],
        )
        crop = preprocess_crop(
            source,
            crop=CropSpec(tuple(entry["origin_xyz"]), tuple(entry["shape_xyz"])),
            normalization_lower=normalization["lower"],
            normalization_upper=normalization["upper"],
            sampling=SamplingSpec(
                entry["sampling_factor"], entry["sampling_offset"]
            ),
        )
        checks = crop_checks(crop, entry)
        row = {
            "crop_id": entry["crop_id"],
            "split": entry["split"],
            "origin_xyz": entry["origin_xyz"],
            "checks": {name: bool(value) for name, value in checks.items()},
            "passed": all(checks.values()),
            "pore_fraction": float(crop.pore_mask.mean()),
        }
        results.append(row)
        print(f"{entry['crop_id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)

    report_path = args.output_dir / "final_preprocessing_qc.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(result["passed"] for result in results)
    print(f"QC: {passed}/{len(results)}")
    if passed != len(results):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(handler=None)
    subparsers = parser.add_subparsers(dest="command")

    for name, handler, help_text in (
        ("audit", _audit, "audit the paired grayscale and binary RAW files"),
        ("qc", _qc, "check all frozen validation and test crops"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument(
            "--config-dir", type=_protocol_dir, default=Path("configs/data")
        )
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
        command.set_defaults(handler=handler)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.handler is None:
        parser.print_help()
        raise SystemExit(2)
    args.handler(args)

