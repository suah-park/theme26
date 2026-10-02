"""AMI cache generation and frozen E1 reconstruction of held-out crops."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import yaml
from theme26.data import read_raw_volume, extract_volume_crop, normalize_grayscale, make_sparse_sample, compose_full_volume
from theme26.models.saint.inference import reconstruct_directions, fuse_directions
from theme26.training.directional_cache import generate_cache
from theme26.training.workflow import build_stage_model, config_hash, file_hash, read_berea_context, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen theme26 SAINT inference")
    parser.add_argument("--config", type=Path, required=True, help="Path to a versioned inference config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint")
    parser.add_argument("--ami-checkpoint", type=Path, help="AMI checkpoint used to train RFN")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--crop-id", help="One crop within the configured held-out split")
    return parser


def _load_model(path, stage, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["experiment"]["stage"] != stage:
        raise ValueError(f"expected {stage} checkpoint")
    model = build_stage_model(checkpoint["config"], stage)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("range_policy") != "raw_intermediate_clip_final":
        raise ValueError("inference must explicitly select raw_intermediate_clip_final")
    requested = args.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = resolve_path(args.output_dir or config["output_dir"])
    batch_size = int(config["batch_size"])
    use_amp = bool(config["mixed_precision"])
    if config["stage"] == "ami-cache":
        if args.crop_id or args.ami_checkpoint:
            raise ValueError("cache includes the complete train/validation manifest")
        ami, checkpoint = _load_model(args.checkpoint, "ami", device)
        data = checkpoint["config"]["data"]
        if data.get("mode") != "berea_manifest":
            raise ValueError("cache requires a Berea manifest AMI checkpoint")
        path = generate_cache(ami, args.checkpoint, data, destination,
                              device=device, batch_size=batch_size, use_amp=use_amp)
        print(f"cache_manifest={path}")
        return
    if config["stage"] != "reconstruct" or args.ami_checkpoint is None:
        raise ValueError("reconstruct requires --ami-checkpoint and RFN --checkpoint")
    if config.get("split") not in ("validation", "test"):
        raise ValueError("reconstruction split must be validation or test")
    ami, ami_ckpt = _load_model(args.ami_checkpoint, "ami", device)
    rfn, rfn_ckpt = _load_model(args.checkpoint, "rfn", device)
    data = ami_ckpt["config"]["data"]
    rfn_data = rfn_ckpt["config"]["data"]
    if (rfn_data.get("mode") != "berea_rfn_cache" or rfn_data["source"] != data
        or file_hash(resolve_path(rfn_data["ami_checkpoint"])) != file_hash(args.ami_checkpoint)):
        raise ValueError("RFN was trained with a different AMI checkpoint or source protocol")
    raw_path, metadata, records, normalization, protocol_id = read_berea_context(data)
    source_hash = file_hash(raw_path)
    for checkpoint in (ami_ckpt, rfn_ckpt):
        provenance = checkpoint.get("run_metadata", {})
        if provenance.get("source_raw_sha256") != source_hash or provenance.get("protocol_sha256") != protocol_id:
            raise ValueError("inference source differs from training checkpoint")
    selected = [r for r in records if r.split == config["split"] and (args.crop_id is None or r.sample_id == args.crop_id)]
    if not selected:
        raise ValueError("no matching held-out crop")
    if destination.exists():
        raise FileExistsError("prediction directory exists; choose a new output directory")
    raw = read_raw_volume(raw_path, metadata=metadata)
    destination.mkdir(parents=True)
    summaries = []
    for index, record in enumerate(selected):
        dense = normalize_grayscale(extract_volume_crop(raw, crop=record.crop),
                                    lower=normalization["lower"], upper=normalization["upper"])
        sample = make_sparse_sample(dense, sampling=record.sampling)
        sag, cor = reconstruct_directions(ami, sample.sparse_volume, sampling=record.sampling,
                                         spacing_um_xyz=metadata.voxel_spacing_um_xyz,
                                         batch_size=batch_size, device=device, use_amp=use_amp)
        learned = fuse_directions(rfn, sag, cor, batch_size=batch_size, device=device, use_amp=use_amp)
        full = compose_full_volume(np.clip(learned, 0, 1), sample)
        crop_dir = destination / f"{index:03d}"
        crop_dir.mkdir()
        np.save(crop_dir / "prediction.npy", full, allow_pickle=False)
        np.save(crop_dir / "primary_mask.npy", sample.primary_metric_mask, allow_pickle=False)
        summary = {"crop_id": record.sample_id, "split": record.split,
                   "origin_xyz": list(record.crop.origin_xyz), "shape_xyz": list(full.shape),
                   "axis_order": "XYZ", "dtype": "float32", "range": [0, 1],
                   "observed_reinserted": True, "primary_mae": float(np.abs(full - dense)[sample.primary_metric_mask].mean()),
                   "primary_voxels": int(sample.primary_metric_mask.sum()),
                   "sampling_factor": record.sampling.factor, "sampling_offset": record.sampling.offset,
                   "voxel_spacing_um_xyz": list(metadata.voxel_spacing_um_xyz),
                   "normalization_id": metadata.normalization_id,
                   "ami_checkpoint_sha256": file_hash(args.ami_checkpoint),
                   "rfn_checkpoint_sha256": file_hash(args.checkpoint),
                   "source_raw_sha256": source_hash, "protocol_sha256": protocol_id,
                   "inference_config_sha256": config_hash(config)}
        (crop_dir / "metadata.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        summaries.append(summary)
        print(f"crop={record.sample_id} primary_mae={summary['primary_mae']:.6f}", flush=True)
    (destination / "manifest.yaml").write_text(yaml.safe_dump({"complete": True, "records": summaries}, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
