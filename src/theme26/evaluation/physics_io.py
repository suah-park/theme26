"""Use existing inference files and data APIs; never load or train a checkpoint."""
from __future__ import annotations
import hashlib
from pathlib import Path
import math
import numpy as np
import yaml
from .physics_metrics import paired_scalar_metrics
from .yang_lbm import result_scalars
from .yang_lbm.result import SolveResult
from .yang_lbm.reporting import write_json, write_csv
from .yang_lbm.validation import validate_pore, validate_spacing


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    return value


def segmentation_settings(data):
    s = data.get('segmentation', {})
    t = s.get('threshold')
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('Set the team-frozen segmentation.threshold in the evaluation YAML')
    if type(s.get('pore_is_dark')) is not bool:
        raise ValueError('segmentation.pore_is_dark must be an explicit boolean')
    if not isinstance(s.get('version'), str) or not s['version'].strip():
        raise ValueError('Set segmentation.version; this is provenance, not solver approval')
    return s


def segment_full_volume(full, *, threshold, pore_is_dark):
    """Input is already clipped and observation-preserving, as infer.py saves it."""
    if not isinstance(full, np.ndarray) or full.ndim != 3 or full.dtype != np.float32:
        raise TypeError('Saved inference must be float32 [X,Y,Z] grayscale')
    if not np.isfinite(full).all() or full.min() < 0 or full.max() > 1:
        raise ValueError('Expected final clipped inference output, not raw AMI/RFN cache')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Explicit frozen threshold in [0,1] required')
    if type(pore_is_dark) is not bool:
        raise TypeError('pore_is_dark must be bool')
    return np.ascontiguousarray(full < threshold if pore_is_dark else full > threshold)


def _solve_volume(pore, spacing, solver):
    """Preserve invalid segmentation as a reported case, not a fictitious K=0."""
    try:
        validate_pore(pore)
        if not pore.any():
            raise ValueError('Empty pore phase is invalid at the public evaluation boundary')
    except (TypeError, ValueError) as exc:
        cfg = getattr(solver, 'config', None)
        result = SolveResult(status='invalid_input', message=str(exc), spacing_um_xyz=spacing,
                             shape_xyz=getattr(pore, 'shape', None),
                             solver_config_hash=getattr(cfg, 'fingerprint', None))
        return result_scalars(result), result.as_dict()
    if hasattr(solver, 'solve_detailed'):
        result = solver.solve_detailed(pore.copy(), spacing_um_xyz=spacing)
        flat = result_scalars(result)
        flat.update(solver_name=solver.name, solver_version=solver.version)
        return flat, result.as_dict()
    result = solver.solve(pore.copy(), spacing_um_xyz=spacing)
    return result, result


def evaluate_pair_arrays(predicted_pore, target_pore, *, spacing_um_xyz, solver):
    """Return a flat public comparison plus full per-volume diagnostics/history."""
    if not isinstance(predicted_pore, np.ndarray) or not isinstance(target_pore, np.ndarray):
        raise TypeError('Expected NumPy arrays')
    if predicted_pore.shape != target_pore.shape:
        raise ValueError('Reference/prediction shape mismatch')
    spacing = validate_spacing(spacing_um_xyz)
    r, rd = _solve_volume(target_pore, spacing, solver)
    p, pd = _solve_volume(predicted_pore, spacing, solver)
    row = paired_scalar_metrics(p, r)
    return row, {'reference': rd, 'prediction': pd, 'comparison': row}


def save_rows(directory, rows):
    directory = Path(directory)
    preferred = ['crop_id', 'split', 'experiment_id', 'seed', 'comparison_status',
                 'reference_kz_mD', 'prediction_kz_mD', 'relative_error_pct',
                 'signed_relative_error_pct', 'kz_log10_abs_error', 'porosity_abs_error']
    columns = list(dict.fromkeys(preferred + sorted({k for r in rows for k in r})))
    write_json(directory / 'results.json', rows)
    write_csv(directory / 'results.csv', rows, columns=columns)


def evaluate_binary_files(prediction_path, reference_path, *, spacing_um_xyz, solver,
                          output_dir, experiment_id, seed):
    p = np.load(prediction_path, allow_pickle=False)
    r = np.load(reference_path, allow_pickle=False)
    if p.shape != (256, 256, 256) or r.shape != p.shape:
        raise ValueError('Project file evaluation requires aligned 256^3 binary arrays')
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    row, detail = evaluate_pair_arrays(p, r, spacing_um_xyz=spacing_um_xyz, solver=solver)
    row.update(experiment_id=experiment_id, seed=seed,
               alignment_basis='caller-provided binary pair',
               prediction_file_sha256=file_sha256(prediction_path),
               reference_file_sha256=file_sha256(reference_path))
    out.mkdir(parents=True)
    write_json(out / 'case.json', detail)
    save_rows(out, [row])
    return [row]


def read_inference_records(prediction_dir):
    """Match scripts/infer.py at base commit 8e7f092; do not guess file order."""
    root = Path(prediction_dir)
    manifest = read_yaml(root / 'manifest.yaml')
    records = manifest.get('records')
    if manifest.get('complete') is not True or not isinstance(records, list) or not records:
        raise ValueError('Expected a complete nonempty inference manifest.yaml')
    seen = set()
    for index, meta in enumerate(records):
        if not isinstance(meta, dict):
            raise ValueError('Invalid inference record')
        if meta.get('crop_id') in seen or not isinstance(meta.get('crop_id'), str):
            raise ValueError('Missing or duplicate crop_id')
        seen.add(meta['crop_id'])
        if (meta.get('axis_order') != 'XYZ' or meta.get('dtype') != 'float32'
                or meta.get('shape_xyz') != [256, 256, 256]
                or meta.get('range') != [0, 1]
                or meta.get('observed_reinserted') is not True):
            raise ValueError('Metadata is not a final observation-preserving XYZ inference')
        if meta.get('split') not in ('validation', 'test'):
            raise ValueError('Only held-out inference is supported')
        if meta.get('sampling_factor') != 4 or type(meta.get('sampling_offset')) is not int or not 0 <= meta['sampling_offset'] < 4:
            raise ValueError('Expected declared x4 sampling')
        origin = meta.get('origin_xyz')
        if not isinstance(origin, list) or len(origin) != 3 or any(type(v) is not int or v < 0 for v in origin):
            raise ValueError('Invalid origin_xyz')
        validate_spacing(meta.get('voxel_spacing_um_xyz'))
        for key in ('normalization_id', 'source_raw_sha256', 'protocol_sha256',
                    'ami_checkpoint_sha256', 'rfn_checkpoint_sha256', 'inference_config_sha256'):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f'Missing inference provenance: {key}')
        directory = root / f'{index:03d}'
        if read_yaml(directory / 'metadata.yaml') != meta:
            raise ValueError('Crop metadata.yaml and inference manifest disagree')
        if not (directory / 'prediction.npy').is_file():
            raise FileNotFoundError(directory / 'prediction.npy')
    return records


def _read_reference(binary_raw, reference_config, first_record):
    # Reuse existing project loader and metadata; do not duplicate RAW conventions.
    from theme26.contracts import VolumeMetadata
    from theme26.data.raw_volume import read_raw_volume
    if reference_config.get('dtype') != 'uint8' or reference_config.get('source_pore_value') not in (0, 1):
        raise ValueError('Reference YAML must declare uint8 source and pore label 0 or 1')
    metadata = VolumeMetadata(
        sample_id=reference_config['core_id'], source_core_id=reference_config['core_id'],
        shape_xyz=tuple(reference_config['shape_xyz']),
        voxel_spacing_um_xyz=tuple(first_record['voxel_spacing_um_xyz']),
        dtype='uint8', endianness='not-applicable')
    return read_raw_volume(binary_raw, metadata=metadata)


def _reference_crop(raw, meta, pore_value):
    from theme26.contracts import CropSpec
    from theme26.data.crop_sampler import extract_volume_crop
    from theme26.data.raw_volume import to_pore_mask
    crop = CropSpec(origin_xyz=tuple(meta['origin_xyz']), shape_xyz=tuple(meta['shape_xyz']))
    return to_pore_mask(extract_volume_crop(raw, crop=crop), source_pore_value=pore_value)


def evaluate_inference_directory(prediction_dir, binary_raw, *, settings, solver,
                                 output_dir, experiment_id, seed):
    """Consume native infer.py outputs without rerunning inference or training."""
    seg = segmentation_settings(settings)
    reference_config = settings.get('reference', {})
    records = read_inference_records(prediction_dir)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    raw = _read_reference(binary_raw, reference_config, records[0])
    binary_hash = file_sha256(binary_raw)
    out.mkdir(parents=True)
    rows = []
    (out / 'evaluation_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    for index, meta in enumerate(records):
        path = Path(prediction_dir) / f'{index:03d}' / 'prediction.npy'
        full = np.load(path, allow_pickle=False)
        if list(full.shape) != meta['shape_xyz']:
            raise ValueError('Saved prediction shape differs from metadata')
        pore = segment_full_volume(full, threshold=seg['threshold'], pore_is_dark=seg['pore_is_dark'])
        target = _reference_crop(raw, meta, reference_config['source_pore_value'])
        row, detail = evaluate_pair_arrays(pore, target, spacing_um_xyz=meta['voxel_spacing_um_xyz'], solver=solver)
        row.update(meta)
        row.update(experiment_id=experiment_id, seed=seed, core_id=reference_config['core_id'],
                   segmentation_protocol_id=seg['version'], segmentation_threshold=seg['threshold'],
                   pore_is_dark=seg['pore_is_dark'], source_binary_sha256=binary_hash,
                   prediction_file_sha256=file_sha256(path))
        write_json(out / 'cases' / f'{index:03d}.json', {**detail, 'metadata': meta})
        rows.append(row)
        save_rows(out, rows)
    return rows
