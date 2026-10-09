"""Repository-native entry point for fixed-budget FP64 post-training evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
from importlib.resources import files
import json
import os
import yaml
from pathlib import Path
from .yang_lbm import LBMConfig, YangLBMSolver
from .yang_lbm.runtime import doctor
from .physics_io import read_yaml, evaluate_binary_files, evaluate_inference_directory

ERROR_STATUSES = {'invalid_input', 'missing_input', 'dependency_unavailable',
                  'backend_unavailable', 'out_of_memory', 'timeout', 'numerical_error'}
DEFAULT_CONFIG = 'configs/evaluation/physics_fp64.yaml'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Inspect packages only; no GPU or fluid run')
    sub.add_parser('reference-report', help='Read the supplied historical FP64 record; no run')
    pair = sub.add_parser('evaluate-pair', help='Aligned bool XYZ 256^3 .npy files')
    pair.add_argument('--prediction', required=True)
    pair.add_argument('--reference', required=True)
    pair.add_argument('--spacing-um', type=float, nargs=3, required=True)
    infer = sub.add_parser('evaluate-inference', help='Read existing scripts/infer.py output')
    infer.add_argument('--prediction-dir', required=True)
    infer.add_argument('--binary-raw', required=True)
    for command in (pair, infer):
        command.add_argument('--config', default=DEFAULT_CONFIG)
        command.add_argument('--output-dir', required=True)
        command.add_argument('--experiment-id', required=True)
        command.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            print(json.dumps({**doctor(), 'precision': 'float64'}, indent=2))
            return 0
        if args.command == 'reference-report':
            print(files('theme26.evaluation.yang_lbm').joinpath('recorded_reference.json').read_text(encoding='utf-8'))
            return 0
        settings = read_yaml(args.config)
        config = LBMConfig.from_yaml(args.config)
        if config.axis != 'z':
            raise ValueError('Project physics evaluation requires z flow')
        os.environ.setdefault('THEME26_LBM_PROGRESS', '1')
        solver = YangLBMSolver(config)
        common = dict(solver=solver, output_dir=args.output_dir,
                      experiment_id=args.experiment_id, seed=args.seed)
        if args.command == 'evaluate-pair':
            rows = evaluate_binary_files(args.prediction, args.reference,
                                         spacing_um_xyz=args.spacing_um, **common)
            Path(args.output_dir, 'evaluation_config.yaml').write_text(
                yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        else:
            rows = evaluate_inference_directory(args.prediction_dir, args.binary_raw,
                                                 settings=settings, **common)
        print(json.dumps({'results': str(Path(args.output_dir) / 'results.csv'),
                          'rows': len(rows), 'status_counts': dict(Counter(r['comparison_status'] for r in rows))}, indent=2))
        invalid = any(r.get(prefix + 'status') in ERROR_STATUSES
                      for r in rows for prefix in ('prediction_', 'reference_'))
        invalid = invalid or any(r.get('comparison_status') in
                                {'incompatible_solver', 'invalid_estimate_record'} for r in rows)
        return 2 if invalid else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'status': 'error', 'message': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
