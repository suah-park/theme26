"""Fit normalization using only the configured Berea train region."""
import argparse
import json
from pathlib import Path
from theme26.data.protocol import default_split, fit_normalization, eval_manifest
from theme26.data.manifest import write_training_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Berea RAW folder or its parent')
    parser.add_argument('--config-dir', type=Path, default=Path('config'))
    parser.add_argument('--training-config-dir', type=Path, default=Path('configs/data'))
    args = parser.parse_args()
    path = args.config_dir / 'splits.json'
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else default_split()
    norm = fit_normalization(config, args.data_root)
    manifest = eval_manifest(config, norm)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for name,value in [('splits',config),('normalization',norm),('eval_crop_manifest',manifest)]:
        (args.config_dir/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    write_training_protocol(args.training_config_dir, config, norm, manifest)
    print(f"Frozen train-region normalization: {norm['lower']}..{norm['upper']}; {norm['histogram_voxels']} voxels")
    print(f"Fixed validation/test crops: {len(manifest['crops'])}")


if __name__ == '__main__':
    main()
