"""Refit canonical train-only XYZ normalization and synchronize training exports."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import json
from theme26.data.protocol import fit_normalization,eval_manifest
from theme26.data.manifest import write_training_protocol
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grayscale',type=Path,required=True)
    p.add_argument('--config-dir',type=Path,default=ROOT/'config')
    p.add_argument('--training-config-dir',type=Path,default=ROOT/'configs/data')
    p.add_argument('--chunk-x',type=int,default=16)
    args=p.parse_args()
    config=json.loads((args.config_dir/'splits.json').read_text(encoding='utf-8'))
    if args.grayscale.name != config['splits']['train'][0]['grayscale_filtered_filename']:
        raise ValueError('grayscale filename differs from canonical protocol')
    norm=fit_normalization(config,args.grayscale.parent,chunk_x=args.chunk_x)
    evaluation=eval_manifest(config,norm)
    for name,value in [('normalization',norm),('eval_crop_manifest',evaluation)]:
        (args.config_dir/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    write_training_protocol(args.training_config_dir,config,norm,evaluation)
    print(f"XYZ train normalization: {norm['lower']}..{norm['upper']} ({norm['histogram_voxels']} voxels)")
if __name__=='__main__':
    main()
