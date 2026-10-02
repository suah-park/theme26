"""Export training CSV from the canonical frozen XYZ protocol."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from theme26.data.protocol import load_protocol
from theme26.data.manifest import records_from_protocol, validate_protocol_records, write_crop_manifest
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'configs/data/berea_spatial_split_v1.csv')
    p.add_argument('--config-dir',type=Path,default=ROOT/'config')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--train-crops',type=int,default=48)
    args=p.parse_args()
    config,norm,evaluation=load_protocol(args.config_dir)
    records=records_from_protocol(config,evaluation,seed=args.seed,train_crops=args.train_crops)
    validate_protocol_records(records,config,evaluation)
    write_crop_manifest(args.output,records)
    print({s:sum(r.split==s for r in records) for s in ('train','validation','test')})
if __name__=='__main__':
    main()
