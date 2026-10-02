"""Verify a saved Berea artifact against the frozen spatial protocol."""
import argparse
import json
from pathlib import Path
from theme26.data.protocol import load_protocol
from theme26.data.qc import inspect_artifact

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,default=Path('config'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config,norm,manifest=load_protocol(args.config_dir)
    report,_=inspect_artifact(args.input,config,norm,manifest)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'berea_preprocessing_qc.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
