"""Read-only Berea source audit; the legacy command name is retained."""
import argparse
import json
from pathlib import Path
from theme26.data.audit import audit_core
from theme26.data.protocol import default_split,source_directory

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    entry=default_split()['splits']['train'][0]
    result,_=audit_core(source_directory(args.data_root,entry))
    # Directory name is not a reliable core ID when files are in Downloads.
    result['core_id']='Berea'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Berea source audit: {result['status']}; failures={result['failures']}")
    if result['status']!='pass':
        raise SystemExit(1)

if __name__=='__main__':
    main()
