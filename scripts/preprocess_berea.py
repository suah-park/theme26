"""Export a fixed evaluation crop through the shared memory-map API."""
import argparse
from pathlib import Path
from theme26.contracts import CropSpec, SamplingSpec
from theme26.data.protocol import load_protocol
from theme26.data.dataset import preprocess_crop, source_from_entry
from theme26.data.qc import save_artifact

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Berea RAW directory or parent')
    parser.add_argument('--config-dir',type=Path,default=Path('config'))
    parser.add_argument('--crop-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config,norm,manifest=load_protocol(args.config_dir)
    entries=[e for e in manifest['crops'] if e['crop_id']==args.crop_id]
    if len(entries)!=1:
        raise ValueError(f'unknown fixed crop ID: {args.crop_id}')
    entry=entries[0]
    source_entry=next(e for e in config['splits'][entry['split']] if e['region_id']==entry['region_id'])
    source=source_from_entry(args.data_dir,config,entry['split'],source_entry,norm['normalization_id'])
    crop=preprocess_crop(source,crop=CropSpec(tuple(entry['origin_xyz']),tuple(entry['shape_xyz'])),
                         normalization_lower=norm['lower'],normalization_upper=norm['upper'],
                         sampling=SamplingSpec(entry['sampling_factor'],entry['sampling_offset']))
    save_artifact(args.output,crop,entry,config,norm,source.source_pore_value)
    print(f'Exported {args.crop_id}: {args.output}')

if __name__=='__main__':
    main()
