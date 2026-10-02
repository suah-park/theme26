"""Check every frozen Berea validation/test crop without writing bulk arrays."""
import argparse
import json
from pathlib import Path
from theme26.contracts import CropSpec, SamplingSpec
from theme26.data.protocol import load_protocol
from theme26.data.dataset import preprocess_crop,source_from_entry
from theme26.data.qc import crop_checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,default=Path('config'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    parser.add_argument('--plots',action='store_true',help='Requires matplotlib; optional PNGs')
    args=parser.parse_args()
    config,norm,manifest=load_protocol(args.config_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for entry in manifest['crops']:
        src_entry=next(e for e in config['splits'][entry['split']] if e['region_id']==entry['region_id'])
        source=source_from_entry(args.data_root,config,entry['split'],src_entry,norm['normalization_id'])
        crop=preprocess_crop(source,crop=CropSpec(tuple(entry['origin_xyz']),tuple(entry['shape_xyz'])),
                             normalization_lower=norm['lower'],normalization_upper=norm['upper'],
                             sampling=SamplingSpec(entry['sampling_factor'],entry['sampling_offset']))
        checks=crop_checks(crop,entry)
        row={'crop_id':entry['crop_id'],'split':entry['split'],'origin_xyz':entry['origin_xyz'],
             'checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),
             'pore_fraction':float(crop.pore_mask.mean())}
        results.append(row)
        print(f"{entry['crop_id']}: {'PASS' if row['passed'] else 'FAIL'}",flush=True)
        if args.plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(8,4))
            axes[0].imshow(crop.grayscale[:,:,128].T,cmap='gray',vmin=0,vmax=1)
            axes[1].imshow(crop.pore_mask[:,:,128].T,cmap='gray',vmin=0,vmax=1)
            fig.suptitle(entry['crop_id']); fig.savefig(args.output_dir/f"{entry['crop_id']}.png"); plt.close(fig)
        del crop
    (args.output_dir/'final_preprocessing_qc.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    passed=sum(r['passed'] for r in results)
    (args.output_dir/'final_preprocessing_qc.md').write_text(f'# Berea spatial preprocessing QC\n\n{passed}/{len(results)} fixed crops passed.\n',encoding='utf-8')
    print(f'QC: {passed}/{len(results)}')
    if passed!=len(results):
        raise SystemExit(1)

if __name__=='__main__':
    main()
