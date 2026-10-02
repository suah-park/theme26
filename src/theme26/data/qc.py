"""Shared numeric QC of frozen crops and saved artifact provenance."""
from pathlib import Path
import numpy as np
from theme26.contracts import SamplingSpec
from theme26.data.sparse_sampling import make_sparse_sample, linear_interpolate_z

def crop_checks(crop,entry):
    shape=tuple(entry['shape_xyz'])
    checks={'shape':crop.grayscale.shape==shape and crop.pore_mask.shape==shape and crop.e0_linear.shape==shape,
            'dtype':crop.grayscale.dtype==np.float32 and crop.pore_mask.dtype==np.bool_ and crop.e0_linear.dtype==np.float32}
    reference=make_sparse_sample(crop.grayscale,sampling=SamplingSpec(entry['sampling_factor'],entry['sampling_offset']))
    s=crop.sparse
    checks.update(observed_indices=s.observed_indices==reference.observed_indices,
                  sparse=np.array_equal(s.sparse_volume,reference.sparse_volume),
                  observed_mask=np.array_equal(s.observed_mask,reference.observed_mask),
                  missing_mask=np.array_equal(s.missing_mask,reference.missing_mask),
                  primary_mask=np.array_equal(s.primary_metric_mask,reference.primary_metric_mask),
                  e0=np.allclose(crop.e0_linear,linear_interpolate_z(reference),rtol=0,atol=1e-6))
    checks['finite_range']=bool(np.isfinite(crop.e0_linear).all() and ((crop.e0_linear>=0)&(crop.e0_linear<=1)).all())
    # Homogeneous blocks are valid; intensity contrast applies only if both phases exist.
    if crop.pore_mask.any() and (~crop.pore_mask).any():
        checks['phase_contrast']=float(crop.grayscale[crop.pore_mask].mean())<float(crop.grayscale[~crop.pore_mask].mean())
    return checks

def save_artifact(path,crop,entry,config,norm,source_pore_value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    s=crop.sparse
    np.savez_compressed(path,grayscale=crop.grayscale,pore_mask=crop.pore_mask,
        sparse_volume=s.sparse_volume,observed_indices=np.asarray(s.observed_indices,dtype=np.int32),
        observed_mask=s.observed_mask,missing_mask=s.missing_mask,primary_metric_mask=s.primary_metric_mask,
        e0_linear=crop.e0_linear,crop_origin_xyz=np.asarray(entry['origin_xyz'],dtype=np.int32),
        sampling_factor=entry['sampling_factor'],sampling_offset=entry['sampling_offset'],
        normalization_lower=norm['lower'],normalization_upper=norm['upper'],source_pore_value=source_pore_value,
        crop_id=entry['crop_id'],split=entry['split'],source_core_id=entry['source_core_id'],region_id=entry['region_id'],
        normalization_id=norm['normalization_id'],split_manifest_version=config['manifest_version'],
        voxel_spacing_um_xyz=np.asarray(config['voxel_spacing_um_xyz']))

def inspect_artifact(path,config,norm,manifest):
    from types import SimpleNamespace
    with np.load(path,allow_pickle=False) as archive:
        a={k:archive[k] for k in archive.files}
    crop_id=str(a['crop_id'].item())
    candidates=[e for e in manifest['crops'] if e['crop_id']==crop_id]
    if len(candidates)!=1:
        raise ValueError('artifact crop is not in frozen manifest')
    entry=candidates[0]
    source=next(e for e in config['splits'][entry['split']] if e['region_id']==entry['region_id'])
    expected={'sampling_factor':entry['sampling_factor'],'sampling_offset':entry['sampling_offset'],
              'normalization_lower':norm['lower'],'normalization_upper':norm['upper'],
              'source_pore_value':source['source_pore_value'],'split':entry['split'],
              'source_core_id':entry['source_core_id'],'region_id':entry['region_id'],
              'normalization_id':norm['normalization_id'],'split_manifest_version':config['manifest_version']}
    checks={f'metadata_{key}':a[key].shape==() and a[key].item()==value for key,value in expected.items()}
    checks['origin']=np.array_equal(a['crop_origin_xyz'],entry['origin_xyz'])
    checks['spacing']=np.array_equal(a['voxel_spacing_um_xyz'],config['voxel_spacing_um_xyz'])
    s=SimpleNamespace(sparse_volume=a['sparse_volume'],observed_indices=tuple(a['observed_indices'].tolist()),
                      observed_mask=a['observed_mask'],missing_mask=a['missing_mask'],primary_metric_mask=a['primary_metric_mask'])
    crop=SimpleNamespace(grayscale=a['grayscale'],pore_mask=a['pore_mask'],e0_linear=a['e0_linear'],sparse=s)
    checks.update(crop_checks(crop,entry))
    checks={k:bool(v) for k,v in checks.items()}
    return {'crop_id':crop_id,'checks':checks,'passed':all(checks.values())},a
