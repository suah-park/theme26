"""Berea spatial split, train-only statistics, and artifact provenance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from theme26.contracts import CropSpec


def default_split() -> dict:
    """Half-open XYZ regions; 32-voxel gaps are a fixed design choice."""
    regions = {'train': (0, 424), 'validation': (456, 712), 'test': (744, 1000)}
    return {
        'manifest_version': 'berea-spatial-v1', 'split_mode': 'within-core-spatial',
        'axis_order': 'XYZ', 'source_shape_xyz': [1000, 1000, 1000],
        'dtype': 'uint8', 'endianness': 'not-applicable',
        'voxel_spacing_um_xyz': [2.25, 2.25, 2.25], 'buffer_voxels_x': 32,
        'crop_shape_xyz': [256, 256, 256],
        'splits': {name: [{
            'source_core_id': 'Berea', 'region_id': f'berea_{name}',
            'grayscale_filtered_filename': 'Berea_2d25um_grayscale_filtered.raw',
            'binary_filename': 'Berea_2d25um_binary.raw', 'source_pore_value': 0,
            'region_origin_xyz': [start, 0, 0], 'region_shape_xyz': [end-start, 1000, 1000],
        }] for name, (start, end) in regions.items()},
    }


def region_spec(entry: dict) -> CropSpec:
    return CropSpec(tuple(entry['region_origin_xyz']), tuple(entry['region_shape_xyz']))


def contains(region: CropSpec, crop: CropSpec) -> bool:
    return all(r <= c and c+s <= r+t for r,t,c,s in zip(region.origin_xyz,region.shape_xyz,crop.origin_xyz,crop.shape_xyz))


def validate_split(config: dict) -> None:
    if config.get('split_mode') != 'within-core-spatial' or config.get('axis_order') != 'XYZ':
        raise ValueError('expected a within-core-spatial XYZ split')
    entries = []
    for split in ('train', 'validation', 'test'):
        if not config['splits'].get(split):
            raise ValueError(f'missing {split} region')
        for entry in config['splits'][split]:
            roi = region_spec(entry)
            if any(not isinstance(v,int) for v in (*roi.origin_xyz,*roi.shape_xyz)):
                raise ValueError('region coordinates must be integers')
            roi.validate_inside(tuple(config['source_shape_xyz']))
            if any(a < b for a,b in zip(roi.shape_xyz,config['crop_shape_xyz'])):
                raise ValueError('region must contain a full model crop')
            if entry['source_pore_value'] not in (0,1):
                raise ValueError('invalid binary polarity')
            entries.append((entry,roi))
    if {entry['source_core_id'] for entry,_ in entries} != {'Berea'}:
        raise ValueError('primary protocol uses Berea only')
    for i,(a,ra) in enumerate(entries):
        for b,rb in entries[i+1:]:
            if all(max(x,y) < min(x+s,y+t) for x,s,y,t in zip(ra.origin_xyz,ra.shape_xyz,rb.origin_xyz,rb.shape_xyz)):
                raise ValueError(f"overlapping split regions: {a['region_id']} / {b['region_id']}")


def source_directory(root: Path, entry: dict) -> Path:
    """Accept either a Berea directory itself or its parent data root."""
    return root if (root / entry['grayscale_filtered_filename']).is_file() else root / entry['source_core_id']


def fit_normalization(config: dict, data_root: Path, *, chunk_x: int = 16) -> dict:
    validate_split(config)
    if chunk_x <= 0:
        raise ValueError('chunk_x must be positive')
    histogram = np.zeros(256,dtype=np.int64)
    sources = []
    shape = tuple(config['source_shape_xyz'])
    for entry in config['splits']['train']:
        path = source_directory(data_root,entry) / entry['grayscale_filtered_filename']
        if path.stat().st_size != int(np.prod(shape)):
            raise ValueError(f'{path}: unexpected RAW size')
        raw = np.memmap(path,dtype=np.uint8,mode='r',shape=shape,order='C')
        roi = region_spec(entry)
        x,y,z = roi.origin_xyz; sx,sy,sz = roi.shape_xyz
        for start in range(x,x+sx,chunk_x):
            histogram += np.bincount(np.asarray(raw[start:min(start+chunk_x,x+sx),y:y+sy,z:z+sz]).ravel(),minlength=256)
        sources.append({'source_core_id':entry['source_core_id'],'region_id':entry['region_id'],
                        'origin_xyz':list(roi.origin_xyz),'shape_xyz':list(roi.shape_xyz)})
        del raw
    count = int(histogram.sum())
    def percentile(p):
        rank = int(np.ceil(p/100*(count-1)))
        return int(np.searchsorted(np.cumsum(histogram),rank+1))
    lower,upper = percentile(.5),percentile(99.5)
    if lower >= upper:
        raise ValueError('train region has degenerate normalization bounds')
    return {'normalization_id':'berea-train-region-p0.5-p99.5-v1','status':'frozen',
            'split_manifest_version':config['manifest_version'], 'lower':lower,'upper':upper,
            'dtype':'float32','output_range':[0.,1.], 'histogram_voxels':count,
            'method':'train-region uint8 histogram; nearest-rank p0.5/p99.5 clipping and scaling',
            'train_regions':sources}


def eval_manifest(config: dict, normalization: dict) -> dict:
    validate_split(config)
    crops = []
    size = tuple(config['crop_shape_xyz'])
    for split in ('validation','test'):
        for entry in config['splits'][split]:
            roi = region_spec(entry)
            for i,(dy,dz) in enumerate(((64,64),(680,64),(64,680),(680,680))):
                crop = CropSpec((roi.origin_xyz[0],roi.origin_xyz[1]+dy,roi.origin_xyz[2]+dz),size)
                if not contains(roi,crop):
                    raise ValueError('fixed evaluation crop exceeds region')
                crops.append({'crop_id':f'berea_{split}_{i:02d}','source_core_id':'Berea',
                              'region_id':entry['region_id'],'split':split,
                              'origin_xyz':list(crop.origin_xyz),'shape_xyz':list(size),
                              'axis_order':'XYZ','sparse_axis':'Z','sampling_factor':4,'sampling_offset':0,
                              'voxel_spacing_um_xyz':config['voxel_spacing_um_xyz']})
    return {'manifest_version':'berea-spatial-eval-v1','split_manifest_version':config['manifest_version'],
            'normalization_id':normalization['normalization_id'],'crops':crops}


def load_protocol(config_dir: Path) -> tuple[dict,dict,dict]:
    split = json.loads((config_dir/'splits.json').read_text(encoding='utf-8'))
    norm = json.loads((config_dir/'normalization.json').read_text(encoding='utf-8'))
    manifest = json.loads((config_dir/'eval_crop_manifest.json').read_text(encoding='utf-8'))
    validate_split(split)
    if norm.get('status') != 'frozen' or norm.get('lower') is None or norm.get('upper') is None:
        raise ValueError('normalization is not frozen; run scripts/freeze_berea_protocol.py first')
    if not np.isfinite([norm['lower'],norm['upper']]).all() or norm['lower'] >= norm['upper']:
        raise ValueError('invalid normalization bounds')
    if manifest != eval_manifest(split,norm) or norm.get('split_manifest_version') != split['manifest_version']:
        raise ValueError('stale protocol metadata or evaluation manifest; freeze again')
    expected = [{'source_core_id':e['source_core_id'],'region_id':e['region_id'],
                 'origin_xyz':e['region_origin_xyz'],'shape_xyz':e['region_shape_xyz']}
                for e in split['splits']['train']]
    if norm.get('train_regions') != expected:
        raise ValueError('normalization was fitted to different train regions')
    return split,norm,manifest
