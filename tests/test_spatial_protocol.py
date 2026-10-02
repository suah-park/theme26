"""Leakage and regression checks using tiny RAW fixtures, not real Berea."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from theme26.contracts import CropSpec,SamplingSpec
from theme26.data.protocol import default_split,validate_split,fit_normalization,eval_manifest,contains,region_spec
from theme26.data.dataset import CoreSource,TrainRandomCropDataset,preprocess_crop
from theme26.data.raw_volume import to_pore_mask
from theme26.data.qc import save_artifact,inspect_artifact


class SpatialTests(unittest.TestCase):
    def test_default_regions_and_eval_blocks_do_not_overlap(self):
        config=default_split(); validate_split(config)
        manifest=eval_manifest(config,{'normalization_id':'test'})
        self.assertEqual(len(manifest['crops']),18)
        crops=[]
        for entry in manifest['crops']:
            roi=region_spec(config['splits'][entry['split']][0])
            crop=CropSpec(tuple(entry['origin_xyz']),tuple(entry['shape_xyz']))
            self.assertTrue(contains(roi,crop)); crops.append(crop)
        for i,a in enumerate(crops):
            for b in crops[i+1:]:
                self.assertFalse(all(max(x,y)<min(x+s,y+t) for x,s,y,t in zip(a.origin_xyz,a.shape_xyz,b.origin_xyz,b.shape_xyz)))

    def test_overlapping_regions_are_rejected(self):
        config=default_split()
        config['splits']['validation'][0]['region_origin_xyz'][0]=399
        with self.assertRaisesRegex(ValueError,'overlapping'):
            validate_split(config)

    def test_normalization_ignores_eval_and_buffer_voxels(self):
        config=default_split(); config['source_shape_xyz']=[40,8,8]; config['crop_shape_xyz']=[4,4,4]
        for name,(start,width) in {'train':(0,12),'validation':(16,12),'test':(32,8)}.items():
            entry=config['splits'][name][0]
            entry['region_origin_xyz']=[start,0,0]; entry['region_shape_xyz']=[width,8,8]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); entry=config['splits']['train'][0]
            raw=np.zeros((40,8,8),dtype=np.uint8)
            raw[:12]=np.arange(12*8*8,dtype=np.uint16).reshape(12,8,8)%101
            path=root/entry['grayscale_filtered_filename']; raw.tofile(path)
            first=fit_normalization(config,root,chunk_x=3)
            raw[12:]=255; raw.tofile(path)
            second=fit_normalization(config,root,chunk_x=2)
            self.assertEqual(first,second)
            self.assertEqual(first['histogram_voxels'],12*8*8)

    def test_sampling_stays_in_train_region_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shape=(12,8,8)
            gray=np.arange(np.prod(shape),dtype=np.uint16).reshape(shape).astype(np.uint8)
            binary=(gray>80).astype(np.uint8)
            gp,bp=root/'g.raw',root/'b.raw'; gray.tofile(gp); binary.tofile(bp)
            roi=CropSpec((2,0,0),(6,8,8))
            source=CoreSource('Berea','train',gp,bp,0,shape,roi,'berea_train')
            dataset=TrainRandomCropDataset([source],normalization_lower=0,normalization_upper=255,crop_shape_xyz=(4,4,4))
            first=dataset.sample(np.random.default_rng(7)); second=dataset.sample(np.random.default_rng(7))
            self.assertEqual(first.crop,second.crop)
            rng=np.random.default_rng(3)
            for _ in range(100):
                self.assertTrue(contains(roi,dataset.sample(rng).crop))
            with self.assertRaisesRegex(ValueError,'boundary'):
                preprocess_crop(source,crop=CropSpec((0,0,0),(4,4,4)),normalization_lower=0,normalization_upper=255)

    def test_train_requires_region_and_rejects_eval(self):
        for source in (CoreSource('Berea','train',Path('g'),Path('b')),
                       CoreSource('Berea','test',Path('g'),Path('b'))):
            with self.assertRaises(ValueError):
                TrainRandomCropDataset([source],normalization_lower=0,normalization_upper=255)

    def test_single_phase_masks_are_valid(self):
        for value in (0,1):
            raw=np.full((4,4,4),value,dtype=np.uint8)
            mask=to_pore_mask(raw,source_pore_value=0)
            self.assertEqual(bool(mask.all()),value==0)
        with self.assertRaises(ValueError):
            to_pore_mask(np.full((4,4,4),2,dtype=np.uint8),source_pore_value=0)

    def test_artifact_qc_rejects_wrong_metadata_and_missing_e0(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shape=(4,4,8)
            gray=np.broadcast_to(np.arange(8,dtype=np.uint8)*30,shape).copy()
            binary=(gray>90).astype(np.uint8)
            gp,bp=root/'g.raw',root/'b.raw'; gray.tofile(gp); binary.tofile(bp)
            source=CoreSource('Berea','validation',gp,bp,0,shape,CropSpec((0,0,0),shape),'berea_validation')
            crop=preprocess_crop(source,crop=CropSpec((0,0,0),shape),normalization_lower=0,normalization_upper=255)
            config=default_split(); norm={'lower':0,'upper':255,'normalization_id':'tiny'}
            entry={'crop_id':'tiny','split':'validation','source_core_id':'Berea','region_id':'berea_validation',
                   'origin_xyz':[0,0,0],'shape_xyz':list(shape),'sampling_factor':4,'sampling_offset':0}
            manifest={'crops':[entry]}; path=root/'crop.npz'
            save_artifact(path,crop,entry,config,norm,0)
            report,arrays=inspect_artifact(path,config,norm,manifest)
            self.assertTrue(report['passed'])
            arrays['sampling_factor']=np.asarray(99)
            arrays['normalization_lower']=np.asarray(250)
            arrays['e0_linear'][...,1]=0
            np.savez(path,**arrays)
            report,_=inspect_artifact(path,config,norm,manifest)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['e0'])
            self.assertFalse(report['checks']['metadata_sampling_factor'])
            self.assertFalse(report['checks']['metadata_normalization_lower'])

if __name__=='__main__':
    unittest.main()
