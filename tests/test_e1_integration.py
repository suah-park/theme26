"""Staged cache, primary metric and epoch-boundary restart regression tests."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data import CropRecord, make_sparse_sample, compose_full_volume
from theme26.models.saint import AnisotropicMetaInterpolator, ResidualFusionNetwork, SaintReconstructor, build_filter_distance_matrix
from theme26.models.saint.inference import reconstruct_directions, fuse_directions
from theme26.training import train_one_epoch, validate_one_epoch
from theme26.training.directional_cache import generate_cache, validate_cache, CachedRfnDataset
from theme26.training.workflow import file_hash

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def test_chunked_inference_matches_whole_forward(self):
        torch.manual_seed(2)
        model = SaintReconstructor(factor=4, feature_channels=4,
            ami_num_blocks=1, ami_layers_per_block=1, ami_growth_rate=2,
            rfn_num_blocks=1, rfn_layers_per_block=1, rfn_growth_rate=2).eval()
        sparse = np.random.default_rng(2).random((4, 5, 3), dtype=np.float32)
        sag, cor = reconstruct_directions(model.ami, sparse, sampling=SamplingSpec(),
            spacing_um_xyz=(1,1,1), batch_size=2, device=torch.device("cpu"), use_amp=False)
        fused = fuse_directions(model.rfn, sag, cor, batch_size=2, device=torch.device("cpu"), use_amp=False)
        with torch.no_grad():
            expected = model(torch.from_numpy(sparse)[None,None],
                build_filter_distance_matrix(factor=4,kernel_size=3,hr_spacing_um_xyz=(1,1,1),marginal_axis="sagittal"))
        np.testing.assert_allclose(sag, expected.sagittal[0,0].numpy(), atol=1e-6)
        np.testing.assert_allclose(cor, expected.coronal[0,0].numpy(), atol=1e-6)
        np.testing.assert_allclose(fused, expected.prediction[0,0].numpy(), atol=1e-6)

    def test_primary_metric_clips_and_ignores_nonprimary(self):
        class Predictor(torch.nn.Module):
            def forward(self, inputs, fdm):
                return inputs
        batch = {"input": torch.tensor([[[[2.,-1.,0.2,0.3]]]]),
                 "target": torch.zeros(1,1,1,4), "fdm": torch.zeros(1),
                 "primary_mask": torch.tensor([[[[True,True,False,False]]]])}
        result = validate_one_epoch(Predictor(), [batch], stage="ami", device=torch.device("cpu"), use_amp=False)
        self.assertAlmostEqual(result.loss, 0.875)
        self.assertAlmostEqual(result.primary_mae, 0.5)
        self.assertEqual(result.primary_voxels, 2)

    def test_real_raw_cache_rfn_and_final_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.random.default_rng(5).integers(0,256,(8,5,12), dtype=np.uint8)
            raw.tofile(root/"source.raw")
            metadata = VolumeMetadata("tiny","core",raw.shape,(1,1,1),"uint8","not-applicable", "tiny-normalization")
            records = tuple(CropRecord("tiny","crop-"+split,"core",split,CropSpec((x,0,0),(4,5,12)),SamplingSpec())
                            for split,x in (("train",0),("validation",4)))
            normalization={"lower":0,"upper":255}
            data={"mode":"berea_manifest"}
            context=(root/"source.raw",metadata,records,normalization,"tiny-protocol")
            ami=AnisotropicMetaInterpolator(feature_channels=4,num_blocks=1,layers_per_block=1,growth_rate=2)
            from theme26.training.digital_rock import ManifestAmiDataset
            ami_dataset=ManifestAmiDataset.from_raw(root/"source.raw",records=records[:1],metadata=metadata,
                                                   normalization_lower=0,normalization_upper=255)
            train_one_epoch(ami,DataLoader(ami_dataset,batch_size=2),torch.optim.Adam(ami.parameters()),
                            stage="ami",device=torch.device("cpu"),max_steps=1,use_amp=False)
            architecture={"feature_channels":4,"num_blocks":1,"layers_per_block":1,"growth_rate":2}
            torch.save({"config":{"experiment":{"stage":"ami"},"data":data,"model":dict(architecture,factor=4)},
                        "model_state":ami.state_dict(),
                        "run_metadata":{"source_raw_sha256":file_hash(root/"source.raw"),"protocol_sha256":"tiny-protocol"}},root/"ami.pt")
            with patch("theme26.training.directional_cache.read_berea_context", return_value=context):
                generate_cache(ami,root/"ami.pt",data,root/"cache",device=torch.device("cpu"),batch_size=2,use_amp=False)
                array, all_records, norms, entries = validate_cache(data,root/"cache",root/"ami.pt")
                train=CachedRfnDataset(array,all_records,norms,entries,root/"cache",split="train")
                val=CachedRfnDataset(array,all_records,norms,entries,root/"cache",split="validation")
                self.assertEqual(len(train),12)
                self.assertEqual(len(val),6)
                rfn=ResidualFusionNetwork(feature_channels=4,num_blocks=1,layers_per_block=1,growth_rate=2)
                train_one_epoch(rfn,DataLoader(train,batch_size=2),torch.optim.Adam(rfn.parameters()),stage="rfn",device=torch.device("cpu"),max_steps=1,use_amp=False)
                result=validate_one_epoch(rfn,DataLoader(val,batch_size=2),stage="rfn",device=torch.device("cpu"),use_amp=False)
                self.assertEqual(result.primary_voxels,4*5*6)
                self.assertIsNotNone(result.primary_mae)
                sag,cor=train.arrays[records[0].sample_id]
                prediction=fuse_directions(rfn,sag,cor,batch_size=2,device=torch.device("cpu"),use_amp=False)
                dense=raw[:4].astype(np.float32)/255
                sample=make_sparse_sample(dense,sampling=SamplingSpec())
                final=compose_full_volume(np.clip(prediction,0,1),sample)
                np.testing.assert_array_equal(final[...,::4],dense[...,::4])
                self.assertTrue(np.all((final>=0)&(final<=1)))
                torch.save({"config":{"experiment":{"stage":"rfn"},"model":architecture,
                    "data":{"mode":"berea_rfn_cache","source":data,"ami_checkpoint":str(root/"ami.pt")}},
                    "model_state":rfn.state_dict(),"run_metadata":{"source_raw_sha256":file_hash(root/"source.raw"),
                    "protocol_sha256":"tiny-protocol"}},root/"rfn.pt")
                spec=importlib.util.spec_from_file_location("inference_test_cli",ROOT/"scripts/infer.py")
                entry=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(entry)
                inference={"stage":"reconstruct","split":"validation","range_policy":"raw_intermediate_clip_final",
                           "batch_size":2,"mixed_precision":False,"output_dir":str(root/"prediction")}
                (root/"infer.yaml").write_text(yaml.safe_dump(inference),encoding="utf-8")
                with patch.object(entry,"read_berea_context",return_value=context), patch.object(sys,"argv",[
                    "infer.py","--config",str(root/"infer.yaml"),"--checkpoint",str(root/"rfn.pt"),
                    "--ami-checkpoint",str(root/"ami.pt"),"--device","cpu"]):
                    entry.main()
                saved=np.load(root/"prediction/000/prediction.npy")
                np.testing.assert_array_equal(saved[...,::4],raw[4:].astype(np.float32)[...,::4]/255)
                self.assertTrue(yaml.safe_load((root/"prediction/manifest.yaml").read_text())["complete"])
                with self.assertRaises(FileExistsError):
                    generate_cache(ami,root/"ami.pt",data,root/"cache",device=torch.device("cpu"),batch_size=2,use_amp=False)
                torch.save({"different":True},root/"ami.pt")
                with self.assertRaises(ValueError):
                    validate_cache(data,root/"cache",root/"ami.pt")
                train.close()
                val.close()
                array._mmap.close()
                del ami_dataset

    def test_resume_equals_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={"experiment":{"id":"resume-test","stage":"ami","seed":7},
                    "model":{"factor":4,"feature_channels":4,"kernel_size":3,"num_blocks":1,"layers_per_block":1,"growth_rate":2},
                    "data":{"mode":"synthetic","length":5,"z_lr":3,"train_samples":4,"validation_samples":3},
                    "training":{"batch_size":2,"epochs":2,"learning_rate":0.001,"mixed_precision":False}}
            path=root/"config.yaml"
            path.write_text(yaml.safe_dump(config),encoding="utf-8")
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            base=[sys.executable,"-B",str(ROOT/"scripts/train.py"),"--config",str(path),"--device","cpu"]
            def run(args):
                result=subprocess.run(base+args,cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run(["--output-dir",str(root/"full")])
            run(["--output-dir",str(root/"resumed"),"--epochs","1"])
            run(["--output-dir",str(root/"resumed"),"--resume",str(root/"resumed/last.pt"),"--epochs","2"])
            full=torch.load(root/"full/last.pt",weights_only=False)
            resumed=torch.load(root/"resumed/last.pt",weights_only=False)
            for key in full["model_state"]:
                torch.testing.assert_close(full["model_state"][key],resumed["model_state"][key],rtol=0,atol=0)
            self.assertEqual(full["best_validation_loss"],resumed["best_validation_loss"])
            self.assertEqual([x["validation_primary_mae"] for x in full["run_metadata"]["history"]],
                             [x["validation_primary_mae"] for x in resumed["run_metadata"]["history"]])


if __name__ == "__main__":
    unittest.main()
