"""Software/IO tests only: no real LBM, Taichi initialization or GPU allocation."""
from pathlib import Path
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from theme26.evaluation.physics_metrics import compute_physics_metrics, paired_scalar_metrics
from theme26.evaluation.physics_io import (
    evaluate_pair_arrays, read_inference_records, segment_full_volume,
    segmentation_settings, evaluate_inference_directory,
)
from theme26.evaluation.yang_lbm import LBMConfig, YangLBMSolver, result_scalars
from theme26.evaluation.yang_lbm.result import SolveResult
from theme26.evaluation.yang_lbm.statistics import RunStatistics, history_statistics
from theme26.evaluation.yang_lbm.permeability import permeability_from_mean, M2_PER_MD
from theme26.evaluation.yang_lbm.cache import load_cache, write_cache, cache_key
from theme26.evaluation.physics_cli import main

ROOT = Path(__file__).resolve().parents[1]
HAS_SCIPY = importlib.util.find_spec('scipy') is not None


def result(k=1e-12, status='completed'):
    return SolveResult(status=status, k_m2=k, k_mD=k/M2_PER_MD, flow_axis='z',
        solver_config_hash='fixture-config', core_sha256='fixture-core',
        shape_xyz=(6,6,6), spacing_um_xyz=(2.25,)*3)


class MockSolver:
    name = 'mock_not_physical'
    version = '1'
    def __init__(self): self.calls = 0
    def solve(self, pore, *, spacing_um_xyz):
        self.calls += 1
        r = result(self.calls * 1e-12)
        r.shape_xyz = pore.shape
        r.spacing_um_xyz = spacing_um_xyz
        return result_scalars(r)


class FP64EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pore = np.zeros((6,6,6), dtype=bool)
        self.pore[2:4,2:4,:] = True

    def test_final_config_no_gates(self):
        c = LBMConfig.from_yaml(ROOT/'configs/evaluation/physics_fp64.yaml')
        self.assertEqual(c.precision, 'float64')
        self.assertEqual(c.max_steps, 50000)
        self.assertEqual(c.force_vector, (0.,0.,1e-6))
        self.assertFalse({'authorization_path','relative_tolerance','allow_unverified'} & set(c.as_dict()))

    def test_invalid_settings(self):
        for kw in ({'precision':'float32'}, {'max_steps':0}, {'axis':'time'},
                   {'boundary':'pressure'}, {'force_density':float('nan')}, {'isolation':'false'}):
            with self.subTest(kw=kw), self.assertRaises((ValueError,TypeError)):
                LBMConfig(**kw)

    def test_descriptive_statistics_no_acceptance(self):
        stats = RunStatistics(LBMConfig())
        r = stats.update(1, {'mean_velocity':1., 'mass_balance_diagnostic':999.})
        self.assertNotIn('passed', r)
        self.assertEqual(stats.update(2, {'mean_velocity':1.})['window_relative_range'], 0.)

    def test_history_std_is_temporal(self):
        r = history_statistics([{'estimated_k_axis_m2':x*M2_PER_MD} for x in [10,11,12]])
        self.assertAlmostEqual(r['tail_kz_mean_mD'],11.)
        self.assertAlmostEqual(r['tail_kz_std_mD'],np.std([10,11,12]))
        self.assertIn('not an accuracy',r['variation_definition'])

    def test_completion_not_convergence(self):
        r = result_scalars(result())
        self.assertNotIn('converged', r)
        self.assertFalse(r['convergence_assessed'])
        self.assertNotIn('passed', r)

    def test_scalar_contract(self):
        r = compute_physics_metrics(self.pore,self.pore,spacing_um_xyz=(2.25,)*3,solver=MockSolver())
        self.assertTrue(all(isinstance(v,(str,int,float,bool)) for v in r.values()))
        self.assertEqual(r['comparison_status'],'completed_pair')
        self.assertAlmostEqual(r['relative_error_pct'],100.)
        self.assertAlmostEqual(r['kz_log10_abs_error'],np.log10(2.))
        json.dumps(r,allow_nan=False)

    def test_public_empty_phase_contract(self):
        for p in (np.zeros_like(self.pore), np.ones_like(self.pore)):
            with self.subTest(value=bool(p.flat[0])), self.assertRaises(ValueError):
                compute_physics_metrics(p,self.pore,spacing_um_xyz=(2.25,)*3,solver=MockSolver())

    def test_bad_dtype_shape_spacing(self):
        cases=[(self.pore.astype(np.float32),(2.25,)*3),
               (self.pore[:3],(2.25,)*3), (self.pore,(2.25,2.25,9.))]
        for p,s in cases:
            with self.subTest(shape=p.shape,s=s),self.assertRaises((TypeError,ValueError)):
                compute_physics_metrics(p,self.pore,spacing_um_xyz=s,solver=MockSolver())

    def test_mutating_solver_cannot_change_inputs(self):
        class Mutating(MockSolver):
            def solve(self,a,*,spacing_um_xyz):
                a[:]=False
                return super().solve(a,spacing_um_xyz=spacing_um_xyz)
        before=self.pore.copy()
        compute_physics_metrics(self.pore,self.pore,spacing_um_xyz=(2.25,)*3,solver=Mutating())
        np.testing.assert_array_equal(self.pore,before)

    def test_budget_result_is_not_hidden(self):
        r=paired_scalar_metrics(result_scalars(result(2e-12,'budget_exhausted')), result_scalars(result()))
        self.assertEqual(r['comparison_status'],'budget_limited_pair')
        self.assertEqual(r['relative_error_pct'],100.)

    def test_nonpercolating_is_not_failure(self):
        r=paired_scalar_metrics(result_scalars(result(0.,'non_percolating')),result_scalars(result()))
        self.assertEqual(r['comparison_status'],'percolation_failure')
        self.assertNotIn('log10_error',r)
        self.assertEqual(r['relative_error_pct'],100.)

    def test_numeric_error_not_zero(self):
        r=paired_scalar_metrics(result_scalars(SolveResult(status='numerical_error')),result_scalars(result()))
        self.assertEqual(r['comparison_status'],'numerical_error')
        self.assertNotIn('prediction_kz_m2',r)
        self.assertNotIn('relative_error_pct',r)

    def test_mixed_settings_rejected(self):
        p=result_scalars(result());r=result_scalars(result());p['solver_config_hash']='different'
        self.assertEqual(paired_scalar_metrics(p,r)['comparison_status'],'incompatible_solver')

    def test_nested_solver_result_rejected(self):
        with self.assertRaises(TypeError):
            paired_scalar_metrics({'status':'completed','nested':{}},result_scalars(result()))

    def test_nonfinite_solver_result_rejected(self):
        p=result_scalars(result());p['k_m2']=float('nan')
        with self.assertRaises(ValueError): paired_scalar_metrics(p,result_scalars(result()))

    def test_cache_does_not_require_quality_gate(self):
        with tempfile.TemporaryDirectory() as d:
            r=result();r.diagnostics={'mass_balance_diagnostic':.026}
            write_cache(d,'a',r);cached=load_cache(d,'a')
            self.assertTrue(cached.cache_hit)
            self.assertFalse(cached.numerical_solve_performed)
            self.assertIsNone(cached.converged)
            self.assertEqual(cached.k_m2,r.k_m2)

    def test_cache_budget_separate(self):
        a=cache_key('a',(2.25,)*3,LBMConfig(),'source',{})
        b=cache_key('a',(2.25,)*3,LBMConfig(max_steps=60000),'source',{})
        self.assertNotEqual(a,b)

    def test_spacing_squared(self):
        a=permeability_from_mean(1e-6,nu=.1,force_density=1e-6,spacing_um=2.25)
        b=permeability_from_mean(1e-6,nu=.1,force_density=1e-6,spacing_um=4.5)
        self.assertEqual(b['k_m2'],4*a['k_m2'])

    def test_segmentation_does_not_clip_raw_output(self):
        p=np.ones((6,6,6),np.float32)*1.1
        with self.assertRaises(ValueError):segment_full_volume(p,threshold=.5,pore_is_dark=True)

    def test_segmentation_polarity(self):
        p=np.where(self.pore,.2,.8).astype(np.float32)
        np.testing.assert_array_equal(segment_full_volume(p,threshold=.5,pore_is_dark=True),self.pore)
        np.testing.assert_array_equal(segment_full_volume(1-p,threshold=.5,pore_is_dark=False),self.pore)

    def test_segmentation_not_invented(self):
        settings=yaml.safe_load((ROOT/'configs/evaluation/physics_fp64.yaml').read_text())
        with self.assertRaises(ValueError):segmentation_settings(settings)
        settings['segmentation']={'threshold':.5,'pore_is_dark':True,'version':'fixture'}
        self.assertEqual(segmentation_settings(settings)['threshold'],.5)

    def test_invalid_segmentation_saved(self):
        r,detail=evaluate_pair_arrays(np.zeros_like(self.pore),self.pore,spacing_um_xyz=(2.25,)*3,solver=MockSolver())
        self.assertEqual(r['prediction_status'],'invalid_input')
        self.assertNotIn('prediction_kz_m2',r)
        self.assertIn('reference',detail)

    def test_adapter_conforms_without_running_taichi(self):
        with patch('theme26.evaluation.yang_lbm.solver.YangLBMSolver.solve',return_value=result()):
            r=YangLBMSolver().solve(self.pore,spacing_um_xyz=(2.25,)*3)
            self.assertIsInstance(r,dict)
            self.assertIn('solver_version',r)
            self.assertIn('kz_mD',r)

    @unittest.skipUnless(HAS_SCIPY,'optional scipy not installed')
    def test_periodic_winding(self):
        from theme26.evaluation.yang_lbm.connectivity import periodic_winding
        self.assertTrue(periodic_winding(self.pore,2))
        self.pore[:,:,3]=False
        self.assertFalse(periodic_winding(self.pore,2))

    @unittest.skipUnless(HAS_SCIPY,'optional scipy not installed')
    def test_engine_same_budget_no_taichi(self):
        from theme26.evaluation.yang_lbm import engine
        class Field:
            def __init__(self,a):self.a=a
            def from_numpy(self,a):self.a=a
            def to_numpy(self):return self.a
        class Core:
            def __init__(self,nx,ny,nz,nu,force,precision):
                assert precision=='float64'
                self.tau_f=3*nu+.5;self.solid=Field(None)
                self.v=Field(np.zeros((nx,ny,nz,3),np.float64));self.v.a[...,2]=1e-6
                self.rho=Field(np.ones((nx,ny,nz),np.float64))
            def init(self):pass
            def step(self):pass
        name='theme26.evaluation.yang_lbm._vendor.periodic_core'
        fake=types.ModuleType(name);fake.LB3D_Solver_Single_Phase=Core
        ti=types.SimpleNamespace(__version__=(1,7,4),sync=lambda:None,reset=lambda:None)
        with patch.dict(sys.modules,{name:fake}),patch.object(engine,'init_runtime',return_value=ti):
            r=engine.run_case(self.pore,(2.25,)*3,LBMConfig(max_steps=3,check_every=1,history_window=2,isolation=False))
        self.assertEqual(r.status,'completed');self.assertEqual(r.iterations,3)
        self.assertIsNone(r.converged)
        self.assertTrue(all('passed' not in x for x in r.history))

    def test_reference_report_no_run(self):
        stream=io.StringIO()
        with redirect_stdout(stream): self.assertEqual(main(['reference-report']),0)
        r=json.loads(stream.getvalue())
        self.assertAlmostEqual(r['kz_mD'],444.7208435554732)
        self.assertFalse(r['same_quantity_confirmed'])
        self.assertGreater(r['absolute_arithmetic_difference_pct'],43.)

    def test_entrypoint_and_import_side_effects(self):
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        code='import sys;from theme26.evaluation.yang_lbm import YangLBMSolver;assert "taichi" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],env=env,check=True)
        run=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_physics.py'),'doctor'],env=env,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(run.stdout)['precision'],'float64')

    def test_worker_module_after_relocation(self):
        from theme26.evaluation.yang_lbm import solver as module
        def fake_run(argv, **kwargs):
            self.assertEqual(argv[2], 'theme26.evaluation.yang_lbm.worker')
            self.assertIn(str(ROOT/'src'), kwargs['env']['PYTHONPATH'])
            req=json.loads(Path(argv[3]).read_text())
            np.testing.assert_array_equal(np.load(req['input']),self.pore)
            Path(req['output']).write_text(json.dumps(result().as_dict()))
            return types.SimpleNamespace(returncode=0,stdout='',stderr='')
        environment={'taichi_installed':True,'packages':{'taichi':'1.7.4'}}
        with patch.object(module,'doctor',return_value=environment),patch.object(module.subprocess,'run',side_effect=fake_run):
            r=YangLBMSolver(LBMConfig()).solve(self.pore,spacing_um_xyz=(2.25,)*3)
        self.assertEqual(r['status'],'completed')

    def test_native_inference_format_bridge(self):
        # Matches the files and metadata written by scripts/infer.py. Mock solver
        # and source loader avoid a real rock input/flow run; arithmetic is not science.
        meta=dict(crop_id='fixture',split='test',origin_xyz=[720,0,0],shape_xyz=[256]*3,
            axis_order='XYZ',dtype='float32',range=[0,1],observed_reinserted=True,
            primary_mae=.01,primary_voxels=256*256*189,sampling_factor=4,sampling_offset=0,
            voxel_spacing_um_xyz=[2.25]*3,normalization_id='fixture',source_raw_sha256='source',
            protocol_sha256='protocol',ami_checkpoint_sha256='ami',rfn_checkpoint_sha256='rfn',
            inference_config_sha256='inference')
        settings={'segmentation':{'threshold':.5,'pore_is_dark':True,'version':'fixture'},
                  'reference':{'core_id':'Berea','source_pore_value':0}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/'prediction';(folder/'000').mkdir(parents=True)
            (folder/'manifest.yaml').write_text(yaml.safe_dump({'complete':True,'records':[meta]}))
            (folder/'000/metadata.yaml').write_text(yaml.safe_dump(meta))
            full=np.full((256,256,256),.8,np.float32);full[100:110,100:110,:]=.2
            np.save(folder/'000/prediction.npy',full,allow_pickle=False)
            target=full<.5;raw=root/'reference.raw';raw.write_bytes(b'fixture')
            with patch('theme26.evaluation.physics_io._read_reference',return_value=None),patch('theme26.evaluation.physics_io._reference_crop',return_value=target):
                rows=evaluate_inference_directory(folder,raw,settings=settings,solver=MockSolver(),output_dir=root/'out',experiment_id='E1',seed=1)
            self.assertEqual(rows[0]['comparison_status'],'completed_pair')
            self.assertEqual(rows[0]['rfn_checkpoint_sha256'],'rfn')
            self.assertEqual(rows[0]['primary_voxels'],256*256*189)
            self.assertTrue((root/'out/results.csv').exists())
            self.assertIn('rfn_checkpoint_sha256',(root/'out/results.csv').read_text(encoding='utf-8-sig'))
            meta['observed_reinserted']=False
            (folder/'manifest.yaml').write_text(yaml.safe_dump({'complete':True,'records':[meta]}))
            with self.assertRaises(ValueError):read_inference_records(folder)


if __name__ == '__main__':
    unittest.main()
