"""Public adapter. Default execution uses a fresh Python process per volume.

No C/C++ executable is invoked. This isolates Taichi fields and leaves the
parent process/PyTorch CUDA context untouched. Workers use the same interpreter.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np
from .result import SolveResult
from .validation import validate_pore,validate_spacing,checksum_array
from .config import LBMConfig
from .runtime import doctor,core_fingerprint,UPSTREAM_COMMIT
from .cache import load_cache,write_cache,cache_key,from_dict

class YangLBMSolver:
    name='yang_derived_d3q19_mrt_periodic'
    version='0.3.0'

    def __init__(self,config: LBMConfig | None=None):
        self.config=config or LBMConfig()

    def solve(self,pore_mask,*,spacing_um_xyz):
        start=time.perf_counter()
        result=SolveResult(status='invalid_input',flow_axis=self.config.axis,
            solver_commit=UPSTREAM_COMMIT,core_sha256=core_fingerprint(),
            solver_config_hash=self.config.fingerprint,precision=self.config.precision)
        try:
            a=validate_pore(pore_mask);spacing=validate_spacing(spacing_um_xyz)
        except (ValueError,TypeError) as e:
            result.message=str(e);return result
        result.porosity=float(a.mean());result.input_checksum=checksum_array(a)
        result.spacing_um_xyz=spacing;result.shape_xyz=a.shape
        environment=doctor()
        key=cache_key(result.input_checksum,spacing,self.config,result.core_sha256,environment['packages'])
        cached=load_cache(self.config.cache_dir,key)
        if cached is not None:return cached
        # Topological zero is valid even when Taichi is unavailable.
        if not a.any():
            result.status='non_percolating';result.percolates=False
            result.k_m2=0.;result.k_mD=0.;result.verification_scope='all_solid_input';return result
        if not environment['taichi_installed']:
            result.status='dependency_unavailable'
            result.message='Taichi is not installed in this interpreter. Use the same working environment as the initial test.'
            return result
        if os.environ.get('THEME26_LBM_PROGRESS')=='1':
            print(f'[LBM] start shape={a.shape}, axis={self.config.axis}, FP64, steps={self.config.max_steps}; first compilation may take minutes',flush=True)
        if not self.config.isolation:
            from .engine import run_case
            result=run_case(a.copy(),spacing,self.config)
        else:
            with tempfile.TemporaryDirectory(prefix='theme26_lbm_') as directory:
                root=Path(directory)
                np.save(root/'pore.npy',a,allow_pickle=False)
                (root/'request.json').write_text(json.dumps({'config':self.config.as_dict(),
                    'spacing':spacing,'input':str(root/'pore.npy'),'output':str(root/'response.json')}))
                env=os.environ.copy()
                src=str(Path(__file__).resolve().parents[3])
                env['PYTHONPATH']=src+os.pathsep+env.get('PYTHONPATH','')
                try:
                    run=subprocess.run([sys.executable,'-m','theme26.evaluation.yang_lbm.worker',str(root/'request.json')],
                        env=env,capture_output=True,text=True,timeout=self.config.max_seconds+60,check=False)
                    if run.returncode!=0 or not (root/'response.json').exists():
                        result.status='numerical_error';result.message=f'worker exit={run.returncode}: '+run.stderr[-3000:]
                        result.diagnostics={'worker_stdout_tail':run.stdout[-2000:]}
                    else:
                        result=from_dict(json.loads((root/'response.json').read_text()))
                        result.diagnostics['worker_stdout_tail']=run.stdout[-2000:]
                        result.diagnostics['worker_stderr_tail']=run.stderr[-2000:]
                except subprocess.TimeoutExpired:
                    result.status='timeout';result.message='Worker deadline (including initialization) exceeded'
                except (ValueError,TypeError,KeyError,OSError) as exc:
                    result.status='numerical_error';result.message=f'Invalid worker response: {exc}'
        result.runtime_seconds=time.perf_counter()-start
        if os.environ.get('THEME26_LBM_PROGRESS')=='1':
            print(f'[LBM] end status={result.status}, K{self.config.axis}={result.k_mD} mD, elapsed={result.runtime_seconds/60:.1f} min',flush=True)
        try:
            write_cache(self.config.cache_dir,key,result)
        except OSError as exc:
            result.diagnostics['cache_write_error']=str(exc)
        return result
