"""Same v0.2.2 FP64 fluid equations; fixed iteration/time budget, report only."""
from __future__ import annotations
import time,os,json
from pathlib import Path
import numpy as np
from .result import SolveResult
from .validation import checksum_array
from .runtime import init_runtime,core_fingerprint,UPSTREAM_COMMIT
from .connectivity import periodic_winding
from .permeability import field_diagnostics,permeability_from_mean
from .statistics import RunStatistics


def run_case(pore,spacing,config):
    start=time.perf_counter()
    result=SolveResult(status='numerical_error',porosity=float(pore.mean()),flow_axis=config.axis,
        input_checksum=checksum_array(pore),solver_config_hash=config.fingerprint,
        spacing_um_xyz=spacing,shape_xyz=pore.shape,solver_commit=UPSTREAM_COMMIT,
        core_sha256=core_fingerprint(),precision='float64')
    ti=None
    try:
        axis='xyz'.index(config.axis)
        result.percolates=periodic_winding(pore,axis)
        if not result.percolates:
            result.status='non_percolating';result.k_m2=0.;result.k_mD=0.
            result.stop_reason='no_periodic_winding_path'
            result.message='No winding path in periodic D3Q19 graph; structural zero, not a failed run'
            return result
        if os.environ.get('THEME26_LBM_PROGRESS')=='1':
            print(f'[LBM] {config.backend}/float64 initializing; first JIT can be slow',flush=True)
        ti=init_runtime(config)
        result.backend=config.backend
        result.taichi_version='.'.join(str(v) for v in ti.__version__)
        from ._vendor.periodic_core import LB3D_Solver_Single_Phase
        lbm=LB3D_Solver_Single_Phase(*pore.shape,config.nu_lattice,config.force_vector,precision='float64')
        lbm.solid.from_numpy(np.ascontiguousarray(~pore,dtype=np.int8))
        lbm.init();ti.sync()
        monitor=RunStatistics(config)
        for step in range(1,config.max_steps+1):
            lbm.step();result.iterations=step;result.numerical_solve_performed=True
            if step%config.check_every and step!=config.max_steps:continue
            ti.sync()
            velocity=lbm.v.to_numpy();density=lbm.rho.to_numpy()
            if velocity.dtype!=np.float64 or density.dtype!=np.float64:
                raise TypeError(f'FP64 required; velocity={velocity.dtype}, density={density.dtype}')
            diag=field_diagnostics(velocity,density,pore,axis)
            diag['field_dtype']='float64'
            diag['total_fluid_mass']=float(density[pore].sum(dtype=np.float64))
            diag['relative_total_mass_drift']=abs(diag['total_fluid_mass']/int(pore.sum())-1.)
            del velocity,density
            k=permeability_from_mean(diag['mean_velocity'],nu=config.nu_lattice,
                force_density=config.force_density,spacing_um=spacing[0])
            stats=monitor.update(step,diag)
            elapsed=time.perf_counter()-start
            row={'iteration':step,'elapsed_seconds':elapsed,**diag,
                 'estimated_kz':k['k_m2'] if axis==2 else None,'estimated_k_axis_m2':k['k_m2'],
                 **stats,'backend':config.backend,'precision':'float64'}
            result.history.append(row)
            diag.update(force_density_vector_xyz=list(config.force_vector),tau=lbm.tau_f,
                        history_window=config.history_window)
            result.diagnostics=diag
            result.k_m2=k['k_m2'];result.k_mD=k['k_mD']
            # Optional progress is factual reporting only, never a quality gate.
            if os.environ.get('THEME26_LBM_PROGRESS')=='1':
                span=stats['window_relative_range']
                span_text='n/a' if span is None else f'{100*span:.6g}%'
                print(f'[LBM] {step}/{config.max_steps} | K{config.axis}={k["k_mD"]:.6g} mD | '
                      f'recorded-window range={span_text} | elapsed={elapsed/60:.1f} min',flush=True)
            progress_path=os.environ.get('THEME26_LBM_PROGRESS_FILE')
            if progress_path:
                target=Path(progress_path);target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix('.tmp')
                temp.write_text(json.dumps({'iteration':step,'k_mD':k['k_mD'],**stats,
                                           'elapsed_seconds':elapsed},allow_nan=False),encoding='utf-8')
                temp.replace(target)
            if step==config.max_steps:
                result.status='completed';result.stop_reason='requested_steps_reached';break
            if elapsed>=config.max_seconds:
                result.status='budget_exhausted';result.stop_reason='time_budget_reached';break
        if result.k_m2 is None or not np.isfinite(result.k_m2) or result.k_m2<=0:
            result.status='numerical_error';result.stop_reason='no_positive_finite_estimate'
            result.message='No positive finite estimate was obtained from this percolating case'
        else:
            result.message='Finite estimate recorded. Completion is not a convergence/accuracy certificate.'
        return result
    except (ImportError,ModuleNotFoundError) as exc:
        result.status='dependency_unavailable';result.message=str(exc);return result
    except MemoryError as exc:
        result.status='out_of_memory';result.message=str(exc);return result
    except Exception as exc:
        text=f'{type(exc).__name__}: {exc}'
        result.status=('out_of_memory' if any(x in text.lower() for x in ('out of memory','bad_alloc'))
                       else 'backend_unavailable' if ti is None else 'numerical_error')
        result.message=text;return result
    finally:
        # Never pass off an earlier valid estimate as the completed result after an error.
        if result.status not in ('completed','budget_exhausted','non_percolating'):
            if result.k_m2 is not None:result.diagnostics['last_estimate_before_error_m2']=result.k_m2
            result.k_m2=None;result.k_mD=None
        result.runtime_seconds=time.perf_counter()-start
        if ti is not None:ti.reset()
