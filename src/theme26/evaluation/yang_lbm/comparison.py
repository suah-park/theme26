"""Paired finite-budget estimates; no scientific pass/fail label."""
import math
from .validation import validate_pore,validate_spacing


def relative_metrics(predicted,reference):
    if (isinstance(predicted,bool) or isinstance(reference,bool) or
        not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (predicted,reference)) or
        predicted<0 or reference<=0):
        raise ValueError('finite predicted>=0 and reference>0 required')
    return {'absolute_error_m2':abs(predicted-reference),
            'relative_error_pct':100*abs(predicted-reference)/reference,
            'signed_relative_error_pct':100*(predicted-reference)/reference,
            'log10_error':abs(math.log10(predicted)-math.log10(reference)) if predicted>0 else None}


def compare_results(prediction,reference):
    p=prediction.as_dict() if hasattr(prediction,'as_dict') else prediction
    r=reference.as_dict() if hasattr(reference,'as_dict') else reference
    out={'comparison_status':'unavailable','relative_error_pct':None,'signed_relative_error_pct':None,
         'absolute_error_m2':None,'log10_error':None,'reference_kz_m2':r.get('kz_m2'),
         'reference_kz_mD':r.get('kz_mD'),'reference_status':r.get('status'),'porosity_abs_error':None}
    if p.get('porosity') is not None and r.get('porosity') is not None:
        out['porosity_abs_error']=abs(p['porosity']-r['porosity'])
    if (any(p.get(k)!=r.get(k) for k in ('flow_axis','solver_config_hash','core_sha256','precision')) or
        any(tuple(p.get(k) or ())!=tuple(r.get(k) or ()) for k in ('spacing_um_xyz','shape_xyz'))):
        out['comparison_status']='incompatible_solver';return out
    valid={'completed','budget_exhausted'}
    ps,rs=p['status'],r['status']
    for row in (p,r):
        if row['status'] in valid:
            k=row.get('k_m2')
            if isinstance(k,bool) or not isinstance(k,(float,int)) or not math.isfinite(k) or k<=0:
                out['comparison_status']='invalid_estimate_record';return out
    if ps in valid and rs in valid:
        out.update(relative_metrics(p['k_m2'],r['k_m2']))
        out['comparison_status']='completed_pair' if ps==rs=='completed' else 'budget_limited_pair'
    elif ps=='non_percolating' and rs in valid:
        out.update(relative_metrics(0.,r['k_m2']));out['comparison_status']='percolation_failure'
    elif ps in valid and rs=='non_percolating':out['comparison_status']='spurious_percolation'
    elif ps==rs=='non_percolating':out['comparison_status']='both_non_percolating'
    else:out['comparison_status']=ps if ps not in valid else 'reference_'+rs
    return out

