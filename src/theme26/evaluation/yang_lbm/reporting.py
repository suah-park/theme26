"""Strict JSON/CSV and grouped descriptive statistics (not pseudo-replication)."""
from pathlib import Path
from collections import defaultdict,Counter
import csv
import json
import os
import tempfile
import math
import numpy as np


def json_safe(value):
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite JSON output rejected')
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'unsupported output type {type(value)}')


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    tmp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,delete=False) as f:
            f.write(text);tmp=f.name
        os.replace(tmp,p)
    finally:
        if tmp and os.path.exists(tmp):os.unlink(tmp)

CSV_COLUMNS=['sample_id','core_id','split','experiment_id','seed','porosity','kz_m2','kz_mD',
 'reference_kz_m2','reference_kz_mD','absolute_error_m2','porosity_abs_error','relative_error_pct','signed_relative_error_pct','log10_error','converged',
 'stop_reason','tail_samples','tail_kz_mean_mD','tail_kz_std_mD','tail_kz_min_mD','tail_kz_max_mD',
 'tail_relative_range_pct','last_relative_change_pct','plane_flux_spread_pct',
 'status','comparison_status','iterations','runtime_seconds','solver_config_hash','core_sha256',
 'input_checksum','segmentation_protocol_id','boundary_condition','spacing_um_xyz','shape_xyz',
 'flow_axis','formulation_id','execution_policy','verification_scope',
 'backend','precision','taichi_version','numerical_solve_performed','cache_hit','message']


def write_csv(path,rows,columns=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    columns=columns or CSV_COLUMNS
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');out.writeheader()
        for row in rows:
            d=json_safe(row)
            for k,v in list(d.items()):
                if isinstance(v,(dict,list)):d[k]=json.dumps(v,ensure_ascii=False,allow_nan=False)
            out.writerow(d)


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        if r.get('experiment_id')=='Reference':continue
        key=tuple(r.get(k) for k in ('experiment_id','seed','solver_config_hash','core_sha256',
             'segmentation_protocol_id','boundary_condition'))
        groups[key].append(r)
    output=[]
    for key,items in groups.items():
        successful=[r for r in items if r.get('comparison_status')=='completed_pair']
        by_core=defaultdict(list)
        for r in successful:by_core[r.get('core_id','not_provided')].append(r['relative_error_pct'])
        core_means={k:float(np.mean(v)) for k,v in by_core.items()}
        output.append({'experiment_id':key[0],'seed':key[1],'solver_config_hash':key[2],
            'core_sha256':key[3],'segmentation_protocol_id':key[4],'boundary_condition':key[5],
            'rows':len(items),'completed_pairs':len(successful),
            'status_counts':dict(Counter(r.get('comparison_status','unknown') for r in items)),
            'core_mean_relative_error_pct':core_means,
            'mean_of_core_means_relative_error_pct':float(np.mean(list(core_means.values()))) if core_means else None,
            'confidence_interval':None,
            'note':'Completed fixed-budget pairs only; not certified accurate. Budget-limited pairs are reported separately in results.'})
    return output
