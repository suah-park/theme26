"""Cache completed finite estimates, NOT a cache of certified accurate values."""
from pathlib import Path
import json,os,tempfile,math
from .result import SolveResult
from .validation import canonical_hash
from .permeability import M2_PER_MD


def cache_key(checksum,spacing,config,core_hash,versions):
    return canonical_hash({'schema':'fp64-report-only-1','input':checksum,'spacing_um':spacing,
        'config':config.numerical_dict(),'core_sha256':core_hash,'packages':versions})

def from_dict(data):
    fields={k:v for k,v in data.items() if k in SolveResult.__dataclass_fields__}
    for key in ('spacing_um_xyz','shape_xyz'):
        if fields.get(key) is not None:fields[key]=tuple(fields[key])
    return SolveResult(**fields)

def _valid(data):
    if data.get('status')!='completed' or data.get('precision')!='float64':return False
    for key in ('k_m2','k_mD'):
        v=data.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:return False
    return math.isclose(data['k_mD'],data['k_m2']/M2_PER_MD,rel_tol=1e-10)

def load_cache(directory,key):
    if not directory:return None
    try:
        record=json.loads((Path(directory)/(key+'.json')).read_text(encoding='utf-8'))
        if record.get('key')!=key or not _valid(record['result']):return None
        result=from_dict(record['result']);result.cache_hit=True
        result.numerical_solve_performed=False;result.runtime_seconds=0.
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def write_cache(directory,key,result):
    if not directory or not _valid(result.as_dict()):return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);tmp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=root,delete=False) as f:
            tmp=f.name;json.dump({'key':key,'result':result.as_dict()},f,allow_nan=False)
        os.replace(tmp,root/(key+'.json'))
    finally:
        if tmp and os.path.exists(tmp):os.unlink(tmp)
