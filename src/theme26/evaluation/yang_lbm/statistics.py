"""Descriptive history only. No pass/fail, adaptive stop, or acceptance level."""
from collections import deque
import numpy as np
from .permeability import M2_PER_MD

class RunStatistics:
    def __init__(self,config):
        self.samples=deque(maxlen=config.history_window)
        self.previous=None
    def update(self,iteration,diag):
        u=float(diag['mean_velocity'])
        change=None if self.previous is None or u==0 else abs(u-self.previous)/abs(u)
        self.previous=u;self.samples.append(u)
        span=None if len(self.samples)<2 or u==0 else (max(self.samples)-min(self.samples))/abs(u)
        return {'relative_change':change,'window_relative_range':span}

def history_statistics(history,window=6):
    """Variation in the last recorded samples, not independent-run uncertainty."""
    rows=history[-window:]
    values=[]
    for row in rows:
        x=row.get('estimated_k_axis_m2',row.get('estimated_kz'))
        if x is not None and np.isfinite(x):values.append(float(x)/M2_PER_MD)
    keys=('tail_kz_mean_mD','tail_kz_std_mD','tail_kz_min_mD','tail_kz_max_mD',
          'tail_relative_range_pct','last_relative_change_pct')
    out={k:None for k in keys}
    out['tail_samples']=len(values)
    out['variation_definition']='Last recorded time samples; population std; not an accuracy CI or replicate std'
    if not values:return out
    a=np.asarray(values,dtype=np.float64);last=float(a[-1])
    out.update(tail_kz_mean_mD=float(a.mean()),tail_kz_std_mD=float(a.std(ddof=0)),
               tail_kz_min_mD=float(a.min()),tail_kz_max_mD=float(a.max()))
    if last!=0:
        out['tail_relative_range_pct']=float(100*np.ptp(a)/abs(last))
        if len(a)>1:out['last_relative_change_pct']=float(100*abs(a[-1]-a[-2])/abs(last))
    return out
