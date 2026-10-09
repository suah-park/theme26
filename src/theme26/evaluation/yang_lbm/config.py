"""FP64 fixed-budget evaluation; descriptive diagnostics, no acceptance gates."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import math
import yaml
from .validation import canonical_hash
from ._vendor.formulation import FORMULATION_ID

@dataclass(frozen=True)
class LBMConfig:
    backend: str = 'cuda'
    axis: str = 'z'
    nu_lattice: float = 0.1
    force_density: float = 1e-6
    boundary: str = 'periodic_all_axes_no_reservoir'
    precision: str = 'float64'
    max_steps: int = 50000
    check_every: int = 500
    history_window: int = 6
    max_seconds: float = 7200.0
    cpu_threads: int = 4
    device_memory_fraction: float = 0.8
    isolation: bool = True
    cache_dir: str | None = None

    def __post_init__(self):
        if self.backend not in {'cpu', 'cuda'} or self.axis not in {'x','y','z'}:
            raise ValueError('backend must be cpu/cuda; axis must be x/y/z')
        if self.boundary != 'periodic_all_axes_no_reservoir':
            raise ValueError('Only the existing all-periodic, no-reservoir formulation is implemented')
        if self.precision != 'float64':
            raise ValueError('This final package uses float64 only; no silent FP32 fallback')
        for key in ('nu_lattice','force_density','max_seconds','device_memory_fraction'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('max_steps','check_every','history_window','cpu_threads'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        if self.check_every>self.max_steps or self.history_window<2 or self.device_memory_fraction>1:
            raise ValueError('invalid recording schedule or memory fraction')
        if type(self.isolation) is not bool:
            raise TypeError('isolation must be bool')
        if self.cache_dir is not None and not isinstance(self.cache_dir,str):
            raise TypeError('cache_dir must be a string or null')

    def as_dict(self): return asdict(self)
    def numerical_dict(self):
        d=asdict(self)
        for key in ('cache_dir','isolation'):d.pop(key)
        d['formulation_id']=FORMULATION_ID
        d['execution_policy']='fixed_budget_report_only_v1'
        return d
    @property
    def fingerprint(self): return canonical_hash(self.numerical_dict())
    @property
    def force_vector(self):
        v=[0.,0.,0.];v['xyz'.index(self.axis)]=float(self.force_density);return tuple(v)
    @classmethod
    def from_yaml(cls,path):
        data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict) or not isinstance(data.get('solver'),dict):
            raise ValueError('YAML requires a solver mapping')
        unknown=set(data['solver'])-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Use the supplied final YAML; obsolete/unknown options: '+', '.join(sorted(unknown)))
        return cls(**data['solver'])
    def changed(self,**kwargs): return replace(self,**kwargs)
