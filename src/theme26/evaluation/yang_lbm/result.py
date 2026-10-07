"""Completed means computation ended, NOT scientific validation or convergence."""
from __future__ import annotations
from dataclasses import dataclass,asdict,field
from typing import Any
from ._vendor.formulation import FORMULATION_ID

STATUSES={'completed','budget_exhausted','non_percolating','invalid_input','missing_input',
          'dependency_unavailable','backend_unavailable','out_of_memory','timeout','numerical_error'}

@dataclass
class SolveResult:
    status: str
    message: str = ''
    porosity: float | None = None
    flow_axis: str = 'z'
    k_m2: float | None = None
    k_mD: float | None = None
    converged: bool | None = None
    iterations: int = 0
    runtime_seconds: float = 0.
    input_checksum: str | None = None
    solver_config_hash: str | None = None
    spacing_um_xyz: tuple | None = None
    shape_xyz: tuple | None = None
    solver_commit: str | None = None
    core_sha256: str | None = None
    backend: str | None = None
    taichi_version: str | None = None
    formulation_id: str = FORMULATION_ID
    precision: str = 'float64'
    numerical_solve_performed: bool = False
    percolates: bool | None = None
    verification_scope: str = 'descriptive_run_no_automatic_acceptance'
    execution_policy: str = 'fixed_budget_report_only_v1'
    stop_reason: str = ''
    diagnostics: dict[str,Any] = field(default_factory=dict)
    history: list[dict[str,Any]] = field(default_factory=list)
    cache_hit: bool = False

    def as_dict(self):
        if self.status not in STATUSES:raise ValueError(f'Unknown status: {self.status}')
        from .statistics import history_statistics
        d=asdict(self)
        d['kz_m2']=self.k_m2 if self.flow_axis=='z' else None
        d['kz_mD']=self.k_mD if self.flow_axis=='z' else None
        d.update(history_statistics(self.history,self.diagnostics.get('history_window',6)))
        d['plane_flux_spread_pct']=(100*self.diagnostics['mass_balance_diagnostic']
            if 'mass_balance_diagnostic' in self.diagnostics else None)
        return d
