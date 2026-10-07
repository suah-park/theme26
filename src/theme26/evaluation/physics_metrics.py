"""Paired post-training physical metrics, with the existing public signatures.

FP64 estimates are reported without numerical acceptance gates. A completed
fixed-budget run is not automatically a converged/validated physical result.
"""
from __future__ import annotations
from typing import Protocol
import math
import numpy as np
import numpy.typing as npt
from .yang_lbm.validation import validate_pore, validate_spacing
from .yang_lbm.comparison import compare_results
from .yang_lbm.result import STATUSES


class PhysicsSolver(Protocol):
    """Adapter contract for a frozen external tortuosity/permeability solver."""
    name: str
    version: str

    def solve(
        self, pore_mask: npt.NDArray[np.bool_], *,
        spacing_um_xyz: tuple[float, float, float],
    ) -> dict[str, float | bool | str]:
        """Return scalar solver outputs for one pore volume."""


def _check_scalar_result(result):
    if not isinstance(result, dict):
        raise TypeError('PhysicsSolver.solve must return a scalar dictionary')
    for key, value in result.items():
        if not isinstance(key, str) or not isinstance(value, (str, bool, int, float)):
            raise TypeError('Solver outputs must contain only named scalar values')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError('Solver returned a nonfinite scalar')
    if result.get('status') not in STATUSES:
        raise ValueError('Solver must return a supported execution status')
    if result.get('flow_axis', 'z') != 'z':
        raise ValueError('Primary physical comparison requires z-direction results')
    return result


def paired_scalar_metrics(prediction, reference):
    """Combine already computed results; useful without repeating simulation."""
    p = _check_scalar_result(prediction)
    r = _check_scalar_result(reference)
    errors = {'invalid_input', 'missing_input', 'dependency_unavailable',
              'backend_unavailable', 'out_of_memory', 'timeout', 'numerical_error'}
    if p['status'] in errors or r['status'] in errors:
        status = p['status'] if p['status'] in errors else 'reference_' + r['status']
        comparison = {'comparison_status': status}
    elif any(p.get(k) != r.get(k) for k in ('solver_name', 'solver_version')):
        comparison = {'comparison_status': 'incompatible_solver'}
    else:
        comparison = compare_results(p, r)
    out = {k: v for k, v in comparison.items() if v is not None}
    out.update({'prediction_' + k: v for k, v in p.items()})
    out.update({'reference_' + k: v for k, v in r.items()})
    if 'log10_error' in out:
        out['kz_log10_abs_error'] = out['log10_error']
    return out


def compute_physics_metrics(
    predicted_pore: npt.NDArray[np.bool_],
    target_pore: npt.NDArray[np.bool_],
    *,
    spacing_um_xyz: tuple[float, float, float],
    solver: PhysicsSolver,
) -> dict[str, float | bool | str]:
    """Run one frozen solver on target and prediction, preserving all statuses.

    Inputs: [X,Y,Z] bool, True=pore; isotropic HR spacing in micrometres.
    Both phases must exist, as required by the repository's public contract.
    Nonempty pore space with no periodic winding path is a structural zero;
    an invalid input or numerical failure is not assigned zero permeability.
    """
    p = validate_pore(predicted_pore)
    r = validate_pore(target_pore)
    s = validate_spacing(spacing_um_xyz)
    if p.shape != r.shape:
        raise ValueError('reference/prediction shape mismatch')
    if not p.any() or not r.any():
        raise ValueError('Public physics metrics require nonempty pore and solid phases')
    rr = solver.solve(r.copy(), spacing_um_xyz=s)
    pr = solver.solve(p.copy(), spacing_um_xyz=s)
    return paired_scalar_metrics(pr, rr)
