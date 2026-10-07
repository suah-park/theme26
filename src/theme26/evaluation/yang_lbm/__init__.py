"""Optional FP64 LBM adapter; importing this package does not initialize Taichi."""
from .config import LBMConfig
from .solver import YangLBMSolver as _DetailedSolver
from .result import SolveResult


def result_scalars(result: SolveResult) -> dict[str, float | bool | str]:
    """Project API view. Detailed histories remain available via solve_detailed.

    Undefined values are omitted, never substituted with zero. Completion is
    an execution status, not a statement of convergence or numerical accuracy.
    """
    data = result.as_dict()
    row = {k: v for k, v in data.items()
           if v is not None and isinstance(v, (str, int, float, bool))}
    for key in ('shape_xyz', 'spacing_um_xyz'):
        if data.get(key) is not None:
            row[key] = ','.join(str(v) for v in data[key])
    row['convergence_assessed'] = False
    for key, value in result.diagnostics.items():
        if value is not None and isinstance(value, (int, float, bool)):
            row['diagnostic_' + key] = value
    return row


class YangLBMSolver(_DetailedSolver):
    """Implements theme26.evaluation.physics_metrics.PhysicsSolver.

    Same FP64 kernels and fixed budget as the final standalone delivery.
    solve() returns scalars as required by the team's existing interface.
    """
    version = '0.3.0-repo1'

    def solve_detailed(self, pore_mask, *, spacing_um_xyz) -> SolveResult:
        return super().solve(pore_mask, spacing_um_xyz=spacing_um_xyz)

    def solve(self, pore_mask, *, spacing_um_xyz) -> dict[str, float | bool | str]:
        row = result_scalars(self.solve_detailed(pore_mask, spacing_um_xyz=spacing_um_xyz))
        row.update(solver_name=self.name, solver_version=self.version)
        return row


__all__ = ['LBMConfig', 'YangLBMSolver']
