"""Physical bookkeeping for periodic constant FORCE DENSITY, rho0=1.

K_lattice = rho0 * nu * U_superficial / force_density.
U_superficial is sum(pore velocity)/TOTAL voxel count; do not multiply phi again.
The lattice spacing squared converts to m^2. No reservoir correction applies.
"""
import math
import numpy as np

M2_PER_MD = 9.869233e-16


def permeability_from_mean(mean_velocity, *, nu, force_density, spacing_um, rho0=1.0):
    values=(mean_velocity,nu,force_density,spacing_um,rho0)
    if not all(math.isfinite(float(v)) for v in values) or min(nu,force_density,spacing_um,rho0)<=0:
        raise ValueError("finite values and positive material/drive parameters required")
    k_lattice = rho0*nu*mean_velocity/force_density
    return {"k_lattice":float(k_lattice),"k_m2":float(k_lattice*(spacing_um*1e-6)**2),
            "k_mD":float(k_lattice*(spacing_um*1e-6)**2/M2_PER_MD)}


def field_diagnostics(velocity, density, pore, axis):
    if velocity.shape != pore.shape+(3,) or density.shape != pore.shape:
        raise ValueError("field shape mismatch")
    if not np.isfinite(velocity).all() or not np.isfinite(density).all():
        raise FloatingPointError("non-finite fluid field")
    if (density[pore]<=0).any(): raise FloatingPointError("non-positive fluid density")
    u=np.where(pore,velocity[...,axis],0).astype(np.float64)
    planes=tuple(a for a in range(3) if a!=axis)
    flux=np.sum(density*u,axis=planes,dtype=np.float64)
    q=float(flux.mean())
    spread=float(np.max(np.abs(flux-q))/max(abs(q),1e-30))
    umax=float(np.sqrt(np.square(velocity[pore].astype(np.float64)).sum(axis=-1)).max())
    return {"mean_velocity":float(u.mean()),"maximum_velocity":umax,
            "max_mach":umax*math.sqrt(3),"mass_balance_diagnostic":spread,
            "mass_flux_mean":q,"density_mean":float(density[pore].mean(dtype=np.float64)),
            "density_deviation":float(np.max(np.abs(density[pore]-1.0)))}
