"""One fixed formulation; local helper, not an unmodified upstream file.

Tau follows Yang (2022) Eq.7 and upstream lbm_solver_3d.py. Force coefficients
follow the D3Q19 quadrature/Guo moment constraints, a separately documented
change to the current class. Numerical verification remains necessary.
"""
import math

FORMULATION_ID = "yang_eq7_guo_d3q19_v02"
CS2 = 1.0 / 3.0
FORCE_LINEAR = 1.0 / CS2
FORCE_QUADRATIC = 1.0 / (CS2 * CS2)


def tau_from_nu(nu):
    if isinstance(nu, bool) or not math.isfinite(float(nu)) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    return 3.0 * float(nu) + 0.5
