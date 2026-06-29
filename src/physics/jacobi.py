"""Jacobi integral for CR3BP."""

import numpy as np

MU = 0.012150585609624

def jacobi_integral(state, mu=MU):
    """
    Compute Jacobi constant (conserved quantity in CR3BP).
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    U = 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2
    return 2*U - (vx**2 + vy**2 + vz**2)
