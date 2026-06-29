"""Lagrange point computation for Earth-Moon system."""

import numpy as np
from scipy.optimize import fsolve

MU = 0.012150585609624

def _collinear_eq(x, mu):
    """Equation for collinear Lagrange points."""
    r1 = abs(x + mu)
    r2 = abs(x - 1 + mu)
    return x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3

def compute_lagrange_points(mu=MU):
    """
    Compute all five Lagrange points.
    Returns dict of (x, y, z) tuples.
    """
    L1 = fsolve(_collinear_eq, 0.8, args=(mu,))[0]
    L2 = fsolve(_collinear_eq, 1.2, args=(mu,))[0]
    L3 = fsolve(_collinear_eq, -1.0, args=(mu,))[0]
    
    L4 = (0.5 - mu, np.sqrt(3)/2, 0)
    L5 = (0.5 - mu, -np.sqrt(3)/2, 0)
    
    return {
        'L1': (L1, 0, 0),
        'L2': (L2, 0, 0),
        'L3': (L3, 0, 0),
        'L4': L4,
        'L5': L5
    }

def stability_eigenvalues(point, mu=MU):
    """
    Compute linear stability eigenvalues at a Lagrange point.
    """
    x, y, z = point
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    Uxx = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*(x + mu)**2/r1**5 + 3*mu*(x - 1 + mu)**2/r2**5
    Uyy = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*y**2/r1**5 + 3*mu*y**2/r2**5
    Uxy = 3*(1 - mu)*(x + mu)*y/r1**5 + 3*mu*(x - 1 + mu)*y/r2**5
    
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [Uxx, Uxy, 0, 2],
        [Uxy, Uyy, -2, 0]
    ])
    
    return np.linalg.eigvals(A)
