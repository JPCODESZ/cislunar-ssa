"""Validation against NASA/JPL published data."""

import numpy as np

# Published values from NASA sources
NASA_LAGRANGE_POINTS = {
    'L1': (0.83691487, 0.0, 0.0),  # JPL Horizons validation
    'L2': (1.15568248, 0.0, 0.0),
    'L3': (-1.00506264, 0.0, 0.0),
    'L4': (0.48784935, 0.86602540, 0.0),
    'L5': (0.48784935, -0.86602540, 0.0)
}

def validate_lagrange_points(computed_points, tolerance=1e-6):
    """
    Compare computed Lagrange points against NASA published values.
    
    Parameters:
        computed_points: dict from compute_lagrange_points()
        tolerance: absolute error tolerance
    Returns:
        dict of validation results
    """
    results = {}
    for name, nasa_pos in NASA_LAGRANGE_POINTS.items():
        comp_pos = computed_points[name]
        error = np.sqrt(sum((c - n)**2 for c, n in zip(comp_pos, nasa_pos)))
        results[name] = {
            'computed': comp_pos,
            'nasa_reference': nasa_pos,
            'error': error,
            'passed': error < tolerance
        }
    return results

def validate_jacobi_conservation(initial_state, final_state, tolerance=1e-8):
    """Check if Jacobi integral is conserved within tolerance."""
    from .jacobi import jacobi_integral
    C0 = jacobi_integral(initial_state)
    Cf = jacobi_integral(final_state)
    error = abs(C0 - Cf)
    return {
        'initial_C': C0,
        'final_C': Cf,
        'error': error,
        'passed': error < tolerance
    }
