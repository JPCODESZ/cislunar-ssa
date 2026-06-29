"""Solar radiation pressure perturbation."""

import numpy as np

def radiation_pressure(state, t, Cr=1.0, A_m=0.01, solar_distance=388.9):
    """
    Solar radiation pressure acceleration.
    
    Parameters:
        state: [x, y, z, vx, vy, vz]
        t: time
        Cr: reflectivity coefficient (1.0 = perfect absorption)
        A_m: area-to-mass ratio (m^2/kg)
        solar_distance: Sun distance / EM distance
    Returns:
        acceleration vector [ax, ay, az]
    """
    x, y, z = state[:3]
    
    # Sun direction (simplified)
    theta = t + np.pi
    sun_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    
    # SRP magnitude (simplified, non-dimensional)
    # P = L_sun / (4*pi*c*r^2), scaled
    P = 4.56e-6  # Approximate non-dimensional pressure
    beta = Cr * A_m * P
    
    return beta * sun_dir
