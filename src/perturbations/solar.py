"""Solar gravity perturbation in CR3BP."""

import numpy as np

def solar_perturbation(state, t, mu, solar_mass_ratio=3.289e5, 
                       solar_distance=388.9, solar_inclination=0.0):
    """
    Add Sun's gravity as perturbation to CR3BP.
    Simplified: Sun in circular orbit around Earth-Moon barycenter.
    
    Parameters:
        state: [x, y, z, vx, vy, vz]
        t: time (non-dimensional)
        mu: Earth-Moon mass ratio
        solar_mass_ratio: M_sun / (M_earth + M_moon)
        solar_distance: Sun distance / Earth-Moon distance
        solar_inclination: Sun orbital inclination (radians)
    Returns:
        acceleration vector [ax, ay, az]
    """
    x, y, z = state[:3]
    
    # Sun position in rotating frame (simplified circular)
    theta = t + np.pi  # Sun opposite to Earth-Moon line roughly
    xs = solar_distance * np.cos(theta)
    ys = solar_distance * np.sin(theta) * np.cos(solar_inclination)
    zs = solar_distance * np.sin(theta) * np.sin(solar_inclination)
    
    # Distance to Sun
    r_sun = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    
    # Solar gravity acceleration
    ax = -solar_mass_ratio * (x - xs) / r_sun**3
    ay = -solar_mass_ratio * (y - ys) / r_sun**3
    az = -solar_mass_ratio * (z - zs) / r_sun**3
    
    return np.array([ax, ay, az])
