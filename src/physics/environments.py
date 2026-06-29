"""Space environment definitions with physical constants."""

import numpy as np

# Standard gravitational parameters (km^3/s^2)
MU_EARTH = 398600.4418
MU_MOON = 4902.800582
MU_SUN = 132712440041.94

# Mean distances (km)
R_EARTH_MOON = 384400.0
R_EARTH_SUN = 149597870.7

# Earth radius (km)
R_EARTH = 6378.137

class Environment:
    """Base class for space environments."""
    
    def __init__(self, name, mu_primary, mu_secondary, distance):
        self.name = name
        self.mu_primary = mu_primary
        self.mu_secondary = mu_secondary
        self.distance = distance
        self.mass_ratio = mu_secondary / (mu_primary + mu_secondary)
        
    def __repr__(self):
        return f"Environment({self.name}, mu={self.mass_ratio:.6f})"

# Predefined environments
EARTH_MOON = Environment("Earth-Moon", MU_EARTH, MU_MOON, R_EARTH_MOON)
SUN_EARTH = Environment("Sun-Earth", MU_SUN, MU_EARTH, R_EARTH_SUN)

# LEO environment (approximate, Earth-centered)
LEO = type('LEO', (), {
    'name': 'Low Earth Orbit',
    'altitude_range': (200, 2000),  # km
    'period_range': (90, 127),  # minutes
    'velocity': lambda h: np.sqrt(MU_EARTH / (R_EARTH + h))
})()

# GEO environment
GEO = type('GEO', (), {
    'name': 'Geostationary Orbit',
    'altitude': 35786,  # km
    'period': 1436,  # minutes (sidereal day)
    'velocity': np.sqrt(MU_EARTH / (R_EARTH + 35786))
})()

def get_cr3bp_mass_ratio(environment):
    """Get mass ratio for CR3BP in given environment."""
    return environment.mass_ratio
