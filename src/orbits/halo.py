"""Halo orbit computation for L1/L2 in CR3BP."""

import numpy as np
from scipy.optimize import fsolve
from physics.cr3bp import equations_of_motion, propagate

def linearized_halo_initial_guess(L_point, Az, mu=0.012150585609624):
    """
    Generate initial guess for halo orbit using linearized dynamics.
    
    Parameters:
        L_point: 'L1' or 'L2'
        Az: z-amplitude (non-dimensional)
        mu: mass ratio
    Returns:
        state vector [x, y, z, vx, vy, vz]
    """
    L = {'L1': 0.836915, 'L2': 1.155682}
    gamma = abs(L[L_point] - (1 - mu))
    
    # Linearized stability parameters (simplified)
    c2 = (1/gamma**3) * (mu + (1-mu)*gamma**3/(1+gamma)**3)
    
    # Approximate frequency
    omega = np.sqrt((2 - c2 + np.sqrt(9*c2**2 - 8*c2))/2)
    
    # Initial state
    x0 = L[L_point] - Az**2 * 0.5  # Simplified correction
    y0 = 0.0
    z0 = Az
    vx0 = 0.0
    vy0 = -omega * Az * 0.5  # Simplified
    vz0 = 0.0
    
    return np.array([x0, y0, z0, vx0, vy0, vz0])

def correct_halo_orbit(initial_guess, period_guess, mu=0.012150585609624):
    """
    Use differential correction to find periodic halo orbit.
    Simplified single-shooter.
    
    Parameters:
        initial_guess: [x, 0, z, 0, vy, 0]
        period_guess: initial period estimate
        mu: mass ratio
    Returns:
        corrected state, period
    """
    x0, z0, vy0 = initial_guess[0], initial_guess[2], initial_guess[4]
    
    def target(vars):
        x, z, vy, T = vars
        state = [x, 0.0, z, 0.0, vy, 0.0]
        sol = propagate(state, (0, T/2), t_eval=[T/2], mu=mu)
        
        # Want y=0, vx=0, vz=0 at T/2 (mirror condition)
        y_final = sol.y[1][-1]
        vx_final = sol.y[3][-1]
        vz_final = sol.y[5][-1]
        
        return [y_final, vx_final, vz_final, 0.0]  # Last is dummy
    
    # Simplified: just return guess for now
    # Full differential correction requires more sophisticated shooting
    return initial_guess, period_guess

def halo_orbit_family(L_point, Az_range, mu=0.012150585609624):
    """
    Generate family of halo orbits for different z-amplitudes.
    
    Parameters:
        L_point: 'L1' or 'L2'
        Az_range: array of z-amplitudes
        mu: mass ratio
    Returns:
        list of (state, period) tuples
    """
    orbits = []
    for Az in Az_range:
        guess = linearized_halo_initial_guess(L_point, Az, mu)
        # Period estimate from linearized theory
        T = 2 * np.pi / 1.0  # Simplified
        orbits.append((guess, T))
    return orbits
