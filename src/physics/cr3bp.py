"""Circular Restricted Three-Body Problem equations of motion."""

import numpy as np
from scipy.integrate import solve_ivp

# Earth-Moon system mass ratio
MU = 0.012150585609624

def equations_of_motion(t, state, mu=MU):
    """
    CR3BP equations in rotating frame.
    
    state = [x, y, z, vx, vy, vz]
    Returns: [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    
    # Distance to Earth and Moon
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    # Accelerations
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = -(1 - mu)*z/r1**3 - mu*z/r2**3
    
    return [vx, vy, vz, ax, ay, az]

def propagate(state0, t_span, t_eval=None, mu=MU):
    """
    Propagate state vector through CR3BP.
    
    Parameters:
        state0: [x, y, z, vx, vy, vz] initial state
        t_span: (t0, tf) non-dimensional time
        t_eval: evaluation points
    Returns:
        scipy.integrate.solve_ivp result
    """
    sol = solve_ivp(
        lambda t, y: equations_of_motion(t, y, mu),
        t_span,
        state0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-12
    )
    return sol
