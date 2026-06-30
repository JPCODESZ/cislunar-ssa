"""Differential correction for periodic halo orbits."""

import numpy as np
from physics.cr3bp import propagate

def single_shooter(state_guess, period_guess, mu=0.012150585609624, 
                   max_iter=50, tol=1e-10):
    """
    Single-shooting differential correction for periodic halo orbits.
    
    Target: y=0, vx=0, vz=0 at x-z plane crossing (T/2).
    """
    x, z, vy = state_guess[0], state_guess[2], state_guess[4]
    T = period_guess
    
    for iteration in range(max_iter):
        # Propagate half period
        state = [x, 0.0, z, 0.0, vy, 0.0]
        sol = propagate(state, (0, T/2), t_eval=[T/2], mu=mu)
        
        if not sol.success:
            print(f"Iteration {iteration}: integration failed")
            break
        
        # Final state
        y_f = float(sol.y[1][-1])
        vx_f = float(sol.y[3][-1])
        vz_f = float(sol.y[5][-1])
        
        # Check convergence
        error = np.sqrt(y_f**2 + vx_f**2 + vz_f**2)
        print(f"Iteration {iteration}: error = {error:.2e}")
        
        if error < tol:
            print("Converged!")
            return [x, 0.0, z, 0.0, vy, 0.0], T
        
        # Numerical Jacobian (finite differences)
        eps = 1e-8
        J = np.zeros((3, 3))
        
        # Vary x
        state_x = [x + eps, 0.0, z, 0.0, vy, 0.0]
        sol_x = propagate(state_x, (0, T/2), t_eval=[T/2], mu=mu)
        if sol_x.success:
            diff = np.array([
                float(sol_x.y[1][-1]) - y_f,
                float(sol_x.y[3][-1]) - vx_f,
                float(sol_x.y[5][-1]) - vz_f
            ]) / eps
            J[:, 0] = diff
        
        # Vary z
        state_z = [x, 0.0, z + eps, 0.0, vy, 0.0]
        sol_z = propagate(state_z, (0, T/2), t_eval=[T/2], mu=mu)
        if sol_z.success:
            diff = np.array([
                float(sol_z.y[1][-1]) - y_f,
                float(sol_z.y[3][-1]) - vx_f,
                float(sol_z.y[5][-1]) - vz_f
            ]) / eps
            J[:, 1] = diff
        
        # Vary vy
        state_vy = [x, 0.0, z, 0.0, vy + eps, 0.0]
        sol_vy = propagate(state_vy, (0, T/2), t_eval=[T/2], mu=mu)
        if sol_vy.success:
            diff = np.array([
                float(sol_vy.y[1][-1]) - y_f,
                float(sol_vy.y[3][-1]) - vx_f,
                float(sol_vy.y[5][-1]) - vz_f
            ]) / eps
            J[:, 2] = diff
        
        # Newton step
        try:
            delta = np.linalg.solve(J, -np.array([y_f, vx_f, vz_f]))
            x += delta[0]
            z += delta[1]
            vy += delta[2]
        except np.linalg.LinAlgError:
            print("Jacobian singular, using least squares")
            delta = np.linalg.lstsq(J, -np.array([y_f, vx_f, vz_f]), rcond=None)[0]
            x += delta[0]
            z += delta[1]
            vy += delta[2]
    
    print("Max iterations reached")
    return [x, 0.0, z, 0.0, vy, 0.0], T

def generate_halo_family(L_point, Az_range, mu=0.012150585609624):
    """
    Generate family of halo orbits using differential correction.
    """
    orbits = []
    
    for Az in Az_range:
        print(f"\n--- Generating halo with Az = {Az:.4f} ---")
        
        from .halo import linearized_halo_initial_guess
        guess = linearized_halo_initial_guess(L_point, Az, mu)
        period_guess = 2 * np.pi / 1.0
        
        corrected_state, corrected_period = single_shooter(
            guess, period_guess, mu, max_iter=20, tol=1e-8
        )
        
        sol = propagate(corrected_state, (0, corrected_period), 
                       t_eval=np.linspace(0, corrected_period, 500), mu=mu)
        
        if sol.success:
            orbits.append({
                'Az': Az,
                'state': corrected_state,
                'period': corrected_period,
                'trajectory': sol
            })
    
    return orbits
