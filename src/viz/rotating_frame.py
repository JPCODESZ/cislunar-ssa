"""Trajectory visualization in rotating frame."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_trajectory(sol, mu=0.012150585609624, title="CR3BP Trajectory"):
    """
    Plot trajectory in rotating frame with Earth and Moon.
    
    Parameters:
        sol: solve_ivp result object
        mu: mass ratio
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Trajectory
    ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, label='Trajectory')
    ax.plot(sol.y[0][0], sol.y[1][0], 'go', markersize=8, label='Start')
    
    # Earth and Moon
    ax.add_patch(Circle((-mu, 0), 0.02, color='blue', label='Earth'))
    ax.add_patch(Circle((1 - mu, 0), 0.008, color='gray', label='Moon'))
    
    ax.set_xlabel('x (non-dimensional)')
    ax.set_ylabel('y (non-dimensional)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax
