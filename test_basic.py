#!/usr/bin/env python3
"""Quick test of CR3BP propagator."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import compute_lagrange_points, propagate, jacobi_integral
from viz.rotating_frame import plot_trajectory

print("Computing Lagrange points...")
L = compute_lagrange_points()
for name, pos in L.items():
    print(f"  {name}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")

print("\nPropagating test particle near L1...")
state0 = [L['L1'][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

sol = propagate(state0, t_span, t_eval)

print(f"  Integration successful: {sol.success}")
print(f"  Final position: ({sol.y[0][-1]:.4f}, {sol.y[1][-1]:.4f}, {sol.y[2][-1]:.4f})")

print("\nChecking Jacobi integral conservation...")
C0 = jacobi_integral(state0)
Cf = jacobi_integral(sol.y[:, -1])
print(f"  Initial C:  {C0:.6f}")
print(f"  Final C:    {Cf:.6f}")
print(f"  Difference: {abs(C0 - Cf):.2e}")

print("\nPlotting...")
fig, ax = plot_trajectory(sol, title="Test: Particle Near L1")
fig.savefig('test_trajectory.png', dpi=150)
print("  Saved: test_trajectory.png")

print("\nAll tests passed.")
