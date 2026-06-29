#!/usr/bin/env python3
"""Test 3D visualization."""

import sys
sys.path.insert(0, 'src')

from physics import compute_lagrange_points, propagate
from viz import plot_3d_system, plot_debris_cloud
from debris import generate_debris_catalog
import numpy as np

print("Computing Lagrange points...")
L = compute_lagrange_points()

print("Propagating trajectory...")
state0 = [L['L1'][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]
t_eval = np.linspace(0, 10, 1000)
sol = propagate(state0, (0, 10), t_eval)

print("Creating 3D system plot...")
fig = plot_3d_system(sol=sol, L_points=L, title="CR3BP Trajectory in 3D")
fig.write_html("docs/3d_system.html")
print("  Saved: docs/3d_system.html")

print("Generating debris catalog...")
catalog = generate_debris_catalog(n_objects=500, seed=42)

print("Creating debris cloud plot...")
fig2 = plot_debris_cloud(catalog, title="Synthetic Cislunar Debris")
fig2.write_html("docs/debris_cloud.html")
print("  Saved: docs/debris_cloud.html")

print("\nOpen these HTML files in your browser to see interactive 3D plots.")
