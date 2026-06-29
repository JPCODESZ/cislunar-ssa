#!/usr/bin/env python3
"""Research-grade 3D visualization with NASA validation data."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import compute_lagrange_points, propagate, validate_lagrange_points
from viz import plot_3d_system, plot_debris_cloud
from debris import generate_debris_catalog

print("Computing Lagrange points with NASA validation...")
L = compute_lagrange_points()
validation = validate_lagrange_points(L)

print("\nLagrange Point Accuracy:")
for name, result in validation.items():
    print(f"  {name}: error = {result['error']:.2e} {'✓' if result['passed'] else '✗'}")

print("\nPropagating trajectory near L1...")
state0 = [L['L1'][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]
t_eval = np.linspace(0, 10, 1000)
sol = propagate(state0, (0, 10), t_eval)

print("Creating research-grade 3D system plot...")
fig = plot_3d_system(sol=sol, L_points=L, title="Cislunar SSA - CR3BP Trajectory (NASA-Validated)")
fig.write_html("docs/3d_system_research.html")
print("  Saved: docs/3d_system_research.html")

print("\nGenerating realistic debris catalog...")
catalog = generate_debris_catalog(n_objects=1000, region='cislunar', seed=42)

print("Creating debris cloud with size coloring...")
fig2 = plot_debris_cloud(catalog, title="Cislunar Debris Population (n=1000, Synthetic)")
fig2.write_html("docs/debris_cloud_research.html")
print("  Saved: docs/debris_cloud_research.html")

print("\n" + "="*60)
print("Open these files in browser:")
print("  docs/3d_system_research.html")
print("  docs/debris_cloud_research.html")
print("="*60)
