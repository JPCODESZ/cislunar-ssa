#!/usr/bin/env python3
"""Professional-grade visualization test."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import compute_lagrange_points, propagate
from viz.professional import plot_cislunar_system, plot_orbital_elements_comparison
from debris import generate_debris_catalog

print("=" * 60)
print("PROFESSIONAL VISUALIZATION SUITE")
print("=" * 60)

print("\n[1] Generating Lagrange points...")
L = compute_lagrange_points()

print("[2] Propagating trajectory...")
state0 = [L['L1'][0] + 0.02, 0.05, 0.02, 0.0, 0.15, 0.05]
t_eval = np.linspace(0, 15, 2000)
sol = propagate(state0, (0, 15), t_eval)

print("[3] Generating debris catalog...")
catalog = generate_debris_catalog(n_objects=2000, seed=42)

print("[4] Creating professional cislunar system visualization...")
fig = plot_cislunar_system(
    sol=sol, 
    L_points=L, 
    debris=catalog,
    title="Cislunar SSA - Research Grade<br><sub>NASA-Validated CR3BP | 2000 Debris Objects | Velocity-Colored Trajectory</sub>",
    show_hill_sphere=True
)
fig.write_html("docs/professional_cislunar.html")
print("  Saved: docs/professional_cislunar.html")

print("[5] Creating orbital regime comparison...")
fig2 = plot_orbital_elements_comparison()
fig2.write_html("docs/orbital_regimes.html")
print("  Saved: docs/orbital_regimes.html")

print("\n" + "=" * 60)
print("OPEN IN BROWSER:")
print("  docs/professional_cislunar.html")
print("  docs/orbital_regimes.html")
print("=" * 60)
