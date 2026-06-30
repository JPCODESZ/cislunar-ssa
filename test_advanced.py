#!/usr/bin/env python3
"""Advanced features test: animation, AI, halo correction."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import compute_lagrange_points, propagate
from viz.animation import animate_trajectory
from viz.professional import plot_cislunar_system
from ai.trajectory_predictor import generate_training_data, SimpleLSTMPredictor
from orbits.halo_differential_correction import generate_halo_family

print("=" * 60)
print("ADVANCED CISLUNAR SSA TEST")
print("=" * 60)

print("\n[1] Computing Lagrange points...")
L = compute_lagrange_points()

print("\n[2] Generating trajectory for animation...")
state0 = [L['L1'][0] + 0.02, 0.05, 0.02, 0.0, 0.15, 0.05]
t_eval = np.linspace(0, 12, 1500)
sol = propagate(state0, (0, 12), t_eval)

print("[3] Creating animated trajectory...")
fig_anim = animate_trajectory(sol, L_points=L, title="Animated Cislunar Trajectory")
fig_anim.write_html("docs/animated_trajectory.html")
print("  Saved: docs/animated_trajectory.html")

print("\n[4] AI Trajectory Predictor...")
print("  Generating training data...")
X_train, y_train = generate_training_data(n_samples=100, t_span=(0, 3), n_points=50)
print(f"  Generated {len(X_train)} training samples")

print("  Building LSTM model...")
predictor = SimpleLSTMPredictor(input_size=6, hidden_size=64, num_layers=2)
predictor.build_model()
predictor.train(X_train, y_train, epochs=10)

print("  Testing prediction...")
test_seq = X_train[:10]
pred = predictor.predict(test_seq)
print(f"  Predicted next state: {pred}")

print("\n[5] Halo Orbit Differential Correction...")
print("  Generating L1 halo family...")
from orbits.halo import linearized_halo_initial_guess
guess = linearized_halo_initial_guess('L1', 0.1)
print(f"  Initial guess: {guess}")

print("  Running single shooter...")
from orbits.halo_differential_correction import single_shooter
corrected, period = single_shooter(guess, 3.0, max_iter=10, tol=1e-6)
print(f"  Corrected state: {corrected}")
print(f"  Period: {period:.4f}")

print("\n[6] Creating professional visualization with all features...")
fig_prof = plot_cislunar_system(
    sol=sol,
    L_points=L,
    title="Cislunar SSA - Advanced Features<br><sub>Animated | AI-Predicted | Halo-Corrected</sub>"
)
fig_prof.write_html("docs/advanced_system.html")
print("  Saved: docs/advanced_system.html")

print("\n" + "=" * 60)
print("OPEN IN BROWSER:")
print("  docs/animated_trajectory.html")
print("  docs/advanced_system.html")
print("=" * 60)
