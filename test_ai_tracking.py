#!/usr/bin/env python3
"""AI trajectory prediction and tracking visualization."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from physics import compute_lagrange_points, propagate
from ai.trajectory_predictor import generate_training_data, SimpleLSTMPredictor
import plotly.graph_objects as go

print("=" * 60)
print("AI TRAJECTORY TRACKING")
print("=" * 60)

print("\n[1] Generating training data from CR3BP...")
X_train, y_train = generate_training_data(n_samples=200, t_span=(0, 3), n_points=50)
print(f"  Generated {len(X_train)} state transitions")

print("\n[2] Training AI predictor...")
predictor = SimpleLSTMPredictor(input_size=6, hidden_size=128, num_layers=3)
predictor.build_model()
predictor.train(X_train, y_train, epochs=20)

print("\n[3] Testing on real trajectory...")
L = compute_lagrange_points()
state0 = [L['L1'][0] + 0.02, 0.05, 0.02, 0.0, 0.15, 0.05]
t_eval = np.linspace(0, 8, 800)
true_sol = propagate(state0, (0, 8), t_eval)

# AI prediction
print("  Running AI prediction...")
ai_states = [state0]
dt = t_eval[1] - t_eval[0]
for i in range(1, len(t_eval)):
    pred = predictor.predict(ai_states[-10:] if len(ai_states) >= 10 else ai_states)
    ai_states.append(pred)

ai_states = np.array(ai_states)

print("\n[4] Creating comparison visualization...")
fig = go.Figure()

# True trajectory
fig.add_trace(go.Scatter3d(
    x=true_sol.y[0], y=true_sol.y[1], z=true_sol.y[2],
    mode='lines',
    line=dict(width=4, color='cyan'),
    name='True Physics'
))

# AI predicted trajectory
fig.add_trace(go.Scatter3d(
    x=ai_states[:, 0], y=ai_states[:, 1], z=ai_states[:, 2],
    mode='lines',
    line=dict(width=2, color='magenta', dash='dash'),
    name='AI Predicted'
))

# Start point
fig.add_trace(go.Scatter3d(
    x=[state0[0]], y=[state0[1]], z=[state0[2]],
    mode='markers',
    marker=dict(size=10, color='green', symbol='diamond'),
    name='Start'
))

# Earth and Moon
fig.add_trace(go.Scatter3d(
    x=[-0.012], y=[0], z=[0],
    mode='markers',
    marker=dict(size=12, color='blue'),
    name='Earth'
))
fig.add_trace(go.Scatter3d(
    x=[0.988], y=[0], z=[0],
    mode='markers',
    marker=dict(size=6, color='gray'),
    name='Moon'
))

fig.update_layout(
    title="AI Trajectory Prediction vs True Physics<br><sub>LSTM-based prediction in CR3BP</sub>",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        aspectmode='data',
        bgcolor='rgb(10,10,10)'
    ),
    paper_bgcolor='rgb(5,5,5)',
    plot_bgcolor='rgb(5,5,5)',
    font=dict(color='white'),
    width=1000,
    height=800
)

fig.write_html("docs/ai_tracking.html")
print("  Saved: docs/ai_tracking.html")

# Calculate error
error = np.sqrt(np.mean((true_sol.y[:3, :len(ai_states)].T - ai_states[:, :3])**2))
print(f"\n  RMS position error: {error:.4f}")

print("\n" + "=" * 60)
print("OPEN: docs/ai_tracking.html")
print("=" * 60)
