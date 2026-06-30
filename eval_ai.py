#!/usr/bin/env python
"""Evaluate trained AI model and create visualizations."""

import sys
import numpy as np
import torch

sys.path.insert(0, 'src')

from ai.lstm_trainer import CislunarLSTM, StateNormalizer
from physics.cr3bp import propagate
from physics.lagrange import compute_lagrange_points

print("=" * 60)
print("CISLUNAR SSA - AI EVALUATION & INFERENCE")
print("=" * 60)

# Load model
print("\n[1] Loading model...")
checkpoint = torch.load('models/cislunar_lstm_final.pt', map_location='cpu')
model = CislunarLSTM(
    input_size=6,
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load normalizer
normalizer = StateNormalizer()
normalizer.load('models/normalizer.npz')

seq_length = checkpoint['seq_length']
print(f"  Sequence length: {seq_length}")

# Test 1: Compare AI vs Physics propagation
print("\n[2] AI vs Physics Comparison...")

# Start near L1
L = compute_lagrange_points()
state0 = [L['L1'][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]

# True physics propagation
t_eval = np.linspace(0, 8, 800)
true_sol = propagate(state0, (0, 8), t_eval)

# AI prediction
print("  Running AI prediction...")
ai_states = [state0]

# Build initial sequence using physics
init_seq = []
for i in range(seq_length):
    t = i * (8 / 800)
    sol = propagate(state0, (0, t), [t])
    if sol.success:
        init_seq.append(sol.y[:, -1])
    else:
        init_seq.append(state0)

# Normalize and predict
current_seq = normalizer.transform(np.array(init_seq))

with torch.no_grad():
    for i in range(len(t_eval) - seq_length):
        inp = torch.FloatTensor(current_seq).unsqueeze(0)
        pred_norm = model(inp).numpy()[0]
        pred = normalizer.inverse_transform(pred_norm)
        ai_states.append(pred)
        
        # Roll sequence
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = pred_norm

ai_states = np.array(ai_states)

# Compute error
min_len = min(len(true_sol.y[0]), len(ai_states))
error = np.sqrt(np.mean((true_sol.y[:3, :min_len].T - ai_states[:min_len, :3])**2))
print(f"  RMS position error: {error:.4f}")

# Save results
print("\n[3] Saving results...")
np.savez('models/evaluation_results.npz',
         true_trajectory=true_sol.y.T,
         ai_trajectory=ai_states,
         time=true_sol.t,
         rms_error=error)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"  Results saved: models/evaluation_results.npz")
print(f"  RMS Error: {error:.4f}")
print(f"  {'EXCELLENT' if error < 0.1 else 'GOOD' if error < 0.5 else 'NEEDS IMPROVEMENT'}")
