#!/usr/bin/env python
"""Visualize AI vs Physics trajectory comparison."""

import sys
import numpy as np

sys.path.insert(0, 'src')

print("=" * 60)
print("CISLUNAR SSA - AI VISUALIZATION")
print("=" * 60)

# Load results
print("\n[1] Loading evaluation results...")
data = np.load('models/evaluation_results.npz')
true_traj = data['true_trajectory']
ai_traj = data['ai_trajectory']
time = data['time']
rms_error = float(data['rms_error'])

print(f"  Trajectory points: {len(true_traj)}")
print(f"  RMS Error: {rms_error:.4f}")

# Create comparison plot
print("\n[2] Creating comparison plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: XY trajectory comparison
ax = axes[0, 0]
ax.plot(true_traj[:, 0], true_traj[:, 1], 'b-', linewidth=1, alpha=0.7, label='Physics (True)')
ax.plot(ai_traj[:, 0], ai_traj[:, 1], 'r--', linewidth=1, alpha=0.7, label='AI Prediction')
ax.plot(true_traj[0, 0], true_traj[0, 1], 'go', markersize=10, label='Start')
ax.set_xlabel('x (non-dimensional)')
ax.set_ylabel('y (non-dimensional)')
ax.set_title('XY Trajectory Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: XZ trajectory comparison
ax = axes[0, 1]
ax.plot(true_traj[:, 0], true_traj[:, 2], 'b-', linewidth=1, alpha=0.7, label='Physics (True)')
ax.plot(ai_traj[:, 0], ai_traj[:, 2], 'r--', linewidth=1, alpha=0.7, label='AI Prediction')
ax.set_xlabel('x (non-dimensional)')
ax.set_zlabel('z (non-dimensional)')
ax.set_title('XZ Trajectory Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Position error over time
ax = axes[1, 0]
min_len = min(len(true_traj), len(ai_traj))
pos_error = np.sqrt(np.sum((true_traj[:min_len, :3] - ai_traj[:min_len, :3])**2, axis=1))
ax.plot(time[:min_len], pos_error, 'g-', linewidth=1)
ax.set_xlabel('Time (non-dimensional)')
ax.set_ylabel('Position Error')
ax.set_title(f'Position Error Over Time (RMS: {rms_error:.4f})')
ax.grid(True, alpha=0.3)

# Plot 4: Velocity comparison
ax = axes[1, 1]
ax.plot(time[:min_len], true_traj[:min_len, 3], 'b-', linewidth=1, alpha=0.7, label='True vx')
ax.plot(time[:min_len], ai_traj[:min_len, 3], 'r--', linewidth=1, alpha=0.7, label='AI vx')
ax.plot(time[:min_len], true_traj[:min_len, 4], 'c-', linewidth=1, alpha=0.7, label='True vy')
ax.plot(time[:min_len], ai_traj[:min_len, 4], 'm--', linewidth=1, alpha=0.7, label='AI vy')
ax.set_xlabel('Time (non-dimensional)')
ax.set_ylabel('Velocity')
ax.set_title('Velocity Component Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'Cislunar SSA - AI vs Physics\nRMS Position Error: {rms_error:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig('docs/ai_vs_physics.png', dpi=150, bbox_inches='tight')
print("  Saved: docs/ai_vs_physics.png")

# Create 3D plot
print("\n[3] Creating 3D plot...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
        'b-', linewidth=0.8, alpha=0.7, label='Physics (True)')
ax.plot(ai_traj[:, 0], ai_traj[:, 1], ai_traj[:, 2], 
        'r--', linewidth=0.8, alpha=0.7, label='AI Prediction')

# Earth and Moon
mu = 0.012150585609624
ax.scatter([-mu], [0], [0], color='blue', s=100, label='Earth')
ax.scatter([1-mu], [0], [0], color='gray', s=50, label='Moon')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Trajectory: AI vs Physics')
ax.legend()
ax.set_box_aspect([1,1,0.5])

plt.savefig('docs/ai_vs_physics_3d.png', dpi=150, bbox_inches='tight')
print("  Saved: docs/ai_vs_physics_3d.png")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("  docs/ai_vs_physics.png")
print("  docs/ai_vs_physics_3d.png")
