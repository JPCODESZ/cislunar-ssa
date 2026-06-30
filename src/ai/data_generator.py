"""CR3BP trajectory data generation for ML training."""

import numpy as np
import sys
sys.path.insert(0, '../src')

from physics.cr3bp import propagate
from physics.lagrange import compute_lagrange_points

MU = 0.012150585609624
L = compute_lagrange_points()

def generate_random_state(region='cislunar', center=None, spread=0.5, velocity_scale=0.2):
    """Generate a random initial state in the specified region."""
    if center is None:
        if region == 'cislunar':
            center = [0.5, 0.0, 0.0]
        elif region == 'near_earth':
            center = [-0.5, 0.0, 0.0]
        elif region == 'near_moon':
            center = [1.2, 0.0, 0.0]
        elif region == 'l1_halo':
            center = [L['L1'][0], 0.0, 0.0]
        elif region == 'l2_halo':
            center = [L['L2'][0], 0.0, 0.0]
        else:
            center = [0.0, 0.0, 0.0]
    
    x = center[0] + np.random.uniform(-spread, spread)
    y = center[1] + np.random.uniform(-spread, spread)
    z = center[2] + np.random.uniform(-spread*0.3, spread*0.3)
    vx = np.random.normal(0, velocity_scale)
    vy = np.random.normal(0, velocity_scale)
    vz = np.random.normal(0, velocity_scale*0.5)
    
    return [x, y, z, vx, vy, vz]

def generate_trajectory_dataset(
    n_trajectories=500,
    t_span=(0, 8),
    n_points=150,
    regions=None,
    add_noise=True,
    noise_std=5e-5,
    seed=None
):
    """Generate dataset of CR3BP trajectories for ML training."""
    if seed is not None:
        np.random.seed(seed)
    
    if regions is None:
        regions = ['cislunar', 'near_earth', 'near_moon', 'l1_halo', 'l2_halo']
    
    trajectories = []
    metadata = []
    
    for i in range(n_trajectories):
        region = np.random.choice(regions)
        state0 = generate_random_state(region=region)
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = propagate(state0, t_span, t_eval)
        
        if sol.success:
            states = sol.y.T
            
            if add_noise:
                states += np.random.normal(0, noise_std, states.shape)
            
            trajectories.append((sol.t, states))
            metadata.append({
                'id': i,
                'region': region,
                'initial_state': state0,
                't_span': t_span,
                'n_points': n_points
            })
    
    return trajectories, metadata

def create_sequence_dataset(trajectories, seq_length=20, pred_horizon=1, stride=1):
    """Convert trajectories into supervised learning sequences."""
    X, y = [], []
    
    for t, states in trajectories:
        n = len(states)
        for i in range(0, n - seq_length - pred_horizon + 1, stride):
            X.append(states[i:i+seq_length])
            if pred_horizon == 1:
                y.append(states[i+seq_length])
            else:
                y.append(states[i+seq_length:i+seq_length+pred_horizon])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def normalize_states(states, mean=None, std=None):
    """Z-score normalization."""
    if mean is None:
        mean = np.mean(states, axis=(0, 1) if states.ndim == 3 else 0)
    if std is None:
        std = np.std(states, axis=(0, 1) if states.ndim == 3 else 0)
        std = np.where(std == 0, 1.0, std)
    return (states - mean) / std, mean, std

def denormalize_states(states_norm, mean, std):
    """Reverse Z-score normalization."""
    return states_norm * std + mean

def augment_trajectory(trajectory, methods=['noise', 'time_reverse', 'scale']):
    """Augment a single trajectory."""
    t, states = trajectory
    augmented = []
    
    for method in methods:
        if method == 'noise':
            noise = np.random.normal(0, 1e-4, states.shape)
            augmented.append((t, states + noise))
        elif method == 'time_reverse':
            states_rev = states[::-1].copy()
            states_rev[:, 3:] *= -1
            augmented.append((t, states_rev))
        elif method == 'scale':
            scale = np.random.uniform(0.95, 1.05)
            states_scaled = states.copy()
            states_scaled[:, :3] *= scale
            states_scaled[:, 3:] *= scale**(-0.5)
            augmented.append((t, states_scaled))
    
    return augmented
