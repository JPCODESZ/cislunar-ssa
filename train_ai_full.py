#!/usr/bin/env python
"""Main training script for cislunar trajectory prediction."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch

from ai.data_generator import generate_trajectory_dataset, create_sequence_dataset, normalize_states
from ai.lstm_predictor import CislunarLSTM, CislunarGRU, PhysicsInformedLSTM
from ai.trainer import TrajectoryTrainer
from ai.utils import split_train_val_test, create_dataloaders
from ai.evaluator import TrajectoryEvaluator

print("=" * 60)
print("CISLUNAR SSA - AI TRAINING PIPELINE")
print("=" * 60)

# 1. Generate data
print("\n[1] Generating trajectory dataset...")
trajectories, metadata = generate_trajectory_dataset(
    n_trajectories=1000,
    t_span=(0, 8),
    n_points=150,
    regions=['cislunar', 'near_earth', 'near_moon', 'l1_halo', 'l2_halo'],
    add_noise=True,
    noise_std=5e-5,
    seed=42
)
print(f"  Generated {len(trajectories)} trajectories")

# 2. Create sequences
print("\n[2] Creating sequence dataset...")
seq_length = 25
pred_horizon = 1
X, y = create_sequence_dataset(trajectories, seq_length=seq_length, pred_horizon=pred_horizon, stride=2)
print(f"  Total sequences: {len(X)}")
print(f"  Sequence shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# 3. Normalize
print("\n[3] Normalizing data...")
X_norm, mean, std = normalize_states(X)
y_norm = (y - mean) / std
print(f"  Mean: {mean}")
print(f"  Std:  {std}")

# 4. Split
print("\n[4] Splitting train/val/test...")
X_train, X_val, X_test = split_train_val_test(X_norm, train_ratio=0.7, val_ratio=0.15, seed=42)
y_train, y_val, y_test = split_train_val_test(y_norm, train_ratio=0.7, val_ratio=0.15, seed=42)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 5. Create dataloaders
print("\n[5] Creating DataLoaders...")
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
)

# 6. Build model
print("\n[6] Building model...")
model = CislunarLSTM(
    input_size=6,
    hidden_size=128,
    num_layers=3,
    output_size=6,
    pred_horizon=1,
    dropout=0.15,
    bidirectional=False
)
print(f"  Model: CislunarLSTM")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 7. Train
print("\n[7] Training...")
trainer = TrajectoryTrainer(
    model,
    device='auto',
    learning_rate=1e-3,
    weight_decay=1e-5,
    physics_weight=0.0
)

history = trainer.train(
    train_loader,
    val_loader,
    epochs=50,
    patience=12,
    checkpoint_path='models/best_lstm.pt'
)

# 8. Evaluate
print("\n[8] Evaluating...")
evaluator = TrajectoryEvaluator(model, device=trainer.device)
metrics, predictions = evaluator.evaluate_dataset(X_test, y_test)
evaluator.print_report(metrics)

# Save normalization params
import json
norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
with open('models/normalization_params.json', 'w') as f:
    json.dump(norm_params, f)
print("\nNormalization params saved to models/normalization_params.json")

# Save training history
trainer.save_history('models/training_history.json')

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("  Best model: models/best_lstm.pt")
print("  Norm params: models/normalization_params.json")
print("  History: models/training_history.json")
print("=" * 60)
