#!/usr/bin/env python
"""Train AI model on CR3BP trajectories."""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, 'src')

from ai.lstm_trainer import (
    CislunarLSTM, TrajectoryDataset, TrajectoryTrainer,
    generate_cr3bp_trajectories, create_sequences, StateNormalizer,
    evaluate_model
)
from torch.utils.data import DataLoader

print("=" * 60)
print("CISLUNAR SSA - AI TRAINING PIPELINE")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

# 1. Generate data
print("\n[1] Generating CR3BP trajectory dataset...")
trajectories = generate_cr3bp_trajectories(n_samples=1000, t_max=10.0, n_points=200)

# 2. Create sequences
print("\n[2] Creating sequences...")
seq_length = 25
X, y = create_sequences(trajectories, seq_length=seq_length)
print(f"  Total sequences: {len(X)}")
print(f"  Sequence shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# 3. Normalize
print("\n[3] Normalizing data...")
normalizer = StateNormalizer()
X_norm = normalizer.fit_transform(X.reshape(-1, 6)).reshape(X.shape)
y_norm = normalizer.transform(y)
normalizer.save('models/normalizer.npz')
print(f"  Mean: {normalizer.mean}")
print(f"  Std:  {normalizer.std}")

# 4. Split data
print("\n[4] Splitting train/val/test...")
n_total = len(X_norm)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

X_train, y_train = X_norm[:n_train], y_norm[:n_train]
X_val, y_val = X_norm[n_train:n_train+n_val], y_norm[n_train:n_train+n_val]
X_test, y_test = X_norm[n_train+n_val:], y_norm[n_train+n_val:]

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 5. Create DataLoaders
print("\n[5] Creating DataLoaders...")
batch_size = 64
train_dataset = TrajectoryDataset(X_train, y_train)
val_dataset = TrajectoryDataset(X_val, y_val)
test_dataset = TrajectoryDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 6. Build model
print("\n[6] Building model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

model = CislunarLSTM(input_size=6, hidden_size=128, num_layers=2, dropout=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# 7. Train
print("\n[7] Training...")
trainer = TrajectoryTrainer(model, device=device, lr=1e-3)
history = trainer.train(train_loader, val_loader, epochs=30, patience=10)

# 8. Evaluate
print("\n[8] Evaluating on test set...")
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.to(device)

true_traj, ai_traj, rms_error = evaluate_model(model, normalizer, X_test, y_test, n_steps=100)
print(f"  RMS position error (100 steps): {rms_error:.4f}")

# 9. Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'seq_length': seq_length
}, 'models/cislunar_lstm_final.pt')

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("  Model saved: models/cislunar_lstm_final.pt")
print("  Best model:  models/best_model.pt")
print("  Normalizer:  models/normalizer.npz")
