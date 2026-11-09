#!/usr/bin/env python
"""Train the CR3BP next-state LSTM and save models/cislunar_lstm.pt.

Settings below are the short CPU demo: a few dozen arcs and about ten
epochs. Run from the repository root. ``train_ai_full.py`` calls this
same function. ``eval_ai.py`` loads the checkpoint written here.
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from ai.lstm_trainer import (
    CHECKPOINT_PATH,
    CislunarLSTM,
    TrajectoryDataset,
    TrajectoryTrainer,
    prepare_datasets,
    save_checkpoint,
)

# Documented demo settings. Changing these changes the checkpoint that
# eval_ai.py rebuilds its holdout from.
CONFIG = {
    "seed": 42,
    "n_samples": 36,
    "t_max": 3.0,
    "n_points": 48,
    "seq_length": 12,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "epochs": 40,
    "patience": 8,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "rollout_steps": 20,
    "position_low": (0.50, -0.45, -0.08),
    "position_high": (1.25, 0.45, 0.08),
    "velocity_scale": (0.03, 0.06, 0.02),
}


def main():
    print("=" * 60)
    print("CISLUNAR SSA - AI TRAINING")
    print("=" * 60)
    print("  Checkpoint:", CHECKPOINT_PATH)
    for key in ("seed", "n_samples", "t_max", "n_points", "seq_length", "epochs"):
        print(f"  {key}: {CONFIG[key]}")

    torch.manual_seed(CONFIG["seed"])

    print("\n[1] Integrating training arcs...")
    data = prepare_datasets(
        n_samples=CONFIG["n_samples"],
        t_max=CONFIG["t_max"],
        n_points=CONFIG["n_points"],
        seq_length=CONFIG["seq_length"],
        seed=CONFIG["seed"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        position_low=CONFIG["position_low"],
        position_high=CONFIG["position_high"],
        velocity_scale=CONFIG["velocity_scale"],
    )
    print(f"  Integrated: {data['n_integrated']}")
    print(f"  Train arcs: {len(data['train_trajs'])}, "
          f"val: {len(data['val_trajs'])}, test: {len(data['test_trajs'])}")
    print(f"  Train windows: {len(data['X_train'])}, val windows: {len(data['X_val'])}")
    print(f"  State mean: {data['normalizer'].mean}")
    print(f"  State std:  {data['normalizer'].std}")

    train_loader = DataLoader(
        TrajectoryDataset(data["X_train"], data["y_train"]),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TrajectoryDataset(data["X_val"], data["y_val"]),
        batch_size=CONFIG["batch_size"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CislunarLSTM(
        input_size=6,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[2] Training on {device} ({n_params:,} parameters)...")
    trainer = TrajectoryTrainer(model, device=device, lr=CONFIG["learning_rate"])
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG["epochs"],
        patience=CONFIG["patience"],
    )

    save_checkpoint(CHECKPOINT_PATH, model, data["normalizer"], CONFIG, history, trainer.best_val_loss)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val MSE (normalized states): {trainer.best_val_loss:.6f}")
    print(f"  Epochs run: {len(history['train_loss'])}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print("  Next: python eval_ai.py && python viz_ai.py")


if __name__ == "__main__":
    main()
