#!/usr/bin/env python
"""Score the trained LSTM against a constant-velocity baseline.

Truth is the CR3BP propagator on held-out arcs from the same seed as
training. The checkpoint ``models/cislunar_lstm.pt`` stores the
architecture, the normalizer, and the data settings used to rebuild
that holdout.

Position RMSE in normalized units is

    sqrt(mean((z_true - z_pred)^2))

on the three position components after the training z-score. The
constant-velocity baseline steps ``x <- x + v dt`` and leaves velocity
unchanged. It uses only the last state of the window; the LSTM uses
the whole window.
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from ai.lstm_trainer import (
    CHECKPOINT_PATH,
    EVAL_RESULTS_PATH,
    load_checkpoint,
    plot_bundle,
    prepare_datasets,
    score_against_baseline,
)


def _line(label, lstm_value, cv_value):
    winner = "LSTM" if lstm_value < cv_value else "constant velocity"
    print(f"  {label}")
    print(f"    LSTM:              {lstm_value:.6f}")
    print(f"    constant velocity: {cv_value:.6f}")
    print(f"    lower error:       {winner}")


def main():
    print("=" * 60)
    print("CISLUNAR SSA - AI EVALUATION")
    print("=" * 60)

    print("\n[1] Loading", CHECKPOINT_PATH)
    model, normalizer, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    print(f"  hidden={checkpoint['hidden_size']} layers={checkpoint['num_layers']} "
          f"seq_length={checkpoint['seq_length']} epochs_ran={checkpoint['epochs_ran']}")
    print(f"  best val MSE: {checkpoint['best_val_loss']:.6f}")

    print("\n[2] Rebuilding the holdout from the checkpoint seed...")
    data = prepare_datasets(
        n_samples=checkpoint["n_samples"],
        t_max=checkpoint["t_max"],
        n_points=checkpoint["n_points"],
        seq_length=checkpoint["seq_length"],
        seed=checkpoint["seed"],
        val_ratio=checkpoint["val_ratio"],
        test_ratio=checkpoint["test_ratio"],
        position_low=checkpoint["position_low"],
        position_high=checkpoint["position_high"],
        velocity_scale=checkpoint["velocity_scale"],
    )
    if not np.allclose(data["normalizer"].mean, normalizer.mean, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Rebuilt training mean does not match the checkpoint normalizer.")
    if not np.allclose(data["normalizer"].std, normalizer.std, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Rebuilt training std does not match the checkpoint normalizer.")

    print("\n[3] Scoring...")
    metrics = score_against_baseline(
        model,
        normalizer,
        data["test_trajs"],
        seq_length=checkpoint["seq_length"],
        dt=data["dt"],
        rollout_steps=checkpoint["rollout_steps"],
    )
    print(f"  Test arcs: {metrics['n_test_trajectories']}")
    print(f"  One-step windows: {metrics['n_onestep']}")
    print(f"  Rollout length: {metrics['rollout_steps']} steps "
          f"(dt = {data['dt']:.6f})")
    print("\n  Normalized position RMSE")
    _line("one-step (teacher forced)",
          metrics["lstm_onestep_pos_rmse_norm"],
          metrics["cv_onestep_pos_rmse_norm"])
    _line(f"{metrics['rollout_steps']}-step autoregressive rollout",
          metrics["lstm_rollout_pos_rmse_norm"],
          metrics["cv_rollout_pos_rmse_norm"])
    print("\n  Non-dimensional position RMSE (Earth-Moon distance = 1)")
    _line("one-step (teacher forced)",
          metrics["lstm_onestep_pos_rmse_nondim"],
          metrics["cv_onestep_pos_rmse_nondim"])
    _line(f"{metrics['rollout_steps']}-step autoregressive rollout",
          metrics["lstm_rollout_pos_rmse_nondim"],
          metrics["cv_rollout_pos_rmse_nondim"])

    beats = metrics["lstm_rollout_pos_rmse_norm"] < metrics["cv_rollout_pos_rmse_norm"]
    print("\n  LSTM rollout beats constant velocity "
          f"(normalized): {'yes' if beats else 'no'}")

    time, true_traj, lstm_traj, cv_traj = plot_bundle(
        model,
        normalizer,
        data["test_trajs"][0],
        seq_length=checkpoint["seq_length"],
        dt=data["dt"],
        rollout_steps=checkpoint["rollout_steps"],
    )
    np.savez(
        EVAL_RESULTS_PATH,
        true_trajectory=true_traj,
        ai_trajectory=lstm_traj,
        cv_trajectory=cv_traj,
        time=time,
        rms_error=metrics["lstm_rollout_pos_rmse_norm"],
        **metrics,
    )
    print(f"\n  Wrote {EVAL_RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
