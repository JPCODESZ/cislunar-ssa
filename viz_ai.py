#!/usr/bin/env python
"""Plot one held-out arc: propagator, LSTM rollout, constant velocity.

Reads ``models/evaluation_results.npz`` written by ``eval_ai.py``.
The x-z panel is a 2D axes, so the vertical axis is labeled with
``set_ylabel``.
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from ai.lstm_trainer import EVAL_RESULTS_PATH


def _finite(traj):
    return np.isfinite(traj).all(axis=1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CISLUNAR SSA - AI VISUALIZATION")
    print("=" * 60)
    data = np.load(EVAL_RESULTS_PATH)
    true_traj = data["true_trajectory"]
    ai_traj = data["ai_trajectory"]
    cv_traj = data["cv_trajectory"]
    time = data["time"]
    lstm_roll = float(data["lstm_rollout_pos_rmse_norm"])
    cv_roll = float(data["cv_rollout_pos_rmse_norm"])
    print(f"  Test-set rollout position RMSE (normalized)")
    print(f"    LSTM: {lstm_roll:.6f}")
    print(f"    constant velocity: {cv_roll:.6f}")

    ai_ok = _finite(ai_traj)
    cv_ok = _finite(cv_traj)
    forecast_start = int(np.argmax(ai_ok))

    def _arc_rmse(mask, pred):
        diff = true_traj[mask, :3] - pred[mask, :3]
        return float(np.sqrt(np.mean(diff ** 2)))

    arc_lstm = _arc_rmse(ai_ok, ai_traj)
    arc_cv = _arc_rmse(cv_ok, cv_traj)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(true_traj[:, 0], true_traj[:, 1], "b-", linewidth=1.2, label="CR3BP propagator")
    ax.plot(ai_traj[ai_ok, 0], ai_traj[ai_ok, 1], "r--", linewidth=1.2, label="LSTM rollout")
    ax.plot(cv_traj[cv_ok, 0], cv_traj[cv_ok, 1], "g:", linewidth=1.4, label="Constant velocity")
    ax.plot(true_traj[forecast_start, 0], true_traj[forecast_start, 1], "ko", markersize=6, label="Forecast start")
    ax.set_xlabel("x (non-dimensional)")
    ax.set_ylabel("y (non-dimensional)")
    ax.set_title("XY trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.plot(true_traj[:, 0], true_traj[:, 2], "b-", linewidth=1.2, label="CR3BP propagator")
    ax.plot(ai_traj[ai_ok, 0], ai_traj[ai_ok, 2], "r--", linewidth=1.2, label="LSTM rollout")
    ax.plot(cv_traj[cv_ok, 0], cv_traj[cv_ok, 2], "g:", linewidth=1.4, label="Constant velocity")
    ax.set_xlabel("x (non-dimensional)")
    ax.set_ylabel("z (non-dimensional)")
    ax.set_title("XZ trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    lstm_err = np.sqrt(np.sum((true_traj[ai_ok, :3] - ai_traj[ai_ok, :3]) ** 2, axis=1))
    cv_err = np.sqrt(np.sum((true_traj[cv_ok, :3] - cv_traj[cv_ok, :3]) ** 2, axis=1))
    ax.plot(time[ai_ok], lstm_err, "r-", linewidth=1.2, label="LSTM")
    ax.plot(time[cv_ok], cv_err, "g:", linewidth=1.4, label="Constant velocity")
    ax.set_xlabel("Time (non-dimensional)")
    ax.set_ylabel("Position error (non-dimensional)")
    ax.set_title(
        f"This arc, non-dimensional RMSE: LSTM {arc_lstm:.3f}, constant velocity {arc_cv:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time, true_traj[:, 3], "b-", linewidth=1, label="True vx")
    ax.plot(time[ai_ok], ai_traj[ai_ok, 3], "r--", linewidth=1, label="LSTM vx")
    ax.plot(time[cv_ok], cv_traj[cv_ok, 3], color="darkgreen", linestyle=":", linewidth=1.2, label="CV vx")
    ax.plot(time, true_traj[:, 4], "c-", linewidth=1, label="True vy")
    ax.plot(time[ai_ok], ai_traj[ai_ok, 4], "m--", linewidth=1, label="LSTM vy")
    ax.set_xlabel("Time (non-dimensional)")
    ax.set_ylabel("Velocity (non-dimensional)")
    ax.set_title("Velocity components")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Held-out arc: propagator vs LSTM vs constant velocity\n"
        f"Test-set {int(data['rollout_steps'])}-step normalized position RMSE: "
        f"LSTM {lstm_roll:.4f}, constant velocity {cv_roll:.4f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("docs/ai_vs_physics.png", dpi=150, bbox_inches="tight")
    print("  Saved: docs/ai_vs_physics.png")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
            "b-", linewidth=1.0, label="CR3BP propagator")
    ax.plot(ai_traj[ai_ok, 0], ai_traj[ai_ok, 1], ai_traj[ai_ok, 2],
            "r--", linewidth=1.0, label="LSTM rollout")
    ax.plot(cv_traj[cv_ok, 0], cv_traj[cv_ok, 1], cv_traj[cv_ok, 2],
            "g:", linewidth=1.2, label="Constant velocity")
    mu = 0.012150585609624
    ax.scatter([-mu], [0], [0], color="blue", s=80, label="Earth")
    ax.scatter([1 - mu], [0], [0], color="gray", s=40, label="Moon")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D trajectory: propagator, LSTM, constant velocity")
    ax.legend()
    fig.savefig("docs/ai_vs_physics_3d.png", dpi=150, bbox_inches="tight")
    print("  Saved: docs/ai_vs_physics_3d.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
