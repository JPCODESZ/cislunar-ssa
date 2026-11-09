"""LSTM training pipeline for one-step CR3BP state prediction.

The network maps a short window of non-dimensional states
``[x, y, z, vx, vy, vz]`` to the next sample. ``train_ai.py`` is the
supported entry point and writes ``models/cislunar_lstm.pt``.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import sys
sys.path.insert(0, "src")
from physics.cr3bp import propagate


CHECKPOINT_PATH = os.path.join("models", "cislunar_lstm.pt")
EVAL_RESULTS_PATH = os.path.join("models", "evaluation_results.npz")

# Initial conditions for the short CPU demo. The box covers the
# Earth–Moon corridor (L1 through L2) with modest rotating-frame speeds.
DEMO_POSITION_LOW = (0.50, -0.45, -0.08)
DEMO_POSITION_HIGH = (1.25, 0.45, 0.08)
DEMO_VELOCITY_SCALE = (0.03, 0.06, 0.02)


class CislunarLSTM(nn.Module):
    """LSTM that predicts the next 6-D CR3BP state."""

    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TrajectoryDataset(Dataset):
    """Windows of states and the single next state."""

    def __init__(self, sequences, targets):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StateNormalizer:
    """Per-component z-score, fit on training states only."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

    def fit(self, data):
        data = np.asarray(data, dtype=float)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, data):
        return (np.asarray(data, dtype=float) - self.mean) / self.std

    def inverse_transform(self, data):
        return np.asarray(data, dtype=float) * self.std + self.mean

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        return self


class TrajectoryTrainer:
    """Adam + MSE trainer. Restores the best validation weights."""

    def __init__(self, model, device="cpu", lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.history = {"train_loss": [], "val_loss": []}
        self.best_state_dict = None
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def train(self, train_loader, val_loader, epochs=12, patience=6):
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.history


def generate_cr3bp_trajectories(
    n_samples,
    t_max,
    n_points,
    seed,
    position_low=DEMO_POSITION_LOW,
    position_high=DEMO_POSITION_HIGH,
    velocity_scale=DEMO_VELOCITY_SCALE,
):
    """Integrate ``n_samples`` random states. Drop arcs that fail or blow up."""
    rng = np.random.default_rng(seed)
    position_low = np.asarray(position_low, dtype=float)
    position_high = np.asarray(position_high, dtype=float)
    velocity_scale = np.asarray(velocity_scale, dtype=float)
    t_eval = np.linspace(0.0, t_max, n_points)
    trajectories = []

    for _ in range(n_samples):
        position = rng.uniform(position_low, position_high)
        velocity = rng.normal(0.0, velocity_scale)
        state0 = np.concatenate([position, velocity])
        sol = propagate(state0, (0.0, t_max), t_eval)
        if not sol.success:
            continue
        states = np.asarray(sol.y.T, dtype=float)
        if states.shape != (n_points, 6) or not np.isfinite(states).all():
            continue
        if np.max(np.abs(states[:, :3])) > 5.0:
            continue
        trajectories.append(states)
    return trajectories


def create_sequences(trajectories, seq_length):
    """Sliding windows. Target is the state immediately after each window."""
    sequences, targets = [], []
    for traj in trajectories:
        for i in range(len(traj) - seq_length):
            sequences.append(traj[i:i + seq_length])
            targets.append(traj[i + seq_length])
    if not sequences:
        raise RuntimeError("No training windows. Increase n_points or lower seq_length.")
    return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float)


def prepare_datasets(
    n_samples,
    t_max,
    n_points,
    seq_length,
    seed,
    val_ratio=0.15,
    test_ratio=0.15,
    position_low=DEMO_POSITION_LOW,
    position_high=DEMO_POSITION_HIGH,
    velocity_scale=DEMO_VELOCITY_SCALE,
):
    """Same seed rebuilds the same train, validation, and test arcs."""
    trajectories = generate_cr3bp_trajectories(
        n_samples, t_max, n_points, seed,
        position_low=position_low,
        position_high=position_high,
        velocity_scale=velocity_scale,
    )
    if len(trajectories) < 6:
        raise RuntimeError(f"Only {len(trajectories)} trajectories integrated.")

    order = np.random.default_rng(seed).permutation(len(trajectories))
    n_test = max(1, int(round(len(trajectories) * test_ratio)))
    n_val = max(1, int(round(len(trajectories) * val_ratio)))
    if n_test + n_val >= len(trajectories):
        n_test, n_val = 1, 1
    test_trajs = [trajectories[i] for i in order[:n_test]]
    val_trajs = [trajectories[i] for i in order[n_test:n_test + n_val]]
    train_trajs = [trajectories[i] for i in order[n_test + n_val:]]

    normalizer = StateNormalizer().fit(np.concatenate(train_trajs, axis=0))

    def _windows(trajs):
        raw_x, raw_y = create_sequences(trajs, seq_length)
        x_norm = normalizer.transform(raw_x.reshape(-1, 6)).reshape(raw_x.shape)
        y_norm = normalizer.transform(raw_y)
        return x_norm, y_norm

    x_train, y_train = _windows(train_trajs)
    x_val, y_val = _windows(val_trajs)
    dt = t_max / (n_points - 1)
    return {
        "train_trajs": train_trajs,
        "val_trajs": val_trajs,
        "test_trajs": test_trajs,
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "normalizer": normalizer,
        "dt": dt,
        "n_integrated": len(trajectories),
    }


def _normalized_position_rmse(true_states, pred_states, normalizer):
    true_n = normalizer.transform(true_states)[:, :3]
    pred_n = normalizer.transform(pred_states)[:, :3]
    return float(np.sqrt(np.mean((true_n - pred_n) ** 2)))


def _nondim_position_rmse(true_states, pred_states):
    diff = np.asarray(true_states)[:, :3] - np.asarray(pred_states)[:, :3]
    return float(np.sqrt(np.mean(diff ** 2)))


def constant_velocity_states(last_state, n_steps, dt):
    """Hold the rotating-frame velocity fixed and step position by v*dt."""
    state = np.asarray(last_state, dtype=float).copy()
    preds = []
    for _ in range(n_steps):
        state = state.copy()
        state[:3] = state[:3] + state[3:] * dt
        preds.append(state)
    return np.asarray(preds)


def lstm_rollout(model, normalizer, window, n_steps):
    """Autoregressive rollout. ``window`` is physical (seq_length, 6)."""
    current = normalizer.transform(np.asarray(window, dtype=float)).copy()
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            inp = torch.from_numpy(current).float().unsqueeze(0)
            pred_n = model(inp).detach().cpu().numpy()[0]
            preds.append(normalizer.inverse_transform(pred_n))
            current = np.roll(current, -1, axis=0)
            current[-1] = pred_n
    return np.asarray(preds)


def score_against_baseline(model, normalizer, test_trajs, seq_length, dt, rollout_steps):
    """Teacher-forced one-step error and an autoregressive rollout.

    Both errors are position RMSE. The normalized figure applies the
    training z-score and then takes the three position components.
    Constant velocity sees only the last state of each window; the LSTM
    sees the whole window. The numerical propagator is the truth, not a
    competitor: its error against itself is zero.
    """
    model.eval()
    one_true = []
    one_lstm = []
    one_cv = []
    roll_true = []
    roll_lstm = []
    roll_cv = []

    for traj in test_trajs:
        horizon = min(rollout_steps, len(traj) - seq_length)
        for i in range(len(traj) - seq_length):
            window = traj[i:i + seq_length]
            target = traj[i + seq_length]
            pred = lstm_rollout(model, normalizer, window, 1)[0]
            cv = constant_velocity_states(window[-1], 1, dt)[0]
            one_true.append(target)
            one_lstm.append(pred)
            one_cv.append(cv)

        window = traj[:seq_length]
        truth = traj[seq_length:seq_length + horizon]
        roll_true.append(truth)
        roll_lstm.append(lstm_rollout(model, normalizer, window, horizon))
        roll_cv.append(constant_velocity_states(window[-1], horizon, dt))

    one_true = np.asarray(one_true)
    one_lstm = np.asarray(one_lstm)
    one_cv = np.asarray(one_cv)
    roll_true = np.concatenate(roll_true, axis=0)
    roll_lstm = np.concatenate(roll_lstm, axis=0)
    roll_cv = np.concatenate(roll_cv, axis=0)

    return {
        "n_test_trajectories": len(test_trajs),
        "n_onestep": len(one_true),
        "rollout_steps": int(min(rollout_steps, test_trajs[0].shape[0] - seq_length)),
        "lstm_onestep_pos_rmse_norm": _normalized_position_rmse(one_true, one_lstm, normalizer),
        "cv_onestep_pos_rmse_norm": _normalized_position_rmse(one_true, one_cv, normalizer),
        "lstm_rollout_pos_rmse_norm": _normalized_position_rmse(roll_true, roll_lstm, normalizer),
        "cv_rollout_pos_rmse_norm": _normalized_position_rmse(roll_true, roll_cv, normalizer),
        "lstm_onestep_pos_rmse_nondim": _nondim_position_rmse(one_true, one_lstm),
        "cv_onestep_pos_rmse_nondim": _nondim_position_rmse(one_true, one_cv),
        "lstm_rollout_pos_rmse_nondim": _nondim_position_rmse(roll_true, roll_lstm),
        "cv_rollout_pos_rmse_nondim": _nondim_position_rmse(roll_true, roll_cv),
    }


def plot_bundle(model, normalizer, trajectory, seq_length, dt, rollout_steps):
    """One held-out arc, with NaNs on the seed so plots start at the forecast."""
    horizon = min(rollout_steps, len(trajectory) - seq_length)
    window = trajectory[:seq_length]
    lstm = np.full_like(trajectory, np.nan)
    cv = np.full_like(trajectory, np.nan)
    lstm[seq_length:seq_length + horizon] = lstm_rollout(model, normalizer, window, horizon)
    cv[seq_length:seq_length + horizon] = constant_velocity_states(window[-1], horizon, dt)
    time = np.arange(len(trajectory)) * dt
    return time, trajectory, lstm, cv


def save_checkpoint(path, model, normalizer, config, history, best_val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_size": 6,
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "seq_length": int(config["seq_length"]),
        "normalizer_mean": [float(v) for v in normalizer.mean],
        "normalizer_std": [float(v) for v in normalizer.std],
        "seed": int(config["seed"]),
        "n_samples": int(config["n_samples"]),
        "t_max": float(config["t_max"]),
        "n_points": int(config["n_points"]),
        "val_ratio": float(config["val_ratio"]),
        "test_ratio": float(config["test_ratio"]),
        "rollout_steps": int(config["rollout_steps"]),
        "position_low": [float(v) for v in config["position_low"]],
        "position_high": [float(v) for v in config["position_high"]],
        "velocity_scale": [float(v) for v in config["velocity_scale"]],
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "epochs_requested": int(config["epochs"]),
        "epochs_ran": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss),
        "train_loss": [float(v) for v in history["train_loss"]],
        "val_loss": [float(v) for v in history["val_loss"]],
    }
    torch.save(payload, path)
    return payload


def load_checkpoint(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = CislunarLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    normalizer = StateNormalizer(checkpoint["normalizer_mean"], checkpoint["normalizer_std"])
    return model, normalizer, checkpoint


def evaluate_model(model, normalizer, test_sequences, test_targets, n_steps=20):
    """Legacy single-window rollout. Prefer ``score_against_baseline``."""
    model.eval()
    idx = 0
    seq = np.asarray(test_sequences[idx], dtype=float).copy()
    true_rows = [seq[-1]]
    for i in range(n_steps):
        if idx + i < len(test_targets):
            true_rows.append(test_targets[idx + i])
    current = seq.copy()
    ai_rows = [seq[-1]]
    with torch.no_grad():
        for _ in range(n_steps):
            inp = torch.from_numpy(current).float().unsqueeze(0)
            pred = model(inp).detach().cpu().numpy()[0]
            ai_rows.append(pred)
            current = np.roll(current, -1, axis=0)
            current[-1] = pred
    true_traj = normalizer.inverse_transform(np.asarray(true_rows))
    ai_traj = normalizer.inverse_transform(np.asarray(ai_rows))
    n = min(len(true_traj), len(ai_traj))
    rms_error = float(np.sqrt(np.mean((true_traj[:n, :3] - ai_traj[:n, :3]) ** 2)))
    return true_traj[:n], ai_traj[:n], rms_error
