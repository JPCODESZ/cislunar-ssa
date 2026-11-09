# Cislunar SSA

Open-source propagator for space situational awareness in the Earth–Moon system. States are integrated in the Circular Restricted Three-Body Problem (CR3BP): the Earth–Moon distance is 1, and the mass parameter is μ = 0.012150585609624.

![Trajectory of a test particle started near L1, in the Earth–Moon rotating frame](docs/test_trajectory.png)

## What this is

The toolkit propagates spacecraft and debris in the rotating frame, locates the Lagrange points, checks the Jacobi integral, and builds synthetic catalogs for close-approach screening. A separate PyTorch pipeline learns a one-step map from a short state history to the next CR3BP state, so a tracked object can be rolled forward and compared with the numerical integrator.

Coordinates are non-dimensional unless a routine says otherwise. Earth sits at (−μ, 0, 0) and the Moon at (1 − μ, 0, 0).

## Physics

- **CR3BP equations of motion** in the rotating frame, integrated with SciPy `RK45` (`rtol=1e-9`, `atol=1e-12`) through `physics.propagate`.
- **Lagrange points** L1–L5. The collinear points are roots of the usual quintic; L4 and L5 are the equilateral solutions. `stability_eigenvalues` returns the planar linearization at a point.
- **Jacobi integral**, the conserved energy-like quantity of the CR3BP, plus a check that it stays within a tolerance along a propagated arc.
- **Reference environments** for Earth–Moon and Sun–Earth mass ratios, with LEO and GEO constants (altitudes, periods, circular speed).
- **Validation helpers** that compare computed Lagrange points with the published NASA/JPL coordinates stored in `physics.validation`.
- **Halo orbits**: a linearized L1/L2 initial guess, and a single-shooting corrector (`orbits.halo_differential_correction.single_shooter`) that adjusts x, z, and vy so the half-period x–z crossing approaches y = vx = vz = 0. The period stays at the value passed in.
- **Perturbations**, implemented as extra accelerations: solar gravity (Sun on a circular path about the Earth–Moon barycenter) and a simplified solar-radiation-pressure term. The default `propagate` call is pure CR3BP; these accelerations are not added inside it.
- **Debris catalog** of synthetic cislunar states (id, 6-vector, size, epoch) and a pairwise conjunction screen on position distance. LEO and GEO catalog regions are not implemented.
- **Plots**: Matplotlib in the rotating frame, and Plotly figures for the 3D Earth–Moon system, debris clouds, and an animated trajectory.

Sources for the constants and the halo literature are listed in [`docs/references.md`](docs/references.md).

## AI tracking

Tracking is next-state regression on synthetic CR3BP trajectories. Each example is a window of 6-vectors `[x, y, z, vx, vy, vz]`; the label is the state one step ahead. Two entry points share that idea and write different checkpoints. No training log or error table is stored in the repository, so this README does not quote an accuracy number.

`src/ai/__init__.py` imports `src/ai/lstm_trainer.py`. That module, as committed, has a pasted shell fragment inside `TrajectoryDataset.__init__` (around line 49), which is a syntax error. Importing the `ai` package therefore fails, and `train_ai.py`, `train_ai_full.py`, and `eval_ai.py` do not run until that fragment is removed. The same file repeats a complete copy of the module after the paste.

### `train_ai.py`

Short pipeline on top of `src/ai/lstm_trainer.py`.

1. Integrate 1000 random cislunar initial conditions to non-dimensional time 10 (200 samples each). Failed integrations are dropped.
2. Cut sliding windows of length 25. The target is the next state.
3. Fit a per-component z-score normalizer and save it to `models/normalizer.npz`.
4. Take a contiguous 70% / 15% / 15% split (the sequences are not shuffled).
5. Train `CislunarLSTM`: unidirectional LSTM, input 6, hidden 128, 2 layers, dropout 0.2, then a linear layer 128 → 64, ReLU, dropout, and a linear layer 64 → 6. Loss is MSE, optimizer is Adam at `1e-3` with `ReduceLROnPlateau`, up to 30 epochs and patience 10.
6. Write the best weights as a raw `state_dict` to `models/best_model.pt`, and a small hyperparameter checkpoint to `models/cislunar_lstm_final.pt`.
7. Roll one random test window forward 100 steps and print the RMS of the position components after undoing the normalization.

### `train_ai_full.py`

Wider data pipeline. It uses `ai.data_generator`, `ai.lstm_predictor.CislunarLSTM`, `ai.trainer.TrajectoryTrainer`, and `ai.evaluator.TrajectoryEvaluator`.

1. Integrate 1000 trajectories on t ∈ [0, 8] (150 samples), drawn from `cislunar`, `near_earth`, `near_moon`, `l1_halo`, and `l2_halo`, with Gaussian state noise of standard deviation `5e-5` and seed 42.
2. Windows of length 25, prediction horizon 1, stride 2.
3. Z-score the states and save `mean` / `std` to `models/normalization_params.json`.
4. Shuffle into 70% / 15% / 15% with seed 42. Batch size 128.
5. Train a second `CislunarLSTM`: input 6, hidden 128, 3 layers, dropout 0.15, unidirectional. The head is linear 128 → 256, ReLU, dropout, linear 256 → 128, ReLU, dropout, linear 128 → 6. Optimizer is AdamW (`1e-3`, weight decay `1e-5`) with gradient clipping. `physics_weight` is 0, so the physics residual is off. Up to 50 epochs, patience 12.
6. Save the best epoch to `models/best_lstm.pt` (weights, optimizer, train loss, validation loss) and the loss history to `models/training_history.json`.
7. Score the test set one step ahead. The printed metrics are overall RMSE, MAE, position RMSE, velocity RMSE, and per-axis RMSE, all in **normalized** units.

The same package also defines a GRU with the same head shape, and a `PhysicsInformedLSTM` whose forward pass returns a next state plus an acceleration that can be penalized against the CR3BP acceleration. `train_ai_full.py` does not instantiate either of those. `TrajectoryEvaluator.evaluate_long_horizon` can roll a trajectory out and report position RMSE, endpoint drift, and Jacobi mismatch at horizons 10, 50, 100, and 200; the training script calls only the one-step evaluation.

`src/ai/trajectory_predictor.py` (`SimpleLSTMPredictor`, used by `test_ai_tracking.py`) is a placeholder. Its `predict` method integrates the last state with the CR3BP propagator for 0.1 time units. It does not train the networks above.

### `eval_ai.py`

Loads `models/cislunar_lstm_final.pt` and `models/normalizer.npz` from the short pipeline, rebuilds that 2-layer network, and compares it with physics on one arc. The initial condition is L1 shifted by +0.01 in x, with vy = 0.1. The truth is `propagate` from t = 0 to 8 (800 samples). The network is seeded with the first `seq_length` states of that arc, then rolled forward by feeding each prediction back into the window. It prints the RMS position error in non-dimensional units, labels the run against fixed cutoffs (below 0.1, below 0.5, or above that), and writes `models/evaluation_results.npz` with the two trajectories, the time grid, and that RMS. Those files are not in the repository.

### `viz_ai.py`

Reads `models/evaluation_results.npz` and draws the physics arc against the AI rollout: rotating-frame x–y, x–z, position error versus time, and the x and y velocity components, plus a 3D overlay with Earth and the Moon. Intended outputs are `docs/ai_vs_physics.png` and `docs/ai_vs_physics_3d.png`. The x–z panel calls `set_zlabel` on a 2D Matplotlib axes, so the script raises before it writes either figure. Neither image is checked in.

### `best_model.pt`

The file at the repository root is a PyTorch zip archive (serialization version 3) holding a raw `state_dict`, with no epoch, optimizer, or loss stored beside the tensors. The keys are a unidirectional 3-layer LSTM and a two-layer head:

| Tensor | Shape |
| --- | --- |
| `lstm.weight_ih_l0` | (1024, 6) |
| `lstm.weight_hh_l{0,1,2}`, `lstm.weight_ih_l{1,2}` | (1024, 256) |
| `lstm.bias_*_l{0,1,2}` | (1024,) |
| `fc.0.weight` / `fc.0.bias` | (64, 256) / (64,) |
| `fc.3.weight` / `fc.3.bias` | (6, 64) / (6,) |

That is input size 6, hidden size 256, three layers, then a linear map 256 → 64 and a linear map 64 → 6. `fc.1` and `fc.2` are present and hold no parameters. The parameter count is 1,339,846. `train_ai.py` is written to save a 2-layer, hidden-128 network under `models/best_model.pt`. `train_ai_full.py` is written to save a 3-layer, hidden-128 network with a 128 → 256 → 128 → 6 head under `models/best_lstm.pt`. The root file is a different network from either of those, and `eval_ai.py` loads the hyperparameter dict in `models/cislunar_lstm_final.pt`.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

Dependencies are NumPy, SciPy, Matplotlib, Plotly, Jupyter, pytest, PyTorch, and tqdm. There is no installable package; scripts add `src` to `sys.path` and expect to be run from the repository root.

## Usage

Propagate a state offset from L1 by +0.01 in x (toward the Moon) and check the Jacobi integral. This is the same initial condition used to draw the figure above (`test_basic.py` writes that figure to `test_trajectory.png`).

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from physics import compute_lagrange_points, propagate, jacobi_integral

L = compute_lagrange_points()
state0 = [L["L1"][0] + 0.01, 0.0, 0.0, 0.0, 0.1, 0.0]
t_eval = np.linspace(0, 10, 1000)
sol = propagate(state0, (0, 10), t_eval)

print(sol.success, sol.y[:3, -1])
print(jacobi_integral(state0), jacobi_integral(sol.y[:, -1]))
```

These demos import `physics`, `viz`, `debris`, or `orbits` and do not load the `ai` package:

```bash
python test_basic.py           # Lagrange points, a short arc, Jacobi check, rotating-frame figure
python test_validation.py      # mass ratio, NASA/JPL Lagrange points, Jacobi check
python test_3d.py              # Plotly Earth–Moon view and debris cloud
python test_3d_research.py     # same views, with the Lagrange-point check printed
python test_professional.py    # dark-theme 3D figure and orbital-regime comparison
```

`test_advanced.py` and `test_ai_tracking.py` import the `ai` package, so they hit the same syntax error as the trainers. After that import is fixed, the tracking scripts are:

```bash
python train_ai.py            # writes models/best_model.pt and models/cislunar_lstm_final.pt
python train_ai_full.py       # writes models/best_lstm.pt
python eval_ai.py             # needs models/cislunar_lstm_final.pt and models/normalizer.npz
python viz_ai.py              # needs models/evaluation_results.npz
```

## Project structure

```
├── train_ai.py              # 2-layer LSTM training entry point
├── train_ai_full.py         # 3-layer LSTM training entry point
├── eval_ai.py               # autoregressive comparison against propagate
├── viz_ai.py                # physics-vs-AI figures from an eval archive
├── best_model.pt            # 3-layer, hidden-256 state dict (see above)
├── test_*.py                # propagator, validation, halo, and visualization demos
├── requirements.txt
├── src/
│   ├── physics/             # CR3BP, Lagrange points, Jacobi, environments, validation
│   ├── orbits/              # halo initial guess and single-shooter
│   ├── perturbations/       # solar gravity and radiation pressure
│   ├── debris/              # synthetic catalog and conjunction screen
│   ├── ai/                  # datasets, LSTM/GRU, trainer, evaluator, inference
│   └── viz/                 # rotating-frame, Plotly 3D, animation
├── docs/                    # HTML views, references, trajectory figure
├── data/
│   ├── nasa_ephemerides/    # placeholder
│   └── synthetic_debris/    # placeholder
└── tests/                   # package marker only; demos live at the repo root
```

## Next steps

- Remove the pasted shell text in `src/ai/lstm_trainer.py` so the `ai` package imports, then rerun training from a fixed seed.
- Keep one checkpoint layout. The root `best_model.pt` is a hidden-256, 3-layer network; the two training scripts save hidden-128 networks to `models/` and do not load this file.
- Record the metrics the evaluators already compute (one-step RMSE in normalized units, multi-step position RMS, Jacobi drift) from a completed run. Nothing in the tree is that run.
- Fix the x–z panel in `viz_ai.py` (`set_ylabel`, or a 3D axes) so the comparison figures can be written.
- Add the solar-gravity and radiation-pressure accelerations inside the propagator, with a switch to turn them off.
- Fill `data/nasa_ephemerides/` and check halo families from the single-shooter against published NRHO and Gateway orbits.
- Implement the LEO and GEO branches of the debris catalog. `generate_debris_catalog` currently raises `NotImplementedError` for those regions.
- Replace `SimpleLSTMPredictor` with the trained network so `test_ai_tracking.py` is a real rollout.
