"""Smoke checks for the tracking package. The demo scripts live at the repo root."""

import numpy as np
import torch

from ai.lstm_trainer import CislunarLSTM, StateNormalizer, constant_velocity_states


def test_package_import():
    import ai
    assert ai.CislunarLSTM is CislunarLSTM


def test_lstm_output_shape():
    model = CislunarLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.1)
    pred = model(torch.zeros(4, 12, 6))
    assert tuple(pred.shape) == (4, 6)


def test_constant_velocity_holds_velocity():
    state = np.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3])
    out = constant_velocity_states(state, n_steps=2, dt=0.5)
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out[0, :3], [1.05, 1.9, 3.15])
    np.testing.assert_allclose(out[1, :3], [1.10, 1.8, 3.30])
    np.testing.assert_allclose(out[:, 3:], np.broadcast_to(state[3:], (2, 3)))


def test_normalizer_roundtrip():
    data = np.arange(12, dtype=float).reshape(2, 6)
    normalizer = StateNormalizer().fit(data)
    np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(data)), data)
