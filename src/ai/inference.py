"""Inference pipeline for trajectory prediction."""

import numpy as np
import torch

class TrajectoryPredictor:
    """High-level inference interface for trajectory prediction."""
    
    def __init__(self, model, device='cpu', normalize_params=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.normalize_params = normalize_params
        self.mean = normalize_params['mean'] if normalize_params else None
        self.std = normalize_params['std'] if normalize_params else None
    
    def _normalize(self, states):
        if self.mean is not None:
            return (states - self.mean) / self.std
        return states
    
    def _denormalize(self, states_norm):
        if self.mean is not None:
            return states_norm * self.std + self.mean
        return states_norm
    
    def predict_next(self, state_sequence):
        if state_sequence.ndim == 1:
            state_sequence = state_sequence.reshape(1, -1, 6)
        
        if state_sequence.ndim == 2:
            state_sequence = state_sequence.reshape(1, *state_sequence.shape)
        
        seq_norm = self._normalize(state_sequence)
        seq_tensor = torch.FloatTensor(seq_norm).to(self.device)
        
        with torch.no_grad():
            output = self.model(seq_tensor)
            
            if isinstance(output, dict):
                pred_norm = output['next_state'].cpu().numpy()
            else:
                pred_norm = output.cpu().numpy()
        
        pred = self._denormalize(pred_norm)
        
        return pred[0] if pred.shape[0] == 1 else pred
    
    def rollout(self, initial_state, n_steps=100, seq_length=20, return_uncertainty=False):
        sequence = np.tile(initial_state, (seq_length, 1))
        
        predictions = []
        uncertainties = []
        
        for i in range(n_steps):
            pred = self.predict_next(sequence[-seq_length:])
            predictions.append(pred)
            
            if return_uncertainty:
                uncertainties.append(np.zeros(6))
            
            sequence = np.vstack([sequence, pred])
        
        predictions = np.array(predictions)
        
        if return_uncertainty:
            return predictions, np.array(uncertainties)
        return predictions
    
    def rollout_with_physics_correction(self, initial_state, n_steps=100, seq_length=20, correction_interval=10, physics_weight=0.5):
        from physics.cr3bp import propagate
        
        sequence = np.tile(initial_state, (seq_length, 1))
        predictions = []
        
        current_state = initial_state.copy()
        
        for i in range(n_steps):
            ml_pred = self.predict_next(sequence[-seq_length:])
            
            if i > 0 and i % correction_interval == 0:
                t_span = (0, 0.05)
                t_eval = [0.05]
                sol = propagate(current_state, t_span, t_eval)
                physics_pred = sol.y[:, -1]
                pred = physics_weight * physics_pred + (1 - physics_weight) * ml_pred
            else:
                pred = ml_pred
            
            predictions.append(pred)
            current_state = pred
            sequence = np.vstack([sequence, pred])
        
        return np.array(predictions)


class EnsemblePredictor:
    """Ensemble of models for uncertainty quantification."""
    
    def __init__(self, models, device='cpu'):
        self.predictors = [TrajectoryPredictor(m, device) for m in models]
    
    def predict_next(self, state_sequence):
        preds = [p.predict_next(state_sequence) for p in self.predictors]
        preds = np.array(preds)
        
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        
        return mean, std
    
    def rollout(self, initial_state, n_steps=100, seq_length=20):
        all_rollouts = []
        
        for predictor in self.predictors:
            rollout = predictor.rollout(initial_state, n_steps, seq_length)
            all_rollouts.append(rollout)
        
        all_rollouts = np.array(all_rollouts)
        
        mean = np.mean(all_rollouts, axis=0)
        std = np.std(all_rollouts, axis=0)
        
        return mean, std, all_rollouts
