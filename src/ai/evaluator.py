"""Evaluation metrics for trajectory prediction models."""

import numpy as np
import torch

def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets)**2))

def compute_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def compute_position_error(predictions, targets):
    pos_pred = predictions[:, :3]
    pos_true = targets[:, :3]
    return compute_rmse(pos_pred, pos_true)

def compute_velocity_error(predictions, targets):
    vel_pred = predictions[:, 3:]
    vel_true = targets[:, 3:]
    return compute_rmse(vel_pred, vel_true)

def compute_jacobi_error(states, mu=0.012150585609624):
    from physics.jacobi import jacobi_integral
    
    C_values = np.array([jacobi_integral(s, mu) for s in states])
    C0 = C_values[0]
    errors = np.abs(C_values - C0)
    
    return {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors),
        'final_error': errors[-1]
    }

def compute_drift_error(predicted_trajectory, true_trajectory):
    pos_pred = predicted_trajectory[:, :3]
    pos_true = true_trajectory[:, :3]
    
    drift = np.sqrt(np.sum((pos_pred - pos_true)**2, axis=1))
    relative_drift = drift / 1.0
    
    return drift, relative_drift

def compute_energy_error(predicted_trajectory, true_trajectory):
    from physics.jacobi import jacobi_integral
    
    C_pred = np.array([jacobi_integral(s) for s in predicted_trajectory])
    C_true = np.array([jacobi_integral(s) for s in true_trajectory])
    
    return {
        'mean_diff': np.mean(np.abs(C_pred - C_true)),
        'max_diff': np.max(np.abs(C_pred - C_true)),
        'pred_conservation': np.std(C_pred),
        'true_conservation': np.std(C_true)
    }

class TrajectoryEvaluator:
    """Comprehensive evaluator for trajectory models."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_dataset(self, X_test, y_test):
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            
            if isinstance(output, dict):
                predictions = output['next_state'].cpu().numpy()
            else:
                predictions = output.cpu().numpy()
        
        targets = y_test
        
        metrics = {
            'overall_rmse': compute_rmse(predictions, targets),
            'overall_mae': compute_mae(predictions, targets),
            'position_rmse': compute_position_error(predictions, targets),
            'velocity_rmse': compute_velocity_error(predictions, targets),
            'x_rmse': np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2)),
            'y_rmse': np.sqrt(np.mean((predictions[:, 1] - targets[:, 1])**2)),
            'z_rmse': np.sqrt(np.mean((predictions[:, 2] - targets[:, 2])**2)),
        }
        
        return metrics, predictions
    
    def evaluate_long_horizon(self, test_trajectories, seq_length=20, horizons=[10, 50, 100, 200]):
        all_results = []
        
        for traj in test_trajectories:
            states = traj[1]
            
            if len(states) < seq_length + max(horizons):
                continue
            
            input_seq = states[:seq_length].copy()
            predictions = []
            
            with torch.no_grad():
                for i in range(max(horizons)):
                    seq_tensor = torch.FloatTensor(input_seq[-seq_length:]).unsqueeze(0).to(self.device)
                    pred = self.model(seq_tensor).cpu().numpy()[0]
                    predictions.append(pred)
                    input_seq = np.vstack([input_seq, pred])
            
            predictions = np.array(predictions)
            
            traj_results = {'trajectory_length': len(states)}
            
            for h in horizons:
                pred_h = predictions[:h]
                true_h = states[seq_length:seq_length+h]
                
                traj_results[f'h{h}_pos_rmse'] = compute_position_error(pred_h, true_h)
                traj_results[f'h{h}_drift'] = compute_drift_error(pred_h, true_h)[0][-1]
                traj_results[f'h{h}_energy_error'] = compute_energy_error(pred_h, true_h)['mean_diff']
            
            all_results.append(traj_results)
        
        aggregated = {}
        for key in all_results[0].keys():
            if key != 'trajectory_length':
                values = [r[key] for r in all_results if key in r]
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'max': np.max(values)
                }
        
        return aggregated, all_results
    
    def print_report(self, metrics):
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval:.6f}")
            else:
                print(f"{key}: {value:.6f}")
        
        print("=" * 60)
