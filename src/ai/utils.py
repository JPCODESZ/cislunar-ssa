"""Utility functions for AI trajectory prediction."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def split_train_val_test(data, train_ratio=0.7, val_ratio=0.15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    if isinstance(data, np.ndarray):
        return data[train_idx], data[val_idx], data[test_idx]
    else:
        return [data[i] for i in train_idx], [data[i] for i in val_idx], [data[i] for i in test_idx]

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def compare_models(models_dict, test_loader, device='cpu'):
    from evaluator import compute_rmse, compute_position_error, compute_velocity_error
    
    results = {}
    
    for name, model in models_dict.items():
        model.to(device)
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_batch)
                if isinstance(output, dict):
                    pred = output['next_state']
                else:
                    pred = output
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        results[name] = {
            'rmse': compute_rmse(preds, targets),
            'pos_rmse': compute_position_error(preds, targets),
            'vel_rmse': compute_velocity_error(preds, targets),
            'params': count_parameters(model),
            'size_mb': get_model_size_mb(model)
        }
    
    return results

def export_to_onnx(model, filepath, seq_length=20, batch_size=1):
    dummy_input = torch.randn(batch_size, seq_length, 6)
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['predicted_state'],
        dynamic_axes={
            'input_sequence': {0: 'batch_size'},
            'predicted_state': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filepath}")
