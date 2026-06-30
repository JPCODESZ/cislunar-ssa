"""Training utilities for cislunar trajectory models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json

class TrajectoryTrainer:
    """Trainer for trajectory prediction models."""
    
    def __init__(
        self,
        model,
        device='auto',
        learning_rate=1e-3,
        weight_decay=1e-5,
        physics_weight=0.1
    ):
        self.model = model
        self.physics_weight = physics_weight
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(X_batch)
            
            if isinstance(output, dict):
                pred_state = output['next_state']
                pred_accel = output['pred_accel']
                physics_accel = output['physics_accel']
                
                loss_state = self.criterion(pred_state, y_batch)
                loss_physics = self.criterion(pred_accel, physics_accel)
                loss = loss_state + self.physics_weight * loss_physics
            else:
                loss = self.criterion(output, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                output = self.model(X_batch)
                
                if isinstance(output, dict):
                    pred_state = output['next_state']
                    loss = self.criterion(pred_state, y_batch)
                else:
                    loss = self.criterion(output, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        patience=15,
        checkpoint_path='models/best_model.pt'
    ):
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={self.optimizer.param_groups[0]['lr']:.2e} | time={epoch_time:.1f}s | patience={patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model (val_loss={self.best_val_loss:.6f})")
        
        return self.history
    
    def save_history(self, path='models/training_history.json'):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")
