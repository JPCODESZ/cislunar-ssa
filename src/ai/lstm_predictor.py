"""PyTorch LSTM trajectory predictor for cislunar space."""

import torch
import torch.nn as nn
import numpy as np

class CislunarLSTM(nn.Module):
    """LSTM-based trajectory predictor for CR3BP states."""
    
    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_layers=3,
        output_size=6,
        pred_horizon=1,
        dropout=0.2,
        bidirectional=False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.pred_horizon = pred_horizon
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, pred_horizon * output_size)
        )
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_final = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h_final = h_n[-1]
        
        out = self.fc(h_final)
        
        if self.pred_horizon > 1:
            out = out.view(-1, self.pred_horizon, self.output_size)
        
        return out


class CislunarGRU(nn.Module):
    """GRU variant for comparison."""
    
    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_layers=3,
        output_size=6,
        pred_horizon=1,
        dropout=0.2
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pred_horizon * output_size)
        )
        self.pred_horizon = pred_horizon
        self.output_size = output_size
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        h_final = h_n[-1]
        out = self.fc(h_final)
        if self.pred_horizon > 1:
            out = out.view(-1, self.pred_horizon, self.output_size)
        return out


class PhysicsInformedLSTM(nn.Module):
    """LSTM with physics-informed loss (CR3BP residual penalty)."""
    
    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_layers=3,
        mu=0.012150585609624,
        dropout=0.2
    ):
        super().__init__()
        
        self.mu = mu
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        self.accel_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def cr3bp_acceleration(self, state):
        """Compute CR3BP acceleration for physics loss."""
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        
        r1 = torch.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = torch.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        ax = 2*state[:, 4] + x - (1 - self.mu)*(x + self.mu)/r1**3 - self.mu*(x - 1 + self.mu)/r2**3
        ay = -2*state[:, 3] + y - (1 - self.mu)*y/r1**3 - self.mu*y/r2**3
        az = -(1 - self.mu)*z/r1**3 - self.mu*z/r2**3
        
        return torch.stack([ax, ay, az], dim=-1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_final = lstm_out[:, -1, :]
        
        next_state = self.state_head(h_final)
        pred_accel = self.accel_head(h_final)
        physics_accel = self.cr3bp_acceleration(next_state)
        
        return {
            'next_state': next_state,
            'pred_accel': pred_accel,
            'physics_accel': physics_accel
        }
