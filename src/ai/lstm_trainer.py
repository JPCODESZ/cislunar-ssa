"""Complete LSTM training pipeline for cislunar trajectory prediction."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, 'src')
from physics.cr3bp import propagate


# ==================== MODEL ====================

class CislunarLSTM(nn.Module):
    """LSTM for predicting next state in CR3BP trajectory."""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
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


# ==================== DATASET ====================

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets =
cd ~/cislunar-ssa

# Create the working LSTM trainer (no deprecated params)
cat > src/ai/lstm_trainer.py << 'PYEOF'
"""Complete LSTM training pipeline for cislunar trajectory prediction."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, 'src')
from physics.cr3bp import propagate


# ==================== MODEL ====================

class CislunarLSTM(nn.Module):
    """LSTM for predicting next state in CR3BP trajectory."""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
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


# ==================== DATASET ====================

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== DATA GENERATION ====================

def generate_cr3bp_trajectories(n_samples=500, t_max=10.0, n_points=200):
    """
    Generate synthetic CR3BP trajectories for training.
    """
    trajectories = []
    
    print(f"  Generating {n_samples} trajectories...")
    
    for i in tqdm(range(n_samples), desc="  Trajectories"):
        # Random initial state in cislunar region
        x0 = np.random.uniform(-1.2, 1.2)
        y0 = np.random.uniform(-1.2, 1.2)
        z0 = np.random.uniform(-0.3, 0.3)
        vx0 = np.random.normal(0, 0.15)
        vy0 = np.random.normal(0, 0.15)
        vz0 = np.random.normal(0, 0.05)
        
        state0 = [x0, y0, z0, vx0, vy0, vz0]
        
        # Propagate
        t_eval = np.linspace(0, t_max, n_points)
        sol = propagate(state0, (0, t_max), t_eval)
        
        if sol.success:
            trajectories.append(sol.y.T)  # (n_points, 6)
    
    return trajectories


def create_sequences(trajectories, seq_length=25):
    """
    Convert trajectories into input sequences and target states.
    """
    X, y = [], []
    
    for traj in trajectories:
        for i in range(len(traj) - seq_length):
            X.append(traj[i:i + seq_length])
            y.append(traj[i + seq_length])
    
    return np.array(X), np.array(y)


# ==================== NORMALIZATION ====================

class StateNormalizer:
    """Normalize/unnormalize state vectors."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
        
    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']


# ==================== TRAINER ====================

class TrajectoryTrainer:
    """Training manager for CislunarLSTM."""
    
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.scheduler.step(val_loss)
            
            # Print every epoch
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        return self.history
    
    def predict(self, state_sequence):
        """Predict next state from sequence."""
        self.model.eval()
        with torch.no_grad():
            seq = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
            pred = self.model(seq)
            return pred.cpu().numpy()[0]


# ==================== EVALUATION ====================

def evaluate_model(model, normalizer, test_sequences, test_targets, n_steps=100):
    """
    Evaluate model by predicting multi-step ahead.
    """
    model.eval()
    
    # Pick a random test sequence
    idx = np.random.randint(0, len(test_sequences))
    seq = test_sequences[idx].copy()
    
    # True trajectory
    true_traj = [seq[-1]]
    for i in range(n_steps):
        if idx + i < len(test_sequences):
            true_traj.append(test_targets[idx + i])
    true_traj = np.array(true_traj)
    
    # AI prediction
    ai_traj = [seq[-1]]
    current_seq = seq.copy()
    
    with torch.no_grad():
        for _ in range(n_steps):
            inp = torch.FloatTensor(current_seq).unsqueeze(0)
            pred = model(inp).numpy()[0]
            ai_traj.append(pred)
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1] = pred
    
    ai_traj = np.array(ai_traj)
    
    # Unnormalize
    true_traj = normalizer.inverse_transform(true_traj)
    ai_traj = normalizer.inverse_transform(ai_traj)
    
    # Compute error
    rms_error = np.sqrt(np.mean((true_traj[:, :3] - ai_traj[:, :3])**2))
    
    return true_traj, ai_traj, rms_error
