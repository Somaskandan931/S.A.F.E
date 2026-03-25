"""
LSTM Autoencoder for S.A.F.E - FIXED VERSION
=============================================
FIXES APPLIED:
  1. Score normalization for consistent thresholds
  2. Proper save/load of normalization parameters
  3. Robust error handling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
from src.models.base_model import BaseAnomalyModel
import joblib


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder architecture."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, (hidden, cell) = self.encoder(x)
        seq_len = x.size(1)
        decoded_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoded_input, (hidden, cell))
        return self.output_layer(decoded)


class LSTMAutoencoderDetector(BaseAnomalyModel):
    """LSTM Autoencoder for anomaly detection (unsupervised)."""

    def __init__(self, sequence_length: int = 30, hidden_size: int = 64,
                 num_layers: int = 2, learning_rate: float = 0.001,
                 batch_size: int = 32, epochs: int = 50, dropout: float = 0.2,
                 device: str = None):
        super().__init__(
            name="LSTMAutoencoder",
            config={
                'sequence_length': sequence_length,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'dropout': dropout,
            }
        )
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.model: Optional[nn.Module] = None
        self.threshold: Optional[float] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

        # NEW: Store score normalization parameters
        self.score_min: float = 0.0
        self.score_max: float = 1.0

    # ------------------------------------------------------------------
    def _normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalise data using training statistics."""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0)
            self.scaler_std = np.where(self.scaler_std == 0, 1, self.scaler_std)
        return (X - self.scaler_mean) / self.scaler_std

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Train on NORMAL sequences only."""
        print(f"  LSTM Input shape: {X.shape}")

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3-D input (n_seq, seq_len, n_feat), got {X.shape}"
            )

        n_seq, seq_len, n_feat = X.shape

        if seq_len != self.sequence_length:
            print(f"  Adjusting sequence_length: {self.sequence_length} → {seq_len}")
            self.sequence_length = seq_len

        self.feature_names = [f"feature_{i}" for i in range(n_feat)]

        # Normalise (fit on training data)
        X_flat = X.reshape(-1, n_feat)
        X_norm = self._normalize(X_flat, fit=True).reshape(n_seq, seq_len, n_feat)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Create model
        self.model = LSTMAutoencoder(
            input_size=n_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.config['dropout'],
        ).to(self.device)

        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"  Training LSTM on {self.device} for {self.epochs} epochs...")
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"    Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

        # Compute training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()

        # NEW: Normalize scores to [0, 1] range for consistent threshold
        self.score_min = float(errors.min())
        self.score_max = float(errors.max())

        if self.score_max > self.score_min:
            errors_norm = (errors - self.score_min) / (self.score_max - self.score_min)
            self.threshold = float(np.percentile(errors_norm, 95))
        else:
            self.threshold = float(np.percentile(errors, 95))

        print(f"  LSTM training threshold (p95): {self.threshold:.6f}")
        print(f"  Score range: [{self.score_min:.6f}, {self.score_max:.6f}]")

        self.train_stats = {
            'n_samples': n_seq,
            'n_features': n_feat,
            'sequence_length': seq_len,
            'score_min': self.score_min,
            'score_max': self.score_max,
            'threshold': self.threshold
        }
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return normalized reconstruction errors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.ndim != 3:
            raise ValueError(f"Expected 3-D input, got {X.shape}")

        n_seq, seq_len, n_feat = X.shape

        # Normalise using training statistics
        X_flat = X.reshape(-1, n_feat)
        X_norm = self._normalize(X_flat, fit=False).reshape(n_seq, seq_len, n_feat)

        # Predict
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        self.model.eval()

        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()

        # NEW: Normalize scores to [0, 1] using stored min/max
        if self.score_max > self.score_min:
            errors = (errors - self.score_min) / (self.score_max - self.score_min)
        else:
            errors = np.clip(errors, 0, 1)

        return errors

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the stored threshold."""
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)

    # ------------------------------------------------------------------
    def save_model(self, filepath: str):
        """Save model using joblib for maximum compatibility."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Prepare data for saving
        save_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'train_stats': self.train_stats,
            'threshold': self.threshold,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'model_state': self.model.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'device': str(self.device),
            # NEW: Save normalization parameters
            'score_min': self.score_min,
            'score_max': self.score_max,
        }

        # Save with joblib
        joblib.dump(save_data, filepath, compress=3)
        print(f"  LSTM model saved → {filepath}")

    # ------------------------------------------------------------------
    @classmethod
    def load_model(cls, filepath: str):
        """Load model using joblib."""
        print(f"  Loading LSTM model from {filepath}...")

        # Load with joblib
        data = joblib.load(filepath)

        # Create instance
        inst = cls(**data['config'])

        # Restore attributes
        inst.feature_names = data['feature_names']
        inst.train_stats = data['train_stats']
        inst.threshold = data['threshold']
        inst.scaler_mean = data['scaler_mean']
        inst.scaler_std = data['scaler_std']

        # NEW: Load normalization parameters
        inst.score_min = data.get('score_min', 0.0)
        inst.score_max = data.get('score_max', 1.0)

        # Rebuild model
        n_feat = len(inst.feature_names)
        inst.model = LSTMAutoencoder(
            input_size=n_feat,
            hidden_size=inst.hidden_size,
            num_layers=inst.num_layers,
            dropout=inst.config['dropout'],
        ).to(inst.device)

        # Load state dict
        inst.model.load_state_dict(data['model_state'])
        inst.is_fitted = True

        print(f"  LSTM model loaded successfully")
        print(f"  Score range: [{inst.score_min:.4f}, {inst.score_max:.4f}]")
        print(f"  Threshold: {inst.threshold:.6f}")
        return inst

    # ------------------------------------------------------------------
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (based on variance)."""
        if not self.is_fitted:
            return {}

        if self.scaler_std is not None:
            importance = self.scaler_std / np.sum(self.scaler_std)
            return dict(zip(self.feature_names, importance))
        return {name: 1.0/len(self.feature_names) for name in self.feature_names}