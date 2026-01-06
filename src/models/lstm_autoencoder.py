"""
LSTM Autoencoder for S.A.F.E
Deep learning based time-series anomaly detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
from src.models.base_model import BaseAnomalyModel


class LSTMAutoencoder( nn.Module ) :
    """LSTM-based Autoencoder architecture"""

    def __init__ ( self, input_size: int, hidden_size: int = 64,
                   num_layers: int = 2, dropout: float = 0.2 ) :
        super( LSTMAutoencoder, self ).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.output_layer = nn.Linear( hidden_size, input_size )

    def forward ( self, x ) :
        # Encoder
        encoded, (hidden, cell) = self.encoder( x )

        # Use last hidden state as compressed representation
        # Repeat for decoder sequence length
        seq_len = x.size( 1 )
        decoded_input = hidden[-1].unsqueeze( 1 ).repeat( 1, seq_len, 1 )

        # Decoder
        decoded, _ = self.decoder( decoded_input, (hidden, cell) )

        # Output
        output = self.output_layer( decoded )

        return output


class LSTMAutoencoderDetector( BaseAnomalyModel ) :
    """LSTM Autoencoder for anomaly detection"""

    def __init__ ( self, sequence_length: int = 30, hidden_size: int = 64,
                   num_layers: int = 2, learning_rate: float = 0.001,
                   batch_size: int = 32, epochs: int = 50, dropout: float = 0.2,
                   device: str = None ) :
        """
        Initialize LSTM Autoencoder

        Args:
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            dropout: Dropout rate
            device: Computing device (cuda/cpu)
        """
        super().__init__(
            name="LSTMAutoencoder",
            config={
                'sequence_length' : sequence_length,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'learning_rate' : learning_rate,
                'batch_size' : batch_size,
                'epochs' : epochs,
                'dropout' : dropout
            }
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        if device is None :
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else :
            self.device = torch.device( device )

        self.model = None
        self.threshold = None
        self.scaler_mean = None
        self.scaler_std = None

    def _create_sequences ( self, X: np.ndarray ) -> np.ndarray :
        """Create sequences from time-series data"""
        sequences = []

        for i in range( len( X ) - self.sequence_length + 1 ) :
            seq = X[i :i + self.sequence_length]
            sequences.append( seq )

        return np.array( sequences )

    def _normalize_data ( self, X: np.ndarray, fit: bool = True ) -> np.ndarray :
        """Normalize data to zero mean and unit variance"""
        if fit :
            self.scaler_mean = np.mean( X, axis=0 )
            self.scaler_std = np.std( X, axis=0 )
            self.scaler_std = np.where( self.scaler_std == 0, 1, self.scaler_std )

        return (X - self.scaler_mean) / self.scaler_std

    def fit ( self, X: np.ndarray, y: np.ndarray = None ) :
        """Train LSTM Autoencoder - FIXED VERSION"""

        # X should be (num_sequences, seq_len, num_features)
        print( f"📥 Input shape: {X.shape}" )

        if len( X.shape ) != 3 :
            raise ValueError(
                f"Expected 3D input (num_sequences, seq_len, features), got shape {X.shape}"
            )

        num_sequences, seq_len, num_features = X.shape

        # Store feature info
        self.feature_names = [f"feature_{i}" for i in range( num_features )]

        # CRITICAL: Update sequence_length if needed
        if seq_len != self.sequence_length :
            print( f"⚠️ Adjusting sequence_length from {self.sequence_length} to {seq_len}" )
            self.sequence_length = seq_len

        # Normalize data (flatten, normalize, reshape)
        X_flat = X.reshape( -1, num_features )
        X_flat_norm = self._normalize_data( X_flat, fit=True )
        X_norm = X_flat_norm.reshape( num_sequences, seq_len, num_features )

        print( f"✅ Normalized shape: {X_norm.shape}" )

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor( X_norm ).to( self.device )

        # Create data loader
        dataset = TensorDataset( X_tensor, X_tensor )  # Input = Target for autoencoder
        dataloader = DataLoader( dataset, batch_size=self.batch_size, shuffle=True )

        # Initialize model with CORRECT input_size
        self.model = LSTMAutoencoder(
            input_size=num_features,  # CRITICAL: Use actual feature count
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.config['dropout']
        ).to( self.device )

        print( f"🧠 Model initialized: input_size={num_features}, hidden_size={self.hidden_size}" )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam( self.model.parameters(), lr=self.learning_rate )

        # Training loop
        self.model.train()
        print( f"\n🔄 Training on {self.device} for {self.epochs} epochs..." )

        for epoch in range( self.epochs ) :
            epoch_loss = 0
            for batch_X, batch_y in dataloader :
                optimizer.zero_grad()

                # Forward pass
                output = self.model( batch_X )
                loss = criterion( output, batch_y )

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len( dataloader )

            if (epoch + 1) % 5 == 0 :
                print( f"   Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}" )

        # Calculate reconstruction errors for threshold
        self.model.eval()
        with torch.no_grad() :
            reconstructed = self.model( X_tensor )
            errors = torch.mean( (X_tensor - reconstructed) ** 2, dim=(1, 2) )
            errors = errors.cpu().numpy()

        # Set threshold at 95th percentile
        self.threshold = np.percentile( errors, 95 )

        # Store training stats
        self.train_stats = {
            'n_samples' : num_sequences,
            'n_features' : num_features,
            'sequence_length' : seq_len
        }

        self.is_fitted = True

        print( f"✅ Training complete! Threshold: {self.threshold:.6f}" )

        return self

    def predict_score ( self, X: np.ndarray ) -> np.ndarray :
        """Calculate reconstruction error as anomaly score - FIXED VERSION"""

        if not self.is_fitted :
            raise ValueError( "Model must be fitted before prediction" )

        if len( X.shape ) != 3 :
            raise ValueError(
                f"Expected 3D input (num_sequences, seq_len, features), got shape {X.shape}"
            )

        num_sequences, seq_len, num_features = X.shape

        # Normalize (flatten, normalize, reshape)
        X_flat = X.reshape( -1, num_features )
        X_flat_norm = self._normalize_data( X_flat, fit=False )
        X_norm = X_flat_norm.reshape( num_sequences, seq_len, num_features )

        X_tensor = torch.FloatTensor( X_norm ).to( self.device )

        # Get reconstructions
        self.model.eval()
        with torch.no_grad() :
            reconstructed = self.model( X_tensor )
            # Calculate mean squared error per sequence
            errors = torch.mean( (X_tensor - reconstructed) ** 2, dim=(1, 2) )
            errors = errors.cpu().numpy()

        # Normalize scores by threshold
        if self.threshold > 0 :
            scores = errors / self.threshold
        else :
            scores = errors

        return scores

    def predict ( self, X: np.ndarray ) -> np.ndarray :
        """Predict anomaly labels based on threshold"""
        scores = self.predict_score( X )
        return (scores > 1.0).astype( int )

    def save_model ( self, filepath: str ) :
        """Save model including PyTorch state"""
        if not self.is_fitted :
            raise ValueError( "Cannot save unfitted model" )

        save_dict = {
            'config' : self.config,
            'feature_names' : self.feature_names,
            'train_stats' : self.train_stats,
            'threshold' : self.threshold,
            'scaler_mean' : self.scaler_mean,
            'scaler_std' : self.scaler_std,
            'model_state' : self.model.state_dict()
        }

        torch.save( save_dict, filepath )
        print( f"Model saved to {filepath}" )

    @classmethod
    def load_model ( cls, filepath: str ) :
        """Load model from disk"""
        save_dict = torch.load( filepath )

        # Recreate model
        model_instance = cls( **save_dict['config'] )
        model_instance.feature_names = save_dict['feature_names']
        model_instance.train_stats = save_dict['train_stats']
        model_instance.threshold = save_dict['threshold']
        model_instance.scaler_mean = save_dict['scaler_mean']
        model_instance.scaler_std = save_dict['scaler_std']

        # Recreate neural network
        input_size = len( model_instance.feature_names )
        model_instance.model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=model_instance.hidden_size,
            num_layers=model_instance.num_layers,
            dropout=model_instance.config['dropout']
        ).to( model_instance.device )

        model_instance.model.load_state_dict( save_dict['model_state'] )
        model_instance.is_fitted = True

        return model_instance


if __name__ == "__main__" :
    print( "Testing LSTM Autoencoder..." )

    # Generate sample data
    from src.data.synthetic_generator import SyntheticCrowdGenerator
    from src.features.feature_engineering import FeatureEngineer

    generator = SyntheticCrowdGenerator()
    df = generator.generate_normal_pattern( n_samples=500 )
    df_anomaly = generator.inject_anomalies( df, anomaly_ratio=0.1 )

    # Select single zone for time-series
    zone_data = df_anomaly[df_anomaly['zone_id'] == 0].copy()
    zone_data = zone_data.sort_values( 'timestamp' )

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_feature_matrix( zone_data )

    # Prepare data
    feature_cols = ['density', 'speed_mean', 'direction_variance', 'entry_exit_ratio']
    X = df_features[feature_cols].values
    y = df_features['is_anomaly'].values

    # Train model
    model = LSTMAutoencoderDetector(
        sequence_length=20,
        hidden_size=32,
        num_layers=1,
        epochs=30,
        batch_size=16
    )

    split_idx = int( len( X ) * 0.8 )
    model.fit( X[:split_idx] )

    # Predict
    scores = model.predict_score( X[split_idx :] )
    predictions = model.predict( X[split_idx :] )

    print( f"\nDetected {predictions.sum()} anomalies" )
    print( f"Actual anomalies: {y[split_idx :].sum()}" )
    print( f"Score range: {scores.min():.3f} - {scores.max():.3f}" )