"""
Transformer-based Predictive Model for Robotic MPC
Captures temporal dependencies in sequential robot motions
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    Preserves temporal ordering information
    """
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for predicting robot dynamics
    Uses multi-head self-attention to model temporal dependencies
    """
    
    def __init__(self, config):
        super(TransformerPredictor, self).__init__()
        
        self.config = config
        robot_config = config['robot']
        tf_config = config['transformer']
        
        self.dof = robot_config['dof']
        self.history_length = tf_config['history_length']
        self.input_dim = tf_config['input_dim']  # 21 (7 pos + 7 vel + 7 torque)
        self.d_model = tf_config['d_model']
        self.output_dim = tf_config['output_dim']  # 14 (7 pos + 7 vel)

        self.register_buffer('pos_mean', torch.zeros(7))
        self.register_buffer('pos_std', torch.ones(7))
        self.register_buffer('vel_mean', torch.zeros(7))
        self.register_buffer('vel_std', torch.ones(7))
        self.register_buffer('tau_mean', torch.zeros(7))
        self.register_buffer('tau_std', torch.ones(7))
        
        # Input embedding: project input features to d_model
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.d_model,
            max_len=tf_config['max_len'],
            dropout=tf_config['dropout']
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=tf_config['num_heads'],
            dim_feedforward=tf_config['dim_feedforward'],
            dropout=tf_config['dropout'],
            activation='relu',
            batch_first=True,  # (batch, seq, feature)
            norm_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_config['num_encoder_layers']
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(tf_config['dropout']),
            nn.Linear(self.d_model // 2, self.output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, history_positions, history_velocities, history_torques):
        """
        Forward pass through transformer
        
        Args:
            history_positions: (batch, history_len, 7)
            history_velocities: (batch, history_len, 7)
            history_torques: (batch, history_len, 7)
            
        Returns:
            next_state: (batch, 14) predicted [next_positions, next_velocities]
        """
        # Normalize inputs across ALL timesteps
        pos_norm = (history_positions - self.pos_mean) / (self.pos_std + 1e-6)
        vel_norm = (history_velocities - self.vel_mean) / (self.vel_std + 1e-6)
        tau_norm = (history_torques - self.tau_mean) / (self.tau_std + 1e-6)
        
        # Concatenate normalized inputs
        x = torch.cat([pos_norm, vel_norm, tau_norm], dim=-1)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Use the last time step for prediction
        last_encoded = encoded[:, -1, :]
        
        # Predict DELTA (normalized space)
        delta_state = self.output_projection(last_encoded)
        
        # Denormalize deltas
        delta_pos = delta_state[:, :self.dof] * self.pos_std
        delta_vel = delta_state[:, self.dof:] * self.vel_std
        
        # Get current state (last timestep)
        current_pos = history_positions[:, -1, :]
        current_vel = history_velocities[:, -1, :]
        
        # Apply residual connection
        next_pos = current_pos + delta_pos
        next_vel = current_vel + delta_vel
        
        return torch.cat([next_pos, next_vel], dim=-1)   

    def predict_next_state(self, history_positions, history_velocities, history_torques):
        """
        Predict next state (convenience method)
        
        Returns:
            next_positions: (batch, 7)
            next_velocities: (batch, 7)
        """
        next_state = self.forward(history_positions, history_velocities, history_torques)
        next_positions = next_state[..., :self.dof]
        next_velocities = next_state[..., self.dof:]
        return next_positions, next_velocities
    
    def get_param_count(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, history_positions, history_velocities, history_torques):
        """
        Extract attention weights for visualization
        
        Returns:
            attention_weights: List of attention weights from each layer
        """
        # This requires modifying the forward pass to return attention
        # For now, we'll return None (can be extended if needed)
        return None


class TransformerPredictorSingleStep(nn.Module):
    """
    Wrapper for single-step prediction compatible with MPC
    Maintains internal buffer for history
    """
    
    def __init__(self, transformer_model, history_length):
        super(TransformerPredictorSingleStep, self).__init__()
        
        self.transformer = transformer_model
        self.history_length = history_length
        self.dof = transformer_model.dof
        
        # Initialize history buffers
        self.reset_history()
    
    def reset_history(self):
        """Reset history buffers"""
        self.position_history = []
        self.velocity_history = []
        self.torque_history = []
    
    def update_history(self, positions, velocities, torques):
        """
        Update history with new state
        
        Args:
            positions: (7,) or (batch, 7)
            velocities: (7,) or (batch, 7)
            torques: (7,) or (batch, 7)
        """
        # Ensure inputs are 2D
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            velocities = velocities.unsqueeze(0)
            torques = torques.unsqueeze(0)
        
        self.position_history.append(positions)
        self.velocity_history.append(velocities)
        self.torque_history.append(torques)

        # Normalization statistics (compute from training data)
        self.register_buffer('pos_mean', torch.zeros(7))
        self.register_buffer('pos_std', torch.ones(7))
        self.register_buffer('vel_mean', torch.zeros(7))
        self.register_buffer('vel_std', torch.ones(7))
        self.register_buffer('tau_mean', torch.zeros(7))
        self.register_buffer('tau_std', torch.ones(7))
        
        # Keep only recent history
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.torque_history.pop(0)
    
    def forward(self, history_positions, history_velocities, history_torques):
        """Forward pass - predicts CHANGE in state"""
        # Normalize inputs
        pos_norm = (positions - self.pos_mean) / (self.pos_std + 1e-6)
        vel_norm = (velocities - self.vel_mean) / (self.vel_std + 1e-6)
        tau_norm = (torques - self.tau_mean) / (self.tau_std + 1e-6)
        
        x = torch.cat([pos_norm, vel_norm, tau_norm], dim=-1)
        delta_state = self.network(x)
        
        # Denormalize outputs
        delta_pos = delta_state[..., :7] * self.pos_std
        delta_vel = delta_state[..., 7:] * self.vel_std
        
        return torch.cat([positions + delta_pos, velocities + delta_vel], dim=-1)

def create_transformer_model(config):
    """Factory function to create transformer model"""
    model = TransformerPredictor(config)
    print(f"Transformer model created with {model.get_param_count():,} parameters")
    return model


if __name__ == "__main__":
    import yaml
    
    # Test model
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_transformer_model(config)
    
    # Test forward pass
    batch_size = 32
    history_len = config['transformer']['history_length']
    
    hist_pos = torch.randn(batch_size, history_len, 7)
    hist_vel = torch.randn(batch_size, history_len, 7)
    hist_tau = torch.randn(batch_size, history_len, 7)
    
    next_state = model(hist_pos, hist_vel, hist_tau)
    print(f"Input shapes: hist_pos={hist_pos.shape}")
    print(f"Output shape: {next_state.shape}")
    
    next_pos, next_vel = model.predict_next_state(hist_pos, hist_vel, hist_tau)
    print(f"Next positions: {next_pos.shape}, Next velocities: {next_vel.shape}")
