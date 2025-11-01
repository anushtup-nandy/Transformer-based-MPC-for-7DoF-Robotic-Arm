"""
Baseline Feed-Forward DNN Model - FIXED with Residual Learning
Predicts CHANGES (deltas) instead of absolute next states
"""

import torch
import torch.nn as nn


class BaselineDNN(nn.Module):
    """
    Feed-forward DNN for predicting DELTA in robot state
    Architecture: input -> [128] -> [32] -> output
    """
    
    def __init__(self, config):
        super(BaselineDNN, self).__init__()
        
        self.config = config
        robot_config = config['robot']
        dnn_config = config['baseline_dnn']
        
        # Input: [positions (7), velocities (7), torques (7)] = 21
        # Output: [delta_positions (7), delta_velocities (7)] = 14
        self.input_dim = 3 * robot_config['dof']  # 21
        self.output_dim = 2 * robot_config['dof']  # 14
        self.dt = config['data']['sampling_time']  # 0.05s

        # Normalization statistics (compute from training data)
        self.register_buffer('pos_mean', torch.zeros(7))
        self.register_buffer('pos_std', torch.ones(7))
        self.register_buffer('vel_mean', torch.zeros(7))
        self.register_buffer('vel_std', torch.ones(7))
        self.register_buffer('tau_mean', torch.zeros(7))
        self.register_buffer('tau_std', torch.ones(7))
        
        # Build network layers
        layers = []
        hidden_layers = dnn_config['hidden_layers']
        
        # Input layer
        prev_dim = self.input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if dnn_config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif dnn_config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout (if specified)
            if dnn_config['dropout'] > 0:
                layers.append(nn.Dropout(dnn_config['dropout']))
            
            prev_dim = hidden_dim
        
        # Output layer - predicts DELTA
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, positions, velocities, torques):
        """
        Forward pass - predicts next state
        """
        # Normalize inputs
        pos_norm = (positions - self.pos_mean) / (self.pos_std + 1e-6)
        vel_norm = (velocities - self.vel_mean) / (self.vel_std + 1e-6)
        tau_norm = (torques - self.tau_mean) / (self.tau_std + 1e-6)
        
        x = torch.cat([pos_norm, vel_norm, tau_norm], dim=-1)
        
        # Predict DELTA in normalized space
        delta_state = self.network(x)
        
        # Denormalize deltas
        delta_pos = delta_state[:, :7] * self.pos_std
        delta_vel = delta_state[:, 7:] * self.vel_std
        
        # Apply residual connection
        next_pos = positions + delta_pos
        next_vel = velocities + delta_vel
        
        return torch.cat([next_pos, next_vel], dim=-1)   

    def predict_next_state(self, positions, velocities, torques):
        """
        Predict next state (convenience method)
        
        Returns:
            next_positions: (batch, 7)
            next_velocities: (batch, 7)
        """
        next_state = self.forward(positions, velocities, torques)
        next_positions = next_state[..., :7]
        next_velocities = next_state[..., 7:]
        return next_positions, next_velocities
    
    def get_param_count(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_model(config):
    """Factory function to create baseline DNN model"""
    model = BaselineDNN(config)
    print(f"Baseline DNN created with {model.get_param_count():,} parameters")
    return model

if __name__ == "__main__":
    import yaml
    
    # Test model
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_baseline_model(config)
    
    # Test forward pass
    batch_size = 32
    positions = torch.randn(batch_size, 7)
    velocities = torch.randn(batch_size, 7)
    torques = torch.randn(batch_size, 7)
    
    next_state = model(positions, velocities, torques)
    print(f"Input shapes: pos={positions.shape}, vel={velocities.shape}, tau={torques.shape}")
    print(f"Output shape: {next_state.shape}")
    
    next_pos, next_vel = model.predict_next_state(positions, velocities, torques)
    print(f"Next positions: {next_pos.shape}, Next velocities: {next_vel.shape}")
