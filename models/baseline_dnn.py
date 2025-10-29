"""
Baseline Feed-Forward DNN Model
Based on El-Hussieny et al. (2024) paper
"""

import torch
import torch.nn as nn


class BaselineDNN(nn.Module):
    """
    Feed-forward DNN for predicting next robot state
    Architecture: input -> [128] -> [32] -> output
    """
    
    def __init__(self, config):
        super(BaselineDNN, self).__init__()
        
        self.config = config
        robot_config = config['robot']
        dnn_config = config['baseline_dnn']
        
        # Input: [positions (7), velocities (7), torques (7)] = 21
        # Output: [next_positions (7), next_velocities (7)] = 14
        self.input_dim = 3 * robot_config['dof']  # 21
        self.output_dim = 2 * robot_config['dof']  # 14
        
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        # No activation on output (linear)
        
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
        Forward pass
        
        Args:
            positions: (batch, 7) current joint positions
            velocities: (batch, 7) current joint velocities  
            torques: (batch, 7) applied torques
            
        Returns:
            next_state: (batch, 14) predicted [next_positions, next_velocities]
        """
        # Concatenate inputs
        x = torch.cat([positions, velocities, torques], dim=-1)
        
        # Forward pass
        next_state = self.network(x)
        
        return next_state
    
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
