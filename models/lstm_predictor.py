import torch
import torch.nn as nn

class LSTMDynamicsPredictor(nn.Module):
    """
    Sequence-to-one LSTM dynamics model.
    Input  (B, T, 21): [q(7), qd(7), tau(7)] for T steps
    Output (B, 14): absolute next state [q_{t+1}, qd_{t+1}]
    """
    def __init__(self,
                 input_dim: int = 21,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 output_dim: int = 14,
                 residual: bool = True):
        super().__init__()
        self.residual = residual

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

        # Normalization buffers expected by train.py (shape [7] each)
        self.register_buffer('pos_mean', torch.zeros(7))
        self.register_buffer('pos_std',  torch.ones(7))
        self.register_buffer('vel_mean', torch.zeros(7))
        self.register_buffer('vel_std',  torch.ones(7))
        self.register_buffer('tau_mean', torch.zeros(7))
        self.register_buffer('tau_std',  torch.ones(7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 21)
        returns: (B, 14) -> [q_next(7), qd_next(7)]
        """
        out, _ = self.lstm(x)          # (B, T, hidden)
        h_last = out[:, -1, :]         # (B, hidden)
        deltas = self.head(h_last)     # (B, 14) -> [Δq, Δqd]

        if self.residual:
            q_last  = x[:, -1, :7]
            dq_last = x[:, -1, 7:14]
            q_next  = q_last  + deltas[:, :7]
            dq_next = dq_last + deltas[:, 7:14]
            return torch.cat([q_next, dq_next], dim=-1)
        else:
            # If residual=False, treat 'deltas' as absolute next state
            return deltas

def create_lstm_model(config, hidden_dim=128, num_layers=2, dropout=0.1):
    """
    Factory to match the repo's model-creation style.
    """
    return LSTMDynamicsPredictor(
        input_dim=21,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=14,
        residual=True
    )
