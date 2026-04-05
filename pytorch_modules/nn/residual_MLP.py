import torch.nn as nn
import torch.nn.functional as F 

class ResidualBlock(nn.Module):

    """Residual block for deep networks"""

    def __init__(self, act: nn.Module, 
                 input_dim: int, 
                 output_dim: int,
                 dropout: float = 0.2):
        
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)

        return self.dropout(out)

class ResidualMLP(nn.Module):

    def __init__(self, num_layers: int, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int, 
                 act: nn.Module,
                 dropout_rate: float):
        
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout_rate),
        )

        blocks = [input_proj]

        for _ in range(num_layers):

            blocks.append(ResidualBlock(act, hidden_dim, hidden_dim, dropout_rate))

        output_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        blocks.append(output_proj)

        self.blocks = nn.ModuleList(blocks)


    def forward_features(self, x):
        """Run all blocks except the last Linear block"""
        
        for block in self.blocks[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.blocks[-1](x)
        return x
