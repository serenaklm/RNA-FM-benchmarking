import torch.nn as nn
import torch.nn.functional as F 


class MLP(nn.Module):

    def __init__(self, num_layers: int, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int, 
                 act: nn.Module,
                 dropout_rate: float):
        
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        blocks = []

        for _ in range(num_layers-1):

            blocks.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        act,
                                        nn.Dropout(dropout_rate)))

            input_dim = hidden_dim 
        
        if num_layers == 1: hidden_dim = input_dim 
        
        output_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        blocks.append(output_proj)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        
        return x 
