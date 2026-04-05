import torch
import torch.nn as nn

class CIndexLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk, time, event):
        """
        risk : (N,) predicted risk scores (higher = higher risk)
        time : (N,) survival/censor times
        event: (N,) 1 if event observed, 0 if censored
        """
        n = risk.shape[0]
        # Broadcast times and events
        time_i  = time.unsqueeze(0).expand(n, n)
        time_j  = time.unsqueeze(1).expand(n, n)
        event_i = event.unsqueeze(0).expand(n, n)

        # Mask of comparable pairs: i failed before j
        mask = (time_i < time_j) & (event_i == 1)

        # Pairwise risk difference
        diff = risk.unsqueeze(0) - risk.unsqueeze(1)

        # logistic loss on comparable pairs
        loss = torch.log1p(torch.exp(-diff))   # = softplus(-diff)
        return loss[mask].mean()