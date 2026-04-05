# pytorch_modules/losses/cox_loss.py

import torch


def cox_ph_loss(risk, time, event, eps: float = 1e-8) -> torch.Tensor:
    """
    Cox partial negative log-likelihood (Breslow approximation).
    Works with minibatches (approximation to full risk sets).

    Args:
        risk:  (B,) or (B,1) predicted risk scores (higher = riskier)
        time:  (B,) survival/censoring times
        event: (B,) event indicators (1=event observed, 0=censored)
        eps:   numerical stability constant

    Returns:
        scalar loss (mean per event)
    """
    # Flatten
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()

    # Sort by descending time (so risk sets are prefixes)
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    # Stabilize exponentials
    risk = risk - risk.max()
    exp_risk = torch.exp(risk)

    # Cumulative sum of exp(risk) over risk sets
    cum_risk = torch.cumsum(exp_risk, dim=0)

    # Partial log-likelihood contributions for events only
    loglik = (risk - torch.log(cum_risk + eps)) * event

    # Normalize by number of observed events (avoid divide-by-zero)
    n_events = event.sum().clamp_min(1.0)
    return -loglik.sum() / n_events
