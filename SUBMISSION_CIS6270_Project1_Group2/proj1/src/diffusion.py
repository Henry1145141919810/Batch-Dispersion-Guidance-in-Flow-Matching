"""Variance-preserving (VP) diffusion schedule, continuous time.

Forward noising time tau: 0 (clean) -> 1 (noise), the convention of
SMG_PRIOR_WORK_AUDIT.md section 3.2, so the probability-flow ODE there is
integrated from tau = 1 down to 0 with h < 0.

    dX = -1/2 beta(tau) X dtau + sqrt(beta(tau)) dW

    x_tau = alpha(tau) x_0 + sigma(tau) eps,   alpha^2 + sigma^2 = 1

    log alpha(tau) = -1/2 * int_0^tau beta(s) ds
                   = -1/2 * (beta_min tau + 1/2 (beta_max - beta_min) tau^2)

for the linear beta schedule of Song et al. (2021). The network predicts eps;
everything downstream is derived from that:

    score     s_theta = -eps_theta / sigma
    x0-hat    (x_tau - sigma eps_theta) / alpha            (conditional mean)
    Sigma     sigma^2 / alpha * d x0-hat / d x_tau          (2nd-order Tweedie)

The last line is the covariance the SMG correction needs; it has the same form
as the flow-matching case with (alpha, sigma) in place of (t, 1 - t).
"""
from __future__ import annotations

import torch

BETA_MIN = 0.1
BETA_MAX = 20.0


def beta(tau: torch.Tensor, beta_min: float = BETA_MIN,
         beta_max: float = BETA_MAX) -> torch.Tensor:
    return beta_min + tau * (beta_max - beta_min)


def log_alpha(tau: torch.Tensor, beta_min: float = BETA_MIN,
              beta_max: float = BETA_MAX) -> torch.Tensor:
    return -0.5 * (beta_min * tau + 0.5 * (beta_max - beta_min) * tau ** 2)


def alpha_sigma(tau: torch.Tensor, beta_min: float = BETA_MIN,
                beta_max: float = BETA_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    """(alpha, sigma) with alpha^2 + sigma^2 = 1. Both shaped like tau."""
    a = torch.exp(log_alpha(tau, beta_min, beta_max))
    s = torch.sqrt((1.0 - a * a).clamp(min=1e-12))
    return a, s
