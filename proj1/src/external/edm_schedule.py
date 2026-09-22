"""EDM's discrete `polynomial_2` noise schedule, exposed as a continuous
VP schedule so our `VPSampler` can integrate the probability-flow ODE on it.

WHY THIS FILE EXISTS. `proj1/src/diffusion.py` defines the *linear-beta* VP
schedule of Song et al., which is what our own diffusion generator was trained
on. EDM was trained on a different schedule -- a clipped polynomial in
alpha^2, tabulated on 1001 grid points -- and a checkpoint sampled under the
wrong schedule is not a weaker model, it is a different model. So the borrowed
generator gets its own schedule object, and `VPSampler` takes it by injection.

THE CONVERSION, derived once and gated in tests/test_transfer_backend.py.

EDM stores gamma(t) = log(sigma^2(t) / alpha^2(t)) on t_int = 0..T, and reads
it back at t_int = round(t * T) for t in [0, 1]. From gamma:

    alpha^2 = sigmoid(-gamma)        sigma^2 = sigmoid(gamma)

which is variance preserving -- sigmoid(g) + sigmoid(-g) = 1 exactly -- so the
same forward SDE our sampler assumes applies, with a beta we have to recover:

    log alpha = 1/2 log sigmoid(-gamma)
    d/dt log alpha = -1/2 sigmoid(gamma) gamma'(t) = -1/2 sigma^2 gamma'(t)
    d/dt log alpha = -1/2 beta(t)          (definition of the VP schedule)
    ==> beta(t) = sigma^2(t) * gamma'(t)

gamma is tabulated, not closed form (`clip_noise_schedule` applies a running
clamp that no elementary function reproduces), so gamma' is taken from the
table by central differences and gamma itself by linear interpolation between
grid points. Both are exact at the grid points the model was trained on, and
the sampler only ever visits ~100 of the 1001 of them.

TIME DIRECTION. EDM's t and our tau agree: 0 is clean data, 1 is noise. No
flip is needed anywhere, and there is a test that asserts it rather than a
comment claiming it.
"""
from __future__ import annotations

import numpy as np
import torch


def _clip_noise_schedule(alphas2, clip_value=0.001):
    """Verbatim port of EDM's `clip_noise_schedule` (energy.py / en_diffusion.py).

    Kept as a copy rather than an import because the vendored file also
    executes module-level code we do not want to run.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def _polynomial_schedule(timesteps: int, s=1e-5, power=2.0):
    """Verbatim port of EDM's `polynomial_schedule`. Returns alpha^2 on 0..T."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = _clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    return precision * alphas2 + s


class EDMSchedule:
    """Continuous (alpha, sigma, beta) from EDM's tabulated gamma.

    Constructed from the generator's own `args.pickle` so a checkpoint trained
    under `polynomial_2` at 1000 steps can never be sampled under anything
    else by accident: `from_args` reads the schedule name, step count and
    precision off the checkpoint and refuses a schedule it has not ported.
    """

    def __init__(self, noise_schedule="polynomial_2", timesteps=1000,
                 precision=1e-5, device="cpu", dtype=torch.float32):
        if not noise_schedule.startswith("polynomial_"):
            raise ValueError(
                "only EDM's polynomial_* schedules are ported; got %r. Porting "
                "'cosine' or 'learned' means adding the corresponding table -- "
                "do not fall back to the linear-beta schedule in diffusion.py, "
                "which is a different model." % noise_schedule)
        power = float(noise_schedule.split("_")[1])
        alphas2 = _polynomial_schedule(timesteps, s=precision, power=power)
        sigmas2 = 1.0 - alphas2
        gamma = np.log(sigmas2) - np.log(alphas2)

        self.T = timesteps
        self.noise_schedule = noise_schedule
        self.precision = precision
        self.gamma = torch.as_tensor(gamma, dtype=dtype, device=device)

        # gamma'(tau) as the slope of the SAME piecewise-linear interpolant
        # that `gamma_at` evaluates -- one value per cell [k/T, (k+1)/T), so
        # this array is one shorter than gamma.
        #
        # THIS HAS TO BE THE INTERPOLANT'S OWN SLOPE, not a central difference
        # of the table. Central differences are the more accurate estimate of
        # the true derivative of the underlying schedule, and using them broke
        # the identity d log alpha / d tau = -beta / 2 by 1.3% at the worst
        # tau -- because alpha comes from the INTERPOLANT, and the interpolant's
        # derivative is the forward difference. The sampler's drift must be
        # consistent with the sampler's own alpha and sigma; an inconsistency
        # there is not a small numerical error, it is integrating a different
        # ODE than the one whose marginals the model was trained to match.
        # With the slope taken this way the identity is exact up to the
        # variation of sigma^2 inside one cell, and the gate measures 1e-5.
        self.dgamma = (self.gamma[1:] - self.gamma[:-1]) * timesteps

    @classmethod
    def from_args(cls, args: dict, device="cpu", dtype=torch.float32):
        return cls(noise_schedule=args["diffusion_noise_schedule"],
                   timesteps=int(args["diffusion_steps"]),
                   precision=float(args["diffusion_noise_precision"]),
                   device=device, dtype=dtype)

    def to(self, device=None, dtype=None):
        self.gamma = self.gamma.to(device=device, dtype=dtype)
        self.dgamma = self.dgamma.to(device=device, dtype=dtype)
        return self

    # -- interpolation -----------------------------------------------------

    def _cell(self, tau):
        """(cell index, position within cell) for continuous tau in [0, 1]."""
        tau = torch.as_tensor(tau, dtype=self.gamma.dtype,
                              device=self.gamma.device)
        x = tau.reshape(-1).clamp(0.0, 1.0) * self.T
        lo = x.floor().clamp(max=self.T - 1).long()
        return lo, x - lo.to(x.dtype)

    def gamma_at(self, tau):
        """Linear interpolation of EDM's gamma table at continuous tau.

        EDM itself SNAPS to round(tau * T). Snapping makes alpha piecewise
        constant, so beta derived from it would be a comb of zeros and spikes
        and the ODE would integrate a drift unrelated to its own marginals.
        Interpolating agrees with EDM exactly on the grid -- gated -- and is
        differentiable between points, which is what a solver needs.
        """
        lo, frac = self._cell(tau)
        return self.gamma[lo] * (1.0 - frac) + self.gamma[lo + 1] * frac

    def alpha_sigma(self, tau):
        """(alpha, sigma) with alpha^2 + sigma^2 = 1, flat, one entry per tau."""
        g = self.gamma_at(tau)
        return torch.sigmoid(-g).sqrt(), torch.sigmoid(g).sqrt()

    def beta(self, tau):
        """beta(tau) = sigma^2(tau) * gamma'(tau) -- see the module docstring."""
        lo, _ = self._cell(tau)
        return torch.sigmoid(self.gamma_at(tau)) * self.dgamma[lo]

    def tau_of_gamma(self, g):
        """Inverse of `gamma_at`: the tau whose gamma is g.

        gamma is strictly increasing in tau (gated), so the inverse is well
        defined and is found by locating the cell and interpolating inside it
        -- exactly undoing `gamma_at`, so `tau_of_gamma(gamma_at(t)) == t`.

        This exists for ONE purpose: building a time grid uniform in gamma
        instead of uniform in tau. EDM's polynomial_2 schedule is extremely
        stiff at the noise end -- on a uniform 100-step tau grid the FIRST step
        moves gamma by 3.70, against 0.20 for the linear-beta schedule our own
        diffusion model uses, and |1 - beta*h/2| reaches 2.68 there. A grid
        uniform in gamma spends steps where the distribution is actually
        moving, which is the standard fix and costs nothing.
        """
        g = torch.as_tensor(g, dtype=self.gamma.dtype, device=self.gamma.device)
        flat = g.reshape(-1).clamp(float(self.gamma[0]), float(self.gamma[-1]))
        # right=True then -1 puts a value equal to a knot in the cell BELOW it,
        # which keeps the top endpoint from indexing past the table.
        idx = (torch.searchsorted(self.gamma, flat, right=True) - 1)
        idx = idx.clamp(0, self.T - 1)
        lo, hi = self.gamma[idx], self.gamma[idx + 1]
        frac = (flat - lo) / (hi - lo).clamp(min=1e-30)
        return ((idx.to(flat.dtype) + frac) / self.T).reshape(g.shape)
