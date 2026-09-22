"""ODE samplers -- Euler and Heun -- for both model families, with guidance.

Implements SMG_PRIOR_WORK_AUDIT.md section 3 exactly:

  Euler         x_{n+1} = x_n + h F(x_n, t_n)
  Heun (RK2)    k1 = F(x_n, t_n);  x~ = x_n + h k1;  k2 = F(x~, t_n + h)
                x_{n+1} = x_n + h/2 (k1 + k2)

with F the FULL guided field, re-evaluated (generator, posterior, f, g, H, c,
v_f, all schedules) at the predicted state in stage two. Holding the guidance
fixed across stages would be a different method.

Fields (section 3.1 / 3.2):

  flow matching   F = v_theta + w (1 - t) / t * G,           t: 0 -> 1, h > 0
  VP diffusion    F = -1/2 beta x - 1/2 beta (s_theta + G),   tau: 1 -> 0, h < 0

with G in score units from guidance.py and s_theta = -eps_theta / sigma.

GUIDANCE CLIP. The (1 - t)/t conversion factor is 19 at t = 0.05 and the raw
property gradient is several Debye per angstrom, so an unclipped correction
dwarfs the base velocity early in the trajectory and throws the sample off the
data manifold (measured: molecule stability 0.16 -> 0.02, a third of
trajectories non-finite). The correction is therefore capped at `clip` times
the base velocity norm, per sample. This is the velocity-relative trust region
of OSCAR (component A in our catalogue) -- prior work, adopted, not claimed.
`clip=None` disables it for the ablation that shows why it is needed.

Endpoint policy, declared once and used by every arm: integrate to the last
grid point, then jump to the posterior mean m there. Every generator, guide,
VJP and JVP pass is counted in `sampler.cost` so cost is measured, not inferred
from step counts.
"""
from __future__ import annotations

import torch

from diffusion import alpha_sigma, beta
from guidance import (Cost, DISPLACEMENT_MODES, fm_posterior,  # noqa: E501
                      guidance_field, vp_posterior)
from guidance_update import GuidanceUpdater
from models.egnn import zero_com


def _clip_to_velocity(G_c, G_f, v_c, v_f, mask, clip):
    """Scale (G_c, G_f) per sample so its norm is at most clip * |(v_c, v_f)|."""
    if clip is None:
        return G_c, G_f
    gn = torch.sqrt((G_c ** 2).sum((1, 2)) + (G_f ** 2).sum((1, 2)) + 1e-12)
    vn = torch.sqrt((v_c ** 2).sum((1, 2)) + (v_f ** 2).sum((1, 2)) + 1e-12)
    f = torch.clamp(clip * vn / gn, max=1.0).view(-1, 1, 1)
    return G_c * f, G_f * f


class _Base:
    def __init__(self, net, mask, f_net=None, y=None, s=1.0, mode="plug",
                 w=1.0, n_probe=1, probe_seed=0, clip=1.0,
                 n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                 band_tau=None, band_eta=1.0, band_radius=None,
                 tau=None, spbc_eta=1.0, spbc_radius=None, schedule=None,
                 update_rule="euler", update_kw=None):
        self.net, self.mask = net, mask
        self.f_net, self.y, self.s, self.mode, self.w = f_net, y, s, mode, w
        self.n_probe, self.clip = n_probe, clip
        self.n_mc, self.sigma_mc = n_mc, sigma_mc
        # arms that need extra state: rch carries a fitted head, band carries
        # its tolerance, kappa3 is a diagnostic column on whatever arm is running
        self.want_kappa3, self.rch = want_kappa3, rch
        self.band_tau, self.band_eta, self.band_radius = band_tau, band_eta, band_radius
        self.tau, self.spbc_eta, self.spbc_radius = tau, spbc_eta, spbc_radius
        # SHG: [(t_lo, t_hi, mode, w), ...]. The first interval containing t
        # wins; outside every interval, guidance is off. `mode`/`w` on the
        # sampler are the fallback when no schedule is given.
        self.schedule = schedule
        # THE UPDATE RULE. The only thing this changes is how the guidance
        # direction returned by guidance_field is turned into the correction
        # added to the field; "euler" is the original first-order behaviour and
        # returns the direction untouched. `_primary_stage` is False during a
        # Heun second stage so one ODE step advances the optimiser state once.
        self.updater = GuidanceUpdater(rule=update_rule, **(update_kw or {}))
        self._primary_stage = True
        self.kappa3_log = []
        self.schedule_log = {}
        self.diag_acc = {}
        self.cost = Cost()
        self.n_field = 0
        self.n_clipped = 0
        self.n_guided = 0
        self.probe_gen = torch.Generator(device=mask.device).manual_seed(probe_seed)
        self.last_diag = None

    def active(self, t_scalar):
        """(mode, weight) in force at this time, or (None, 0.0) for unguided.

        A handoff is a hard switch, so a Heun step that straddles a boundary
        would otherwise blend two fields inside one step. Both Heun stages call
        this with their own time, so each stage uses one regime -- which is
        what the memo requires (section 9.5).

        THE SCHEDULE'S THIRD FIELD IS A MULTIPLIER ON self.w, NOT AN ABSOLUTE
        WEIGHT. Returning it directly made every SHG arm ignore the strength
        knob entirely: the screening sweep produced five cells per arm at
        w = 0.01 ... 4 that were identical to the last digit (alpha MAE 4.811
        at every strength). That is the same signature as the earlier `band`
        dead-code bug, and it is why the multiplier is applied here."""
        if self.schedule is None:
            return self.mode, self.w
        for (lo, hi, mode, mult) in self.schedule:
            if lo <= t_scalar < hi:
                return mode, self.w * mult
        return None, 0.0

    def note_guided(self, mode):
        """Record that a step was ACTUALLY guided with `mode`.

        Counting inside `active()` overcounted, because the caller may still
        skip the step on t_min_guide or t >= 1: a schedule starting at 0.50
        with t_min_guide = 0.75 logged 10 steps of which 5 never ran."""
        self.schedule_log[mode] = self.schedule_log.get(mode, 0) + 1

    def reset_counts(self):
        self.cost.reset()
        self.n_field = self.n_clipped = self.n_guided = 0
        self.schedule_log = {}
        self.diag_acc = {}
        self.kappa3_log = []      # was omitted; would double-count on reuse
        self.updater.reset()      # moment buffers are per-trajectory

    # Diagnostics worth carrying out of the run. Deliberately a short list:
    # these are the quantities that decide whether BTVG's variance model held
    # and whether SPBC's forecast was moving anything, and without them the
    # sweep cannot answer either question.
    DIAG_KEYS = ("btvg_V_over_tau2", "btvg_V_nonpositive", "btvg_V_raw",
                 "spbc_bias", "spbc_increment", "spbc_live_fraction",
                 "spbc_mean_response", "c", "v_f")

    def _accumulate_diag(self, diag):
        """Running mean of each diagnostic over every guided step and batch."""
        for key in self.DIAG_KEYS:
            v = diag.get(key)
            if v is None:
                continue
            try:
                x = float(v.double().mean()) if hasattr(v, "double") else float(v)
            except Exception:
                continue
            if x != x:                       # NaN carries no information here
                continue
            tot, n = self.diag_acc.get(key, (0.0, 0))
            self.diag_acc[key] = (tot + x, n + 1)

    def diag_summary(self):
        return {k: tot / n for k, (tot, n) in self.diag_acc.items() if n}

    def _guide(self, coords, feats, t, post_fn, mode=None, t_scalar=None):
        G_c, G_f, diag = guidance_field(
            self.f_net, post_fn, coords, feats, self.mask, self.y, self.s,
            mode or self.mode, self.n_probe, self.probe_gen, self.cost,
            self.n_mc, self.sigma_mc, want_kappa3=self.want_kappa3,
            rch=self.rch, band_tau=self.band_tau, band_eta=self.band_eta,
            band_radius=self.band_radius, t_scalar=t,
            tau=self.tau, spbc_eta=self.spbc_eta, spbc_radius=self.spbc_radius)
        self.last_diag = diag
        self._accumulate_diag(diag)
        if self.want_kappa3 and "kappa3_skew" in diag:
            self.kappa3_log.append(diag["kappa3_skew"].detach().cpu())
        # THE ONE PLACE THE GUIDANCE DIRECTION BECOMES AN UPDATE. Everything
        # downstream -- the (1-t)/t conversion, the strength w, the
        # velocity-relative clip -- is unchanged and sees a direction of the
        # same per-sample norm, so only the DIRECTION is under test.
        G_c, G_f = self.updater.apply(G_c, G_f, self.mask, t_scalar=t_scalar,
                                      update_state=self._primary_stage)
        return zero_com(G_c, self.mask), G_f * self.mask.unsqueeze(-1)


class FlowSampler(_Base):
    def __init__(self, net, mask, t_min_guide=0.05, **kw):
        super().__init__(net, mask, **kw)
        self.t_min_guide = t_min_guide

    def time_grid(self, n_steps, span=None):
        a, b = (0.0, 1.0) if span is None else span
        return torch.linspace(a, b, n_steps + 1)

    def field(self, coords, feats, t_scalar):
        self.n_field += 1
        B = coords.shape[0]
        t = torch.full((B,), float(t_scalar), device=coords.device)
        with torch.no_grad():
            v_c, v_f = self.net(coords, feats, self.mask, t)
        self.cost.gen_fwd += 1
        mode, w = self.active(t_scalar)
        if (self.f_net is None or mode is None or t_scalar < self.t_min_guide
                or t_scalar >= 1.0):
            return v_c, v_f

        self.n_guided += 1
        self.note_guided(mode)
        post_fn = lambda c, f: fm_posterior(self.net, c, f, self.mask, t)  # noqa: E731
        G_c, G_f = self._guide(coords, feats, t, post_fn, mode,
                               t_scalar=t_scalar)
        if mode in DISPLACEMENT_MODES:
            # Already a state displacement. Converting it again through the
            # score-to-velocity factor (1-t)/t would apply that factor twice
            # and make the edit scale wrongly in t.
            mult = w
        else:
            mult = w * (1.0 - t_scalar) / t_scalar
        G_c, G_f = mult * G_c, mult * G_f
        if self.clip is not None:
            gn = torch.sqrt((G_c ** 2).sum((1, 2)) + (G_f ** 2).sum((1, 2)))
            vn = torch.sqrt((v_c ** 2).sum((1, 2)) + (v_f ** 2).sum((1, 2)))
            self.n_clipped += int((gn > self.clip * vn).sum().item())
            G_c, G_f = _clip_to_velocity(G_c, G_f, v_c, v_f, self.mask, self.clip)
        return v_c + G_c, v_f + G_f

    @torch.no_grad()
    def terminal(self, coords, feats, t_scalar):
        return coords, feats            # t = 1: m = x


class VPSampler(_Base):
    """Probability-flow ODE for a VP diffusion model.

    `noise_schedule` selects which VP schedule the checkpoint was trained
    under. The default is `diffusion.py`'s linear-beta schedule, which is what
    OUR diffusion generator uses. A borrowed checkpoint trained under a
    different schedule must pass its own (see
    `external/edm_schedule.EDMSchedule`): sampling EDM under the linear-beta
    schedule does not give a weaker EDM, it gives a different model, and the
    failure is silent -- samples still come out, they are just not draws from
    the trained distribution.

    `tau_max_guide` is the VP counterpart of `FlowSampler.t_min_guide`: guidance
    is skipped while `tau > tau_max_guide`, i.e. in the noisy part of the
    trajectory. The two are the SAME window under the time convention of each
    family -- flow time runs 0 (noise) -> 1 (data) and VP time runs 1 (noise)
    -> 0 (data), so `t_min_guide = w` corresponds to `tau_max_guide = 1 - w`.

    It defaults to `None` (guide everywhere), which is what this class did
    before the parameter existed, so nothing already measured changes. But
    `None` is not a neutral choice and a caller should not take it as one: the
    guidance window is the largest single effect measured anywhere in this
    project -- alpha MAE 10.02 -> 5.32 as `t_min_guide` goes 0.05 -> 0.5,
    bigger than every arm difference combined -- and guiding at tau near 1 asks
    the guide to read a property off a posterior mean formed by dividing by
    alpha(1) ~ 0.003, which amplifies any epsilon error by ~300x. A sweep that
    leaves this at `None` is measuring arms at the bad end of that curve and
    must say so.
    """

    def __init__(self, net, mask, tau_min=1e-3, noise_schedule=None,
                 tau_max_guide=None, **kw):
        super().__init__(net, mask, **kw)
        self.tau_min = tau_min
        self.tau_max_guide = tau_max_guide
        self.alpha_sigma = (noise_schedule.alpha_sigma if noise_schedule
                            else alpha_sigma)
        self.beta = noise_schedule.beta if noise_schedule else beta

    def time_grid(self, n_steps, span=None):
        a, b = (1.0, self.tau_min) if span is None else span
        return torch.linspace(a, b, n_steps + 1)                  # decreasing

    def field(self, coords, feats, tau_scalar):
        self.n_field += 1
        B = coords.shape[0]
        tau = torch.full((B,), float(tau_scalar), device=coords.device)
        a, s = self.alpha_sigma(tau)
        b = self.beta(tau).view(-1, 1, 1)
        sb = s.view(-1, 1, 1)
        with torch.no_grad():
            e_c, e_f = self.net(coords, feats, self.mask, tau)
        self.cost.gen_fwd += 1
        score_c = -e_c / sb
        score_f = -e_f / sb
        base_c = -0.5 * b * coords - 0.5 * b * score_c
        base_f = -0.5 * b * feats - 0.5 * b * score_f
        mode, w = self.active(tau_scalar)
        if (self.tau_max_guide is not None
                and tau_scalar > self.tau_max_guide):
            mode = None
        if self.f_net is not None and mode is not None:
            self.n_guided += 1
            self.note_guided(mode)
            post_fn = lambda c, f: vp_posterior(self.net, c, f, self.mask, tau, a, s)  # noqa: E731
            G_c, G_f = self._guide(coords, feats, tau, post_fn, mode,
                                   t_scalar=tau_scalar)
            # Same units rule as the flow sampler: a displacement arm must not
            # be pushed through the score-to-drift conversion. Without this,
            # every SHG cell on a diffusion checkpoint would hand a schedule
            # NAME into guidance_field and raise.
            if mode in DISPLACEMENT_MODES:
                G_c, G_f = w * G_c, w * G_f
            else:
                G_c, G_f = -0.5 * b * w * G_c, -0.5 * b * w * G_f
            if self.clip is not None:
                gn = torch.sqrt((G_c ** 2).sum((1, 2)) + (G_f ** 2).sum((1, 2)))
                vn = torch.sqrt((base_c ** 2).sum((1, 2)) + (base_f ** 2).sum((1, 2)))
                self.n_clipped += int((gn > self.clip * vn).sum().item())
                G_c, G_f = _clip_to_velocity(G_c, G_f, base_c, base_f, self.mask, self.clip)
            base_c, base_f = base_c + G_c, base_f + G_f
        m = self.mask.unsqueeze(-1)
        return zero_com(base_c, self.mask), base_f * m

    @torch.no_grad()
    def terminal(self, coords, feats, tau_scalar):
        B = coords.shape[0]
        tau = torch.full((B,), float(tau_scalar), device=coords.device)
        p = vp_posterior(self.net, coords, feats, self.mask, tau,
                         *self.alpha_sigma(tau))
        self.cost.gen_fwd += 1
        return zero_com(p.mean_coords, self.mask), p.mean_feats * self.mask.unsqueeze(-1)


def integrate(sampler, coords, feats, n_steps, solver="euler", span=None,
              terminal=True):
    """Run the declared ODE. Returns (coords, feats, n_field_evals); the full
    cost breakdown is in sampler.cost afterwards.

    span=(a, b) integrates an interior interval; terminal=False skips the
    endpoint policy. Both exist for the same-field convergence check, which
    needs a smooth field away from the endpoints and the guidance switch-on."""
    sampler.reset_counts()
    ts = sampler.time_grid(n_steps, span)
    for i in range(n_steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        h = t1 - t0
        k1_c, k1_f = sampler.field(coords, feats, t0)
        if solver == "euler":
            coords = coords + h * k1_c
            feats = feats + h * k1_f
        elif solver == "heun":
            xt_c = coords + h * k1_c
            xt_f = feats + h * k1_f
            # The second stage reads the moment buffers but must not write to
            # them, or one ODE step would take two optimiser steps and Heun
            # would no longer be a controlled comparison against Euler.
            sampler._primary_stage = False
            k2_c, k2_f = sampler.field(xt_c, xt_f, t1)
            sampler._primary_stage = True
            coords = coords + 0.5 * h * (k1_c + k2_c)
            feats = feats + 0.5 * h * (k1_f + k2_f)
        else:
            raise ValueError(solver)
    if terminal:
        coords, feats = sampler.terminal(coords, feats, float(ts[-1]))
    return coords, feats, sampler.n_field


def initial_noise(mask, n_types, generator=None):
    """x at t = 0 (flow) or tau = 1 (diffusion): N(0, I) on the zero-CoM
    subspace for coordinates, N(0, I) masked for atom types."""
    B, N = mask.shape
    c = torch.randn(B, N, 3, generator=generator, device=mask.device)
    f = torch.randn(B, N, n_types, generator=generator, device=mask.device)
    return zero_com(c, mask), f * mask.unsqueeze(-1)
