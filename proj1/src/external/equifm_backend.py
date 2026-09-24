"""EquiFM (Song et al., NeurIPS 2023) as a borrowed flow-matching base model.

WHAT IT IS. The released unconditional QM9 generator, byte-identical in the
MolFM repo, OC-Flow's repo and the NeurIPS supplement (SHA-256 47b40ae6...,
docs/protocol/EQUIFM_USABILITY_AUDIT.md). 256 x 9 EGNN, explicit hydrogens,
H/C/N/O/F, trained on the FULL 100K QM9 training partition. Loaded here
through the project's safe `definitions()` reader: no external module-level
code runs.

ITS PATH, read off the released training code (supplement
`equivariant_diffusion/cnflows.py`, HB_path), in its NATIVE time tau
(1 = noise, 0 = data):

  coordinates   x_tau = (1 - tau) x_0 + s_x(tau) z,  s_x = eps + (1-eps) tau,
                eps = 1e-4; the net's coordinate head q_x regresses
                (1-eps) z - x_0. z is rotation- and permutation-ALIGNED to
                x_0 during training (Kabsch + Hungarian).
  atom types    h_tau = a h_0 + sqrt(1-a^2) z',  a = exp(-T/2),
                T = 0.1 tau + 9.95 tau^2 (VP, beta 0.1 -> 20), z' fresh;
                the type head q_h regresses a h_0 - a^2 h_tau.
  charge        one extra channel (charge / 10) on the same VP path.

State layout: coords [B, N, 3] in angstroms (normalize_factors[0] = 1),
features [B, N, 6] = one-hot / 4 (5) + charge / 10 (1).

WHAT THE GUIDANCE ARMS GET, and how exact each piece is.

  posterior mean   m_x = (1-eps) x - s_x q_x,   m_h = a h + q_h / a.
                   Exact algebra from the regression targets. Alignment does
                   not break a conditional-mean identity.
  covariance       Sigma = K J with K = diag(k_x I, k_h I),
                   k_x = s_x^2 / (1 - tau),  k_h = (1 - a^2) / a:
                   Tweedie's identity for an INDEPENDENT Gaussian channel per
                   block. For the type block that is exactly the training
                   path. For coordinates it is an APPROXIMATION: the training
                   noise was aligned to x_0, so p(x_tau | x_0) is not the
                   isotropic Gaussian the identity assumes. Every covariance-
                   using arm (tmpd, lgd_mc, btvg, btvg_var) inherits the same
                   surrogate; plug uses no covariance and is exact.
  guided velocity  the model's own velocity map evaluated at the guided mean
                   m + K G (G = the arm's score-space field), i.e.
                   dv_x = -(k_x / s_x) G_x = -s_x/(1-tau) G_x and
                   dv_h = M(tau) a k_h G_h ~ -beta/2 G_h, with M the native
                   feature multiplier -beta/2/(1 - a^2 + 1e-5). Consistent
                   with the checkpoint's own parametrisation by construction.

PER-BLOCK K WITHOUT TOUCHING guidance.py. guidance.py forms every Sigma v as
k * (forward-mode JVP of the mean) and every pull-back as a reverse-mode VJP.
`_ScaleTangents` is the identity on values and on reverse-mode gradients, and
multiplies FORWARD-MODE tangents by (k_x, k_h). With the posterior's scalar
k set to 1, every Sigma v in guidance.py becomes K J v, and every J^T w stays
exact. tests/test_equifm_backend.py checks both against explicit references,
including BTVG's reverse-over-forward variance gradient.

WHAT IS GUIDED. Coordinates and the five one-hot channels. The charge
channel is held fixed inside each guidance evaluation (the property networks
never read it) and is advanced by the generator alone; restricting Sigma to a
sub-block keeps Sigma = K J exact for that sub-block.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from guidance import Posterior, tfg_components
from models.egnn import zero_com
from sampling import _Base, _clip_to_velocity

from .tfg_assets import TFGOracle, definitions, dense_edges, metadata

ROOT = Path(__file__).resolve().parents[3]
EQUIFM_DIR = ROOT / "audit" / "equifm_20260922" / "OC-Flow" / "molecule" / "equifm"
EQUIFM_WEIGHTS = EQUIFM_DIR / "generative_model_ema_0.npy"
EQUIFM_ARGS = EQUIFM_DIR / "args.pickle"
EQUIFM_SHA256 = "47b40ae673c1117219432f0c81ae61f72ac380761200144ed04ebb9f179d3527"

EPS = 1e-4                       # the path's coordinate floor, from the training code
BETA_MIN, BETA_MAX = 0.1, 20.0
N_TYPES = 5                      # one-hot channels; channel 5 is the charge


def T(tau):
    return 0.5 * (BETA_MAX - BETA_MIN) * tau ** 2 + BETA_MIN * tau


def beta(tau):
    return (BETA_MAX - BETA_MIN) * tau + BETA_MIN


def a_of(tau):
    return math.exp(-0.5 * T(tau))


def s_x_of(tau):
    return EPS + (1.0 - EPS) * tau


def feat_multiplier(tau):
    """The native sampler's factor on the feature head (cnf_models.forward)."""
    return -0.5 * beta(tau) / (1.0 - math.exp(-T(tau)) + 1e-5)


def block_k(tau):
    """(k_x, k_h): the independent-Gaussian Tweedie scalars per block."""
    a = a_of(tau)
    return s_x_of(tau) ** 2 / (1.0 - tau), (1.0 - a * a) / a


class _ScaleTangents(torch.autograd.Function):
    """Identity on values and reverse-mode gradients; forward-mode tangents are
    multiplied by (k_c, k_f). See the module docstring."""

    @staticmethod
    def forward(m_c, m_f, k_c, k_f):
        # clones, not views: a custom Function returning views of its inputs
        # falls under autograd's view rules
        return m_c.clone(), m_f.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.k = (float(inputs[2]), float(inputs[3]))

    @staticmethod
    def backward(ctx, g_c, g_f):
        return g_c, g_f, None, None

    @staticmethod
    def jvp(ctx, t_c, t_f, _t_kc, _t_kf):
        k_c, k_f = ctx.k
        return (None if t_c is None else t_c * k_c,
                None if t_f is None else t_f * k_f)


class EquiFMGenerator(torch.nn.Module):
    """The released EMA generator; `raw(coords, feats, mask, tau)` is the
    network head (q_x, q_h) BEFORE the sampler's feature multiplier."""

    def __init__(self, device="cpu", weights=EQUIFM_WEIGHTS, args_path=EQUIFM_ARGS):
        super().__init__()
        a = metadata(args_path)
        self.args = a
        if a.get("discrete_path") != "HB_path" or not a.get("include_charges"):
            raise ValueError("this adapter implements the released HB_path, "
                             "include_charges=True checkpoint only; got %r / %r"
                             % (a.get("discrete_path"), a.get("include_charges")))
        if float(a.get("cat_loss_step", -1)) > 0:
            raise ValueError("cat_loss_step > 0 rescales the type channels' "
                             "time; this adapter does not implement that")
        self.norm_values = [float(v) for v in a["normalize_factors"]]
        if self.norm_values[0] != 1.0:
            raise ValueError("coordinates must be unscaled angstroms")
        gd = definitions(EQUIFM_DIR / "egnn.py",
                         {"GCL", "EquivariantUpdate", "EquivariantBlock", "EGNN",
                          "SinusoidsEmbeddingNew", "coord2diff",
                          "unsorted_segment_sum"})
        cd = definitions(EQUIFM_DIR / "cnf_models.py",
                         {"Cnflows", "EGNN_dynamics_QM9", "T", "T_hat",
                          "remove_mean_with_mask", "assert_correctly_masked",
                          "assert_mean_zero_with_mask",
                          "sample_center_gravity_zero_gaussian_with_mask",
                          "sample_gaussian_with_mask"}, gd)
        net = cd["EGNN_dynamics_QM9"](
            7, a["context_node_nf"], 3, hidden_nf=a["nf"], device=device,
            n_layers=a["n_layers"], attention=a["attention"], tanh=a["tanh"],
            norm_constant=a["norm_constant"], inv_sublayers=a["inv_sublayers"],
            sin_embedding=a["sin_embedding"],
            normalization_factor=a["normalization_factor"],
            aggregation_method=a["aggregation_method"])
        self.flow = cd["Cnflows"](
            net, 6, 3, norm_values=a["normalize_factors"], include_charges=True,
            discrete_path=a["discrete_path"], cat_loss=a["cat_loss"],
            cat_loss_step=a["cat_loss_step"], on_hold_batch=a["on_hold_batch"],
            sampling_method=a["sampling_method"],
            weighted_methods=a["weighted_methods"], ode_method=a["ode_method"],
            without_cat_loss=a["without_cat_loss"],
            angle_penalty=a["angle_penalty"])
        state = torch.load(weights, weights_only=True, map_location="cpu")
        self.flow.load_state_dict(state, strict=True)
        self.flow.to(device).eval().requires_grad_(False)

    def raw(self, coords, feats, mask, tau):
        m3 = mask.unsqueeze(-1)
        z = torch.cat([coords, feats], dim=-1) * m3
        t = torch.as_tensor(float(tau), device=coords.device, dtype=coords.dtype)
        # rebuilt every call on purpose: it is cheap next to the network, and a
        # cache keyed on the mask's storage address could return a stale edge
        # list if that address were ever reused for a different mask
        out = self.flow.phi(t, z, m3, dense_edges(mask)[1], None)
        return out[..., :3], out[..., 3:]

    def velocity(self, coords, feats, mask, tau):
        """The native sampler's dz/dtau (cnf_models.forward, HB_path)."""
        q_x, q_h = self.raw(coords, feats, mask, tau)
        return q_x, q_h * feat_multiplier(tau)


def velocity_delta(tau, G_c, G_f):
    """Change in the native tau-velocity when the posterior mean moves by
    K G (G in score units): the model's own velocity map is
        v_x = ((1-eps) x - m_x) / s_x,   v_h = M(tau) (a m_h - a^2 h),
    so dv_x = -(k_x / s_x) G_x and dv_h = M(tau) a k_h G_h."""
    k_c, k_f = block_k(tau)
    return (-(k_c / s_x_of(tau)) * G_c,
            feat_multiplier(tau) * a_of(tau) * k_f * G_f)


def tfg_block_geometry(tau, tau1):
    """TFG's step geometry for EACH block's own path, from tau to tau1.

    TFG (Ye et al. 2024) is written for a VP state. A block whose path is
    x = s(tau) x_0 + n(tau) z is that VP state rescaled by c = sqrt(s^2 + n^2)
    (abar = s^2 / c^2), so TFG's two displacements map back as
        variance step  rho G_vp / sqrt(alpha)  ->  rho (c s' / s) G_vp
        mean step      sqrt(abar') D0          ->  s' D0
    with G_vp = c grad_x L (the gradient in the VP-normalised state) and '
    denoting tau1. Coordinates: s = 1 - tau, n = s_x(tau) -- the flow mapping
    sampling.FlowSampler uses on our own model. Types: s = a(tau),
    n = sqrt(1 - a^2), so c = 1 and this is exactly VPSampler's mapping.
    Returns {"x": (c, k_var, k_0), "h": (c, k_var, k_0)}.
    """
    out = {}
    for blk, (s0, n0, s1) in {
            "x": (1.0 - tau, s_x_of(tau), 1.0 - tau1),
            "h": (a_of(tau), math.sqrt(max(1.0 - a_of(tau) ** 2, 0.0)), a_of(tau1))}.items():
        c = math.sqrt(s0 * s0 + n0 * n0)
        out[blk] = (c, (c * s1 / s0) if s0 > 0 else 0.0, s1)
    return out


def equifm_posterior(gen, coords, types, charge, mask, tau):
    """Posterior over the GUIDED state (coords, 5 one-hot channels), with the
    charge channel held at its current value. k is 1 on purpose: the per-block
    K rides on the forward-mode tangents (see _ScaleTangents)."""
    q_x, q_h = gen.raw(coords, torch.cat([types, charge], dim=-1), mask, tau)
    a = a_of(tau)
    m3 = mask.unsqueeze(-1)
    m_c = zero_com(((1.0 - EPS) * coords - s_x_of(tau) * q_x) * m3, mask)
    m_f = (a * types + q_h[..., :N_TYPES] / a) * m3
    k_c, k_f = block_k(tau)
    m_c, m_f = _ScaleTangents.apply(m_c, m_f, k_c, k_f)
    return Posterior(m_c, m_f, torch.ones(coords.shape[0], device=coords.device,
                                          dtype=coords.dtype))


class EquiFMSampler(_Base):
    """EquiFM's native Euler sampler (tau: 1 -> 0) with guidance.

    Guidance is skipped while tau > tau_max_guide (0.5 mirrors the main sweep's
    t_min_guide = 0.5). The clip is the project's velocity-relative trust
    region, applied to the guided channels (coordinates + one-hot).
    """

    def __init__(self, net, mask, tau_max_guide=0.5, **kw):
        super().__init__(net, mask, **kw)
        self.tau_max_guide = tau_max_guide

    def time_grid(self, n_steps, span=None):
        a, b = (1.0, 0.0) if span is None else span
        return torch.linspace(a, b, n_steps + 1)

    # ---- TFG (Ye et al. 2024), full update, ported to EquiFM's two clocks.
    #
    # WHAT IS PER BLOCK AND WHAT IS NOT. The step GEOMETRY is per block and
    # exact for each block's path (tfg_block_geometry): coordinates get TFG's
    # flow mapping, atom types its VP mapping. The gradient TFG's rescale_grad
    # sees is the VP-state gradient per block, obtained by handing
    # tfg_components the coordinates in their VP-normalised form (x / c_x),
    # so autograd produces c_x grad_x L for coordinates and grad_h L for
    # types (c_h = 1). The SCHEDULES (rho_i, mu_i normalised over the grid,
    # and the smoothing std) run on ONE clock, the coordinate path's, because
    # tfg_components draws one std and takes one mu_step: the coordinate clock
    # is the linear-path clock TFG runs on in the main sweep, so its per-step
    # schedule on EquiFM matches how it runs on our own model. Stated choice.

    def _tfg_abar(self, ts):
        t = ts.to(torch.float64)
        sig = 1.0 - t
        n = EPS + (1.0 - EPS) * t
        return sig ** 2 / (sig ** 2 + n ** 2)

    def _tfg_step(self, coords, types, t_scalar, post_fn, w, base_c, base_f):
        if not self._primary_stage:
            raise NotImplementedError("tfg is defined per Euler step; use solver='euler'")
        ts = getattr(self, "_grid_ts", None)
        i = getattr(self, "_grid_i", None)
        if ts is None or i is None or float(ts[i]) != float(t_scalar):
            raise RuntimeError("tfg needs the time grid: run it through "
                               "sampling.integrate, which records it")
        tau, tau1 = float(t_scalar), float(ts[i + 1])
        h = tau1 - tau                                     # < 0: tau runs 1 -> 0
        rho_i, mu_i, std_i, _, _, _ = self._tfg_schedule(i)
        geo = tfg_block_geometry(tau, tau1)
        c_x, kv_x, k0_x = geo["x"]
        _, kv_h, k0_h = geo["h"]
        cfg = self.tfg
        a_var = float(w) * rho_i
        post_vp = lambda cv, f: post_fn(cv * c_x, f)       # noqa: E731
        G_c, G_f, D_c, D_f, dg = tfg_components(
            self.f_net, post_vp, coords / c_x, types, self.mask, self.y,
            cfg["mad"], std_i, float(w) * mu_i, n_iter=cfg["n_iter"],
            eps_bsz=cfg["eps_bsz"], want_var=(a_var != 0.0), var_scale=1.0,
            clip_scale=cfg["clip_scale"], generator=self.probe_gen,
            cost=self.cost)
        m3 = self.mask.unsqueeze(-1)
        dv_c, dv_f = a_var * kv_x * G_c, a_var * kv_h * G_f   # displacements
        d0_c, d0_f = k0_x * D_c, k0_h * D_f
        C_c = zero_com((dv_c + d0_c) / h * m3, self.mask)
        C_f = (dv_f + d0_f) / h * m3

        def nrm(a, b):
            return torch.sqrt((a ** 2).sum((1, 2)) + (b ** 2).sum((1, 2)))
        nv, n0 = nrm(dv_c, dv_f), nrm(d0_c, d0_f)
        both = nv + n0
        dg["tfg_d0_frac"] = torch.where(both > 0, n0 / both.clamp(min=1e-300),
                                        torch.zeros_like(both))
        dg["tfg_corr_over_v"] = nrm(C_c, C_f) / nrm(base_c, base_f).clamp(min=1e-12)
        dg.update({"tfg_rho_i": rho_i, "tfg_mu_i": mu_i, "tfg_std_i": std_i})
        self.last_diag = dg
        self._accumulate_diag(dg)
        return C_c, C_f

    def field(self, coords, feats, tau_scalar):
        self.n_field += 1
        tau = float(tau_scalar)
        m3 = self.mask.unsqueeze(-1)
        with torch.no_grad():
            v_c, v_f = self.net.velocity(coords, feats, self.mask, tau)
        self.cost.gen_fwd += 1
        v_c = zero_com(v_c * m3, self.mask)
        v_f = v_f * m3
        mode, w = self.active(tau)
        if (self.f_net is None or mode is None or tau > self.tau_max_guide
                or tau <= 0.0):
            return v_c, v_f
        self.n_guided += 1
        self.note_guided(mode)
        types, charge = feats[..., :N_TYPES], feats[..., N_TYPES:].detach()
        post_fn = lambda c, f: equifm_posterior(  # noqa: E731
            self.net, c, f, charge, self.mask, tau)
        vt = v_f[..., :N_TYPES]
        if mode == "tfg":
            # TFG builds its own velocity (displacements over h); the shared
            # clip below still applies, exactly as on the other samplers
            dv_c, dv_f = self._tfg_step(coords, types, tau, post_fn, w, v_c, vt)
        else:
            tb = torch.full((coords.shape[0],), tau, device=coords.device)
            G_c, G_f = self._guide(coords, types, tb, post_fn, mode, t_scalar=tau)
            # the native velocity map at the guided mean m + K w G, minus at m
            dv_c, dv_f = velocity_delta(tau, w * G_c, w * G_f)
        if self.clip is not None:
            gn = torch.sqrt((dv_c ** 2).sum((1, 2)) + (dv_f ** 2).sum((1, 2)))
            vn = torch.sqrt((v_c ** 2).sum((1, 2)) + (vt ** 2).sum((1, 2)))
            self.n_clipped += int((gn > self.clip * vn).sum().item())
            dv_c, dv_f = _clip_to_velocity(dv_c, dv_f, v_c, vt, self.mask,
                                           self.clip)
        v_f = torch.cat([vt + dv_f * m3, v_f[..., N_TYPES:]], dim=-1)
        return zero_com((v_c + dv_c) * m3, self.mask), v_f

    @torch.no_grad()
    def terminal(self, coords, feats, tau_scalar):
        # the native sampler integrates to tau = 0 and stops; no extra step
        return coords, feats


# --------------------------------------------------------------------------
# a second, independently trained oracle
# --------------------------------------------------------------------------

OC_PROP = ROOT / "audit" / "equifm_20260922" / "OC-Flow" / "molecule" / "qm9" / "property_prediction"


class OCFlowOracle(TFGOracle):
    """OC-Flow's clean property EGNN (same architecture as EDM's classifier,
    DIFFERENT weights -- checked tensor by tensor in the usability audit).

    Trained on the FIRST QM9 half, like TFG's `evaluate_<p>`, so it is a
    second oracle, not a second half: it tells whether a guided gain is a
    quirk of one network rather than a property of the molecules. It is
    disjoint from the guide (TFG `tf_predict_<p>`, second half) exactly as
    `evaluate_<p>` is. Takes raw one-hot, like TFGOracle.
    """

    def __init__(self, prop: str, device="cpu"):
        torch.nn.Module.__init__(self)
        folder = OC_PROP / "outputs" / ("exp_class_%s" % prop)
        a = metadata(folder / "args.pickle")
        self.args, self.prop = a, prop
        gcl = definitions(OC_PROP / "models" / "gcl.py",
                          {"E_GCL", "unsorted_segment_sum", "unsorted_segment_mean"})
        d = definitions(OC_PROP / "models_property.py", {"E_GCL_mask", "EGNN"}, gcl)
        self.egnn = d["EGNN"](in_node_nf=5, in_edge_nf=0, hidden_nf=a["nf"],
                              n_layers=a["n_layers"], attention=a["attention"],
                              node_attr=a["node_attr"])
        self.egnn.load_state_dict(torch.load(folder / "best_checkpoint.npy",
                                             map_location="cpu", weights_only=True))
        self.egnn.to(device).eval().requires_grad_(False)
        if a["node_attr"]:
            raise ValueError("exp_class_%s has node_attr=1" % prop)
