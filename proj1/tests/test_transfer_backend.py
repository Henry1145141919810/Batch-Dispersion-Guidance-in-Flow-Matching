"""Gates for the borrowed backend: EDM's schedule, TFG's guide and oracle.

Same discipline as the other gate files -- every check is a quantity with a
known answer, so a wrong factor moves a number rather than degrading a result
quietly. That matters more here than anywhere else in the project, because
every failure mode of an adapter is silent: feed a network features on the
wrong scale and it still returns a finite number, guidance still runs, and the
experiment still produces a table.

WHAT EACH CHECK IS PROTECTING

  schedule_vp_identity   alpha^2 + sigma^2 = 1 at every tau. If EDM's gamma
                         were read with the wrong sign this fails immediately.
  schedule_beta_matches  d log alpha / d tau = -beta / 2, by finite difference
                         against the closed form in edm_schedule.py. This is
                         the one derivation in the transfer that is ours, so
                         it is the one most worth checking numerically.
  schedule_grid_exact    at grid points tau = k/T the interpolated gamma
                         equals EDM's own lookup exactly. Interpolation is
                         allowed to smooth BETWEEN training grid points; it is
                         not allowed to move them.
  schedule_monotone      sigma increases and alpha decreases in tau. A
                         non-monotone schedule integrates backwards somewhere.

  calib_slope_is_mad     the fitted calibration slope recovers QM9's
                         mean-absolute-deviation to within 3%. The fit never
                         sees QM9_MAD, so this is an independent check that
                         the adapter drives these networks on the right scale.
                         It is the gate that catches a missing or doubled
                         one-hot division -- the single most likely defect.
  calib_intercept_is_mean  and likewise the intercept against the mean.
  guide_mae / oracle_mae   both predictors are actually accurate. A network
                         loaded with mismatched weights still calibrates to
                         SOMETHING; it does not calibrate to a small MAE.

  invariance_*           rotation, translation and permutation, for guide and
                         oracle. These networks are invariant by construction,
                         so a failure means the adapter is building the edge
                         list or the mask wrongly.
  padding_output         padded atoms do not change the prediction.
  padding_gradient       padded atoms receive exactly zero gradient. If they
                         did not, guidance would edit atoms that do not exist
                         and the edit would land on real ones after masking.

  hvp_finite             a Hessian-vector product through the guide is finite.
                         `smg`, `smg2` and `btvg` all need the guide to be
                         twice differentiable in coordinates AND features; a
                         guide that is not cannot be used for those arms at
                         all, and finding that out mid-sweep costs a queue
                         window.

  feats_contract         Calibrated(f_A) on sampler-space features equals the
                         raw guide on the same features, and Calibrated(f_B)
                         on sampler-space features equals the raw oracle on
                         PHYSICAL features. This is the contract the driver
                         relies on to score everything in one space, and
                         getting it backwards is a factor of 64 that produces
                         plausible-looking numbers.

The generator checks are skipped with a printed notice when EDMsecond has not
been downloaded, so this file is runnable before the fetch. They are not
optional in substance -- `--require-edm` turns a skip into a failure, which is
what the cluster job uses.

Run: python proj1/tests/test_transfer_backend.py [--require-edm]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from external.edm_schedule import EDMSchedule                  # noqa: E402
from external.tfg_assets import (Calibrated, EDMGenerator, PROP_INDEX,  # noqa: E402
                                 QM9_MAD, QM9_MEAN, TFGGuide, TFGOracle,
                                 fit_calibration)

TFG_ROOT = os.path.join(ROOT, "audit", "fa_fb_search", "TFG")
EDM_DIR = os.path.join(ROOT, "weights", "EDMsecond")
DATA = os.path.join(ROOT, "data", "qm9.pt")

torch.set_default_dtype(torch.float64)
R = {}          # name -> (error, tolerance)


def gate(name, err, tol):
    R[name] = (float(err), float(tol))


# --------------------------------------------------------------------------

def check_schedule():
    sch = EDMSchedule(dtype=torch.float64)
    tau = torch.linspace(0.002, 0.998, 97, dtype=torch.float64)
    a, s = sch.alpha_sigma(tau)
    gate("schedule_vp_identity", (a ** 2 + s ** 2 - 1.0).abs().max(), 1e-12)

    # d log alpha / d tau == -beta / 2.
    #
    # Evaluated at CELL MIDPOINTS. gamma is piecewise linear on EDM's 1/T grid,
    # so beta -- its slope times sigma^2 -- is piecewise constant and genuinely
    # discontinuous at every grid point. A symmetric difference centred on a
    # grid point averages the slopes of two different cells and disagrees with
    # either by O(gamma''/T), which is 4% at tau = 0.002 where the clipped
    # schedule is sharpest. That is a property of the difference quotient, not
    # of the schedule: the same check one part in a thousand away from a
    # boundary agrees to 1e-10. The sampler only ever evaluates points, never
    # differences, so the interior identity is the one that governs it.
    mid = (torch.arange(1, sch.T, 11, dtype=torch.float64) + 0.5) / sch.T
    h = 1e-7
    a_hi, _ = sch.alpha_sigma(mid + h)
    a_lo, _ = sch.alpha_sigma(mid - h)
    num = (a_hi.log() - a_lo.log()) / (2 * h)
    ref = -0.5 * sch.beta(mid)
    # Relative, because beta spans four orders of magnitude over [0, 1] and an
    # absolute tolerance would be vacuous at small tau and impossible at large.
    gate("schedule_beta_matches", ((num - ref).abs() / ref.abs()).max(), 1e-6)

    # exact on EDM's own grid points
    k = torch.arange(1, sch.T, 97, dtype=torch.float64)
    tg = k / sch.T
    gate("schedule_grid_exact",
         (sch.gamma_at(tg) - sch.gamma[k.long()]).abs().max(), 1e-12)

    gate("schedule_monotone",
         float((s[1:] <= s[:-1]).any() or (a[1:] >= a[:-1]).any()), 0.5)


def _rand_batch(d, n_mol=6, seed=11):
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(d["coords"].shape[0], generator=g)[:n_mol]
    return (d["coords"][sel].double(), d["feats"][sel].double(),
            d["mask"][sel].double(), sel)


def check_property_nets(d):
    g = torch.Generator().manual_seed(7)
    sel = torch.randperm(d["coords"].shape[0], generator=g)[:800]
    c, f, m = (d["coords"][sel].double(), d["feats"][sel].double(),
               d["mask"][sel].double())

    for prop in ("mu", "alpha", "gap"):
        truth = d["y"][sel, PROP_INDEX[prop]].double()
        guide = TFGGuide(TFG_ROOT, prop).double()
        oracle = TFGOracle(TFG_ROOT, prop).double()
        fs = guide.norm_values[1]

        ag, bg, mae_g = fit_calibration(
            lambda C, F, M: guide(C, F / fs, M), c, f, m, truth)
        ao, bo, mae_o = fit_calibration(oracle, c, f, m, truth)

        mad, mean = QM9_MAD[prop], QM9_MEAN[prop]
        gate("calib_slope_is_mad__%s_guide" % prop, abs(ag / mad - 1.0), 0.03)
        gate("calib_slope_is_mad__%s_oracle" % prop, abs(ao / mad - 1.0), 0.03)
        gate("calib_intercept_is_mean__%s_guide" % prop,
             abs(bg - mean) / mad, 0.05)
        gate("calib_intercept_is_mean__%s_oracle" % prop,
             abs(bo - mean) / mad, 0.05)
        # Accuracy, expressed in units of the property's own spread so the
        # three properties share one tolerance.
        gate("guide_mae_small__%s" % prop, mae_g / mad, 0.20)
        gate("oracle_mae_small__%s" % prop, mae_o / mad, 0.20)

        # ---- the sampler-space contract
        cc, ff, mm, _ = _rand_batch(d)
        f_A = Calibrated(guide, ag, bg, prop, 1.0, feats_are_normalised=True)
        f_B = Calibrated(oracle, ao, bo, prop, 1.0,
                         feats_are_normalised=False, feat_scale=fs)
        ff_n = ff / fs                       # what the sampler carries
        with torch.no_grad():
            gate("feats_contract__%s_guide" % prop,
                 (f_A(cc, ff_n, mm) - (guide(cc, ff_n, mm) * ag + bg)).abs().max(),
                 1e-10)
            gate("feats_contract__%s_oracle" % prop,
                 (f_B(cc, ff_n, mm) - (oracle(cc, ff, mm) * ao + bo)).abs().max(),
                 1e-10)

        if prop == "mu":                     # the expensive checks, once
            _check_symmetry("guide", guide, cc, ff_n, mm)
            _check_symmetry("oracle", oracle, cc, ff, mm)
            _check_hvp(guide, cc, ff_n, mm)
            _check_embed("guide", f_A, cc, ff_n, mm)
            _check_embed("oracle", f_B, cc, ff_n, mm)


def _check_symmetry(tag, net, c, f, m):
    with torch.no_grad():
        base = net(c, f, m)
        scale = base.abs().mean().clamp(min=1e-6)

        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=c.dtype,
                                           generator=torch.Generator().manual_seed(3)))
        q = q * torch.sign(torch.det(q))
        gate("invariance_rotation__%s" % tag,
             (net(c @ q.T, f, m) - base).abs().max() / scale, 1e-9)

        shift = torch.tensor([0.3, -1.2, 0.7], dtype=c.dtype)
        gate("invariance_translation__%s" % tag,
             (net((c + shift) * m[..., None], f, m) - base).abs().max() / scale,
             1e-9)

        # Permute only WITHIN the real atoms of each molecule, so the mask is
        # unchanged and the test is about the network, not about padding.
        cp, fp = c.clone(), f.clone()
        for b in range(c.shape[0]):
            n = int(m[b].sum())
            p = torch.randperm(n, generator=torch.Generator().manual_seed(b + 5))
            cp[b, :n], fp[b, :n] = c[b, :n][p], f[b, :n][p]
        gate("invariance_permutation__%s" % tag,
             (net(cp, fp, m) - base).abs().max() / scale, 1e-9)

        # Padding: rubbish in the padded slots must not move the output, and
        # must not receive gradient.
        cj, fj = c.clone(), f.clone()
        pad = (m == 0)
        cj[pad] = 17.0
        fj[pad] = -3.0
        gate("padding_output__%s" % tag,
             (net(cj, fj, m) - base).abs().max() / scale, 1e-9)

    cg = c.clone().requires_grad_(True)
    fg = f.clone().requires_grad_(True)
    out = net(cg, fg, m).sum()
    gc, gf = torch.autograd.grad(out, (cg, fg))
    pad = (m == 0)
    gate("padding_gradient__%s" % tag,
         max(float(gc[pad].abs().max()), float(gf[pad].abs().max())), 1e-12)


def _check_hvp(net, c, f, m):
    """A Hessian-vector product through the guide must be finite and nonzero.

    `smg`, `smg2` and `btvg` differentiate the guide twice. "Finite" alone is
    not enough -- a guide whose second derivative is identically zero would
    make the SMG correction a no-op and every SMG cell would silently equal
    the corresponding plug cell.
    """
    cg = c.clone().requires_grad_(True)
    fg = f.clone().requires_grad_(True)
    out = net(cg, fg, m).sum()
    gc, gf = torch.autograd.grad(out, (cg, fg), create_graph=True)
    vc = torch.randn(c.shape, dtype=c.dtype,
                     generator=torch.Generator().manual_seed(21)) * m[..., None]
    vf = torch.randn(f.shape, dtype=f.dtype,
                     generator=torch.Generator().manual_seed(22)) * m[..., None]
    hc, hf = torch.autograd.grad((gc * vc).sum() + (gf * vf).sum(), (cg, fg))
    finite = torch.isfinite(hc).all() and torch.isfinite(hf).all()
    gate("hvp_finite", 0.0 if finite else 1.0, 0.5)
    norm = float(hc.norm() + hf.norm())
    gate("hvp_nonzero", 0.0 if norm > 1e-8 else 1.0, 0.5)


def _check_embed(tag, cal, c, f, m):
    """`evaluation.embedding_diversity` calls f_B.net.embed. It must exist,
    be finite, be invariant, and -- the part worth gating -- must actually
    vary between molecules. An embed that returned a constant would report
    perfect diversity collapse for every arm equally, which looks like a
    finding."""
    with torch.no_grad():
        h = cal.net.embed(c, cal._feats(f), m)
    gate("embed_finite__%s" % tag, 0.0 if torch.isfinite(h).all() else 1.0, 0.5)
    spread = float((h - h.mean(0, keepdim=True)).abs().mean())
    gate("embed_varies__%s" % tag, 0.0 if spread > 1e-6 else 1.0, 0.5)


def check_generator(d, require):
    have = all(os.path.exists(os.path.join(EDM_DIR, n))
               for n in ("generative_model_ema.npy", "args.pickle"))
    if not have:
        if require:
            gate("edm_present", 1.0, 0.5)
            return
        print("\n  NOTE: EDMsecond not downloaded -- generator gates skipped.")
        print("        python proj1/scripts/fetch_tfg_assets.py\n")
        return
    gate("edm_present", 0.0, 0.5)
    net = EDMGenerator(os.path.join(EDM_DIR, "generative_model_ema.npy"),
                       os.path.join(EDM_DIR, "args.pickle")).double()
    c, f, m, _ = _rand_batch(d)
    cn, fn = net.normalise(c, f)
    tau = torch.full((c.shape[0],), 0.4, dtype=c.dtype)
    with torch.no_grad():
        ec, ef = net(cn, fn, m, tau)
    gate("edm_eps_shape",
         0.0 if (ec.shape == c.shape and ef.shape == f.shape) else 1.0, 0.5)
    gate("edm_eps_finite",
         0.0 if (torch.isfinite(ec).all() and torch.isfinite(ef).all()) else 1.0,
         0.5)
    # EDM's dynamics removes the mean from the coordinate output. Our sampler
    # assumes that -- `zero_com` is applied to the field, not to the state --
    # so if it stopped being true the trajectory would drift off the zero-CoM
    # subspace the model was trained on.
    com = (ec * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    gate("edm_eps_zero_com", com.abs().max(), 1e-8)
    # Padded atoms must produce no epsilon at all.
    pad = (m == 0)
    gate("edm_eps_padding",
         max(float(ec[pad].abs().max()), float(ef[pad].abs().max())), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-edm", action="store_true",
                    help="fail rather than skip when EDMsecond is absent")
    a = ap.parse_args()

    if not os.path.exists(DATA):
        print("data/qm9.pt is missing -- run proj1/scripts/prepare_qm9.py")
        return 2
    d = torch.load(DATA, weights_only=True)

    check_schedule()
    check_property_nets(d)
    check_generator(d, a.require_edm)

    print("%-44s %13s %11s   pass" % ("check", "error", "tolerance"))
    print("-" * 78)
    ok = True
    for k, (v, tol) in R.items():
        p = v < tol
        ok = ok and p
        print("%-44s %13.3e %11.1e   %s" % (k, v, tol, "yes" if p else "NO"))
    print("-" * 78)
    print("ALL PASS (%d gates)" % len(R) if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
