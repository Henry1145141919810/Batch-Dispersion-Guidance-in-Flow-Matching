"""Read-only BTVG audit: saved pilot outcomes, counterexamples, optional GPU geometry.

This does not change guidance or select a new arm on test data. The molecular
check uses an unguided trajectory with validation-set sizes/targets; it tests
local geometry, not comparative generation performance. Output is exploratory.
Run: python proj1/scripts/audit_btvg_mechanism.py --geometry --n 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "proj1" / "src"))
sys.path.insert(0, str(ROOT / "proj1" / "scripts"))


def dot(a, b):
    return sum((x * y).flatten(1).sum(1) for x, y in zip(a, b))


def cosine(a, b):
    return dot(a, b) / (dot(a, a) * dot(b, b)).sqrt().clamp_min(1e-30)


def exact_checks():
    # m-space perpendicularity does not survive the pullback J^T.
    g = torch.tensor([[1., 1.]], dtype=torch.float64)
    h = torch.tensor([[1., 2.]], dtype=torch.float64)
    J = torch.diag(torch.tensor([3., 1.], dtype=torch.float64))
    Sm = -(h - (h * g).sum() / (g * g).sum() * g)
    Sx, ax, bx = Sm @ J, g @ J, h @ J
    fixed = -(bx - (ax * bx).sum() / (ax * ax).sum() * ax)
    assert abs(float((g * Sm).sum())) < 1e-12
    assert float((bx * Sx).sum()) > 0  # actual variance INCREASE
    assert abs(float((ax * fixed).sum())) < 1e-12
    assert float((bx * fixed).sum()) < 0

    def cdf(z):
        return 0.5 * math.erfc(-z / math.sqrt(2))

    def band(e, sd, delta=1.):
        return cdf((delta-e)/sd) - cdf((-delta-e)/sd)

    p_wide, p_narrow = band(3., 1.), band(3., 0.5)
    assert p_wide > 100 * p_narrow
    # A shared norm cap steals baseline progress, even if the extra direction
    # is orthogonal in the correct state space before clipping.
    baseline = torch.tensor([1., 0.], dtype=torch.float64)
    added = torch.tensor([0., 1.], dtype=torch.float64)
    clipped = (baseline+added)/torch.linalg.vector_norm(baseline+added)
    assert float(baseline @ clipped) < float(baseline @ baseline)
    # Matching a mean or reducing variance alone is insufficient.
    return {
        "pullback_counterexample": {
            "J": J.tolist(), "mean_gradient_m": g.tolist(),
            "variance_gradient_m": h.tolist(), "correction_m": Sm.tolist(),
            "mean_derivative_m": float((g*Sm).sum()),
            "mean_derivative_after_pullback": float((ax*Sx).sum()),
            "variance_derivative_after_pullback": float((bx*Sx).sum()),
            "corrected_mean_derivative": float((ax*fixed).sum()),
            "corrected_variance_derivative": float((bx*fixed).sum()),
        },
        "band_counterexample": {
            "residual_in_delta": 3., "wide_sd_in_delta": 1.,
            "narrow_sd_in_delta": .5, "wide_coverage": p_wide,
            "narrow_coverage": p_narrow, "coverage_ratio": p_wide/p_narrow,
        },
        "gaussian_sample_variance_relative_sd_K4": math.sqrt(2/3),
        "clip_counterexample": {"baseline_progress": 1.,
                                "progress_after_orthogonal_addition_and_clip": float(baseline @ clipped)},
    }


def pilot_audit():
    directory = ROOT / "results" / "pilot_btvg2" / "seed20261001_n2048"
    rows, comparisons = [], []
    cache = {}
    for path in sorted(directory.glob("*.json")):
        cell = json.loads(path.read_text())
        pm = torch.load(path.with_suffix(".permol.pt"), map_location="cpu", weights_only=False)
        finite = pm["finite"].bool()
        hit = (pm["f_B"] - pm["y"]).abs().le(cell["delta"]) & finite
        ahit = (pm["f_A"] - pm["y"]).abs().le(cell["delta"]) & finite
        useful = hit & pm["valid"].bool()
        key = (cell["prop"], cell["arm"], cell["w"])
        cache[key] = (pm, hit, useful)
        n = hit.numel()
        distinct = {s for s, ok in zip(pm["smiles"], useful.tolist()) if ok and s}
        row = {
            "property": key[0], "arm": key[1], "w": key[2], "n": n,
            "coverage": float(hit.double().mean()),
            "valid_and_in_band": float(useful.double().mean()),
            "stable_and_in_band": float((hit & pm["mol_stable"].bool()).double().mean()),
            "distinct_valid_in_band_per_attempt": len(distinct)/n,
            "guide_hit_precision": float((hit & ahit).sum())/max(int(ahit.sum()), 1),
            "guide_false_hit_per_attempt": float((ahit & ~hit).double().mean()),
            "mae_delta": cell["prop_mae_eval"]/cell["delta"],
            "molecule_stability": cell["mol_stability"],
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No pilot cells in {directory}")
    for key, (pm, hit, useful) in cache.items():
        prop, arm, w = key
        if arm == "lgd_mc":
            continue
        base, bhit, buseful = cache[(prop, "lgd_mc", w)]
        assert torch.equal(pm["mol_idx"], base["mol_idx"])
        assert torch.equal(pm["y"], base["y"])
        def paired(a, b):
            d = a.double()-b.double()
            se = float(d.std(unbiased=True)/math.sqrt(len(d)))
            return {"difference": float(d.mean()), "paired_se": se,
                    "rescued": int((a & ~b).sum()), "lost": int((b & ~a).sum()),
                    "retained": int((a & b).sum())}
        comparisons.append({"property": prop, "arm": arm, "w": w,
                            "coverage": paired(hit, bhit),
                            "valid_and_in_band": paired(useful, buseful)})
    return {"rows": rows, "paired_vs_lgd_mc": comparisons,
            "note": "Exploratory reanalysis of existing overlapping test[:2048]; no new independent evidence. f_B received continuous features in the original evaluator."}


def molecular_geometry(n, seed, props):
    from checkpoint_paths import default_generator, require_predictor
    from m1_signed_bias import PhysicalProperty, load_fm
    from guidance import (_pullback, btvg2_weighted_grad, fm_posterior,
                          sigma_mc_weighted_grad, trace_scale)
    from models.egnn import zero_com
    from sampling import initial_noise, _clip_to_velocity

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(ROOT/"data"/"qm9.pt", map_location="cpu", weights_only=False)
    if not 0 < n <= len(data["split"]["val"]):
        raise ValueError("n must fit the validation split")
    idx = data["split"]["val"][:n]
    mask = data["mask"][idx].to(dev)
    path = default_generator()
    net, _ = load_fm(path, len(data["types"]), dev)
    guides = {p: PhysicalProperty(require_predictor(f"f_A_{p}.pt"), len(data["types"]), dev) for p in props}
    tolerances = {}
    for p in props:
        ck = torch.load(require_predictor(f"f_B_{p}.pt"), map_location="cpu", weights_only=False)
        tolerances[p] = 2*float(ck["val_mae"])

    def projected(v):
        return zero_com(v[0], mask), v[1]*mask.unsqueeze(-1)

    def gen():
        return torch.Generator(device=dev).manual_seed(seed+1)

    c, f = initial_noise(mask, len(data["types"]), torch.Generator(device=dev).manual_seed(seed))
    rows = []
    for step in range(91):
        t = torch.full((n,), step/100, device=dev)
        with torch.no_grad():
            vc, vf = net(c, f, mask, t)
        if step in (50, 65, 80, 90):
            post_fn = lambda cc, ff: fm_posterior(net, cc, ff, mask, t)
            with torch.no_grad():
                post = post_fn(c, f)
            mc, mf, k = post.mean_coords, post.mean_feats, post.k
            for p, guide in guides.items():
                y = data["y"][idx, data["props"].index(p)].to(dev)
                delta, s = tolerances[p], guide.y_std
                Lc, Lf, _ = sigma_mc_weighted_grad(guide, post_fn, c, f, mask, mc, mf, k, y, s, 4, gen())
                Bc, Bf, dg = btvg2_weighted_grad(guide, post_fn, c, f, mask, mc, mf, k, y, s, delta/1.96, 4, gen())
                # Independent replay of the exact same detached-radius draws.
                rng = gen()
                r = trace_scale(post_fn, c, f, mask, k, 1, rng).sqrt().view(-1, 1, 1)
                values, grads = [], []
                for _ in range(4):
                    zc = torch.randn(mc.shape, generator=rng, device=dev)*mask.unsqueeze(-1)
                    zf = torch.randn(mf.shape, generator=rng, device=dev)*mask.unsqueeze(-1)
                    cc = (mc+r*zc).detach().requires_grad_(True)
                    ff = (mf+r*zf).detach().requires_grad_(True)
                    value = guide(cc, ff, mask)
                    gc, gf = torch.autograd.grad(value.sum(), (cc, ff))
                    values.append(value.detach()); grads.append((gc.detach(), gf.detach()))
                F = torch.stack(values)
                GC, GF = (torch.stack([v[i] for v in grads]) for i in range(2))
                dF = (F-F.mean(0)).view(4, n, 1, 1)
                gm = (GC.mean(0), GF.mean(0))
                hm = (2*(dF*GC).sum(0)/3, 2*(dF*GF).sum(0)/3)
                sm = (Bc-Lc, Bf-Lf)
                a = projected(_pullback(post_fn, c, f, *gm))
                b = projected(_pullback(post_fn, c, f, *hm))
                sx = projected(_pullback(post_fn, c, f, *sm))
                active = dot(sx, sx).sqrt() > 1e-7
                def active_median(values):
                    return float(values[active].median()) if bool(active.any()) else None
                frac = (dot(a, b)/dot(a, a).clamp_min(1e-30)).view(-1, 1, 1)
                corrected = tuple(-(bb-frac*aa) for aa, bb in zip(a, b))
                old = projected(_pullback(post_fn, c, f, Lc, Lf))
                both = tuple(v+u for v, u in zip(old, sx))
                mult = 4*(1-step/100)/(step/100)
                oldclip = _clip_to_velocity(*(mult*v for v in old), vc, vf, mask, 1.)
                newclip = _clip_to_velocity(*(mult*v for v in both), vc, vf, mask, 1.)
                increment = tuple(u-v for u, v in zip(newclip, oldclip))
                # All claims concern local, fixed-draw surrogates, not endpoint response.
                row = {"property": p, "t": step/100, "n": n, "active": int(active.sum()),
                       "m_space_mean_abs_cos": active_median(cosine(gm, sm).abs()),
                       "x_space_mean_abs_cos": active_median(cosine(a, sx).abs()),
                       "x_space_variance_increase_count": int(((cosine(b, sx) > 1e-5) & active).sum()),
                       "x_space_variance_increase_fraction": float((cosine(b, sx)[active] > 1e-5).float().mean()) if bool(active.any()) else None,
                       "after_clip_increment_mean_abs_cos": active_median(cosine(a, increment).abs()),
                       "corrected_x_space_mean_abs_cos": float(cosine(a, corrected).abs().max()),
                       "corrected_x_space_variance_max_cos": float(cosine(b, corrected).max()),
                       "median_V_over_tau2": float(dg["btvg2_V_over_tau2"].median())}
                assert row["corrected_x_space_mean_abs_cos"] < 1e-4
                assert row["corrected_x_space_variance_max_cos"] <= 1e-4
                rows.append(row)
                print(json.dumps(row), flush=True)
        c, f = (c+.01*vc).detach(), (f+.01*vf).detach()
    return {"rows": rows, "seed": seed, "split": "val", "mol_idx": idx.tolist(),
            "device": dev, "checkpoint": path,
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "note": "Unguided validation trajectories; active means correction norm > 1e-7. No inference about molecular benefit, no terminal guarantees. Geometry correction is an offline control only."}


def decoding_audit(n, seed, props):
    """Paired scoring of the SAME generated states; not an arm comparison."""
    from checkpoint_paths import default_generator, require_predictor
    from m1_signed_bias import PhysicalProperty, load_fm
    from sampling import FlowSampler, initial_noise, integrate

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(ROOT/"data"/"qm9.pt", map_location="cpu", weights_only=False)
    if not 0 < n <= len(data["split"]["val"]):
        raise ValueError("n must fit the validation split")
    idx = data["split"]["val"][:n]
    mask = data["mask"][idx].to(dev)
    net, _ = load_fm(default_generator(), len(data["types"]), dev)
    rows = []
    for prop in props:
        ga, gb = (require_predictor(f"f_{s}_{prop}.pt") for s in ("A", "B"))
        fa = PhysicalProperty(ga, len(data["types"]), dev)
        fb = PhysicalProperty(gb, len(data["types"]), dev)
        delta = 2*float(torch.load(gb, map_location="cpu", weights_only=False)["val_mae"])
        y = data["y"][idx, data["props"].index(prop)].to(dev)
        c0, f0 = initial_noise(mask, len(data["types"]), torch.Generator(device=dev).manual_seed(seed))
        for arm in ("unguided", "lgd_mc"):
            sampler = FlowSampler(net, mask, f_net=None if arm == "unguided" else fa,
                                  y=y, s=fa.y_std, mode=arm, w=4., n_mc=4,
                                  t_min_guide=.5, clip=1.)
            c, f, _ = integrate(sampler, c0.clone(), f0.clone(), 100, "euler")
            if not bool(torch.isfinite(c).all() & torch.isfinite(f).all()):
                raise RuntimeError("Non-finite endpoint: do not interpret a decoding comparison")
            hard = torch.nn.functional.one_hot(f.argmax(-1), len(data["types"])).to(f.dtype)*mask.unsqueeze(-1)
            with torch.no_grad():
                soft_score, hard_score = fb(c, f, mask), fb(c, hard, mask)
                real_score = fb(data["coords"][idx].to(dev), data["feats"][idx].to(dev), mask)
            shift = (hard_score-soft_score).abs()/delta
            sh, hh = (soft_score-y).abs() <= delta, (hard_score-y).abs() <= delta
            row = {"property": prop, "arm": arm, "n": n,
                   "median_abs_decoding_shift_delta": float(shift.median()),
                   "p90_abs_decoding_shift_delta": float(shift.quantile(.9)),
                   "soft_coverage": float(sh.float().mean()), "hard_coverage": float(hh.float().mean()),
                   "soft_hits": int(sh.sum()), "soft_hits_lost_on_decode": int((sh & ~hh).sum()),
                   "hard_hits_added": int((hh & ~sh).sum()),
                   "soft_mae_delta": float((soft_score-y).abs().mean()/delta),
                   "hard_mae_delta": float((hard_score-y).abs().mean()/delta),
                   "real_molecule_mae_delta": float((real_score-y).abs().mean()/delta)}
            rows.append(row)
            print("DECODING", json.dumps(row), flush=True)
    return {"rows": rows, "seed": seed, "split": "val", "mol_idx": idx.tolist(),
            "note": "Small paired scoring audit, not efficacy evidence; no geometry relaxation or physical oracle. f_B used only after generation."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--out", type=Path, default=ROOT/"results"/"btvg_audit"/"mechanism.json")
    args = ap.parse_args()
    result = {"exact_checks": exact_checks(), "pilot": pilot_audit(),
              "torch_version": torch.__version__,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__).resolve(), ROOT/"proj1"/"src"/"guidance.py",
                                          ROOT/"proj1"/"src"/"sampling.py", ROOT/"proj1"/"src"/"evaluation.py")}}
    if args.geometry:
        result["geometry"] = molecular_geometry(args.n, args.seed, args.props.split(","))
    if args.decode:
        result["decoding"] = decoding_audit(args.n, args.seed, args.props.split(","))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
