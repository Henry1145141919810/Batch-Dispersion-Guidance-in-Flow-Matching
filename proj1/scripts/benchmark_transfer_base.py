"""Unconditional benchmark of a BORROWED base model, through OUR evaluator.

    python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond \
        --n 10000 --steps 100 --grid gamma --seed 0 \
        --out results/bench/edmsecond_nfe100_gamma_s0.json

WHY THIS SCRIPT EXISTS, AND WHY IT IS NOT `benchmark_base.py --ckpt`.
`docs/results/BASE_MODEL_BENCHMARK.md` compares our generator against numbers
READ OUT OF PAPERS. That comparison carries an irreducible caveat: their number
came from their evaluator on their preprocessing of QM9, ours from ours. Six
published rows in that document are quoted from three different papers, and the
document spends a whole section (4.1) showing that one fixed set of molecules
scores 38%, 84% or 93% molecule-stable depending on whose rule is applied.

Running TFG's released EDM through THIS repository's evaluator removes that
caveat for one row. `score_samples` and `dataset_smiles` below are imported
from `benchmark_base.py` -- not reimplemented, imported -- so the borrowed
model and ours go through the same bond tables, the same largest-fragment
validity, the same SMILES canonicalisation, the same everything. A difference
in the output is then a difference in the models.

It is also the hard gate of the transfer experiment
(`docs/protocol/TRANSFER_EXPERIMENT_PLAN.md` §6 step 4): if this does not land
near EDM's published ~98.4 / ~81.7, the adapter or the sampler is wrong, and no
guidance cell should run until that is resolved -- every guided number would
inherit the fault while looking plausible.

WHAT IS AND IS NOT LIKE-FOR-LIKE HERE

  like-for-like    the evaluator, the bond tables, the validity convention,
                   the molecule-size distribution, the sample count, the seed
                   handling, and the JSON schema -- so the rows can sit in one
                   table
  NOT the sampler  EDM samples a 1000-step ancestral SDE. This integrates the
                   probability-flow ODE. That is deliberate -- it is the
                   sampler our guidance fields are defined on, and the point of
                   the transfer experiment is to hold the sampler fixed -- but
                   it means a shortfall against EDM's published row is NOT by
                   itself evidence of an adapter bug. `--steps` and `--grid`
                   exist to separate the two; see "Reading the result" below.
  NOT novelty      `novelty_vs_train_a` is computed against OUR half, which is
                   the half our generator trained on. For a borrowed model
                   whose own training half is unknown, that number measures
                   nothing about the model and must not be quoted. The field is
                   still written, because `score_samples` writes it and
                   suppressing it would make the two JSONs differ in schema;
                   it is flagged in `caveats` inside the output.

READING THE RESULT, in the order to check things:

  1. atom stability >= ~0.97 and molecule stability >= ~0.70 at any setting
     => the adapter is driving the checkpoint correctly. Proceed.
  2. numbers far below that at `--grid uniform` but fine at `--grid gamma`
     => discretisation, not the adapter. EDM's polynomial_2 schedule is stiff
     at the noise end: on a uniform 100-step tau grid the first Euler step
     crosses 3.70 nats of log-SNR (our own linear-beta schedule: 0.20), and a
     gamma-uniform grid flattens that to 0.23 everywhere. Use gamma and say so.
  3. numbers far below at BOTH grids, and not improving with `--steps 1000`
     => the adapter is wrong. Do not proceed to guidance. Suspect, in order:
     the epsilon sign, the time direction, the one-hot scale, the EMA key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, HERE)

from benchmark_base import dataset_smiles, score_samples        # noqa: E402
from external.tfg_assets import EDMGenerator                    # noqa: E402
from sampling import VPSampler, initial_noise, integrate        # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")

# EDM's published unconditional rows, for the gate. Half-data figure is
# EEGSDE Table 5 row 1; full-data is EDM Table 1.
PUBLISHED = {
    "EDMsecond": {"atom": 0.9837, "mol": 0.8174, "source": "EEGSDE Tab. 5"},
    "EDMfull":   {"atom": 0.987,  "mol": 0.820,  "source": "EDM Tab. 1"},
    "EDM":       {"atom": 0.987,  "mol": 0.820,  "source": "EDM Tab. 1"},
}


def size_histogram_gap(d):
    """How different are the two halves' molecule-size distributions?

    Sizes are drawn from `train_a` for BOTH models so the comparison is paired
    -- the borrowed model is conditioned on the same atom counts ours was. That
    is only fair if the halves have the same size distribution, which they do
    by construction (a seeded random partition), but "by construction" is the
    kind of claim that is worth one line of arithmetic. Returns the total
    variation distance; anything under ~0.01 means the choice is immaterial.
    """
    na = d["mask"][d["split"]["train_a"]].sum(1).long()
    nb = d["mask"][d["split"]["train_b"]].sum(1).long()
    m = int(max(na.max(), nb.max())) + 1
    pa = torch.bincount(na, minlength=m).double()
    pb = torch.bincount(nb, minlength=m).double()
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float(0.5 * (pa - pb).abs().sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edm-dir", default=os.path.join(ROOT, "weights", "EDMsecond"),
                    help="directory with generative_model_ema.npy + args.pickle")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--grid", default="gamma", choices=["uniform", "gamma"],
                    help="tau grid. 'gamma' spaces steps evenly in log-SNR, "
                         "which is what EDM's stiff polynomial_2 schedule "
                         "needs at low step counts; 'uniform' reproduces the "
                         "naive grid so the difference can be measured.")
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--sizes-from", default="train_a",
                    choices=["train_a", "train_b", "all"],
                    help="which split's size histogram to draw molecule sizes "
                         "from. train_a pairs this run with our own benchmark.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    wpath = os.path.join(args.edm_dir, "generative_model_ema.npy")
    apath = os.path.join(args.edm_dir, "args.pickle")
    for p in (wpath, apath):
        if not os.path.exists(p):
            raise SystemExit(
                "missing %s\nRun: python proj1/scripts/fetch_tfg_assets.py" % p)

    d = torch.load(DATA, weights_only=True)
    types = d["types"]
    net = EDMGenerator(wpath, apath, n_types=len(types), device=dev)
    tag = os.path.basename(os.path.normpath(args.edm_dir))

    print("%s | EDM nf=%d layers=%d attention=%s | schedule=%s steps=%d"
          % (tag, net.args["nf"], net.args["n_layers"], net.args["attention"],
             net.args["diffusion_noise_schedule"], net.args["diffusion_steps"]))
    print("  n=%d  steps=%d  grid=%s  solver=%s  seed=%d  sizes<-%s  device=%s"
          % (args.n, args.steps, args.grid, args.solver, args.seed,
             args.sizes_from, dev))

    ref = dataset_smiles(d, types, ["train_a", "train_b"])
    ref_ab = ref["train_a"] | ref["train_b"]
    tv = size_histogram_gap(d)
    print("  train_a vs train_b size-histogram total variation: %.4f" % tv)

    # Sizes from the SAME histogram, with the SAME generator and seed as
    # benchmark_base.py, so a run of each at one seed is paired molecule for
    # molecule rather than merely equal in distribution.
    g = torch.Generator().manual_seed(args.seed)
    pool = (torch.arange(d["mask"].shape[0]) if args.sizes_from == "all"
            else d["split"][args.sizes_from])
    pick = pool[torch.randint(0, len(pool), (args.n,), generator=g)]
    mask_all = d["mask"][pick].to(dev)

    gdev = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    C, F, nfe = [], [], None
    for i in range(0, args.n, args.batch):
        m = mask_all[i:i + args.batch]
        # Initial noise is drawn in EDM's NORMALISED space, which is where this
        # sampler runs end to end; the one conversion back is after the loop.
        c0, f0 = initial_noise(m, len(types), gdev)
        smp = VPSampler(net, m, tau_min=args.tau_min,
                        noise_schedule=net.schedule, grid=args.grid)
        c, f, _ = integrate(smp, c0, f0, args.steps, args.solver)
        nfe = smp.n_field
        C.append(c.cpu())
        F.append(f.cpu())
        if i == 0:
            print("  first batch %.1fs -> projected %.0fs total"
                  % (time.time() - t0, (time.time() - t0) * args.n / args.batch))
    C, F = torch.cat(C), torch.cat(F)
    M = mask_all.cpu()
    t_sample = time.time() - t0

    # ONE conversion out of EDM's normalised space. norm_values[0] is 1 (the
    # loader refuses anything else), so coordinates are already angstroms;
    # features are multiplied back so the saved samples and the scored tensors
    # are in the same units as every other row in results/bench/.
    C, F = net.denormalise(C, F)

    finite = torch.isfinite(C).all((1, 2)) & torch.isfinite(F).all((1, 2))
    C = torch.where(finite.view(-1, 1, 1), C, torch.zeros_like(C))
    F = torch.where(finite.view(-1, 1, 1), F, torch.zeros_like(F))

    t1 = time.time()
    res = score_samples(C, F, M, finite, types, ref, ref_ab)
    t_eval = time.time() - t1

    pub = PUBLISHED.get(tag)
    res.update({
        "ckpt": os.path.relpath(wpath, ROOT),
        "ckpt_sha": hashlib.sha256(open(wpath, "rb").read()).hexdigest()[:12],
        "ckpt_md5": hashlib.md5(open(wpath, "rb").read()).hexdigest(),
        "model": tag, "family": "vp_diffusion_external",
        "provenance": "TFG (Ye et al. 2024) release; not trained by us",
        "edm_args": {k: net.args.get(k) for k in
                     ("nf", "n_layers", "attention", "tanh", "inv_sublayers",
                      "diffusion_noise_schedule", "diffusion_steps",
                      "diffusion_noise_precision", "normalize_factors",
                      "include_charges", "remove_h", "dataset", "n_epochs",
                      "batch_size", "ema_decay")},
        "weights": "EMA", "steps": args.steps, "solver": args.solver,
        "grid": args.grid, "nfe": nfe, "seed": args.seed,
        "tau_min": args.tau_min, "sizes_from": args.sizes_from,
        "size_hist_tv_a_vs_b": tv,
        "published_reference": pub,
        "t_sample_s": t_sample, "t_eval_s": t_eval,
        "evaluator": "proj1/scripts/benchmark_base.py::score_samples "
                     "(imported, not reimplemented)",
        "caveats": [
            "Sampler is OUR probability-flow ODE, not EDM's 1000-step "
            "ancestral SDE. A shortfall against the published row is not by "
            "itself an adapter defect.",
            "novelty_vs_train_a is computed against OUR half of QM9. This "
            "model's own training half is unknown, so its novelty numbers "
            "measure nothing about it and must not be quoted.",
            "Molecule sizes are drawn from %s, the same histogram our own "
            "benchmark uses, so the two runs are paired. Total variation "
            "between the halves' size histograms is %.4f." % (args.sizes_from, tv),
            "QM9 preprocessing is ours (133,885 molecules, private random "
            "split), not the published protocol. Same caveat as "
            "BASE_MODEL_BENCHMARK.md Section 2, and it applies to both rows "
            "equally here, which is the point.",
        ],
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    torch.save({"coords": C.float(), "types": F.argmax(-1).to(torch.int8),
                "mask": M.bool(), "finite": finite, "type_names": types,
                "seed": args.seed, "steps": args.steps, "solver": args.solver,
                "grid": args.grid, "model": tag},
               args.out.replace(".json", "_samples.pt"))

    print("  atom %.4f  mol %.4f  valid(EDM) %.4f  uniq %.4f  VxU %.4f  "
          "connected %.4f  nonfinite %d  [sample %.0fs eval %.0fs]"
          % (res["atom_stability"], res["mol_stability"], res["validity_edm"],
             res["uniqueness_edm"], res["valid_x_unique_edm"],
             res["connected_of_valid"], res["n_nonfinite"], t_sample, t_eval))

    if pub:
        da = res["atom_stability"] - pub["atom"]
        dm = res["mol_stability"] - pub["mol"]
        print("  vs %s published (%s): atom %+.4f  mol %+.4f"
              % (tag, pub["source"], da, dm))
        # The gate. Deliberately loose: it is asking "is the adapter driving
        # this checkpoint at all", not "does our ODE reproduce their SDE".
        if res["atom_stability"] < 0.95 or res["mol_stability"] < 0.60:
            print("\n  *** GATE FAILED ***\n"
                  "  This is far below the published row. Do NOT run guidance\n"
                  "  cells on this backend yet -- every guided number would\n"
                  "  inherit the fault while looking plausible. Next checks,\n"
                  "  in order: re-run with --grid %s and --steps 1000; if that\n"
                  "  fixes it the cause is discretisation, not the adapter.\n"
                  "  If not, suspect the epsilon sign, the time direction, the\n"
                  "  one-hot scale, or the EMA key."
                  % ("uniform" if args.grid == "gamma" else "gamma"))
            return 1
        print("  GATE PASSED: the adapter drives this checkpoint correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
