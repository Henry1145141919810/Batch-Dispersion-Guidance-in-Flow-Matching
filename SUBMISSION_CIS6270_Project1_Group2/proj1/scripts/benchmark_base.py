"""Unconditional-generation benchmark for a base model, EDM protocol.

    python proj1/scripts/benchmark_base.py --ckpt betty_pull/fm.pt --n 10000 \
        --steps 1000 --solver euler --seed 0 --out results/bench/fm_nfe1000_euler_s0.json

Reproduces the metric conventions of Hoogeboom et al. (EDM, 2022) so the
numbers can sit in the same table as published ones:

  atom stability   fraction of atoms whose inferred valency (from distances,
                   Hoogeboom bond tables, WITH hydrogens) matches the element
  mol stability    fraction of molecules in which EVERY atom is stable
  validity         RDKit sanitises the inferred graph; EDM takes the LARGEST
                   CONNECTED FRAGMENT of each sample, so a disconnected sample
                   counts as valid through its biggest piece. Both that and the
                   whole-graph figure are recorded; the EDM convention is what
                   goes in the comparison table.
  uniqueness       distinct canonical SMILES among the valid ones
  novelty          fraction of unique valid SMILES absent from the training set,
                   where training-set SMILES are inferred from 3D by the SAME
                   bond-inference code (that is how EDM's retrieve_qm9_smiles
                   does it, so the comparison is like for like). Reported
                   against train_a (what this model saw) and train_a+train_b.
  connected        fraction of valid samples that are a single fragment. Not an
                   EDM metric; recorded because validity alone hides
                   disconnected output.

Molecule sizes are drawn from the train_a size distribution, as EDM draws n
from the training distribution. The EMA weights are used, as in EDM.

Every per-molecule record (SMILES, stable-atom count, fragment flag) is written
to the JSON so an independent check can recompute every aggregate.
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
from models.egnn import EGNNVelocity  # noqa: E402
from evaluation import stability, to_smiles  # noqa: E402
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
DATASET_SMILES = os.path.join(ROOT, "results", "bench", "dataset_smiles.json")


def largest_fragment_smiles(bonds, syms):
    """EDM's convention: sanitise, then keep the largest connected fragment."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    m = Chem.RWMol()
    for s in syms:
        m.AddAtom(Chem.Atom(s))
    for (i, j), o in bonds.items():
        m.AddBond(i, j, order[o])
    try:
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, default=mol, key=lambda x: x.GetNumAtoms())
        Chem.SanitizeMol(largest)
        return Chem.MolToSmiles(largest)
    except Exception:
        return None


def dataset_smiles(d, types, splits):
    """Inferred-from-3D canonical SMILES for the named splits, cached."""
    if os.path.exists(DATASET_SMILES):
        with open(DATASET_SMILES) as fh:
            cache = json.load(fh)
        if all(s in cache for s in splits):
            return {s: set(cache[s]) for s in splits}
    cache = {}
    for s in splits:
        idx = d["split"][s]
        out = []
        t0 = time.time()
        for i in range(0, len(idx), 2000):
            sel = idx[i:i + 2000]
            st = stability(d["coords"][sel].float(), d["feats"][sel].float(),
                           d["mask"][sel].float(), types)
            out += [largest_fragment_smiles(r[3], r[4]) for r in st]
        cache[s] = [x for x in out if x is not None]
        print("  dataset SMILES %-8s %6d molecules, %6d valid, %.0fs"
              % (s, len(idx), len(cache[s]), time.time() - t0))
    os.makedirs(os.path.dirname(DATASET_SMILES), exist_ok=True)
    with open(DATASET_SMILES, "w") as fh:
        json.dump(cache, fh)
    return {s: set(cache[s]) for s in splits}

def score_samples(C, F, M, finite, types, ref, ref_ab):
    """All EDM-convention aggregates from raw samples. One code path for the
    live benchmark and for re-scoring saved samples after an evaluator fix."""
    st = stability(C.float(), F.float(), M.float(), types)
    recs = []
    for r, ok in zip(st, finite.tolist()):
        smi_whole = to_smiles(r[3], r[4]) if ok else None
        smi_edm = largest_fragment_smiles(r[3], r[4]) if ok else None
        recs.append({"n_atoms": r[1], "n_stable": r[0] if ok else 0,
                     "mol_stable": bool(r[2]) and ok, "finite": ok,
                     "smiles_whole": smi_whole, "smiles_edm": smi_edm,
                     "connected": (smi_whole is not None and "." not in smi_whole)})
    n = len(recs)
    n_at = sum(r["n_atoms"] for r in recs)
    n_sa = sum(r["n_stable"] for r in recs)
    n_sm = sum(r["mol_stable"] for r in recs)
    valid_edm = [r["smiles_edm"] for r in recs if r["smiles_edm"] is not None]
    valid_whole = [r["smiles_whole"] for r in recs if r["smiles_whole"] is not None]
    uniq_edm = set(valid_edm)
    return {
        "n": n, "n_nonfinite": int((~finite).sum()),
        "atom_stability": n_sa / n_at,
        "mol_stability": n_sm / n,
        "validity_edm": len(valid_edm) / n,
        "validity_whole": len(valid_whole) / n,
        "uniqueness_edm": len(uniq_edm) / max(len(valid_edm), 1),
        "valid_x_unique_edm": len(uniq_edm) / n,
        "novelty_vs_train_a": sum(1 for x in uniq_edm if x not in ref["train_a"]) / max(len(uniq_edm), 1),
        "novelty_vs_train_ab": sum(1 for x in uniq_edm if x not in ref_ab) / max(len(uniq_edm), 1),
        "connected_of_valid": sum(1 for r in recs if r["connected"]) / max(len(valid_whole), 1),
        "connected_of_all": sum(1 for r in recs if r["connected"]) / n,
        "mean_atoms": n_at / n,
        "records": recs,
    }


def rescore(samples_path, out_path):
    """Re-evaluate a saved *_samples.pt with the CURRENT evaluator; keeps the
    run metadata from the original JSON beside it."""
    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    ref = dataset_smiles(d, types, ["train_a", "train_b"])
    ref_ab = ref["train_a"] | ref["train_b"]
    s = torch.load(samples_path, weights_only=False)
    F1 = torch.nn.functional.one_hot(s["types"].long(), len(types)).float()
    res = score_samples(s["coords"].float(), F1, s["mask"].float(), s["finite"].bool(),
                        s["type_names"], ref, ref_ab)
    orig = samples_path.replace("_samples.pt", ".json")
    if os.path.exists(orig):
        with open(orig) as fh:
            meta = json.load(fh)
        for k in ("ckpt", "ckpt_sha", "family", "epoch", "weights", "steps", "solver", "nfe",
                  "seed", "t_sample_s"):
            if k in meta:
                res[k] = meta[k]
    res["rescored_from"] = os.path.basename(samples_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    print("  %-40s atom %.4f  mol %.4f  valid %.4f  VxU %.4f  novel(a) %.4f  conn %.4f"
          % (os.path.basename(out_path), res["atom_stability"], res["mol_stability"],
             res["validity_edm"], res["valid_x_unique_edm"], res["novelty_vs_train_a"],
             res["connected_of_valid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    ap.add_argument("--out", required=True, help="output JSON (or directory with --rescore)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rescore", nargs="*", default=None,
                    help="re-score these *_samples.pt files with the current evaluator "
                         "(no sampling); --out is then the output DIRECTORY")
    args = ap.parse_args()

    if args.rescore is not None:
        for sp in args.rescore:
            rescore(sp, os.path.join(args.out, os.path.basename(sp).replace("_samples.pt", ".json")))
        return 0

    dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    is_last = "model" in ck and "state_dict" not in ck
    sd = ck["ema" if (is_last and not args.raw) else
            ("model" if is_last else ("ema_state_dict" if not args.raw else "state_dict"))]
    cargs = ck["args"]
    family = ck.get("family") or cargs.get("family")
    if family not in ("flow", "vp_diffusion"):
        raise SystemExit("checkpoint has no usable 'family' key; refusing to guess")
    tau_min = ck.get("tau_min", 1e-3)

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    net = EGNNVelocity(len(types), cargs["hidden"], cargs["layers"]).to(dev).eval()
    net.load_state_dict(sd)
    for p in net.parameters():
        p.requires_grad_(False)

    ref = dataset_smiles(d, types, ["train_a", "train_b"])
    ref_ab = ref["train_a"] | ref["train_b"]

    # molecule sizes from the TRAINING distribution (EDM convention)
    g = torch.Generator().manual_seed(args.seed)
    tr = d["split"]["train_a"]
    pick = tr[torch.randint(0, len(tr), (args.n,), generator=g)]
    mask_all = d["mask"][pick].to(dev)

    gdev = torch.Generator(device=dev).manual_seed(args.seed)
    print("%s | %s | n=%d steps=%d solver=%s seed=%d weights=%s"
          % (os.path.basename(args.ckpt), family, args.n, args.steps, args.solver,
             args.seed, "raw" if args.raw else "EMA"))
    t0 = time.time()
    C, F, nfe = [], [], None
    for i in range(0, args.n, args.batch):
        m = mask_all[i:i + args.batch]
        c0, f0 = initial_noise(m, len(types), gdev)
        smp = (VPSampler(net, m, tau_min=tau_min) if family == "vp_diffusion"
               else FlowSampler(net, m))
        c, f, info = integrate(smp, c0, f0, args.steps, args.solver)
        nfe = getattr(smp, "n_field", None)
        C.append(c.cpu()); F.append(f.cpu())
        if i == 0:
            print("  first batch %.1fs -> projected %.0fs total"
                  % (time.time() - t0, (time.time() - t0) * args.n / args.batch))
    C, F = torch.cat(C), torch.cat(F)
    M = mask_all.cpu()
    t_sample = time.time() - t0

    finite = torch.isfinite(C).all((1, 2)) & torch.isfinite(F).all((1, 2))
    C = torch.where(finite.view(-1, 1, 1), C, torch.zeros_like(C))
    F = torch.where(finite.view(-1, 1, 1), F, torch.zeros_like(F))

    t1 = time.time()
    res = score_samples(C, F, M, finite, types, ref, ref_ab)
    t_eval = time.time() - t1
    res.update({
        "ckpt": args.ckpt, "ckpt_sha": hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest()[:12],
        "family": family, "epoch": ck.get("epoch"), "weights": "raw" if args.raw else "EMA",
        "steps": args.steps, "solver": args.solver, "nfe": nfe, "seed": args.seed,
        "t_sample_s": t_sample, "t_eval_s": t_eval,
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    # Raw samples beside the JSON, so any other metric convention (heavy-atom
    # only, lenient valence, bond-length histograms) can be computed later
    # without re-sampling.
    torch.save({"coords": C.float(), "types": F.argmax(-1).to(torch.int8),
                "mask": M.bool(), "finite": finite, "type_names": types,
                "seed": args.seed, "steps": args.steps, "solver": args.solver},
               args.out.replace(".json", "_samples.pt"))
    print("  atom %.4f  mol %.4f  valid(EDM) %.4f  uniq %.4f  VxU %.4f  "
          "novel(a) %.4f  novel(ab) %.4f  connected %.4f  nonfinite %d  "
          "[sample %.0fs eval %.0fs]"
          % (res["atom_stability"], res["mol_stability"], res["validity_edm"],
             res["uniqueness_edm"], res["valid_x_unique_edm"], res["novelty_vs_train_a"],
             res["novelty_vs_train_ab"], res["connected_of_valid"], res["n_nonfinite"],
             t_sample, t_eval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
