"""Reanalyse consumed chemistry pilots; no tuning, generation or source mutation.

The independent chemistry rule reconstructs neutral bonds from XYZ with RDKit.
Its disagreement with the table is a diagnostic, not a chemical ground truth.
All decisions in the opportunity audit use f_A; f_B only scores outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("H", "C", "N", "O", "F")


def independent_chemistry(pm):
    RDLogger.DisableLog("rdApp.*")
    valid, connected = [], []
    for c, t, m in zip(pm["coords"], pm["types"], pm["mask"]):
        ix = m.bool()
        c, t = c[ix], t[ix]
        lines = [str(len(c)), ""]
        lines += [f"{TYPES[int(a)]} {float(x):.6f} {float(y):.6f} {float(z):.6f}"
                  for a, (x, y, z) in zip(t, c)]
        ok, conn = False, False
        try:
            mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            Chem.SanitizeMol(mol)
            ok = True
            conn = len(Chem.GetMolFrags(mol)) == 1
        except Exception:
            pass
        valid.append(ok)
        connected.append(conn)
    return torch.tensor(valid), torch.tensor(connected)


def paired(new, base):
    d = new.double() - base.double()
    return {"rescued": int((new & ~base).sum()),
            "lost": int((base & ~new).sum()),
            "retained": int((new & base).sum()),
            "net_count": int(d.sum()), "difference": float(d.mean()),
            "paired_se": float(d.std(unbiased=True) / len(d)**.5)}


def state(hit, chem):
    # 0 neither, 1 chemistry only, 2 hit only, 3 both.
    return hit.long()*2 + chem.long()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w", type=int, nargs="+", default=[16])
    ap.add_argument("--out", type=Path, default=ROOT/"results/btvg_joint_audit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT/"results/pilot_chem/seed20261001_n2048"
    rows, contrasts = [], []
    cache = {}
    for prop in ("mu", "alpha", "gap"):
        for w in args.w:
            for arm in ("lgd_mc", "lgd_mc_chem", "lgd_mc_chemn"):
                stem = f"{prop}__{arm}__dist__w{w}__tmin0.5__tgt"
                path = source/f"{stem}.permol.pt"
                pm = torch.load(path, map_location="cpu", weights_only=False)
                meta = json.loads((source/f"{stem}.json").read_text())
                delta = meta["delta"]
                ahit = ((pm["f_A_dec"]-pm["y"]).abs() <= delta) & pm["finite"].bool()
                hit = ((pm["f_B_dec"]-pm["y"]).abs() <= delta) & pm["finite"].bool()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                cp = args.out/f"{stem}.chem.pt"
                if cp.exists():
                    saved = torch.load(cp, map_location="cpu", weights_only=False)
                    if saved["source_sha256"] != digest:
                        raise RuntimeError(f"Source changed: {path}")
                    valid, conn = saved["valid"], saved["connected"]
                else:
                    valid, conn = independent_chemistry(pm)
                    torch.save({"source_sha256": digest, "valid": valid, "connected": conn}, cp)
                chemistry = {"table_stable": pm["mol_stable"].bool(),
                             "table_rdkit_valid": pm["valid"].bool(),
                             "independent_valid": valid, "independent_connected_valid": conn}
                allowed = torch.tensor([1, 4, 3, 2, 1])
                odd = ((allowed[pm["types"].long()] * pm["mask"]).sum(1).long() % 2) == 1
                both_fail = ~chemistry["table_stable"] & ~valid
                assert not (odd & chemistry["table_stable"]).any()
                row = {"property": prop, "arm": arm, "w": w,
                       "n": len(hit), "source_sha256": digest,
                       "decoded_A_hits": int(ahit.sum()), "decoded_B_hits": int(hit.sum()),
                       "decoded_AB_hits": int((ahit & hit).sum()), "chemistry": {}}
                row["composition"] = {
                    "odd_allowed_valence_sum": int(odd.sum()),
                    "fail_both_rules": int(both_fail.sum()),
                    "fail_both_rules_and_odd": int((both_fail & odd).sum()),
                    "B_hit_fail_both": int((hit & both_fail).sum()),
                    "B_hit_fail_both_and_odd": int((hit & both_fail & odd).sum()),
                    "A_hit_fail_both": int((ahit & both_fail).sum()),
                    "AB_hit_fail_both_even": int((ahit & hit & both_fail & ~odd).sum()),
                    "AB_hit_fail_both_odd": int((ahit & hit & both_fail & odd).sum())}
                for name, chem in chemistry.items():
                    counts = torch.bincount(state(hit, chem), minlength=4).tolist()
                    selected = ahit & ~chem
                    row["chemistry"][name] = {
                        "quadrants_neither_chem_only_hit_only_both": counts,
                        "A_hit_chem_failure": int(selected.sum()),
                        "AB_hit_chem_failure": int((selected & hit).sum()),
                        "A_hit_chem_failure_B_precision": float(hit[selected].double().mean()) if selected.any() else None}
                row["B_hit_table_fail_but_independent_valid"] = int((hit & ~chemistry["table_stable"] & valid).sum())
                rows.append(row)
                cache[(prop, w, arm)] = (pm, ahit, hit, chemistry)
                print(f"{stem}: H={int(hit.sum())}, table U={int((hit & chemistry['table_stable']).sum())}, independent U={int((hit & valid).sum())}", flush=True)
            base, ba, bh, bc = cache[(prop, w, "lgd_mc")]
            for arm in ("lgd_mc_chem", "lgd_mc_chemn"):
                pm, a, h, cc = cache[(prop, w, arm)]
                assert torch.equal(base["mol_idx"], pm["mol_idx"])
                assert torch.equal(base["y"], pm["y"])
                comp = {"property": prop, "w": w, "arm": arm,
                        "hits": paired(h, bh), "chemistry": {}}
                for name, c in cc.items():
                    b = bc[name]
                    transition = torch.bincount(4*state(bh, b)+state(h, c), minlength=16).reshape(4,4)
                    comp["chemistry"][name] = {
                        "useful": paired(h & c, bh & b), "chemistry": paired(c, b),
                        "transition_rows_base_cols_new": transition.tolist(),
                        "new_chem_passes": int((~b & c).sum()),
                        "new_chem_passes_that_hit": int((~b & c & h).sum()),
                        "repair_existing_hit": int((bh & ~b & h & c).sum()),
                        "lose_existing_useful": int((bh & b & ~(h & c)).sum())}
                contrasts.append(comp)
    output = {"note": "Exploratory consumed test[:2048], one seed; w16 chosen for mechanism audit, no strength selection or new independent evidence.",
              "quadrant_order": ["neither", "chemistry_only", "hit_only", "both"],
              "independent_rule": "RDKit DetermineBonds(charge=0), sanitize; connectedness reported separately. Not a physical chemistry assay.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows, "paired_vs_lgd_mc": contrasts}
    (args.out/"joint_failures.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
