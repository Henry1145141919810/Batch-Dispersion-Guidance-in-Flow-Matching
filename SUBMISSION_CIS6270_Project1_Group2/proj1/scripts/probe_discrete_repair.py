"""Exploratory one-atom repair control on consumed LGD-MC pilot endpoints.

Protocol: docs/methods/BTVG_EVOLUTION_AFTER_CHEM_GUARD.md. No production arm is
changed. f_B is loaded only after both candidate selections have been frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"proj1/src"))
sys.path.insert(0, str(ROOT/"proj1/scripts"))
from checkpoint_paths import require_predictor
from evaluation import ALLOWED_VALENCE, BONDS1, BONDS2, BONDS3, MARGIN1, MARGIN2, MARGIN3
from guidance import soft_valence_violation
from m1_signed_bias import PhysicalProperty
from audit_joint_failures import paired

TYPES = ("H", "C", "N", "O", "F")
VAL = torch.tensor([ALLOWED_VALENCE[x] for x in TYPES])


def exact_stable(c, t, m):
    """Vectorized copy of the nested evaluator rule, checked against sidecars."""
    d = torch.linalg.vector_norm(c[:, :, None]-c[:, None, :], dim=-1)*100
    inside = []
    for table, margin in ((BONDS1, MARGIN1), (BONDS2, MARGIN2), (BONDS3, MARGIN3)):
        th = torch.tensor([[table.get(a, {}).get(b, -10000)+margin for b in TYPES] for a in TYPES])
        inside.append(d < th[t[:, :, None], t[:, None, :]])
    bonds = inside[0].long()*(1+inside[1].long()*(1+inside[2].long()))
    mask = m.bool()
    bonds *= (mask[:, :, None] & mask[:, None, :] & ~torch.eye(t.shape[1], dtype=torch.bool)[None])
    return ((bonds.sum(-1) == VAL[t]) | ~mask).all(-1)


def independent_molecule(c, t, m):
    ix = m.bool()
    c, t = c[ix], t[ix]
    lines = [str(len(c)), ""]
    lines += [f"{TYPES[int(a)]} {float(x):.6f} {float(y):.6f} {float(z):.6f}" for a, (x,y,z) in zip(t,c)]
    try:
        mol = Chem.MolFromXYZBlock("\n".join(lines)+"\n")
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


@torch.no_grad()
def predict(net, c, t, m, dev, batch=256):
    pieces = []
    for i in range(0, len(c), batch):
        h = F.one_hot(t[i:i+batch].long(), 5).float()*m[i:i+batch, :, None]
        pieces.append(net(c[i:i+batch].to(dev), h.to(dev), m[i:i+batch].to(dev)).cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def main():
    torch.set_num_threads(4)
    RDLogger.DisableLog("rdApp.*")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    source = ROOT/"results/pilot_chem/seed20261001_n2048"
    out = ROOT/"results/btvg_discrete_probe"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Validate the necessary parity condition on the real dataset as well.
    data = torch.load(ROOT/"data/qm9.pt", map_location="cpu", weights_only=False)
    odd_data = ((VAL[data["feats"].argmax(-1)]*data["mask"]).sum(-1).long()%2).sum().item()
    data_info = {"n": len(data["mask"]), "odd_allowed_valence_sum": int(odd_data)}
    del data
    for prop in ("mu", "alpha", "gap"):
        stem = f"{prop}__lgd_mc__dist__w16__tmin0.5__tgt"
        path = source/f"{stem}.permol.pt"
        pm = torch.load(path, map_location="cpu", weights_only=False)
        meta = json.loads((source/f"{stem}.json").read_text())
        cached = torch.load(ROOT/"results/btvg_joint_audit"/f"{stem}.chem.pt", weights_only=False)
        assert cached["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        c, t, m, y = pm["coords"], pm["types"].long(), pm["mask"].float(), pm["y"]
        stable, valid = pm["mol_stable"].bool(), cached["valid"]
        assert torch.equal(exact_stable(c,t,m), stable)
        eligible = ~stable & ~valid
        start = time.perf_counter()
        parent, ct = [], []
        for b in eligible.nonzero().flatten().tolist():
            odd = int((VAL[t[b]]*m[b]).sum()) % 2
            for atom in m[b].bool().nonzero().flatten().tolist():
                old = int(t[b, atom])
                for new in range(5):
                    if new == old or (odd + int(VAL[new]-VAL[old])) % 2:
                        continue
                    proposal = t[b].clone()
                    proposal[atom] = new
                    parent.append(b)
                    ct.append(proposal)
        parent, ct = torch.tensor(parent), torch.stack(ct)
        enumerated = len(parent)
        table_ok = torch.cat([exact_stable(c[parent[i:i+256]], ct[i:i+256], m[parent[i:i+256]]) for i in range(0,len(parent),256)])
        parent, ct = parent[table_ok], ct[table_ok]
        table_passed = len(parent)
        accepted, candidate_smiles = [], []
        for k, b in enumerate(parent.tolist()):
            mol = independent_molecule(c[b],ct[k],m[b])
            ok = mol is not None and len(Chem.GetMolFrags(mol)) == 1
            accepted.append(ok)
            candidate_smiles.append(Chem.MolToSmiles(mol) if ok else None)
        accepted = torch.tensor(accepted, dtype=torch.bool)
        candidate_smiles = [s for s, ok in zip(candidate_smiles, accepted.tolist()) if ok]
        parent, ct = parent[accepted], ct[accepted]
        # Smooth chemical fit chooses a target-blind control; all candidates
        # already pass both hard chemistry checks and connectedness.
        with torch.no_grad():
            penalties = []
            for i in range(0,len(parent),256):
                ix = parent[i:i+256]
                h = F.one_hot(ct[i:i+256],5).float()*m[ix,:,None]
                penalties.append(soft_valence_violation(c[ix].to(dev),h.to(dev),m[ix].to(dev)).cpu())
            penalty = torch.cat(penalties) if penalties else torch.empty(0)
        a_path = require_predictor(f"f_A_{prop}.pt")
        fa = PhysicalProperty(a_path,5,dev)
        original_a = predict(fa,c,t,m,dev)
        if not torch.allclose(original_a,pm["f_A_dec"],atol=2e-4,rtol=2e-5):
            raise RuntimeError("Guide checkpoint/decoded convention differs from source")
        ca = predict(fa,c[parent],ct,m[parent],dev)
        selections = {"chemistry_only": torch.full((len(c),),-1,dtype=torch.long),
                      "property_aware": torch.full((len(c),),-1,dtype=torch.long)}
        counts = torch.bincount(parent,minlength=len(c))
        for b in (counts>0).nonzero().flatten().tolist():
            ix = (parent==b).nonzero().flatten()
            selections["chemistry_only"][b] = ix[penalty[ix].argmin()]
            selections["property_aware"][b] = ix[(ca[ix]-y[b]).abs().argmin()]
        if dev == "cuda": torch.cuda.synchronize()
        search_seconds = time.perf_counter()-start
        # Freeze selections on disk before loading the evaluator.
        payload = {"source_sha256": cached["source_sha256"], "parent": parent,
                   "candidate_types": ct, "candidate_f_A": ca, "penalty": penalty,
                   "candidate_smiles": candidate_smiles, "selections": selections}
        torch.save(payload,out/f"{prop}.proposals.pt")
        fb = PhysicalProperty(require_predictor(f"f_B_{prop}.pt"),5,dev)
        cb = predict(fb,c[parent],ct,m[parent],dev)
        original_b = predict(fb,c,t,m,dev)
        assert torch.allclose(original_b,pm["f_B_dec"],atol=2e-4,rtol=2e-5)
        bhit = (pm["f_B_dec"]-y).abs() <= meta["delta"]
        output = {"property":prop,"n":len(c),"eligible":int(eligible.sum()),
                  "enumerated_even_parity_candidates": enumerated,
                  "table_passed_candidates":table_passed,"all_checks_passed_candidates":len(parent),
                  "endpoints_with_candidate":int((counts>0).sum()),
                  "endpoints_with_multiple_candidates":int((counts>1).sum()),
                  "max_candidates":int(counts.max()),"search_seconds":search_seconds,
                  "f_A_candidate_evaluations":len(parent),
                  "note":"Runtime includes guide load/reproduction check; excludes independent baseline audit and final evaluator scoring. No end-to-end throughput claim.",
                  "source_sha256":cached["source_sha256"],
                  "guide_sha256":hashlib.sha256(Path(a_path).read_bytes()).hexdigest(),"arms":{}}
        final_hits={}
        for arm,sel in selections.items():
            changed=sel>=0
            assert not (changed & (stable|valid)).any()
            score=pm["f_B_dec"].clone();score[changed]=cb[sel[changed]]
            hit=(score-y).abs()<=meta["delta"]
            ast=stable|changed;av=valid|changed
            conn=cached["connected"]|changed
            final_hits[arm]=hit & ast
            row={"changed":int(changed.sum()),"A_hits_among_repairs":int(((ca[sel[changed]]-y[changed]).abs()<=meta['delta']).sum()),
                 "B_hits_among_repairs":int((hit&changed).sum()),
                 "table_useful":paired(hit&ast,bhit&stable),
                 "independent_useful":paired(hit&av,bhit&valid),
                 "independent_connected_useful":paired(hit&conn,bhit&cached["connected"]),
                 "property_hits":paired(hit,bhit)}
            assert row["table_useful"]["lost"]==row["independent_useful"]["lost"]==0
            successful=[candidate_smiles[int(k)] for k in sel[changed&hit]]
            row["distinct_successful_repair_smiles"] = len(set(successful))
            output["arms"][arm]=row
        output["target_choice_vs_chemistry_choice"] = paired(final_hits["property_aware"], final_hits["chemistry_only"])
        torch.save({"candidate_f_B":cb,"original_f_B":original_b},out/f"{prop}.evaluation.pt")
        rows.append(output)
        print(json.dumps(output),flush=True)
    result={"protocol":"docs/methods/BTVG_EVOLUTION_AFTER_CHEM_GUARD.md: locked one-edit diagnostic",
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "real_data_parity":data_info,"rows":rows,
            "interpretation":"Exploratory postprocessing control on consumed test targets, not a new validated method. No f_B proposal selection."}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
