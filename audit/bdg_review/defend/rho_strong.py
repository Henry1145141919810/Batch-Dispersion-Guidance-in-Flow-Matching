"""Paired correlation of per-molecule mol_stable between unguided and guided arms at
strong strength (v1 dist full run, seed 20261001, n=5000; same noise across arms)."""
import glob, os, torch
R = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/results/full/n5000/seed20261001"
for prop in ("mu", "alpha", "gap"):
    U = torch.load(glob.glob(os.path.join(R, f"{prop}__unguided__*.permol.pt"))[0], weights_only=False)
    for p in sorted(glob.glob(os.path.join(R, f"{prop}__*.permol.pt"))):
        if "__unguided__" in p: continue
        A = torch.load(p, weights_only=False)
        assert torch.equal(A["mol_idx"], U["mol_idx"])
        u, a = U["mol_stable"].float(), A["mol_stable"].float()
        r = torch.corrcoef(torch.stack([u, a]))[0, 1].item()
        b = os.path.basename(p).split("__")
        print(f"{prop:5s} {b[1]:9s} w{b[3][1:]:5s} rho {r:+.3f}  unguided {u.mean():.4f}  guided {a.mean():.4f}")
