"""Fill the PLACEHOLDER_* slots in BASE_MODEL_BENCHMARK.md from results/bench/*.json."""
import glob, json, os, io, statistics as st
R = "results/bench/rescored"
def load(p):
    with open(p) as fh:
        d = json.load(fh); d.pop("records", None); return d
def row(d, label):
    return "| %s | %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
        label, d["nfe"] or d["steps"], d["n"], d["atom_stability"], d["mol_stability"], d["validity_edm"],
        d["uniqueness_edm"], d["valid_x_unique_edm"], d["novelty_vs_train_a"], d["novelty_vs_train_ab"],
        d["connected_of_valid"])
HDR = ("| run | NFE | n | atom stab | mol stab | validity | uniqueness | valid x unique | novelty (train_a) | novelty (train_a+b) | connected |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|")
def pm(vals): 
    return "%.4f +- %.4f" % (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

out = {}
# main: 3 seeds NFE100
seeds = [load(p) for p in sorted(glob.glob(R + "/fm_nfe100_euler_s[0-9].json"))]
lines = [HDR] + [row(d, "seed %d" % d["seed"]) for d in seeds]
if len(seeds) > 1:
    keys = ["atom_stability", "mol_stability", "validity_edm", "uniqueness_edm", "valid_x_unique_edm",
            "novelty_vs_train_a", "novelty_vs_train_ab", "connected_of_valid"]
    lines.append("| **mean +- std (%d seeds)** | 100 | %d | %s |" % (len(seeds), seeds[0]["n"],
                 " | ".join(pm([d[k] for d in seeds]) for k in keys)))
lines.append("")
lines.append("Standard error on one 10,000-sample run: atom stability ~0.0006, molecule stability ~0.005, "
             "validity ~0.004. The seed-to-seed spread above is the honest uncertainty.")
out["PLACEHOLDER_MAIN_TABLE"] = "\n".join(lines)

# NFE / solver table
nfe = []
for p in ("fm_nfe100_euler_s0", "fm_nfe100_heun_s0", "fm_nfe250_euler_s0", "fm_nfe500_euler_s0", "fm_nfe1000_euler_s0"):
    f = R + "/%s.json" % p
    if os.path.exists(f):
        d = load(f); nfe.append(row(d, "%s, %d steps" % (d["solver"], d["steps"])))
out["PLACEHOLDER_NFE_TABLE"] = "\n".join([HDR] + nfe) if nfe else "(sweep incomplete)"

# epoch curve
ep = []
for p in sorted(glob.glob(R + "/ep[0-9]*_nfe100_s7.json")):
    d = load(p); ep.append("| %d | %.4f | %.4f | %.4f | %.4f |" % (d["epoch"], d["atom_stability"], d["mol_stability"], d["validity_edm"], d["connected_of_valid"]))
out["PLACEHOLDER_EPOCH_TABLE"] = "\n".join(["| epoch | atom stab | mol stab | validity | connected |", "|---|---|---|---|---|"] + ep) if ep else "(sweep incomplete)"

# selection checks
sel = []
a = R + "/fm_nfe100_euler_s0.json"; b = R + "/fmlast_ep1500_nfe100_euler_s0.json"; c = R + "/fm_raw_nfe100_euler_s0.json"
if os.path.exists(a) and os.path.exists(b):
    da, db = load(a), load(b)
    sel.append("**Selected epoch 1300 vs final epoch 1500** (10,000 samples each, seed 0, NFE 100, EMA weights):\n")
    sel.append("| checkpoint | atom stab | mol stab | validity |\n|---|---|---|---|")
    sel.append("| epoch 1300 (selected, `fm.pt`) | %.4f | %.4f | %.4f |" % (da["atom_stability"], da["mol_stability"], da["validity_edm"]))
    sel.append("| epoch 1500 (`fm_last.pt`) | %.4f | %.4f | %.4f |" % (db["atom_stability"], db["mol_stability"], db["validity_edm"]))
    da_, db_ = da["atom_stability"], db["atom_stability"]
    sel.append("\n**The selection rule picked the wrong checkpoint, by a small but real margin.** The trainer "
               "scored epoch 1300 at 0.9390 and epoch 1500 at 0.9365 on its fixed 512-sample draw -- a "
               "difference of one standard error at that sample size -- and kept 1300. On 10,000 fresh "
               "samples with the same seed, epoch 1500 is better by %+.4f atom, %+.4f molecule and %+.4f "
               "validity (paired comparison; several SE). This is the winner's curse acting on the choice, "
               "not on the number. The curve in 3.3 was still rising at 1500, so the last epoch was also the "
               "principled a-priori pick. **Recommendation: ship the epoch-1500 EMA weights (`fm_last.pt`, "
               "key `ema`) as the base model, and for the diffusion run raise `--stab-n` to 2048 so the "
               "selection signal's SE halves.**\n"
               % (db_ - da_, db["mol_stability"] - da["mol_stability"], db["validity_edm"] - da["validity_edm"]))
if os.path.exists(a) and os.path.exists(c):
    dc = load(c)
    sel.append("**EMA vs raw weights** (epoch 1300, NFE 100, seed 0):\n")
    sel.append("| weights | n | atom stab | mol stab | validity |\n|---|---|---|---|---|")
    sel.append("| EMA (what we ship) | %d | %.4f | %.4f | %.4f |" % (da["n"], da["atom_stability"], da["mol_stability"], da["validity_edm"]))
    sel.append("| raw | %d | %.4f | %.4f | %.4f |" % (dc["n"], dc["atom_stability"], dc["mol_stability"], dc["validity_edm"]))
out["PLACEHOLDER_SELECTION"] = "\n".join(sel) if sel else "(sweep incomplete)"

p = "BASE_MODEL_BENCHMARK.md"
s = io.open(p, encoding="utf-8").read()
for k, v in out.items():
    s = s.replace(k, v)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("filled:", ", ".join(k for k in out if out[k] != "(sweep incomplete)"))
print("pending:", ", ".join(k for k in out if out[k] == "(sweep incomplete)") or "none")
