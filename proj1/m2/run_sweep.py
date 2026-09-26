"""Modality 2 sweep driver. Emits every cell and runs them in sequence.

Cells are independent and m2_sweep.py skips any whose output JSON already
exists, so this is resumable: re-run it after an interruption and it picks up
where it stopped. Pass --dry to print the plan without running.

All arms read the SAME frozen checkpoint, the same sampler at the same NFE, the
same seeds and the same metrics -- only the steering rule changes. That is what
makes the comparison controlled, and it is why no baseline is retrained.
"""
import argparse, itertools, os, subprocess, sys, time

ARMS_EXT = ["plug", "tmpd", "lgd_mc", "tfg_mc"]   # DPS, TMPD/PiGDM, LGD, TFG-ingredient
W_GRID = [1, 4, 16, 64]
# eta = feedback gain, t = tau/s setpoint, trailing 'o' = one-sided (contract only)
BDG_VARIANTS = ["e0t1",                                  # MUST equal plug exactly
                "e1t0.5", "e1t1", "e1t1.5",
                "e4t0.5", "e4t1", "e4t1.5",
                "e4t1.5o"]                               # widening setpoint, one-sided
BDG_W = [16]                                             # the strength that worked
PROPS = ["gc", "cpg"]
TARGETS = ["q50", "q90"]
SEEDS = [20260921, 20260922, 20260923]


def cells():
    for prop, tgt, seed in itertools.product(PROPS, TARGETS, SEEDS):
        yield dict(prop=prop, arm="unguided", variant="-", w=1, target=tgt, seed=seed)
        for arm, w in itertools.product(ARMS_EXT, W_GRID):
            yield dict(prop=prop, arm=arm, variant="-", w=w, target=tgt, seed=seed)
        for v, w in itertools.product(BDG_VARIANTS, BDG_W):
            yield dict(prop=prop, arm="bdg", variant=v, w=w, target=tgt, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="proj1/m2/blade_bundle/fm_m2_dfb500.pt")
    ap.add_argument("--out-dir", default="results/m2_dfb")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)      # NFE
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    plan = list(cells())
    print(f"{len(plan)} cells -> {a.out_dir}   (n={a.n}, NFE={a.steps})")
    if a.dry:
        for c in plan[:12]:
            print("  ", c)
        print(f"   ... and {len(plan)-12} more")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    t0, done, failed = time.time(), 0, []
    for i, c in enumerate(plan, 1):
        cmd = [a.python, "proj1/m2/m2_sweep.py", "--ckpt", a.ckpt,
               "--prop", c["prop"], "--arm", c["arm"], "--variant", c["variant"],
               "--w", str(c["w"]), "--target", c["target"], "--seed", str(c["seed"]),
               "--n", str(a.n), "--steps", str(a.steps), "--device", a.device,
               "--out-dir", a.out_dir]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((c, r.stderr.strip().splitlines()[-1:] or ["?"]))
            print(f"[{i}/{len(plan)}] FAIL {c['prop']} {c['arm']} {c['variant']} w{c['w']}")
        else:
            done += 1
        if i % 10 == 0 or i == len(plan):
            el = (time.time() - t0) / 60
            print(f"[{i}/{len(plan)}] {el:.1f} min elapsed, "
                  f"~{el/i*(len(plan)-i):.1f} min left, {len(failed)} failed",
                  flush=True)
    print(f"\ndone: {done}/{len(plan)}  failed: {len(failed)}")
    for c, err in failed[:10]:
        print("  ", c, err)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
