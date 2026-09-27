"""Modality 2 sweep driver, on the v3 protocol.

Plans every cell of one stage and runs them in sequence. Cells are independent
and m2_sweep.py skips any whose output JSON already exists, so this is
resumable: re-run it after an interruption and it picks up where it stopped.
Pass --dry to print the plan without running.

All arms read the SAME frozen checkpoint, the same sampler at the same NFE, the
same seeds, the same batch and the same metrics -- only the steering rule
changes. That is what makes the comparison controlled, and it is why no
baseline is retrained.

THE TWO STAGES, pre-registered in docs/protocol/MODALITY2_V3_PROTOCOL.md:

  m2      headline. 7 arms at ONE strength, target q50.   7 x 2 x 3 =  42 cells
  m2abl   ablation. BDG's 17-arm eta x tau grid at TWO
          strengths, plus the t_min window rung.         204 + 18 = 222 cells

Run:
  python proj1/m2/run_sweep.py --stage m2    --w 16 --n 2000 --batch 500
  python proj1/m2/run_sweep.py --stage m2abl --w 16 --n 2000 --batch 500
"""
import argparse, itertools, os, subprocess, sys, time

# The headline arm set, matching Modality 1's v3 one-for-one. tfg_mc is TFG's
# MC-smoothing INGREDIENT, not the full TFG update that M1's `tfg` arm runs --
# the write-up must preserve that distinction (protocol section 3).
ARMS_EXT = ["plug", "tmpd", "lgd_mc", "tfg_mc"]
HEADLINE_BDG = ["e4t0.5", "e4t1"]

# The ablation grid: eta x tau_mult, exactly M1's ABL_ARMS. eta = 0 kills the
# feedback whatever tau is, so it appears once -- 1 + 4 x 4 = 17 arms.
ABL_BDG = ["e0t1"] + ["e%gt%g" % (e, t)
                      for e in (1, 2, 4, 8)
                      for t in (0.5, 0.75, 1, 1.5)]
ABL_WS = (1, 4)                  # multipliers on the headline strength

PROPS = ["gc", "cpg"]            # affine and quadratic; the contrast is the point
TARGET = "q50"                   # v3 is a q50 run
SEEDS = [20260921, 20260922, 20260923]
WINDOW_RUNG = 0.5                # the t_min the ablation measures against t_min=0


def cells(stage, w):
    if stage == "m2":
        for prop, seed in itertools.product(PROPS, SEEDS):
            yield dict(prop=prop, arm="unguided", variant="-", w=w, seed=seed,
                       t_min=0.0)
            for arm in ARMS_EXT:
                yield dict(prop=prop, arm=arm, variant="-", w=w, seed=seed,
                           t_min=0.0)
            for v in HEADLINE_BDG:
                yield dict(prop=prop, arm="bdg", variant=v, w=w, seed=seed,
                           t_min=0.0)
    elif stage == "m2abl":
        for prop, seed in itertools.product(PROPS, SEEDS):
            for v, mult in itertools.product(ABL_BDG, ABL_WS):
                yield dict(prop=prop, arm="bdg", variant=v, w=w * mult,
                           seed=seed, t_min=0.0)
            # The window rung. Protocol section 2.1 departs from v3's t >= 0.5;
            # this is what lets a reader see the departure was necessary instead
            # of taking a comment's word for it.
            for arm, v in (("plug", "-"), ("bdg", "e4t0.5"), ("bdg", "e4t1")):
                yield dict(prop=prop, arm=arm, variant=v, w=w, seed=seed,
                           t_min=WINDOW_RUNG)
    else:
        raise SystemExit("unknown stage %r (expected m2 or m2abl)" % stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="proj1/m2/blade_bundle/fm_m2_dfb500.pt")
    ap.add_argument("--out-dir", default="results/m2")
    ap.add_argument("--stage", default="m2", choices=["m2", "m2abl"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)      # NFE; protocol 2.2
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--seeds", default="",
                    help="comma list; default all of %s. Splitting the sweep by "
                         "seed is how it fans across GPUs -- cells are "
                         "independent and each is skipped if it already exists, "
                         "so two workers on disjoint seeds never collide."
                         % (SEEDS,))
    ap.add_argument("--props", default="",
                    help="comma list; default %s" % (PROPS,))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    if a.w is None:
        raise SystemExit(
            "--w is not set, and this protocol has no default headline "
            "strength.\nIt is fixed by a recorded measurement, not copied from "
            "Modality 1: the\nsmallest w in {1,4,16,64} at which plug separates "
            "from unguided by more\nthan the seed-to-seed se. Run\n"
            "  python proj1/m2/measure_strength.py\n"
            "and pass the w it records (docs/results/M2_STRENGTH.md).")
    if a.n % a.batch:
        raise SystemExit(
            "batch %d does not divide n %d. BDG's controller IS the batch; a "
            "remainder batch is a second, noisier controller pooled in as an "
            "equal." % (a.batch, a.n))

    plan = list(cells(a.stage, a.w))
    if a.seeds:
        keep = {int(s) for s in a.seeds.split(",") if s.strip()}
        unknown = keep - set(SEEDS)
        if unknown:
            raise SystemExit("--seeds %s is not in the protocol's %s"
                             % (sorted(unknown), SEEDS))
        plan = [c for c in plan if c["seed"] in keep]
    if a.props:
        keep = {p.strip() for p in a.props.split(",") if p.strip()}
        unknown = keep - set(PROPS)
        if unknown:
            raise SystemExit("--props %s is not in the protocol's %s"
                             % (sorted(unknown), PROPS))
        plan = [c for c in plan if c["prop"] in keep]
    print("%d cells -> %s/%s/n%d   (n=%d, batch=%d, NFE=%d, w=%g)"
          % (len(plan), a.out_dir, a.stage, a.n, a.n, a.batch, a.steps, a.w))
    if a.dry:
        for c in plan[:12]:
            print("  ", c)
        print("   ... and %d more" % (len(plan) - 12))
        return 0
    t0, done, failed = time.time(), 0, []
    for i, c in enumerate(plan, 1):
        cmd = [a.python, "proj1/m2/m2_sweep.py", "--ckpt", a.ckpt,
               "--prop", c["prop"], "--arm", c["arm"], "--variant", c["variant"],
               "--w", str(c["w"]), "--target", TARGET, "--seed", str(c["seed"]),
               "--t-min", str(c["t_min"]), "--n", str(a.n),
               "--batch", str(a.batch), "--steps", str(a.steps),
               "--stage", a.stage, "--device", a.device, "--out-dir", a.out_dir]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append((c, (r.stderr.strip().splitlines() or ["?"])[-1]))
            print("[%d/%d] FAIL %s %s %s w%g"
                  % (i, len(plan), c["prop"], c["arm"], c["variant"], c["w"]))
        else:
            done += 1
        if i % 10 == 0 or i == len(plan):
            el = (time.time() - t0) / 60
            print("[%d/%d] %.1f min elapsed, ~%.1f min left, %d failed"
                  % (i, len(plan), el, el / i * (len(plan) - i), len(failed)),
                  flush=True)
    print("\ndone: %d/%d  failed: %d" % (done, len(plan), len(failed)))
    for c, err in failed[:10]:
        print("  ", c, err)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
