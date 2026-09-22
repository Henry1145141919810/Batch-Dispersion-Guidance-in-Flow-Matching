"""Let a missing checkpoint skip ONE arm instead of killing the whole job.

THE PROBLEM. `rch` needs a fitted head per property and correctly refuses to run
without one. But `--preflight` runs under `set -e` in the SLURM script and only
exercises the FIRST property, so with per-property jobs a missing
`rch_alpha.pt` made the alpha job exit at minute two having written ZERO of its
114 cells -- 112 of which have nothing to do with `rch`. That turned a two-minute
fitting job into a hard dependency of the entire night.

THE DISTINCTION THAT WAS MISSING. A missing checkpoint and a wiring bug are not
the same class of failure:

  * a wiring bug means every cell this job writes is suspect -- fail fast, which
    is what preflight is for;
  * a missing checkpoint means ONE arm cannot run yet. The other thirteen are
    fine, and the arm's cells should be retried when the file appears.

`MissingState` marks the second kind. In preflight it is reported and skipped
without failing the job; in a real run it still writes a `.failed` sidecar, so
the cell is retried by the next link once the checkpoint exists.

Consequence: `fit_rch.slurm` stops being a dependency. Submit it whenever; the
v2 chain runs the other arms tonight and picks up `rch` on a later link.
"""
import io

NL = chr(10)
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

# ---- the exception type
OLD_C = '''DATA = os.path.join(ROOT, "data", "qm9.pt")'''
NEW_C = '''class MissingState(RuntimeError):
    """An arm's prerequisite file does not exist yet.

    Distinct from every other error on purpose: this one means ONE arm cannot
    run, not that the job is broken. Preflight skips it; a real run records a
    .failed sidecar so the cell is retried when the file appears.
    """


DATA = os.path.join(ROOT, "data", "qm9.pt")'''
assert s.count(OLD_C) == 1, "data anchor"
s = s.replace(OLD_C, NEW_C)

# ---- raise it for a missing rch head
OLD_R = '''            if not os.path.exists(rp):
                raise RuntimeError("rch needs %s -- fit it with "
                                   "proj1/scripts/fit_rch.py first" % rp)'''
NEW_R = '''            if not os.path.exists(rp):
                raise MissingState(
                    "rch needs %s -- run proj1/cluster/fit_rch.slurm. The rest "
                    "of this job is unaffected; these cells retry on the next "
                    "link once the head exists." % rp)'''
assert s.count(OLD_R) == 1, "rch raise anchor"
s = s.replace(OLD_R, NEW_R)

# ---- preflight: skip, do not fail
OLD_P = '''            failed.append(name)
            if args.preflight:
                print("  PREFLIGHT FAILED %-22s %s: %s"
                      % (arm, type(e).__name__, e))
                continue'''
NEW_P = '''            if args.preflight and isinstance(e, MissingState):
                # not a failure: one arm's prerequisite is absent, the job is
                # fine. Skipping it here is what stops a two-minute fitting job
                # from gating a fourteen-hour sweep.
                skipped.append(arm)
                print("  PREFLIGHT SKIP   %-22s %s" % (arm, e))
                continue
            failed.append(name)
            if args.preflight:
                print("  PREFLIGHT FAILED %-22s %s: %s"
                      % (arm, type(e).__name__, e))
                continue'''
assert s.count(OLD_P) == 1, "preflight except anchor"
s = s.replace(OLD_P, NEW_P)

OLD_I = '''    done = 0
    failed = []'''
NEW_I = '''    done = 0
    failed = []
    skipped = []'''
assert s.count(OLD_I) == 1, "init anchor"
s = s.replace(OLD_I, NEW_I)

OLD_S = '''    if args.preflight:
        print("\\npreflight: %d arms ran, %d failed" % (done, len(failed)))
        if failed:
            print("PREFLIGHT FAILED -- fix these before submitting the sweep")
            return 1
        print("PREFLIGHT OK -- every arm runs end to end")
        return 0'''
NEW_S = '''    if args.preflight:
        print("\\npreflight: %d arms ran, %d failed, %d skipped"
              % (done, len(failed), len(skipped)))
        if skipped:
            print("SKIPPED (prerequisite missing, job proceeds): %s"
                  % ",".join(sorted(set(skipped))))
        if failed:
            print("PREFLIGHT FAILED -- fix these before submitting the sweep")
            return 1
        print("PREFLIGHT OK -- every runnable arm works end to end")
        return 0'''
assert s.count(OLD_S) == 1, "preflight summary anchor"
s = s.replace(OLD_S, NEW_S)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("MissingState added: a missing checkpoint now skips one arm, not the job")
