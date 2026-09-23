"""Add the field's target protocol, and fix the batch-slice bug it would trip. Run once.

THE PROTOCOL. EDM, EEGSDE and TFG do not condition on a fixed quantile. Each
generated molecule gets ITS OWN target, drawn jointly with its size from real
held-out molecules, and MAE is scored per molecule against that target -- the
vendored TFG evaluator does exactly this (`target[cur_slice]`,
audit/fa_fb_search/TFG/evaluations/molecule.py:60,77).

We already take each molecule's SIZE from a real validation molecule
(`va = d["split"]["val"][:n]`). Taking its TARGET from that same molecule's real
property value gives the joint (property, size) draw with no new choices. This
is target name "dist". It is the headline protocol for the full run because it
is the one the comparison literature uses -- which also means WE do not choose
the target, so there is nothing to cherry-pick.

q50 / q90 remain as the two diagnostic tasks: q50 is concentration (the
unguided generator already sits within 0.05-0.27 sd of it), q90 is steering
(1.1-1.6 sd away).

THE BUG. The batch loop passed `y_t[: m.shape[0]]` -- sliced from 0 for EVERY
batch. n=512 runs as 4 batches of 128, so batches 2-4 received batch 1's
targets. Harmless for every cell run so far, since q50/q90 targets are
constant and every slice is identical. With per-molecule targets it would guide
75% of molecules toward someone else's target while `evaluate_samples` scored
them against their own. Fixed to `y_t[i : i + m.shape[0]]`.
"""
import io

NL = chr(10)
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

# ---- the bug: slice from the batch offset, not from 0
OLD1 = "                    net, f_A, m, c0, f0, y_t[: m.shape[0]],"
NEW1 = "                    net, f_A, m, c0, f0, y_t[i:i + m.shape[0]],"
assert s.count(OLD1) == 1, "dflow slice"
s = s.replace(OLD1, NEW1)

OLD2 = "                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w_applied,"
NEW2 = "                      y=y_t[i:i + m.shape[0]], s=s, mode=arm, w=w_applied,"
assert s.count(OLD2) == 1, "guided slice"
s = s.replace(OLD2, NEW2)

# ---- the protocol
OLD3 = "        target = TARGETS[prop][tgt]"
NEW3 = '''        if tgt == "dist":
            # the field's protocol: each molecule's target is the real property
            # of the SAME validation molecule whose size it already takes
            y_real = d["y"][va, d["props"].index(prop)].to(dev).float()
            target = float(y_real.mean())
        else:
            target = TARGETS[prop][tgt]'''
assert s.count(OLD3) == 1, "target anchor"
s = s.replace(OLD3, NEW3)

OLD4 = "        y_t = torch.full((args.n,), target, device=dev)"
NEW4 = '''        y_t = (y_real if tgt == "dist"
               else torch.full((args.n,), target, device=dev))'''
assert s.count(OLD4) == 1, "y_t anchor"
s = s.replace(OLD4, NEW4)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("batch-slice bug fixed; 'dist' target added")
