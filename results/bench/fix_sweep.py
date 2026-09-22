"""Fixes from the sweep audit. Run once."""
import io

# ---------------------------------------------------------------- B1
# "tmpd" was never a mode. guidance_field silently fell through to the plug-in
# branch, so 36 of 288 cells would have been duplicate plug-in results labelled
# TMPD -- and the test that "covers" it runs smg_var and merely NAMES the
# result field_tmpd_exact, so the suite was green while the arm was wrong.
p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

old = """    with torch.no_grad():
        post = post_fn(coords, feats)
    cost.gen_fwd += 1"""
new = """    # An unknown mode used to fall through to the plug-in branch and return a
    # plausible field under the wrong name. Alias the published name, then
    # refuse anything unrecognised.
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in KNOWN_MODES:
        raise ValueError("unknown guidance mode %r; known: %s"
                         % (mode, ", ".join(sorted(KNOWN_MODES))))

    with torch.no_grad():
        post = post_fn(coords, feats)
    cost.gen_fwd += 1"""
assert s.count(old) == 1
s = s.replace(old, new)

anchor = "def guidance_field(f_net, post_fn, coords, feats, mask, y, s,"
consts = '''# "tmpd" is the published name for the uncertainty-denominator field; it is
# exactly our smg_var ablation. Aliased rather than duplicated so the two can
# never drift apart.
_MODE_ALIASES = {"tmpd": "smg_var", "pigdm": "smg_var", "dps": "plug"}
KNOWN_MODES = {"plug", "smg_mean", "smg_var", "smg", "smg2", "smg2_curv",
               "tfg_mc", "lgd_mc", "osc", "rch", "band"}


'''
assert s.count(anchor) == 1
s = s.replace(anchor, consts + anchor)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("B1: tmpd aliased to smg_var; unknown modes now raise")

# ---------------------------------------------------------------- B3
# A failed cell was written under the COMPLETED filename, so no later link ever
# retried it and the sweep still printed SWEEP COMPLETE. One transient OOM on a
# pass-1 cell would have permanently destroyed the arm comparison.
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

old = """        cs = time.time()
        try:
            r = run_cell(prop, arm, tgt, w, win)
        except Exception as e:                       # one bad cell must not kill the sweep
            r = {"prop": prop, "arm": arm, "target_name": tgt, "w": w,
                 "t_min_guide": win, "error": "%s: %s" % (type(e).__name__, e)}
            print("  CELL FAILED %s: %s" % (cell_name(prop, arm, tgt, w, win), e))
        tmp = os.path.join(OUT, cell_name(prop, arm, tgt, w, win) + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(r, fh)
        os.replace(tmp, os.path.join(OUT, cell_name(prop, arm, tgt, w, win)))
        done += 1"""
new = """        cs = time.time()
        name = cell_name(prop, arm, tgt, w, win)
        try:
            r = run_cell(prop, arm, tgt, w, win)
        except Exception as e:
            # A failure must NOT be written under the completed-cell name, or
            # no later link ever retries it and the sweep reports success with
            # a hole in it. Record it beside the cell and move on.
            failed.append(name)
            with open(os.path.join(OUT, name + ".failed"), "w") as fh:
                json.dump({"cell": name, "error": "%s: %s" % (type(e).__name__, e)}, fh)
            print("  CELL FAILED (will be retried on the next link) %s: %s" % (name, e))
            continue
        tmp = os.path.join(OUT, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(r, fh)
        os.replace(tmp, os.path.join(OUT, name))
        done += 1"""
assert s.count(old) == 1
s = s.replace(old, new)

s = s.replace("    t0 = time.time()\n    done = 0",
              "    t0 = time.time()\n    done = 0\n    failed = []")

old_tail = '''    print("\\n%d cells this job, %.1f min total" % (done, (time.time() - t0) / 60.0))
    left = [c for c in cells if not os.path.exists(os.path.join(OUT, cell_name(*c)))]
    print("SWEEP COMPLETE" if not left else "%d cells remain -- re-submit" % len(left))
    return 0'''
new_tail = '''    print("\\n%d cells this job, %.1f min total" % (done, (time.time() - t0) / 60.0))
    if failed:
        print("%d cells FAILED this job and will be retried:" % len(failed))
        for nm in failed[:10]:
            print("    " + nm)
    left = [c for c in cells if not os.path.exists(os.path.join(OUT, cell_name(*c)))]
    if not left:
        print("SWEEP COMPLETE")
    else:
        print("%d cells remain -- re-submit (%d of them previously failed)"
              % (len(left), len(failed)))
    return 0 if not failed else 1'''
assert s.count(old_tail) == 1
s = s.replace(old_tail, new_tail)

# ---------------------------------------------------------------- baseline + strengths
# `unguided` was documented but absent, so the plan's elimination criterion
# "no better than unguided" had nothing to compare against. And plug needs a
# far smaller strength than the grid offered: at 0.25 it was already 75-94%
# clip-saturated, i.e. untunable.
s = s.replace('ARMS = ["plug", "tmpd", "tfg_mc", "lgd_mc", "osc", "smg", "smg2", "smg2_curv"]',
              'ARMS = ["unguided", "plug", "tmpd", "tfg_mc", "lgd_mc", "osc",\n'
              '        "smg", "smg2", "smg2_curv"]')
s = s.replace("STRENGTHS = [0.25, 0.5, 1.0, 2.0, 4.0]",
              "# plug-in needs ~0.003-0.01 to avoid catastrophe (TOP6 memo section 8); at\n"
              "# 0.25 it was already 75-94% clip-saturated, so the old grid could not tune it.\n"
              "STRENGTHS = [0.01, 0.05, 0.25, 1.0, 4.0]")

# the unguided arm ignores strength and window -- run it once per property
s = s.replace('''    for prop in props:                       # pass 1: the arm comparison
        for arm in arms:
            add(prop, arm, "q50", DEFAULT_W, DEFAULT_WIN)''',
'''    for prop in props:                       # pass 1: the arm comparison
        for arm in arms:
            add(prop, arm, "q50", DEFAULT_W, DEFAULT_WIN)

    def expand(arm):
        """unguided has no strength or window, so it gets one cell per target."""
        return arm != "unguided"''')
s = s.replace('''    for prop in props:                       # pass 2: strength
        for arm in arms:
            for w in STRENGTHS:
                add(prop, arm, "q50", w, DEFAULT_WIN)
    for prop in props:                       # pass 3: guidance window
        for arm in arms:
            for win in WINDOWS:
                add(prop, arm, "q50", DEFAULT_W, win)''',
'''    for prop in props:                       # pass 2: strength
        for arm in arms:
            if not expand(arm):
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, DEFAULT_WIN)
    for prop in props:                       # pass 3: guidance window
        for arm in arms:
            if not expand(arm):
                continue
            for win in WINDOWS:
                add(prop, arm, "q50", DEFAULT_W, win)''')
s = s.replace('''            for tgt in TARGETS[prop]:
                if tgt == "q50":
                    continue
                for w in STRENGTHS:
                    add(prop, arm, tgt, w, DEFAULT_WIN)''',
'''            for tgt in TARGETS[prop]:
                if tgt == "q50":
                    continue
                if not expand(arm):
                    add(prop, arm, tgt, DEFAULT_W, DEFAULT_WIN)
                    continue
                for w in STRENGTHS:
                    add(prop, arm, tgt, w, DEFAULT_WIN)''')

# record every setting that affects a number, so cells are self-describing
s = s.replace('''                  "clipped_sample_steps": clipped, "seed": args.seed,
                  "fm": os.path.basename(args.fm)})''',
'''                  "clipped_sample_steps": clipped, "seed": args.seed,
                  "fm": os.path.basename(args.fm),
                  "clip": args.clip, "k_delta": args.k_delta,
                  "sigma_mc": args.sigma_mc, "batch": args.batch})''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("B3: failures retried, not cached; unguided baseline added; strengths widened")
