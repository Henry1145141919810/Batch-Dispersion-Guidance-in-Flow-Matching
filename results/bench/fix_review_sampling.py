"""Adversarial-review fixes to sampling.py: D11, D12, D13. Run once.

D11 VPSampler called `self._guide(...)` with no mode, so `active()` was never
    consulted and a `schedule=` was silently dropped; and it applied the
    score-to-velocity factor to every arm with no DISPLACEMENT_MODES check.
    Latent while the checkpoint is family='flow', but guidance_sweep.py
    branches on family and would hand mode="shg_plug_spbc" straight into
    guidance_field -> ValueError on every SHG cell.

D12 `schedule_log` was incremented BEFORE the t_min_guide / t>=1 guard, so it
    counted steps that were then skipped (measured: {'plug': 7, 'spbc': 3} when
    5 of those were never guided); `reset_counts()` did not clear it; and the
    sweep read it from the last batch's sampler only.

D13 No v2 diagnostic ever reached the cell JSON. `last_diag` was overwritten
    every step and discarded. Given D4 and D10, "did V_F go non-positive, and
    how often was the variance term above tau^2?" is the decisive question for
    BTVG, and the sweep produced no evidence either way. A small set of scalars
    is now accumulated over every step of every batch.
"""
import io

NL = chr(10)
p = "proj1/src/sampling.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- D12 (count)
OLD = '''        for (lo, hi, mode, mult) in self.schedule:
            if lo <= t_scalar < hi:
                self.schedule_log[mode] = self.schedule_log.get(mode, 0) + 1
                return mode, self.w * mult
        return None, 0.0'''
NEW = '''        for (lo, hi, mode, mult) in self.schedule:
            if lo <= t_scalar < hi:
                return mode, self.w * mult
        return None, 0.0

    def note_guided(self, mode):
        """Record that a step was ACTUALLY guided with `mode`.

        Counting inside `active()` overcounted, because the caller may still
        skip the step on t_min_guide or t >= 1: a schedule starting at 0.50
        with t_min_guide = 0.75 logged 10 steps of which 5 never ran."""
        self.schedule_log[mode] = self.schedule_log.get(mode, 0) + 1'''
assert s.count(OLD) == 1, "active anchor"
s = s.replace(OLD, NEW)

# ---------------------------------------------------------------- D12 (reset)
OLD_R = '''    def reset_counts(self):
        self.cost.reset()
        self.n_field = self.n_clipped = self.n_guided = 0'''
NEW_R = '''    def reset_counts(self):
        self.cost.reset()
        self.n_field = self.n_clipped = self.n_guided = 0
        self.schedule_log = {}
        self.diag_acc = {}'''
assert s.count(OLD_R) == 1, "reset anchor"
s = s.replace(OLD_R, NEW_R)

# ---------------------------------------------------------------- D13
OLD_D = '''        self.last_diag = diag'''
NEW_D = '''        self.last_diag = diag
        self._accumulate_diag(diag)'''
assert s.count(OLD_D) == 1, "last_diag anchor"
s = s.replace(OLD_D, NEW_D)

OLD_A = '''    def _guide(self, coords, feats, t, post_fn, mode=None):'''
NEW_A = '''    # Diagnostics worth carrying out of the run. Deliberately a short list:
    # these are the quantities that decide whether BTVG's variance model held
    # and whether SPBC's forecast was moving anything, and without them the
    # sweep cannot answer either question.
    DIAG_KEYS = ("btvg_V_over_tau2", "btvg_V_nonpositive", "btvg_V_raw",
                 "spbc_bias", "spbc_increment", "spbc_live_fraction",
                 "spbc_mean_response", "c", "v_f")

    def _accumulate_diag(self, diag):
        """Running mean of each diagnostic over every guided step and batch."""
        for key in self.DIAG_KEYS:
            v = diag.get(key)
            if v is None:
                continue
            try:
                x = float(v.double().mean()) if hasattr(v, "double") else float(v)
            except Exception:
                continue
            if x != x:                       # NaN carries no information here
                continue
            tot, n = self.diag_acc.get(key, (0.0, 0))
            self.diag_acc[key] = (tot + x, n + 1)

    def diag_summary(self):
        return {k: tot / n for k, (tot, n) in self.diag_acc.items() if n}

    def _guide(self, coords, feats, t, post_fn, mode=None):'''
assert s.count(OLD_A) == 1, "guide anchor"
s = s.replace(OLD_A, NEW_A)

OLD_I = '''        self.kappa3_log = []
        self.schedule_log = {}'''
NEW_I = '''        self.kappa3_log = []
        self.schedule_log = {}
        self.diag_acc = {}'''
assert s.count(OLD_I) == 1, "init anchor"
s = s.replace(OLD_I, NEW_I)

# ---------------------------------------------------------------- D12 (flow)
OLD_F = '''        self.n_guided += 1
        post_fn = lambda c, f: fm_posterior(self.net, c, f, self.mask, t)  # noqa: E731
        G_c, G_f = self._guide(coords, feats, t, post_fn, mode)'''
NEW_F = '''        self.n_guided += 1
        self.note_guided(mode)
        post_fn = lambda c, f: fm_posterior(self.net, c, f, self.mask, t)  # noqa: E731
        G_c, G_f = self._guide(coords, feats, t, post_fn, mode)'''
assert s.count(OLD_F) == 1, "flow guided anchor"
s = s.replace(OLD_F, NEW_F)

# ---------------------------------------------------------------- D11
OLD_V = '''        if self.f_net is not None:
            self.n_guided += 1
            post_fn = lambda c, f: vp_posterior(self.net, c, f, self.mask, tau, a, s)  # noqa: E731
            G_c, G_f = self._guide(coords, feats, tau, post_fn)
            G_c, G_f = -0.5 * b * self.w * G_c, -0.5 * b * self.w * G_f'''
NEW_V = '''        mode, w = self.active(tau_scalar)
        if self.f_net is not None and mode is not None:
            self.n_guided += 1
            self.note_guided(mode)
            post_fn = lambda c, f: vp_posterior(self.net, c, f, self.mask, tau, a, s)  # noqa: E731
            G_c, G_f = self._guide(coords, feats, tau, post_fn, mode)
            # Same units rule as the flow sampler: a displacement arm must not
            # be pushed through the score-to-drift conversion. Without this,
            # every SHG cell on a diffusion checkpoint would hand a schedule
            # NAME into guidance_field and raise.
            if mode in DISPLACEMENT_MODES:
                G_c, G_f = w * G_c, w * G_f
            else:
                G_c, G_f = -0.5 * b * w * G_c, -0.5 * b * w * G_f'''
assert s.count(OLD_V) == 1, "vp guided anchor"
s = s.replace(OLD_V, NEW_V)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sampling.py: D11, D12, D13 applied")
