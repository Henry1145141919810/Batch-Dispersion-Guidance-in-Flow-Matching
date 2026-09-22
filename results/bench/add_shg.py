"""Sampler support for the v2 arms: displacement units and SHG schedules. Run once."""
import io

p = "proj1/src/sampling.py"
s = io.open(p, encoding="utf-8").read()

s = s.replace("from guidance import Cost, fm_posterior, guidance_field, vp_posterior",
              "from guidance import (Cost, DISPLACEMENT_MODES, fm_posterior,  # noqa: E501\n"
              "                      guidance_field, vp_posterior)")

s = s.replace('''                 n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                 band_tau=None, band_eta=1.0, band_radius=None):''',
'''                 n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                 band_tau=None, band_eta=1.0, band_radius=None,
                 tau=None, spbc_eta=1.0, spbc_radius=None, schedule=None):''')

s = s.replace('''        self.band_tau, self.band_eta, self.band_radius = band_tau, band_eta, band_radius
        self.kappa3_log = []''',
'''        self.band_tau, self.band_eta, self.band_radius = band_tau, band_eta, band_radius
        self.tau, self.spbc_eta, self.spbc_radius = tau, spbc_eta, spbc_radius
        # SHG: [(t_lo, t_hi, mode, w), ...]. The first interval containing t
        # wins; outside every interval, guidance is off. `mode`/`w` on the
        # sampler are the fallback when no schedule is given.
        self.schedule = schedule
        self.kappa3_log = []
        self.schedule_log = {}''')

# active (mode, w) at a given time
s = s.replace('''    def reset_counts(self):''',
'''    def active(self, t_scalar):
        """(mode, w) in force at this time, or (None, 0.0) for unguided.

        A handoff is a hard switch, so a Heun step that straddles a boundary
        would otherwise blend two fields inside one step. Both Heun stages call
        this with their own time, so each stage uses one regime -- which is
        what the memo requires (section 9.5)."""
        if self.schedule is None:
            return self.mode, self.w
        for (lo, hi, mode, w) in self.schedule:
            if lo <= t_scalar < hi:
                self.schedule_log[mode] = self.schedule_log.get(mode, 0) + 1
                return mode, w
        return None, 0.0

    def reset_counts(self):''')

s = s.replace('''    def _guide(self, coords, feats, t, post_fn):
        G_c, G_f, diag = guidance_field(
            self.f_net, post_fn, coords, feats, self.mask, self.y, self.s,
            self.mode, self.n_probe, self.probe_gen, self.cost, self.n_mc,
            self.sigma_mc, want_kappa3=self.want_kappa3, rch=self.rch,
            band_tau=self.band_tau, band_eta=self.band_eta,
            band_radius=self.band_radius, t_scalar=t)''',
'''    def _guide(self, coords, feats, t, post_fn, mode=None):
        G_c, G_f, diag = guidance_field(
            self.f_net, post_fn, coords, feats, self.mask, self.y, self.s,
            mode or self.mode, self.n_probe, self.probe_gen, self.cost,
            self.n_mc, self.sigma_mc, want_kappa3=self.want_kappa3,
            rch=self.rch, band_tau=self.band_tau, band_eta=self.band_eta,
            band_radius=self.band_radius, t_scalar=t,
            tau=self.tau, spbc_eta=self.spbc_eta, spbc_radius=self.spbc_radius)''')

# the flow field: honour the schedule and the displacement units
OLD = '''        if self.f_net is None or t_scalar < self.t_min_guide or t_scalar >= 1.0:
            return v_c, v_f

        self.n_guided += 1
        post_fn = lambda c, f: fm_posterior(self.net, c, f, self.mask, t)  # noqa: E731
        G_c, G_f = self._guide(coords, feats, t, post_fn)
        mult = self.w * (1.0 - t_scalar) / t_scalar
        G_c, G_f = mult * G_c, mult * G_f'''
NEW = '''        mode, w = self.active(t_scalar)
        if (self.f_net is None or mode is None or t_scalar < self.t_min_guide
                or t_scalar >= 1.0):
            return v_c, v_f

        self.n_guided += 1
        post_fn = lambda c, f: fm_posterior(self.net, c, f, self.mask, t)  # noqa: E731
        G_c, G_f = self._guide(coords, feats, t, post_fn, mode)
        if mode in DISPLACEMENT_MODES:
            # Already a state displacement. Converting it again through the
            # score-to-velocity factor (1-t)/t would apply that factor twice
            # and make the edit scale wrongly in t.
            mult = w
        else:
            mult = w * (1.0 - t_scalar) / t_scalar
        G_c, G_f = mult * G_c, mult * G_f'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sampling.py: displacement units + SHG schedule wired in")
