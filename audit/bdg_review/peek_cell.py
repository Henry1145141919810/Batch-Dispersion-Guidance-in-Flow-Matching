import json
import sys

c = json.load(open(sys.argv[1]))
keep = ["prop", "arm", "variant", "target_name", "target", "w", "w_applied",
        "batch", "n", "delta", "bdg_eta", "bdg_tau", "bdg_tau_mult", "bdg_s",
        "bdg_onesided", "in_band_fraction", "in_band_fraction_dec",
        "prop_mae_eval", "mol_stability", "validity", "uniqueness_of_valid",
        "f_B_mean", "n_nonfinite", "clipped_sample_steps", "cost"]
for k in keep:
    if k in c:
        print("%-24s %s" % (k, c[k]))
print("diag:")
for k, v in sorted(c.get("diag", {}).items()):
    print("  %-22s %s" % (k, v))
