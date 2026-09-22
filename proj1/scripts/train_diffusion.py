"""VP diffusion generator on QM9 -- the second required model family.

Structured identically to train_fm.py (same data, masks, EMA, resume, time
guard, pre-registered selection rule) so the SLURM script and the chaining
procedure carry over unchanged. Only the corruption and the regression target
differ:

    x_tau = alpha(tau) x_0 + sigma(tau) eps,     target = eps,

with the VP schedule in src/diffusion.py. The same EGNNVelocity architecture is
the eps-network: its coordinate output is equivariant and zero-CoM, which is
what eps for the coordinates must be.

tau is drawn uniformly on [tau_min, 1]; tau_min keeps sigma away from zero so
the target is never divided by a vanishing noise scale downstream.

MODEL SELECTION follows PROJECT_GUIDE.md section 4.5: highest validation atom
stability at NFE 100, ties broken by validation loss. Validation loss alone
cannot see the ~0.1 A precision that decides whether a bond is recognised.

Usage:
  python proj1/scripts/train_diffusion.py --epochs 1500 --resume --max-minutes 225
  python proj1/scripts/train_diffusion.py --epochs 3 --limit 2000 --tag d_smoke --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from models.egnn import EGNNVelocity, zero_com  # noqa: E402
from diffusion import alpha_sigma  # noqa: E402
from evaluation import stability  # noqa: E402
from sampling import VPSampler, initial_noise, integrate  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT_DIR = os.path.join(ROOT, "proj1", "checkpoints")
TAU_MIN = 1e-3


def atomic_save(obj, path, keep_prev=False):
    """Write to a temp file then rename.

    torch.save writes ~60 MB in place; a SIGTERM (or a wall-clock kill) landing
    inside that leaves a truncated archive that torch cannot read at all, which
    destroys a run that was otherwise fully resumable. os.replace is atomic on
    POSIX and Windows, so a reader sees either the old file or the new one.
    keep_prev also rotates one backup, so even a corrupt rename leaves a
    recoverable state.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if keep_prev and os.path.exists(path):
        prev = path.replace(".pt", "_prev.pt")
        try:
            os.replace(path, prev)
        except OSError:
            pass
    os.replace(tmp, path)


def batches(idx, bs, shuffle, gen=None):
    order = idx[torch.randperm(len(idx), generator=gen)] if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def sample_noise(coords_like, mask):
    """Standard Gaussian, projected onto the zero-CoM subspace of valid atoms."""
    eps = torch.randn_like(coords_like)
    return zero_com(eps, mask)


def diffusion_loss(model, x0_c, x0_f, mask, tau, feat_weight):
    """Masked eps-prediction loss. Returns (total, coord, feat)."""
    m = mask.unsqueeze(-1)
    eps_c = sample_noise(x0_c, mask)
    eps_f = torch.randn_like(x0_f) * m

    a, s = alpha_sigma(tau)
    a = a.view(-1, 1, 1)
    s = s.view(-1, 1, 1)
    xt_c = (a * x0_c + s * eps_c) * m
    xt_f = (a * x0_f + s * eps_f) * m

    pred_c, pred_f = model(xt_c, xt_f, mask, tau)

    n_c = (m.sum() * 3.0).clamp(min=1.0)
    n_f = (m.sum() * x0_f.shape[-1]).clamp(min=1.0)
    l_c = ((pred_c - eps_c) ** 2).sum() / n_c
    l_f = ((pred_f - eps_f) ** 2).sum() / n_f
    return l_c + feat_weight * l_f, l_c, l_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_a",
                    choices=["train_a", "train_b", "train_ab"],
                    help="train_a (51,527) is the PRE-REGISTERED choice, PROJECT_GUIDE section 4.1. "
                         "The generator must NOT see train_b: that is f_B's training data, and a "
                         "generator that has seen it produces samples f_B scores optimistically -- "
                         "a leak that biases every downstream guidance number. The cost, to "
                         "disclose in the paper: 50k where E(3)-EDM uses 100k. TFG-Flow and "
                         "PropMolFlow use the same 50/50, so our numbers stay comparable to "
                         "theirs. train_ab exists only as a deliberate, disclosed ablation.")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-min-frac", type=float, default=0.05,
                    help="cosine floor as a fraction of lr. A schedule that reaches "
                         "exactly zero wastes its final epochs on an unconverged model.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9999,
                    help="EDM's value. 0.999 is a ~1k-step horizon, far too short for a "
                         "several-hundred-thousand-step run, and the EMA copy is what "
                         "gets sampled. Warmed up so early epochs are not frozen.")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="diff")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint for this tag and start over. "
                         "WITHOUT this, an existing <tag>.pt is read and can only be "
                         "improved on, never silently replaced by a worse model.")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--stab-every", type=int, default=25,
                    help="epochs between sampling-based stability evaluations")
    ap.add_argument("--stab-n", type=int, default=2048,
                    help="molecules per stability eval. 512 could not separate the last 200 epochs of the first FM run (SE ~0.0025 on atom stability, real differences ~0.002); 2048 halves the SE at ~15% extra wall time. If the curve is still rising at the end, also benchmark the final epoch before shipping the selected one.")
    ap.add_argument("--stab-steps", type=int, default=100,
                    help="NFE for stability eval")
    ap.add_argument("--save-every", type=int, default=100,
                    help="also keep <tag>_ep<N>.pt every N epochs (0 = off). <tag>.pt "
                         "holds only the current best, so these are what a later "
                         "harness run at higher NFE, or an epoch-vs-stability curve, "
                         "is read from. Must match the FM trainer or the two families "
                         "have different diagnostic capability.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    # Loud, unmissable guard. This exact mistake was made once: the generator was
    # set to train_ab, which is NOT the protocol and cost a day of compute before
    # it was caught. See SPLIT_PROTOCOL.md.
    if args.split != "train_a":
        print("")
        print("  " + "!" * 72)
        print("  !!  SPLIT WARNING: --split %s" % args.split)
        print("  !!")
        print("  !!  The protocol is --split train_a. The generator and the guide f_A")
        print("  !!  share train_a; the evaluator f_B gets train_b ALONE. This matches")
        print("  !!  TFG (which guides EDMsecond, a half-trained generator, even though")
        print("  !!  EDMfull was available) and EDM's conditional protocol.")
        print("  !!")
        print("  !!  Training on train_ab lets the generator see the evaluator's data and")
        print("  !!  doubles the data behind every number we quote beside published work.")
        print("  !!  Only do this as a DELIBERATE, DISCLOSED ablation.")
        print("  !!  See SPLIT_PROTOCOL.md before continuing.")
        print("  " + "!" * 72)
        print("")

    torch.manual_seed(args.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if dev == "cuda":
        # TF32 matmuls: large speedup on Ampere and later, irrelevant to this
        # model's accuracy since the loss is a mean-squared regression.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(CKPT_DIR, exist_ok=True)

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    # The whole dataset is ~140 MB. Put it on the GPU once instead of copying
    # every batch across PCIe for hundreds of thousands of steps.
    coords = d["coords"].to(dev)
    feats = d["feats"].to(dev)
    mask = d["mask"].to(dev)

    if args.split == "train_ab":
        tr_idx = torch.cat([d["split"]["train_a"], d["split"]["train_b"]])
    else:
        tr_idx = d["split"][args.split]
    va_idx = d["split"]["val"]
    if args.limit:
        tr_idx = tr_idx[:args.limit]
        va_idx = va_idx[:max(256, args.limit // 8)]
    tr_idx = tr_idx.to(dev)
    va_idx = va_idx.to(dev)

    print("%s | VP diffusion on %s: %d train, %d val | device=%s"
          % (args.tag, args.split, len(tr_idx), len(va_idx), dev))

    model = EGNNVelocity(len(types), args.hidden, args.layers).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac)
    print("  parameters: %d  hidden=%d layers=%d  ema=%g  lr=%g->%g"
          % (n_par, args.hidden, args.layers, args.ema, args.lr, args.lr * args.lr_min_frac))

    step = [0]

    @torch.no_grad()
    def ema_update():
        # Warmup: a 0.9999 decay applied from step 0 would leave the EMA copy
        # near its random initialisation for tens of thousands of steps.
        step[0] += 1
        dcy = min(args.ema, (1.0 + step[0]) / (10.0 + step[0]))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(dcy).add_(pm.detach(), alpha=1.0 - dcy)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)

    @torch.no_grad()
    def evaluate(net, idx, bs=256):
        """Validation loss on a FIXED (tau, eps) stream, so epochs are comparable.

        fork_rng is load-bearing: seeding the global generator here without it
        would rewind the stream the TRAINING loop draws tau and eps from, making
        the corruptions repeat every --eval-every epochs.
        """
        net.eval()
        tot, n = 0.0, 0
        devices = [torch.cuda.current_device()] if dev == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for k, b in enumerate(batches(idx, bs, shuffle=False)):
                u = (torch.arange(len(b), dtype=torch.float32, device=dev) + 0.5) / len(b)
                tau = TAU_MIN + (1.0 - TAU_MIN) * u
                torch.manual_seed(1234 + k)
                loss, _, _ = diffusion_loss(net, coords[b], feats[b], mask[b],
                                            tau, args.feat_weight)
                tot += loss.item() * len(b)
                n += len(b)
        net.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def stability_eval(net, n_mol, steps, bs=128):
        """(atom stability, molecule stability) over sampled molecules.

        This is the pre-registered selection signal. Masks are drawn from the
        validation split so the atom-count distribution matches the data.
        """
        net.eval()
        devices = [torch.cuda.current_device()] if dev == "cuda" else []
        n_stab_a = n_at = n_stab_m = n_mols = 0
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(4321)
            g = torch.Generator(device=dev).manual_seed(4321)
            sel = va_idx[:n_mol]
            for i in range(0, len(sel), bs):
                m = mask[sel[i:i + bs]]
                c0, f0 = initial_noise(m, len(types), g)
                c, f, _ = integrate(VPSampler(net, m, tau_min=TAU_MIN),
                                    c0, f0, steps, "euler")
                fin = torch.isfinite(c).all((1, 2)) & torch.isfinite(f).all((1, 2))
                c = torch.where(fin.view(-1, 1, 1), c, torch.zeros_like(c))
                f = torch.where(fin.view(-1, 1, 1), f, torch.zeros_like(f))
                for (sa, na, ok, _, _), good in zip(
                        stability(c.float().cpu(), f.float().cpu(), m.float().cpu(), types),
                        fin.cpu()):
                    n_at += na
                    n_mols += 1
                    if good:
                        n_stab_a += sa
                        n_stab_m += int(ok)
        net.train()
        return n_stab_a / max(n_at, 1), n_stab_m / max(n_mols, 1)

    gen = torch.Generator().manual_seed(args.seed)
    best_stab, best_loss, best_mol = -1.0, float("inf"), float("nan")
    hist, t0 = [], time.time()

    ckpt_path = os.path.join(CKPT_DIR, args.tag + ".pt")
    last_path = os.path.join(CKPT_DIR, args.tag + "_last.pt")
    start_ep = 1

    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        for k in ("hidden", "layers", "split", "lr", "batch"):
            if st["args"][k] != vars(args)[k]:
                raise SystemExit("--resume: %s was %r, now %r" % (k, st["args"][k], vars(args)[k]))
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        # The batch-order generator alone is not enough: t/tau and eps are drawn
        # from the GLOBAL stream, so without this a resume replays the same
        # corruptions the run already saw.
        if st.get("torch_rng") is not None:
            torch.set_rng_state(st["torch_rng"].cpu().to(torch.uint8))
        if dev == "cuda" and st.get("cuda_rng") is not None:
            try:
                torch.cuda.set_rng_state_all([r.cpu().to(torch.uint8) for r in st["cuda_rng"]])
            except Exception as e:
                print("  note: could not restore CUDA RNG (%s); continuing" % type(e).__name__)
        step[0] = st.get("step", 0)
        # Scheduler LAST: it steps recursively from the optimiser's current LR.
        if st["args"]["epochs"] == args.epochs:
            sched.load_state_dict(st["sched"])
        else:
            print("  NOTE: total epochs changed %d -> %d; rebuilding the cosine "
                  "schedule and fast-forwarding (LR will jump)"
                  % (st["args"]["epochs"], args.epochs))
            for gp in opt.param_groups:
                gp["lr"] = args.lr
                gp["initial_lr"] = args.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac)
            for _ in range(st["epoch"]):
                sched.step()
        best_stab = st.get("best_stab", -1.0)
        best_mol = st.get("best_mol", float("nan"))
        best_loss = st.get("best_loss", st.get("best", float("inf")))
        hist, start_ep = st["hist"], st["epoch"] + 1
        print("  resumed from %s_last.pt at epoch %d (best atom_stab %.4f, loss %.5f)"
              % (args.tag, st["epoch"], best_stab, best_loss))
    elif args.resume:
        print("  --resume given but no %s_last.pt; starting from scratch" % args.tag)

    # B2 guard. Selection compares against best_stab, so if that starts at -1.0
    # the FIRST evaluation always looks like an improvement and overwrites
    # <tag>.pt -- which is the deliverable. That happens whenever _last.pt is
    # absent (forgotten --resume, quota purge, corrupt file) even though a
    # perfectly good <tag>.pt is sitting right there. Seed from it unless the
    # user explicitly asked for a fresh start.
    if not args.fresh and os.path.exists(ckpt_path):
        prev_best = None
        for cand in (ckpt_path, ckpt_path.replace(".pt", "_prev.pt")):
            if not os.path.exists(cand):
                continue
            try:
                prev_best = torch.load(cand, map_location="cpu", weights_only=False)
                if cand != ckpt_path:
                    print("  NOTE: %s is unreadable; recovered from %s"
                          % (os.path.basename(ckpt_path), os.path.basename(cand)))
                break
            except Exception as e:
                # A truncated checkpoint must not make the trainer unstartable.
                print("  WARNING: could not read %s (%s). Trying the backup."
                      % (os.path.basename(cand), type(e).__name__))
        if prev_best is None:
            print("  WARNING: no readable existing checkpoint for tag %r. Training will "
                  "write a new one; delete %s.pt by hand if it is corrupt."
                  % (args.tag, args.tag))
        else:
            # Seeding across a CONFIG CHANGE is the same error the --resume guard
            # above refuses, arriving by a quieter route. An old <tag>.pt trained
            # on a different split saw different data, so its atom_stab is not
            # comparable; inheriting it would mean this run has to beat a number
            # it cannot fairly be measured against, and might never write a
            # deliverable at all while the file on disk stays the OLD model.
            pargs = prev_best.get("args", {})
            mism = [k for k in ("hidden", "layers", "split")
                    if k in pargs and pargs[k] != vars(args)[k]]
            if mism:
                raise SystemExit(
                    "%s.pt was trained with %s, but this run uses %s.\n"
                    "Its score is not comparable, so it must not seed selection.\n"
                    "Either move it aside, or pass --fresh to start selection over.\n"
                    "See SPLIT_PROTOCOL.md if the difference is --split."
                    % (args.tag,
                       ", ".join("%s=%r" % (k, pargs[k]) for k in mism),
                       ", ".join("%s=%r" % (k, vars(args)[k]) for k in mism)))
            pb = float(prev_best.get("atom_stability", -1.0))
            if pb > best_stab:
                best_stab = pb
                best_loss = float(prev_best.get("val_loss", float("inf")))
                best_mol = float(prev_best.get("mol_stability", float("nan")))
                print("  existing %s.pt has atom_stab %.4f (epoch %s); it will only be "
                      "replaced by something better. Pass --fresh to override."
                      % (args.tag, pb, prev_best.get("epoch", "?")))

    def save_last(ep):
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state_all() if dev == "cuda" else None),
                    "epoch": ep, "step": step[0],
                    "best_stab": best_stab, "best_loss": best_loss,
                    "best_mol": best_mol,
                    "hist": hist, "args": vars(args),
                    "family": "vp_diffusion", "tau_min": TAU_MIN},
                   last_path, keep_prev=True)

    ep = start_ep - 1
    for ep in range(start_ep, args.epochs + 1):
        # Accumulate on the GPU: .item() per batch forces a sync every step.
        run = torch.zeros((), device=dev)
        rc = torch.zeros((), device=dev)
        rf = torch.zeros((), device=dev)
        nb = 0
        for b in batches(tr_idx, args.batch, shuffle=True, gen=gen):
            b = b.to(dev, non_blocking=True) if b.device.type != dev else b
            tau = TAU_MIN + (1.0 - TAU_MIN) * torch.rand(len(b), device=dev)
            loss, l_c, l_f = diffusion_loss(model, coords[b], feats[b], mask[b],
                                            tau, args.feat_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema_update()
            run += loss.detach()
            rc += l_c.detach()
            rf += l_f.detach()
            nb += 1
        sched.step()
        tr_loss, tr_c, tr_f = (run / nb).item(), (rc / nb).item(), (rf / nb).item()

        do_loss = ep % args.eval_every == 0 or ep == args.epochs or ep == 1
        do_stab = ep % args.stab_every == 0 or ep == args.epochs or ep == 1
        vl = float("nan")
        a_st = m_st = float("nan")
        if do_loss or do_stab:
            vl = evaluate(ema, va_idx) if do_loss else float("nan")
            star = ""
            if do_stab:
                a_st, m_st = stability_eval(ema, args.stab_n, args.stab_steps)
                better = (a_st > best_stab + 1e-6) or \
                         (abs(a_st - best_stab) <= 1e-6 and vl == vl and vl < best_loss)
                if better:
                    # These must describe the SELECTED checkpoint, not the best
                    # ever seen for each quantity separately -- otherwise the
                    # summary line pairs a stability from one epoch with a loss
                    # from another, and the tie-break compares against a loss
                    # belonging to a different model.
                    best_stab = a_st
                    if vl == vl:
                        best_loss = vl
                    best_mol = m_st
                    star = "  <- best"
                    atomic_save({"state_dict": model.state_dict(),
                                "ema_state_dict": ema.state_dict(),
                                "args": vars(args), "val_loss": vl,
                                "atom_stability": a_st, "mol_stability": m_st,
                                "epoch": ep, "types": types,
                                "max_atoms": d["max_atoms"],
                                "family": "vp_diffusion", "tau_min": TAU_MIN},
                               ckpt_path)
            hist.append({"epoch": ep, "train": tr_loss, "coord": tr_c, "feat": tr_f,
                         "val_ema": vl, "atom_stab": a_st, "mol_stab": m_st,
                         "lr": sched.get_last_lr()[0]})
            print("  ep %4d  train %.5f (c %.5f f %.5f)  val %.5f  atom_stab %s  mol_stab %s%s  (%.0fs)"
                  % (ep, tr_loss, tr_c, tr_f, vl,
                     "  n/a" if a_st != a_st else "%.4f" % a_st,
                     "  n/a" if m_st != m_st else "%.4f" % m_st, star, time.time() - t0))

        save_last(ep)
        if args.save_every and ep % args.save_every == 0:
            atomic_save({"state_dict": model.state_dict(),
                         "ema_state_dict": ema.state_dict(),
                         "args": vars(args), "val_loss": vl,
                         "atom_stability": a_st, "mol_stability": m_st,
                         "epoch": ep, "types": types, "max_atoms": d["max_atoms"],
                         "family": "vp_diffusion"},
                        os.path.join(CKPT_DIR, "%s_ep%04d.pt" % (args.tag, ep)))
        elapsed_min = (time.time() - t0) / 60.0
        if args.max_minutes and ep < args.epochs:
            per_epoch = elapsed_min / max(ep - start_ep + 1, 1)
            if elapsed_min + per_epoch > args.max_minutes:
                print("  time guard: %.1f min used, %.2f min/epoch, stopping at "
                      "epoch %d of %d; resume with --resume"
                      % (elapsed_min, per_epoch, ep, args.epochs))
                break

    # Do not let a short run destroy a long history: the file is a deliverable,
    # and a relaunch without --resume starts with an empty `hist`.
    hp = os.path.join(CKPT_DIR, args.tag + "_history.json")
    if os.path.exists(hp) and not args.fresh:
        try:
            with open(hp) as fh:
                old_hist = json.load(fh).get("history", [])
            if len(old_hist) > len(hist):
                print("  keeping the existing %d-point history (this run has %d); "
                      "the longer one is written to %s_history_prev.json"
                      % (len(old_hist), len(hist), args.tag))
                with open(hp.replace(".json", "_prev.json"), "w") as fh:
                    json.dump({"history": old_hist}, fh, indent=2)
        except Exception:
            pass
    with open(hp + ".tmp", "w") as fh:
        json.dump({"history": hist, "best_atom_stability": best_stab,
                   "best_mol_stability": best_mol,
                   "selected_val_loss": best_loss, "last_epoch": ep,
                   "params": n_par, "args": vars(args)}, fh, indent=2)
    os.replace(hp + ".tmp", hp)
    print("%s %s: epoch %d/%d | SELECTED checkpoint: atom_stab %.4f  mol_stab %.4f  "
          "val_loss %.5f  (this job %.0fs)"
          % (args.tag, "done" if ep == args.epochs else "paused",
             ep, args.epochs, best_stab, best_mol, best_loss, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
