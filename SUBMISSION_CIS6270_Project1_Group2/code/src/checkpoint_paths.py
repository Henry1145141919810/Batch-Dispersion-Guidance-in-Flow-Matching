"""Where the checkpoints are, for a working copy and for a fresh clone.

THE PROBLEM THIS SOLVES. Two directories hold the same checkpoints:

  betty_pull/          cluster sync -- whatever the last job produced.
                       .gitignore'd, so it exists only on a machine that has
                       pulled from Betty.
  proj1/checkpoints/   local training output. Its *.pt are .gitignore'd too;
                       only the *_history.json files are committed.
  weights/             the inference-ready set we publish deliberately, and
                       the ONLY one of the three a fresh clone has.

Every script defaulted to the first two. On the machine they were written on
that is right -- those are the newest files. On a teammate's clone it is a
`FileNotFoundError` on `betty_pull/fm_last.pt` before anything runs, which is
what step 5 of ONBOARDING.md hit.

So: prefer the cluster copy when it is there, fall back to `weights/`, and say
which one was used. Naming the choice matters more than it looks -- two
machines silently running different checkpoints is exactly the kind of thing
that makes two sets of numbers disagree for a week before anyone works out why.

THE FILENAMES DIFFER BETWEEN THE TWO, which is why this cannot be a plain
directory swap: the generator is `fm_last.pt` under `betty_pull/` and
`fm_ema.pt` under `weights/` (same weights -- epoch 1500 EMA -- different
name), while the predictors keep the same `f_A_<prop>.pt` name in both.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

BETTY = os.path.join(ROOT, "betty_pull")
LOCAL_CKPT = os.path.join(ROOT, "proj1", "checkpoints")
WEIGHTS = os.path.join(ROOT, "weights")

# Generator, in preference order. Same model where the names differ.
_GENERATOR = (
    os.path.join(BETTY, "fm_last.pt"),
    os.path.join(LOCAL_CKPT, "fm_last.pt"),
    os.path.join(WEIGHTS, "fm_ema.pt"),
)

# OUR VP diffusion generator. NOT weights/EDMsecond -- that is TFG's borrowed
# checkpoint, reached by --edm-dir.
#
# `diff.pt` COMES FIRST, and the order is the point. train_diffusion.py writes
# TWO files: `diff_last.pt` every epoch as resume state, and `diff.pt` only
# when the pre-registered rule selects (highest validation atom stability at
# NFE 100). Preferring `diff_last.pt` would silently hand sampling whatever
# epoch the trainer last happened to reach -- including an abandoned partial
# run, which is exactly the state job 8590174 left behind at epoch ~80.
#
# The published slim copy is `weights/diff_ema.pt` (epoch 1475, the SELECTED
# checkpoint -- see weights/README.md), by analogy with fm_ema.pt. It is LAST
# so that when nothing exists, the error names the file a fresh clone is
# supposed to have. `vp_ema.pt` was this file's first guess at that name,
# before the weights shipped; it is kept, second to last, so an older clone
# that followed the guess still resolves.
_VP_GENERATOR = (
    os.path.join(BETTY, "diff.pt"),
    os.path.join(LOCAL_CKPT, "diff.pt"),
    os.path.join(BETTY, "diff_last.pt"),
    os.path.join(LOCAL_CKPT, "diff_last.pt"),
    os.path.join(WEIGHTS, "vp_ema.pt"),
    os.path.join(WEIGHTS, "diff_ema.pt"),
)

# Directories to look in for f_A_<prop>.pt, f_B_<prop>.pt, rch_<prop>.pt.
_PREDICTOR_DIRS = (LOCAL_CKPT, BETTY, WEIGHTS)


def default_generator() -> str:
    """The generator checkpoint to use when `--fm` is not given.

    Returns the first that exists; if none do, returns the LAST candidate so
    the error names `weights/fm_ema.pt` -- the one a teammate is supposed to
    have -- rather than a cluster path that means nothing to them.
    """
    for p in _GENERATOR:
        if os.path.exists(p):
            return p
    return _GENERATOR[-1]


def default_vp_generator() -> str:
    """The VP-diffusion generator checkpoint when `--vp-ckpt` is not given.

    Same contract as `default_generator`: first that exists, else the LAST
    candidate, so the error names `weights/diff_ema.pt` -- the committed file a
    fresh clone should have -- rather than a cluster path. It never falls back
    to some other model: a missing VP checkpoint fails loudly in `--backend vp`.
    """
    for p in _VP_GENERATOR:
        if os.path.exists(p):
            return p
    return _VP_GENERATOR[-1]


def find_predictor(name: str) -> str | None:
    """Locate `name` (e.g. 'f_A_mu.pt') across the checkpoint directories."""
    for d in _PREDICTOR_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def require_predictor(name: str, why: str = "") -> str:
    """`find_predictor`, but raise a message a teammate can act on."""
    p = find_predictor(name)
    if p:
        return p
    raise SystemExit(
        "cannot find %s in any of:\n%s\n\n%sThe published set lives in "
        "weights/ and is committed; if it is missing your clone is "
        "incomplete (`git checkout -- weights/`). Checkpoints that are NOT "
        "published -- rch_*.pt, per-epoch snapshots -- have to be built or "
        "pulled from Betty."
        % (name, "\n".join("  " + d for d in _PREDICTOR_DIRS),
           (why + "\n\n") if why else ""))


def describe(path: str) -> str:
    """A short label saying which of the three directories a path came from."""
    for d, label in ((BETTY, "betty_pull"), (LOCAL_CKPT, "proj1/checkpoints"),
                     (WEIGHTS, "weights")):
        if os.path.abspath(path).startswith(os.path.abspath(d)):
            return label
    return "custom"
