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
