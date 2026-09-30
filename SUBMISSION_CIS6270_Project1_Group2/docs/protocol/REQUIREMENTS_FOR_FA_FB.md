# Wanted: a matched pair of QM9 property predictors, $f_A$ and $f_B$

**18 September 2026.** Circulate this to anyone who might have, or be willing to train, what we
need. It states the requirements precisely enough to judge a candidate without a conversation.

**Short version.** We need **two** property predictors for QM9, trained on **disjoint halves** of
the data, each predicting **several properties**, each **differentiable with respect to atomic
coordinates and atom types**. One steers generation; the other grades it. The disjointness is the
whole point and is not negotiable.

---

## 1. Why two, and why disjoint

We do property-targeted molecular generation. At inference time a predictor $f_A$ supplies the
gradient that steers sampling toward a target property value. A second predictor $f_B$ then scores
the results.

Guidance is an optimiser pointed at whatever network it can see. If the same network both steers
and scores, a good score may only mean the optimiser found that network's blind spot. Training the
two on **disjoint data** decorrelates their errors, so the disagreement $\lvert f_A - f_B\rvert$
becomes a usable reward-hacking detector.

This is not hypothetical. On our first guided run the two agreed to **0.197 D** on real molecules
and disagreed by **4.597 D** on guided samples — guidance had walked into a region where the guide
was confidently wrong. With a single shared predictor that run would have looked like a success.

**Therefore:** a predictor that is more accurate but shares training data with its partner is
**worse than useless to us.** Accuracy we can live without; the independence we cannot.

---

## 2. Hard requirements

A candidate must satisfy all of these.

| # | Requirement | Why |
|---|---|---|
| 1 | **Two separate models** trained on **disjoint** molecule sets | the entire diagnostic |
| 2 | Differentiable w.r.t. **atomic coordinates** and **atom-type features**, gradients available through a standard autodiff framework | guidance needs $\nabla f$ at every sampling step |
| 3 | **Twice**-differentiable — Hessian-vector products must work | our method needs $\operatorname{tr}(H\Sigma)$; a network with non-differentiable activations or detached branches is unusable |
| 4 | Takes **3D coordinates + atom types**, no bond graph, no SMILES | generated intermediates have no bonds; they are point clouds |
| 5 | **Invariant** to rotation, translation and atom permutation | the generator works on the zero-centre-of-mass subspace; a frame-dependent predictor injects spurious gradients |
| 6 | Handles **variable atom count** via a padding mask, with padded atoms contributing exactly zero to output *and gradient* | phantom atoms would otherwise be pushed around by guidance |
| 7 | Trained on **clean molecules only** | the method evaluates it at an estimate of the clean molecule; a noise-conditioned predictor would absorb the effect we are trying to measure and **invalidate the experiment** |
| 8 | Weights loadable, architecture inspectable | we freeze it and differentiate through it |

### Strongly wanted

| # | Want | Why |
|---|---|---|
| 9 | **Multiple properties**, ideally dipole $\mu$, polarisability $\alpha$, HOMO–LUMO gap | our effect scales with property **curvature**; one property gives one data point, three give a trend |
| 10 | $f_B$ with a **different architecture** from $f_A$ | disjoint data decorrelates *data* blind spots, not *inductive-bias* blind spots. Same architecture on both sides means a shared failure mode stays invisible |
| 11 | Validation MAE on $\mu$ **below 0.09 D** | our current pair sits at 0.0897 / 0.0840; anything worse is a downgrade |
| 12 | Reported MAE on a held-out set neither model trained on | so we can set the acceptance tolerance $\delta$ from real error |

---

## 3. The interface we need

Whatever the internals, we need to be able to call it like this:

```python
value = f(coords, feats, mask)      # coords [B,N,3] float, angstrom, zero centre of mass
                                    # feats  [B,N,K] float, one-hot atom type (H,C,N,O,F)
                                    # mask   [B,N]   float, 1 = real atom, 0 = padding
                                    # value  [B]     or [B,P] for P properties, PHYSICAL units
```

with `torch.autograd.grad(value.sum(), (coords, feats))` returning finite gradients, and a second
differentiation through those gradients also working. We can write an adapter around a different
calling convention; we cannot work around a missing derivative.

**Units must be physical and stated** — Debye for $\mu$, and the normalisation constants supplied
if the model predicts standardised values.

---

## 4. What disqualifies a candidate

- **Trained on the full QM9 training set.** Nearly every public checkpoint is, which means it
  overlaps whatever half its partner used. This is the usual reason a candidate fails.
- **Graph-input only** (needs bonds or SMILES). Our intermediates are point clouds with no bonds.
- **Non-differentiable anywhere on the path** from coordinates to output.
- **Noise-conditioned or trained on corrupted inputs** — see requirement 7. This one silently
  destroys the measurement rather than failing loudly, so it matters most.
- **Ensembles** where members share training data, or where the interface hides the gradient.

---

## 5. How we will verify a candidate

Anything offered gets put through these before use. They take minutes.

1. **Invariance:** random rotation, translation and atom permutation change the output by less than
   $10^{-6}$ relative.
2. **Padding:** garbage in padded slots changes neither output nor gradient; padded gradient rows
   are exactly zero.
3. **Gradient finiteness:** no NaN or Inf in first or second derivatives, including for molecules
   with coincident or near-coincident atoms.
4. **Curvature is nonzero:** $\operatorname{tr}(H)$ measurably different from zero. A near-linear
   predictor is unusable for us regardless of its accuracy.
5. **Independence:** the two models' errors on a common held-out set should be **substantially
   less correlated** than two models trained on the same data. We will measure this; a correlation
   near 1 means the pair is effectively one model.
6. **Accuracy:** MAE on our validation split, in physical units.

---

## 6. Our fallback, for calibration

If nothing suitable turns up we train our own, and it is not expensive — so please do not spend
days searching.

- E(n)-equivariant GNN, 429,825 parameters, hidden 128, 4 layers
- $f_A$ on 51,527 molecules, $f_B$ on 51,527 **disjoint** molecules
- 120 epochs, **23 minutes each** on one GPU
- Validation MAE: $f_A$ 0.0897 D, $f_B$ 0.0840 D on $\mu$, against a 1.53 D chance baseline

So the bar is: **better than 0.09 D, or multi-property, or a genuinely different architecture for
$f_B$ — while keeping the halves disjoint.** A candidate that only beats us on accuracy but shares
data is a step backwards.

---

## 7. What to send

- the two checkpoints, or a link
- the **exact molecule indices or IDs** each model trained on, so we can verify disjointness
  ourselves rather than take it on trust
- the architecture definition or loading code
- properties predicted, with units and any normalisation constants
- reported MAE and the split it was measured on

Questions to **<email redacted>**.
