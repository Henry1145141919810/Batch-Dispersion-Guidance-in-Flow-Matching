# The deck contract

Everything here is a grading requirement or a decision already made. Follow it exactly.

---

## 1. Slide sequence (mandatory, in this order)

Each item below is graded separately. The label in brackets must appear in the
**top-right corner** of every content slide. More than one slide per item is allowed,
and the suggested counts below fit 15 minutes.

### Part 1: Set Up the Study (1.3 pts)

| Label | Item | Slides | Points |
|---|---|---|---|
| **1a** | Frame the study: problem, central question, both modalities and their state spaces | 1 | .1 |
| **1b** | Data and representations: datasets, splits, leakage, representations, conditioning, modality-specific handling | 2 | .275 |
| **1c** | Flow-matching baseline: formulation, architecture/training, model selection, sampling | 2 | .275 |
| **1d** | Diffusion baseline: same four items | 2 | .275 |
| **1e** | Comparison plan: metrics (with limitations), fairness | 2 | .275 |
| **1f** | Study overview figure | 1 | .1 |

### Part 2: Build the Evidence for Modality 1 (2.25 pts)

| Label | Item | Slides | Points |
|---|---|---|---|
| **2a** | Choose the base model: the §4.2 table, and which model is carried forward and why | 1 | .275 |
| **2b** | Guidance strategy, then the §4.3 table | 2 | .3 |
| **2c** | Introduce the innovation: motivate it, specify it, show it working | 3-4 | .475 |
| **2d** | Test it: the §4.4 ablation table, then the conclusions | 2 | .45 |
| **2e** | Position against recent work: the §4.5 table, then fair study and fair discussion | 2 | .75 |

### Part 3: Test Transfer and Close (0.95 pts)

| Label | Item | Slides | Points |
|---|---|---|---|
| **3a** | The transferred method: what stays, what changes, training, sampling | 1 | .3 |
| **3b** | Transfer results: the §4.6 table, fair study, fair discussion | 1 | .275 |
| **3c** | Does the innovation transfer? Direction and magnitude across modalities | 1 | .1 |
| **3d** | Limitations and failure modes: at least two, meaningful | 1 | .1 |
| **3e** | Conclude: back to the hypothesis, mechanism, trade-offs, next experiments | 1 | .175 |

Then a **Bibliography** slide (or two). Title slide and bibliography carry no label.

**Total: about 26 content slides plus title and bibliography.**

---

## 2. Formatting rules (graded)

**Labels.** `1a` through `3e`, top-right corner, on every content slide. Make them
clearly visible, not a faint watermark.

**Citations.** Numbered, in a consistent style, on the slide itself
(for example "Diffusion models are great [1]"). The matching sources go in a **very
small font at the bottom-right** of the same slide, for example
"[1] Hoogeboom et al. 2022". Numbering may restart on each slide. Citations are
required for every intellectual contribution **and every image taken from elsewhere**.
Any uncited figure must be original.

**Bibliography.** A slide at the end, consistent style, order does not matter.

**Pending numbers.** Render every `[TODO: ...]` marker in **red**, visible. They mark
numbers the in-flight run has not produced. See `08_PLACEHOLDERS.md`.

---

## 3. Design guidance

- **16:9**, clean and readable from the back of a room. Tables are the content here, so
  favour legible tables over decoration.
- Every table slide should carry **one takeaway line** so the point survives if the
  audience cannot read every cell.
- Bold the numbers that matter. Do not bold everything.
- The two figures are the only images. They are vector PDFs and 300 dpi PNGs.
- Keep each slide to one idea. If a slide needs more than about 45 seconds, split it.

---

## 4. The claim the deck must carry, consistently

**Central claim.** At a centred target, band coverage is limited by the batch
dispersion of the guide's endpoint prediction rather than by centring error. Servoing
the *deviation* gain of a property-gradient guidance term against a dispersion setpoint
turns that limit into a directly requested quantity at no additional sampling cost, and
the same controller transfers unchanged to the probability simplex.

**And the honest second half:** it does **not** raise band coverage over a tuned
plug-in baseline, and we can say why.

Every slide should advance that claim or support it. Nothing should contradict it.
