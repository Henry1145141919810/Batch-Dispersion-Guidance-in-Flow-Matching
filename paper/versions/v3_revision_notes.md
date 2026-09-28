# Why the paper was revised

Manuscript v3 | 27 September 2026 | Revision from the live v2 draft

## 1. Scope and editorial decision

BDG is the sole proposed innovation in the current paper. BTVG is absent from the new manuscript and comparison tables. Earlier paper versions, experimental code, and pilot records remain available for traceability. This revision changes the paper; it does not delete historical experiments or launch new ones.

The central question is now consistent throughout: does feedback on the batch dispersion of endpoint properties improve band coverage at comparable quality beyond ordinary guidance-strength adjustment? The method is defined, but its final empirical benefit remains unresolved.

The user's instruction that the data are not yet available governs the evidence status. Existing pilot findings are not presented as final results. This includes unfavorable findings: a pilot null cannot establish that the final BDG comparison will also be null. Every final result cell uses P, explicitly defined as pending, and the abstract and discussion make no completed-performance claim.

The assignment and template supply the required structure, evidence, and page limit. The Chatterjee Lab guide supplies writing and formatting rules. They are reference requirements for this revision, not authorization to run experiments, submit the assignment, or modify unrelated work.

## 2. Main changes and reasons

### A. Rebuilt the abstract and contribution statements

The previous abstract asserted a coverage null, measured spread ordering, and transfer to DNA. Those statements combined pilots with incomplete final comparisons. The new abstract names the problem, both datasets, the guidance mechanism, BDG, provisional backbone choice, and pending empirical questions. Its 211 words fall within the template's approximate 200-250-word range. Contributions describe the method, derivation, implementation, and evaluation design already available.

### B. Restored the required argument structure

The introduction contains exactly three paragraphs followed by four contribution bullets. Related Work uses four methodological paragraphs covering continuous models, nearby guidance mechanisms, molecular comparators, and sequence comparators. All seven Methods and seven Results subsections remain. Discussion interprets the method's scope without predicting the experiment's outcome.

### C. Separated final protocol from pilot configurations

The old draft mixed one-seed, 256-sample pilot results and best-strength selections with the final molecular protocol. The revision follows FULL_RUN_V3_PROTOCOL.md: q50 targets, nominal strength w=1, batch 500, three sampling seeds, and an operator-selected sample count divisible by 500. It removes the asserted n=5,000 commitment and the old chemistry-floor exclusion. All arms remain visible beside chemistry metrics.

### D. Corrected Modality 2 to the current dataset

The old paper used the 200-base-pair, 6,126-sequence Kenyon-cell setup. The current protocol and shipped model identify DeepFlyBrain at 500 base pairs, with 83,722/10,505/10,434 retained train/validation/test sequences. Four ambiguous training examples were removed. The new paper describes this current setup and explicitly leaves final sequence timing, batch size, seeds, and comparator alignment unresolved.

<!-- PAGEBREAK -->

## 3. Technical corrections that affect the scientific claim

### E. Corrected the setpoint definition

The previous ablation caption defined the multiplier relative to the unguided batch's measured spread. The implementation and final protocol use tau = tau_mult times f_A.y_std, the guide's training-property scale. The new definition matches the code. Tau is a property standard deviation; t is sampling time; delta is the evaluation tolerance. These are different quantities.

### F. Corrected the widening condition

The previous prose said variance below the setpoint expands the batch. A negative residual only reduces the deviation gain. Net repulsion requires w_eff = 1 + eta e < 0, equivalently eta > 1 and V_b < tau^2(1 - 1/eta). The distinction matters because weaker contraction is not the same as expansion. Realized spread additionally depends on the generator, Jacobian, clipping, and projection.

### G. Bounded the algebraic reduction

BDG's numerator is affine in the predicted property. Its effective-target factorization is defined only when w_eff is nonzero. The direct numerator remains finite at zero gain. The revision states that boundary and removes any implication that factorization proves a final performance null. The stationary-variance calculation is confined to an explicitly simplified property-space ODE in the appendix; it is not a convergence guarantee for the implemented sampler.

### H. Corrected the identity controls and implementation description

Eta=0 recovers plug-in guidance exactly. A one-sided residual recovers plug-in only when its added term vanishes; requesting a large setpoint does not guarantee that condition at every step. Pseudocode now makes coefficient detachment explicit during the vector-Jacobian product. Differentiating through those coefficients would implement a different update. Batch size is identified as part of the variance estimator, and uncertainty must respect coupled generation batches.

### I. Narrowed the efficiency and geometry claims

BDG has the same neural-call pattern as plug-in guidance, plus batch reductions. This supports a call-count claim, not an assertion of identical runtime. On DNA, clamping and renormalization enforce simplex feasibility. Multiplying a guidance field by a scalar does not by itself establish that the update is tangent or remains feasible.

### J. Made comparison fairness explicit

The trained VP comparison is separate from borrowed EDMsecond. Flow matching remains a development choice until the matched comparison supports it. External EquiFM uses a different guide/evaluator pair and acceptance width; band-coverage values cannot be pooled across backends. Equal nominal strength does not equalize applied force. Adapted DPS/TMPD/LGD-MC/TFG rules are labeled as adaptations, and planned sequence-model comparisons are not labeled as completed reproductions.

### K. Applied the lab's presentation rules

The revision uses direct statements, evidence-matched tense, defined symbols, and bounded novelty language. It removes em dashes, rhetorical claims, and large author-instruction boxes from the paper. The overview is readable and consistent with the provisional model choice. Manuscript prose, tables, figure, algorithm, and appendix now live in one main.tex; body.tex is a compatibility notice. Template fonts, margins, and spacing are retained.

### L. Repaired and checked references

The active literature was checked against primary source records. Dirichlet FM now cites its ICML proceedings record, EquiFM's title is corrected, and the QM9 dataset citation is added. Invalid percent-comment lines inside BibTeX entries were removed so the bibliography compiles. MOG-DFM uses one citation key in the active manuscript. This is a focused verification of cited work, not a claim of an exhaustive novelty search.

<!-- PAGEBREAK -->

## 4. Rubric map and remaining evidence

The paper/code rubric totals 7.5 points. The separate defense also carries 7.5 points and is outside this paper revision. The table below maps requirements to the revised manuscript; it is not a predicted grade. A populated section or a named baseline does not substitute for empirical evidence.

| Rubric item | Points | Addressed in v3 | Still required |
| --- | ---: | --- | --- |
| Abstract | 0.225 | Compact problem, method, modalities, status | Strongest final results and conclusion |
| Introduction | 0.525 | Three paragraphs, hypothesis, four bullets | Align bullets with completed findings |
| Related Work | 0.425 | Closest mechanisms and modality baselines | Final comparison fidelity checks |
| 3.1 Data | 0.300 | States, splits, preprocessing, conditioning | Freeze final sequence protocol |
| 3.2 Flow matching | 0.475 | Path, loss, architecture, training, sampler | Final checkpoint identifiers |
| 3.3 Diffusion | 0.475 | VP process, loss, ODE, configured budget | Confirm training and matched evaluation |
| 3.4 Guidance | 0.300 | Energy, gradient, schedule, clip, frozen parts | Final guided/unguided evidence |
| 3.5 Innovation | 0.625 | BDG equation, limits, controls, pseudocode | Causal ablation measurements |
| 3.6 Transfer method | 0.200 | Same coefficient rule, new state/property | Final sequence run settings |
| 3.7 Overview | 0.200 | Updated two-modality figure | Update selection when measured |
| 4.1 Evaluation | 0.400 | Metrics, pairing, seeds, batch, costs | Actual n, manifests, uncertainty |
| 4.2 Base comparison | 0.450 | Matched FM/VP table | Results and justified model selection |
| 4.3 Guidance | 0.325 | Target and chemistry table | Paired improvement or failure analysis |
| 4.4 Ablations | 0.675 | Gain/setpoint/strength and other controls | Results and mechanism attribution |
| 4.5 Recent methods, M1 | 0.450 | Four named guidance adaptations | At least three meaningful comparisons |
| 4.6 Transfer, M2 | 0.400 | Three named external sequence models | Aligned runs and transfer analysis |
| 4.7 Failure modes | 0.200 | Predictor/decoder mismatch, clip/projection | Measured cross-modality interpretation |
| Discussion | 0.325 | Bounded mechanism and limitations | Evidence-based final conclusion |
| Code/documentation | 0.525 | Source map and paper checking utilities | Complete reproducible experiment record |

The largest unresolved evidence block is Results (2.9 points). The matched base-model choice, innovation ablations, and three recent comparisons on each modality are substantive requirements. Naming baselines, citing their published scores on different tasks, or inserting pilot values cannot complete them. The revised draft preserves places for this evidence without awarding itself credit for pending work.

<!-- PAGEBREAK -->

## 5. How to update the paper when results arrive

1. Verify run identity before copying any values: checkpoint, backend, guide/evaluator pair, target, band width, sample count, batch, solver, window, strength, and seeds.
2. Fill the matched FM/VP comparison and justify the chosen family. Keep external EDM/EquiFM comparisons separate from that requirement.
3. Fill guided versus unguided, then BDG versus plug-in. Report property control and quality together, including negative or mixed outcomes.
4. Analyze the ablation grid at each strength separately. Use more than two setpoints for a monotonicity claim. Check clipping and gain diagnostics before attributing a difference to feedback.
5. Complete the sequence protocol and external comparisons. Fix length, conditioning, target, acceptance width, decoder, reference data, seeds, and budget before comparing outcomes.
6. Report paired seed differences and uncertainty that respects BDG batches. Do not use coupled molecules as independent replicates or present sampling-seed variability as training-seed variability.
7. Replace the pending abstract/discussion statements with supported findings. A null, trade-off, or mechanism-only result is acceptable to report accurately.
8. Recompile and inspect the final result after replacing P cells. The current five-page layout does not guarantee that expanded per-property tables will still fit.

## 6. Source record

Local requirements: CIS_6270_Fall_2026_Project1_Assignment.pdf; CIS_6270_Fall_2026_Project1_Paper_Template.pdf; results/chatterjee_lab_scientific_writing_master_guide (1).md.

Protocol sources: docs/protocol/FULL_RUN_V3_PROTOCOL.md; ABLATION_V3_PROTOCOL.md; MODALITY2_V3_PLAN.md. Method correspondence: proj1/src/guidance.py, proj1/src/diffusion.py, proj1/scripts/train_diffusion.py, and proj1/m2/m2_sweep.py. Sequence provenance: proj1/m2/blade_bundle/HANDOFF.md, REPORT.txt, train_blade.sh, and train.log.

Primary literature checks include [Moment Guided Diffusion](https://arxiv.org/abs/2602.17211), [Variance-Tilted Diffusion](https://arxiv.org/abs/2606.22239), [Particle Guidance](https://arxiv.org/abs/2310.13102), [Dirichlet FM](https://proceedings.mlr.press/v235/stark24b.html), [Fisher Flow](https://arxiv.org/abs/2405.14664), [MOG-DFM](https://arxiv.org/abs/2505.07086), [EquiFM](https://arxiv.org/abs/2312.07168), and [QM9](https://www.nature.com/articles/sdata201422). These records support the literature descriptions and metadata corrections, not project performance claims.

## 7. Version tracking and validation

The v1 and v2 numbered snapshots are preserved. The exact live files found before this edit, including PDF, source, bibliography, style, and figures, are archived in versions/v2_pre_v3_2026-09-27/. The new release is manuscript v3; experiment protocol v3 is a separate identifier that happens to share the number.

The v3 release includes a numbered PDF and a self-contained source snapshot, plus these revision notes as Markdown and PDF. The editable manuscript is paper/main.tex. The built-in preview compiler could not initialize in this environment; the existing local LaTeX installation generated the checked PDF. Final validation covers page count, citations, references, float references, terminology, and visual layout. Results remain intentionally pending.
