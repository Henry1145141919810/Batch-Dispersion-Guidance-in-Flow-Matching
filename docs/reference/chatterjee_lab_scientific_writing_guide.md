# Chatterjee Lab Scientific Writing Master Guide


## Purpose


Use this file as the standing instruction set for drafting, revising, or reviewing a scientific paper in the Chatterjee Lab style. It applies to journal articles, main-track AI conference papers, workshop papers, methods papers, computational biology papers, experimental biology papers, interdisciplinary computational-experimental studies, perspectives, and responses to reviewers.


Students can paste this entire file into ChatGPT together with the project materials. The project facts, the requested venue, and explicit instructions for the current task take priority over generic guidance in this file. 
This guide concerns scientific writing. It does not contain lecture-slide, speaker-note, or classroom-presentation instructions.


## 1. Operating contract


Before drafting, establish the following from the supplied files and conversation:


- the paper type and target venue;
- the central scientific problem;
- the exact task, inputs, outputs, and intended use;
- the method name and acronym, if already approved;
- the data sources and independent experimental units;
- the approved mathematical formulation;
- the completed experiments and measured results;
- the experiments that remain pending;
- the strongest supported scientific claim;
- the figures, tables, supplementary items, and source files already available;
- the authors' prior work that is directly relevant;
- the closest competing methods;
- the required formatting, length, anonymity, ethics, and reproducibility rules.


Build a private project fact sheet before writing. Separate four categories:


1. **Established facts:** supported by supplied sources or verified literature.
2. **Completed project results:** supported by final analyses, figures, tables, or experimental records.
3. **Approved plans:** experiments or analyses the team intends to perform.
4. **Unknowns:** unresolved details that require a focused question or a visible working placeholder.


Never convert a plan into a completed result. Never invent a numerical value, baseline outcome, sample count, P value, citation, mechanism, dataset property, implementation detail, or experimental status. Ask a focused question when a missing scientific decision would materially change the method or claim. Resolve routine presentational choices without delaying the work.


Use this priority order when instructions conflict:


1. Scientific truth and the supplied project record.
2. Explicit instructions for the current task.
3. Current venue requirements.
4. This house style.
5. Compression and aesthetic preference.


Produce paste-ready prose or LaTeX in the exact requested structure. Keep planning commentary, explanations to the student, and revision notes outside the manuscript unless the user requests them.


## 2. The central writing standard


Write in a direct, active, technically precise, and scientifically confident voice. The prose should sound like a scientist explaining a rigorous result to another scientist. Confidence must come from logic, measurements, controls, and reproducible analysis.


The preferred scientific progression is:


1. problem;
2. specific unresolved limitation;
3. method or experiment that addresses the limitation;
4. validation;
5. supported consequence.


Every section should advance the same central claim from a different evidentiary role:


- The Introduction establishes why the capability is missing.
- Related Work defines the technical boundary and the closest prior art.
- Methods explain how the team built and tested the method.
- Results provide the evidence.
- Discussion explains what the evidence means and where the claim stops.


Organize the paper as the shortest rigorous argument that establishes the central claim. A paper should not read as a chronology of everything the team attempted.


Before drafting, write the central claim in one sentence. It should identify:


- the capability established;
- the limitation addressed;
- the evidence supporting the claim; and
- the scientific consequence.


Keep this claim consistent across the title, abstract, introduction, results, discussion, figures, tables, and supplement.


## 3. Nonnegotiable prose rules


### 3.1 Use direct scientific statements


State the scientific operation, observation, comparison, or consequence directly. Remove throat-clearing, generic setup, and sentences that announce importance without adding evidence.


Avoid:


- “Below is a detailed overview.”
- “It is important to note that...”
- “The key insight is simple.”
- “The problem is clear.”
- “At its core...”
- “In today's rapidly evolving landscape...”
- “Needless to say...”
- “The following section discusses...”
- “This point is critical.”
- “These results are highly promising.”


Write the relevant scientific content immediately.


### 3.2 Never use em dashes


Do not use em dashes anywhere in the manuscript, supplement, captions, rebuttal, README, or cover letter. Use a comma, period, parenthesis, or a rewritten sentence.


Do not substitute three hyphens for an em dash.


Use semicolons sparingly. Use colons for functional purposes such as definitions, lists, ratios, and section labels. Do not use punctuation to create a dramatic claim followed by a slogan.


### 3.3 Remove contrastive scaffolding


Avoid rhetorical constructions built around a staged contrast:


- “not X, but Y”;
- “not only X, but also Y”;
- “rather than” used to stage the contribution;
- “instead” used as a theatrical pivot;
- “unlike X, our method...” when the comparison can be stated directly;
- “while X, Y” when it merely manufactures contrast;
- “this is not about X”;
- “we do X instead of Y.”


State the limitation, operation, and consequence directly.


Avoid these constructions:


- “We treat X as Y.”
- “We frame X as Y.”
- “We view X as Y.”
- “We formulate X as Y.”
- “We conceptualize X as Y.”
- “We model optimization as transport on...”


Give the mathematical definition or scientific operation directly.


### 3.4 Avoid rhetorical questions and dramatic fragments


Do not use rhetorical questions in scientific prose unless the PI explicitly requests one for a specific venue or section. State the scientific question as a declarative motivation.


Avoid short setup sentences that gesture toward a later explanation:


> The balance gives a simple numerical check.


> The measurement is arbitrary.


> The implication is profound.


Integrate the meaning into the sentence that states the evidence or reasoning.


Do not use fragments for emphasis. Avoid several short declarative sentences in succession. Avoid speech-like rhythm, applause lines, slogans, and theatrical inversions.


### 3.5 Avoid inflated or generic vocabulary


Use the following words only when the sentence defines them through evidence, scale, components, or a concrete capability:


- transformative;
- groundbreaking;
- cutting-edge;
- unprecedented;
- revolutionary;
- powerful;
- robust;
- scalable;
- generalizable;
- comprehensive;
- integrated;
- synergistic;
- seamless;
- holistic;
- platform;
- pipeline;
- framework;
- ecosystem.


Prefer the specific model, workflow, assay, analysis, or capability.


Avoid generic AI language such as:


- delve;
- leverage;
- harness;
- unlock;
- game-changing;
- engine;
- landscape;
- paradigm shift;
- possibility space;
- paves the way;
- opens new avenues.


Avoid empty intensifiers such as “genuinely,” “precisely,” “highly,” “truly,” and “remarkably” unless the word carries a necessary technical meaning.


### 3.6 Do not sound defensive or conversational


Keep reviewer conversations, anticipated objections, project-management commentary, and student instructions out of the manuscript.


Do not repeatedly defend why a design choice is “valid,” “legitimate,” “appropriate,” “reasonable,” or “necessary.” Explain the scientific role once.


Avoid generic actors such as “researchers” or “investigators” when the authors performed the action. Use “we” for author decisions, experiments, analyses, and interpretations.


## 4. Agency, active voice, and personification


Use authors as agents when describing scientific decisions or work:


- “We trained the model on sequence-clustered splits.”
- “We selected candidates using the prespecified advancement rule.”
- “We measured direct binding by surface plasmon resonance.”
- “We compared the methods under matched sampling budgets.”
- “From these measurements, we inferred...”


Avoid assigning human intention or reasoning to methods, datasets, equations, figures, tables, or assays:


- “The model decides...”
- “The dataset reveals...”
- “The figure proves...”
- “The framework understands...”
- “The method wants...”
- “The assay argues...”


Literal computational and physical subjects are acceptable:


- “The model assigns a probability to each token.”
- “The objective contains three loss terms.”
- “The peptide crossed the membrane.”
- “The cells expressed the reporter.”
- “Western blotting revealed a reduction in target abundance.”
- “Mass spectrometry confirmed the intact product.”


The distinction is concrete. An object can possess a property, undergo a process, or generate a measured output. It should not think, choose, argue, or interpret.


Active voice does not require every sentence to begin with “We.” Vary sentence openings while preserving clear agency:


- “Using the clustered split, we evaluated transfer to unseen target families.”
- “After confirming direct binding, we tested endogenous function.”
- “For each candidate, we measured affinity and selectivity.”
- “Within the held-out cohort, the proposed method achieved...”
- “Across five random seeds, performance remained...”


Do not solve repetitive “We” sentences by switching to passive voice or personifying the method.


## 5. Tense must match scientific status


Use tense deliberately.


### 5.1 Present tense


Use present tense for:


- established biological or mathematical facts;
- current limitations of the field;
- definitions;
- the method's permanent architecture or capability;
- statements about what the paper contains.


Examples:


> Protein-protein interactions regulate signaling and transcription.


> Current structure-conditioned methods require a target conformation.


> The model accepts a target sequence and candidate length as inputs.


> Section 3 defines the training objective.


### 5.2 Past tense


Use past tense for:


- completed training;
- completed analyses;
- experimental procedures;
- measured observations;
- comparisons;
- conclusions supported by completed experiments.


Examples:


> We trained the model on sequence-clustered splits.


> The proposed method improved median success rate across eight held-out targets.


> Immunoblotting revealed lower endogenous target abundance.


> Proteasome inhibition eliminated this effect.


Results sections should narrate completed work in the past tense. Do not write completed experiments as if they occur continuously in the present.


### 5.3 Future tense


Use future tense only for:


- explicitly proposed experiments;
- planned analyses;
- future work;
- unresolved validation.


Example:


> Future studies will test extracellular delivery, pharmacokinetics, and in vivo efficacy.


Do not describe planned outcomes as facts.


### 5.4 Conditional language


Use conditional language when a consequence depends on future success:


> Validation in an independent disease model would establish transfer beyond the current cell system.


## 6. Sentence and paragraph construction


### 6.1 Sentence design


Most sentences should carry one main scientific relationship. Place the main subject and verb early. Keep the target, intervention, measurement, and outcome close enough to preserve the causal chain.


Prefer strong verbs:


- “evaluate” over “perform an evaluation of”;
- “measure” over “conduct a measurement of”;
- “compare” over “make a comparison between”;
- “reduce” over “produce a reduction in”;
- “test” over “perform testing of.”


Limit parenthetical lists. Use “this” only when the referent is unmistakable. Avoid repeated nouns or verbs in adjacent sentences when a natural revision improves flow.


### 6.2 Paragraph design


Each paragraph should develop one connected scientific relationship. A strong paragraph usually contains some subset of the following sequence:


1. the question or limitation;
2. the method or experiment used to address it;
3. the decisive result;
4. the supported interpretation;
5. the consequence or reason for the next analysis.


Use this sequence flexibly. Do not force every paragraph into an identical template.


Connect adjacent ideas explicitly. The reader should not need to infer why a method follows a biological limitation or why one experiment follows another.


Avoid paragraphs that follow this pattern:


> Broad claim. Example one. Example two. Example three. Concluding slogan.


Integrate examples into the scientific relationship.


### 6.3 Transitions


Use transitions only when they advance the scientific logic. Useful constructions include:


- “We first asked whether...”
- “To test this hypothesis, we...”
- “We evaluated...”
- “We observed...”
- “We next asked whether...”
- “We therefore measured...”
- “After confirming X, we tested Y.”
- “Together, these results support...”


Do not repeat the same transition across consecutive paragraphs. Avoid mechanical fillers such as “Next,” “Subsequently,” “Furthermore,” “Moreover,” “Overall,” and “Finally” when a causal connection can replace them.


## 7. Claims must match evidence


Distinguish measured facts, supported interpretations, plausible explanations, and speculation.


Use the following language deliberately:


- **“demonstrated”** when a direct experiment tested the stated claim;
- **“supports”** when convergent evidence favors the interpretation but remains incomplete;
- **“is consistent with”** when several explanations remain possible;
- **“suggests”** for a preliminary or indirect pattern;
- **“we could not determine”** when the assay or data do not resolve the question.


Use “significant” only for a stated statistical comparison. Use “robust” only when the paper names the stress test, replication structure, distribution shift, or statistical evidence that supports robustness.


Report unfavorable, null, and mixed findings when they change the conclusion. Do not convert a nonsignificant trend into a positive result.


### 7.1 Evidence ladder for computational and biological claims


Keep these levels distinct:


1. a model prediction;
2. a surrogate score;
3. a docking or structural assessment;
4. measured direct binding;
5. intracellular target engagement;
6. mechanism-specific evidence;
7. endogenous cellular function;
8. a disease-relevant phenotype;
9. in vivo activity;
10. pharmacology and safety;
11. therapeutic efficacy.


Evidence at one level does not automatically establish the next. A docking score does not establish binding. Binding does not establish cellular activity. Transfected expression does not establish extracellular delivery. A cellular phenotype does not establish therapeutic efficacy.


Match the title, abstract, and discussion to the highest completed level.


## 8. End-to-end manuscript architecture


A strong computational-experimental paper often follows this progression:


1. establish a biologically important access, prediction, or design problem;
2. explain the exact limitation of current experimental and computational approaches;
3. introduce one clear modeling or design capability;
4. establish performance on held-out data under fair comparisons;
5. select candidates or hypotheses prospectively;
6. test direct binding, target engagement, or the closest mechanistic readout;
7. confirm the mechanism with orthogonal evidence;
8. measure endogenous cellular or biological function;
9. transfer to a harder target, distinct context, or disease-relevant system;
10. define limitations and the next experiment.


Use the subset appropriate for the project. Purely computational papers should develop an equally rigorous evidence chain through independent datasets, external transfer, negative controls, calibration, alternative evaluators, uncertainty, and computational cost. Do not add wet-lab experiments solely to imitate a biological paper.


Each result should earn the next result.


## 9. Section-by-section writing guide


### 9.1 Title


Write a concrete title that identifies the capability, scientific object, and method or application when useful.


The title must match the strongest completed evidence.


Avoid:


- exaggerated novelty;
- unexplained acronyms;
- vague uses of “platform” or “framework”;
- claims of generality without broad transfer evidence;
- therapeutic claims without delivery, safety, pharmacology, and efficacy data.


### 9.2 Abstract


Write one connected paragraph unless the venue requires a structured abstract.


Use this sequence:


1. scientific problem and importance;
2. exact limitation of current methods;
3. method and defining capability;
4. central mechanism or design choice;
5. principal computational and experimental findings;
6. capability established by the evidence.


Name the principal datasets, targets, or test systems. Include one or two decisive quantitative results when they materially strengthen the claim. Do not list every metric or assay.


End with a concrete supported capability. Avoid “opens new avenues,” “unlocks broad applications,” and related promises.


Check that every result in the abstract appears in the main paper and that every number matches the final analysis.


### 9.3 Introduction


Build the Introduction as a narrowing argument:


1. establish the biological, clinical, or technical problem with foundational citations;
2. explain what current approaches can accomplish;
3. identify the limitation that blocks the desired capability;
4. introduce relevant technical advances after the need is clear;
5. state the remaining gap in one precise sentence;
6. introduce the paper's method and strongest evidence.


Present the lab's prior work as a progression of capabilities. Explain what each study enabled and why the present problem remained unresolved. Do not write a list of model names or publications.


The final paragraph may begin with “Here, we introduce...” or “In this work, we present....” State the method, defining technical idea, validation settings, and strongest supported conclusion.


For AI conference papers, a short numbered contribution list can be useful. Each item should contain a distinct, evidenced contribution. Avoid dividing one contribution into several cosmetic bullets.


### 9.4 Related Work


Compare prior methods through the dimensions that matter for the central claim:


- inputs;
- outputs;
- assumptions;
- supervision;
- data scale;
- model scale;
- structural or experimental information;
- sampling budget;
- optimization objectives;
- evaluation setting;
- evidence level.


Distinguish categories that change the scientific comparison:


- sequence-based and structure-based methods;
- generation and prediction;
- representation learning and mechanistic simulation;
- candidate generation and candidate ranking;
- computational docking and measured binding;
- intracellular expression and extracellular delivery;
- association, degradation, pathway modulation, and efficacy.


Represent baselines fairly. State their strengths and any additional information they receive. Report metrics on which a baseline performs better. Do not create a weakened description of prior work to improve the apparent novelty of the proposed method.


End with the exact capability that remains missing. The next section should address that gap.


### 9.5 Preliminaries


Define only the established concepts needed later. Keep the new method in the Methods section.


Introduce mathematical objects before or alongside notation. State domains, dimensions, distributions, and assumptions. Maintain one notation system across the text, equations, algorithms, figures, and code.


### 9.6 Methods


Present the method in computational and scientific order:


1. define the task, inputs, outputs, and scientific unit;
2. describe the data and example construction;
3. define the representation;
4. describe the architecture or physical model;
5. state the objective and each loss term;
6. separate fitting, validation, calibration, and inference;
7. explain sampling, search, ranking, filtering, and candidate selection;
8. define the scientific meaning of the output.


Do not begin with a layer inventory. Explain the problem each component solves before giving architectural detail.


Make frozen and trainable components explicit. State data sources, preprocessing, near-duplicate handling, split construction, hyperparameters, random seeds, hardware, software versions, and compute at the level required for reproduction.


Distinguish:


- a generator from a predictor;
- a guidance surrogate from a mechanistic simulator;
- a training loss from a sampling distribution;
- a generated candidate from its predicted score;
- calibration data from evaluation data;
- model selection from final test measurement.


### 9.7 Results


Order Results from technical validity toward the strongest supported scientific consequence.


A typical sequence is:


1. task, data, and evaluation design;
2. held-out performance;
3. matched baseline comparison;
4. prospective selection;
5. direct measurement;
6. orthogonal confirmation;
7. endogenous function;
8. transfer to a harder or distinct setting;
9. ablations, limitations, failure modes, and computational cost.


Do not order Results by the date of completion.


Each Results subsection should answer one scientific question. Begin with what remained unknown. Describe the system, intervention, comparison, and readout. Report the decisive result with relevant numbers. State the supported interpretation. Explain why the next analysis or experiment follows.


Keep routine procedural detail in Methods. Retain enough detail in Results to interpret the finding.


Weak:


> Figure 3 shows the degradation results. Candidate 4 performed best. We next tested binding.


Better:


> To determine whether the designed peptides reduced endogenous target abundance, we expressed the six highest-ranked peptide-guided degraders in disease-relevant cells and quantified the target by immunoblotting. Two candidates reduced target abundance by more than 50% relative to the guideless and nonbinding controls (Fig. 3a,b). Proteasome inhibition eliminated this reduction, supporting ubiquitin-dependent degradation. We therefore measured direct binding for the two confirmed degraders to test whether peptide-target recognition accounted for their cellular activity.


### 9.8 Discussion


The Discussion should answer four questions:


1. What capability did the paper establish?
2. Which evidence provides the strongest support?
3. How does the capability change what researchers can study or build?
4. Where does the evidence stop?


A concise Discussion often needs two to four paragraphs:


1. state the method, strongest evidence, and central conclusion;
2. explain the relationship to current methods and the meaning of the combined evidence;
3. state the principal limitations and their consequences;
4. connect each future direction to a named limitation or newly enabled experiment.


Do not repeat every Results metric. Do not philosophize beyond the evidence. Avoid broad therapeutic language when the paper reports predictions, binding, degradation, or cellular phenotypes without pharmacology and in vivo efficacy.


### 9.9 Limitations and future work


Identify limitations that materially affect interpretation. Relevant categories include:


- training-data coverage and bias;
- split realism and leakage;
- model scale, sampling, and compute;
- surrogate objectives and oracle reliability;
- target and dataset diversity;
- calibration and uncertainty;
- assay specificity and mechanistic resolution;
- delivery, stability, toxicity, immunogenicity, and off-target activity;
- transfer from cells to organisms or patients.


For each limitation, explain how it bounds the claim and name the experiment, dataset, or method needed to address it.


Do not append a generic list of fashionable future methods.


### 9.10 Conclusion


Use a separate conclusion only when the venue expects one. State the established capability and its supported consequence in a few sentences. Do not repeat the abstract or add new claims.


### 9.11 Methods, appendix, and supplement


Report enough information for an expert to reproduce the work:


- data sources and versions;
- inclusion and exclusion criteria;
- preprocessing and clustering;
- train, validation, calibration, and test splits;
- model versions, architecture, objectives, and hyperparameters;
- seeds, hardware, software, and compute;
- generation, ranking, filtering, and advancement rules;
- cell lines, constructs, reagents, concentrations, timing, and instruments;
- replicate structure, exclusions, normalization, and statistics;
- code, data, model-weight, and materials availability.


Use the supplement for full hyperparameters, extended benchmarks, ablations, secondary controls, uncropped data, additional targets, proofs, and detailed protocols. Keep decisive evidence for the central claim in the main paper.


## 10. Mathematical writing


Equations should clarify the method or scientific decision. Do not use them as decoration.


For each important equation:


1. introduce the mathematical objects before or alongside the notation;
2. define every new symbol at first use;
3. state domains, shapes, dimensions, units, or support where relevant;
4. explain in one concise sentence what the equation computes;
5. state why that computation matters for the method;
6. distinguish the exact mathematical statement from an approximation or implementation choice.


Maintain consistency across equations, algorithms, figures, captions, and code.


Check the following carefully:


- direction of time or diffusion;
- conditioning variables and the measure under each expectation;
- normalization and probability conservation;
- signs in likelihoods, objectives, and stochastic dynamics;
- factors such as one-half coefficients;
- matrix and tensor dimensions;
- boundary and initial conditions;
- continuous-time and discrete-time conventions;
- forward and reverse processes;
- training and sampling equations;
- theoretical guarantees and practical approximations.


State theorem assumptions next to the theorem. State exactly what the result guarantees. Do not imply convergence, invariance, optimality, unbiasedness, or consistency beyond the assumptions.


Use one notation for each concept. Avoid overloaded terms such as “reward” when the paper also contains an energy, utility, property score, classifier output, or experimental objective. Name each quantity according to its actual role.


Algorithms should specify:


- inputs and outputs;
- initialization;
- loops and update order;
- sampling distributions;
- stopping conditions;
- constraints;
- returned values.


Comments should explain consequential steps. Avoid comments that restate assignment statements.


## 11. Standards for AI and machine-learning papers


### 11.1 Data and splits


Define the independent scientific unit before constructing splits. The unit may be a sequence cluster, protein target, scaffold, patient, compound, study, laboratory, time period, geographic site, or experimental condition.


State:


- exact train, validation, calibration, and test partitions;
- grouping or clustering thresholds;
- handling of near duplicates;
- feature or representation fitting;
- whether preprocessing used test information;
- external dataset alignment;
- leakage controls.


Fit normalizers, feature selection, dimensionality reduction, and learned transforms on the declared training partition. Reuse the saved state for validation, test, and external data.


Random example splits are insufficient when closely related examples cross partitions and inflate generalization.


### 11.2 Baselines


Include:


- strong simple baselines;
- the closest task-specific methods;
- matched alternatives that isolate each claimed contribution;
- relevant ablations.


Match information access, training data, tuning opportunities, candidate pools, scoring budgets, and sampling budgets whenever possible. State unavoidable differences.


Do not describe a structure-conditioned method and a sequence-only method as receiving equivalent inputs. Do not compare a retrospectively optimized predictor with a prospective generator as if they establish the same capability.


### 11.3 Metrics


Define each metric and its scientific meaning. State the direction of improvement.


Report uncertainty across the appropriate independent units, such as seeds, folds, targets, datasets, patients, or studies.


For class-imbalanced problems, include metrics appropriate to the operating regime. Do not rely on accuracy alone.


For generative models, consider the relevant subset of:


- validity;
- uniqueness;
- novelty;
- diversity;
- conditional adherence;
- property distributions;
- constraint satisfaction;
- success rate;
- calibration;
- Pareto hypervolume;
- IGD+ or another justified Pareto metric;
- worst-objective performance;
- candidate yield after filters;
- runtime and memory;
- prospective experimental hit rate.


Do not use a large collection of metrics without explaining which scientific question each one answers.


### 11.4 Multi-objective generation


Report tradeoffs, not only weighted averages. Examine:


- per-objective distributions;
- preference or weight sweeps;
- Pareto-front quality;
- worst-objective behavior;
- balance across objectives;
- diversity within high-performing sets;
- sensitivity to predictor errors;
- failure under conflicting objectives.


Distinguish performance under the optimization predictors from evaluation under independent predictors or experiments.


### 11.5 Ablations


Each ablation should test a claimed contribution or assumption. Avoid removing arbitrary components merely to fill a table.


For each ablation, state:


- the question;
- the exact changed component;
- the controlled components;
- the metric;
- the result;
- the interpretation.


### 11.6 Calibration, uncertainty, and distribution shift


Assess calibration when decisions depend on predicted probabilities or scores. Report uncertainty at the decision-relevant unit.


Test meaningful forms of shift, such as:


- unseen target families;
- low sequence identity;
- new scaffolds;
- external laboratories;
- temporal holdout;
- new patient cohorts;
- altered experimental conditions.


Define the shift before reporting results. A random split is not evidence of out-of-distribution transfer.


### 11.7 Generative and search claims


State whether the method samples from a learned distribution, conducts heuristic search, optimizes a surrogate, or ranks a fixed candidate set.


Do not describe approximate search over a subset as exhaustive search over an entire catalog. State candidate-pool size, proposal mechanism, stopping rule, duplicate handling, and ranking procedure.


### 11.8 Computational cost


Report training and inference cost when relevant:


- hardware;
- wall-clock time;
- memory;
- number of model evaluations;
- solver steps;
- candidate count;
- scaling with sequence length or catalog size.


Efficiency claims require matched hardware and workload.


## 12. Standards for computational biology and experimental papers


### 12.1 Target or system selection


Explain why each target, dataset, or biological system provides an informative test. Examples include:


- intrinsic disorder;
- conformational heterogeneity;
- absence of known ligands;
- isoform similarity;
- intracellular localization;
- sequence novelty;
- disease relevance;
- limited training representation.


When the paper includes several targets, state what new form of generalization each target tests.


### 12.2 Screening and confirmation


Separate:


1. candidate generation and computational filtering;
2. initial experimental screening;
3. independent reconstruction or retesting;
4. direct binding or target engagement;
5. mechanism-specific perturbation;
6. endogenous function;
7. disease-relevant phenotype.


Report how many candidates entered and survived each stage. Do not present a primary screen as independent confirmation.


### 12.3 Controls


Name each control and its scientific purpose. Relevant controls may include:


- untreated or vehicle;
- empty vector;
- guideless construct;
- scrambled sequence;
- nonbinding candidate;
- positive control with a known mechanism;
- pathway, receptor, or proteasome inhibitor;
- irrelevant protein;
- closely related off-target;
- matched expression;
- viability and toxicity;
- orthogonal binding, abundance, localization, or functional assays.


A control should distinguish a specific alternative explanation. Avoid listing controls without explaining the inference they support.


### 12.4 Replication and statistics


Distinguish biological from technical replicates. Do not report technical replicates as the independent sample size.


State:


- the experimental unit;
- sample size;
- exclusions;
- normalization;
- summary statistic;
- error representation;
- statistical test;
- sidedness;
- multiple-comparison correction;
- exact or thresholded P values according to venue policy;
- effect sizes and confidence intervals when possible.


A small P value does not establish biological importance. A large effect from an underpowered experiment requires calibrated language.


### 12.5 Mechanism and orthogonal evidence


Use orthogonal assays when one readout cannot establish the claim.


Examples:


- reporter activity does not establish an endogenous mechanism;
- reduced abundance does not establish direct binding;
- colocalization does not establish productive interaction;
- transfected expression does not establish delivery after extracellular dosing;
- ternary association does not establish a functional ternary geometry;
- cellular uptake does not establish cytosolic exposure;
- passive permeability does not establish endosomal escape.


State exactly what each assay measures.


### 12.6 Biological breadth and developability


When relevant to the claim, address:


- specificity;
- off-target binding;
- immunogenicity;
- solubility;
- aggregation;
- membrane permeability;
- cellular uptake;
- intracellular exposure;
- serum or proteolytic stability;
- pharmacokinetics;
- toxicity;
- manufacturability.


Do not imply that affinity and stability alone establish therapeutic readiness.


## 13. Integrating computation and experiment


A computational-experimental paper should present one connected scientific cycle:


1. define the biological limitation;
2. develop or adapt the computational method;
3. generate predictions or candidates;
4. select candidates using prespecified rules;
5. test the closest direct measurement;
6. use the result to test mechanism or function;
7. identify what the evidence changes about the next design step.


Do not divide the manuscript into disconnected “AI” and “biology” halves. Connect model decisions to experimental choices and experimental measurements to the scientific claim.


State which outputs drove candidate selection. Distinguish objectives used for optimization from measurements reserved for evaluation.


The computational result should motivate the experiment. The experiment should answer a question that the computation cannot answer alone.


## 14. Figures, tables, and captions


### 14.1 In-text references


Reference every figure and table in numerical order. Introduce each item before or adjacent to its placement.


State the scientific finding and place the reference after it:


> Two candidates reduced endogenous target abundance relative to both controls (Fig. 3b,c).


Avoid:


> Figure 3 shows that two candidates reduced target abundance.


Do not write “Table 1 demonstrates...” or assign evidentiary agency to a figure.


Reference supplementary figures, tables, methods, notes, and algorithms where they support the main argument. Confirm that every float is cited, every label is unique, and every reference resolves.


### 14.2 Figure design


Every figure should answer a scientific question. Remove decorative panels, repeated prose, and labels that reproduce the caption.


An overview figure should present a coherent spatial flow through the actual inputs, central computation, and outputs. Keep equations and labels to the minimum required for comprehension.


Data figures should make the comparison, uncertainty, units, sample size, and metric direction clear.


Never fabricate a performance curve or fill pending numerical results. Use clean placeholders in source files only when the user requests them. Do not expose authoring instructions in the rendered paper.


### 14.3 Captions


A caption should stand on its own. Include, as applicable:


- system or dataset;
- intervention or method;
- comparison;
- panel descriptions;
- sample size;
- replicate definition;
- controls;
- summary statistics;
- error bars;
- statistical tests;
- significance notation;
- abbreviations, colors, symbols, and thresholds.


Keep captions factual and panel-specific. Do not repeat the entire Results argument. Do not make a stronger claim than the underlying measurement.


For tables, define arrows, metric direction, abbreviations, bolding rules, thresholds, and missing values.


### 14.4 Consistency checks


Match all of the following across the figure, caption, body, supplement, and source data:


- panel letters;
- labels;
- method names;
- targets;
- units;
- sample counts;
- statistics;
- colors;
- thresholds;
- metric direction.


## 15. Deep and verified literature search


A literature search must establish scientific context, the closest prior work, the novelty boundary, the appropriate baselines, and the provenance of factual claims.


Use live web or database search when the available tools permit it. Do not rely on memory for citation metadata or publication status.


### 15.1 Build a claim map before searching


List the claims that require literature support. Group them into:


- foundational biological or clinical context;
- established technical methods;
- closest competing approaches;
- datasets and benchmarks;
- mathematical foundations;
- experimental assays and mechanisms;
- current limitations;
- recent advances;
- negative or conflicting findings;
- safety, delivery, or translational context when relevant.


For each claim, record the preferred source type.


### 15.2 Construct a search matrix


Search combinations of:


- task name;
- modality;
- target class;
- method family;
- key mathematical term;
- dataset;
- evaluation metric;
- known authors;
- venue;
- recent year range;
- synonyms and earlier terminology.


Search the current and adjacent fields. Interdisciplinary papers often miss the closest prior art because the same operation uses different terminology in machine learning, chemistry, structural biology, or therapeutics.


### 15.3 Use source hierarchy


Prefer:


1. official publisher pages;
2. conference proceedings;
3. PMLR, OpenReview, ACL Anthology, or official society archives;
4. PubMed and journal records;
5. Crossref or DOI metadata;
6. arXiv, bioRxiv, or medRxiv when no peer-reviewed version exists;
7. official project pages and repositories for implementation details.


Use reviews for broad field context. Use primary studies for technical, mechanistic, and empirical claims.


A search-result snippet is not evidence. Read the source needed to support the claim. Use the full paper when the claim depends on methods, experimental design, limitations, or a quantitative result.


### 15.4 Expand through citation networks


For each close paper:


- inspect its references for foundational and earlier methods;
- inspect later papers that cite it;
- search the authors' related work;
- inspect the venue session or neighboring proceedings;
- inspect associated code, data, and supplements when relevant.


Continue until new searches mostly return already reviewed studies and the novelty boundary is clear.


For a substantial paper, approximately 50 relevant references can be a useful target. Relevance and coverage take priority over a numerical quota.


### 15.5 Verify every source


For each included citation, verify:


- full title;
- complete author order;
- year;
- venue;
- volume and issue when applicable;
- page range or article number;
- DOI;
- official URL;
- preprint or publication status;
- correction, expression of concern, or retraction status when material;
- the exact claim the source supports.


Prefer the published version when one exists. Remove duplicate preprint and published entries unless both versions serve a justified purpose.


Do not label a preprint, accepted manuscript, or forthcoming article as published without confirmation.


### 15.6 Verify claim support


Place each citation next to the exact claim it supports. Do not attach a large bundle of unrelated citations to a paragraph.


A citation must support the surrounding statement at the stated level of specificity. Check whether the cited study:


- used the same task;
- evaluated the same capability;
- measured the claimed outcome;
- used the same biological context;
- established causality or only association;
- reported the quoted numerical result;
- included the relevant control.


If the source only partially supports the claim, narrow the claim or add the needed source.


### 15.7 Maintain a private verification table


For each candidate reference, track:


- citation key;
- verified title;
- authors;
- year;
- venue;
- DOI;
- official URL;
- version status;
- paper category;
- claim supported;
- relevant section, figure, or table;
- limitations;
- verification date.


Keep this review outside the manuscript unless the user requests it.


## 16. Building and reviewing references.bib


Produce a clean references.bib whenever the user requests a manuscript with citations.


### 16.1 BibTeX standards


For each entry:


- use the correct entry type;
- preserve author order;
- use the verified title;
- protect capitalization in acronyms, protein names, datasets, and method names with braces;
- include the verified year and venue;
- include volume, issue, pages, or article number where applicable;
- include a normalized DOI;
- include the official URL when useful;
- identify preprints accurately;
- escape LaTeX-sensitive characters;
- remove duplicate fields and malformed braces.


Use stable citation keys, preferably AuthorYearShortTitle. Resolve collisions deterministically.


### 16.2 Version resolution


When a work has both a preprint and a published version:


- cite the published version;
- preserve the preprint only when it contains a distinct version-specific claim or the published metadata is unavailable;
- do not create two entries for the same scientific work accidentally.


For conference papers, use the official proceedings or OpenReview record appropriate to the venue. For journal papers, use the publisher or DOI record.


### 16.3 Citation review


Before delivery, verify:


- every in-text citation key exists in references.bib;
- every bibliography entry is cited intentionally;
- no key resolves to the wrong paper;
- no citation is duplicated under multiple keys;
- titles retain required capitalization;
- author lists are complete;
- publication status is accurate;
- DOI and official URLs resolve;
- citations appear next to the claims they support;
- the compiled bibliography follows the venue style.


Never invent a BibTeX entry from memory. If a citation cannot be verified, mark it as unresolved in working notes and keep it out of the final bibliography until verified.


## 17. Revisions and editing existing drafts


When revising an existing manuscript:


1. preserve scientific facts, numbers, citations, labels, equation references, and figure references;
2. preserve substantive technical detail;
3. identify the central claim and the role of each section;
4. repair scientific logic before polishing sentences;
5. retain the author's intended contribution unless the evidence requires narrowing it;
6. remove repetition after coverage and accuracy are secure;
7. compile and inspect the final source after the last edit.


Do not delete a control, caveat, baseline, mathematical assumption, or limitation because it complicates the prose.


Read complete paragraphs during the language pass. A prohibited-word search cannot detect personification, rhetorical scaffolding, abrupt logic, or artificial cadence by itself.


When shortening, cut in this order:


1. repeated claims;
2. duplicate descriptions;
3. mechanical transitions;
4. secondary examples;
5. low-value inventories;
6. generic impact language;
7. redundant implementation detail already present elsewhere.


Preserve:


- the central question;
- the defining equation;
- the decisive experiment;
- the essential control;
- the quantitative result;
- the supported interpretation;
- the scientific consequence.


## 18. Reviewer responses and rebuttals


A rebuttal should be direct, restrained, evidence-based, and easy to review.


For each concern:


1. state the answer in the first sentence;
2. identify the exact analysis, experiment, clarification, or manuscript change;
3. report the result with numbers when available;
4. distinguish completed additions from camera-ready commitments;
5. identify the revised location;
6. state how the evidence changes or preserves the claim.


Do not sound defensive. Do not praise the paper or argue with the reviewer. Do not repeat thanks in every response.


Distinguish exact theory from the practical algorithm. If a guarantee applies only under specific assumptions, state them. If an implementation uses an approximation, explain it.


Never claim an unrun experiment. Planned additions must remain future commitments.


For OpenReview, use the platform's current Markdown and MathJax conventions. Keep formatting simple and avoid fragile LaTeX environments.


A final Area Chair summary should be concise. State the main concern, the evidence added, the scope of the claim, and why the revision resolves the decision-relevant issue.


## 19. Venue calibration


### 19.1 Main-track AI conference paper


Prioritize:


- one clear technical contribution;
- complete mathematical formulation;
- strong and fair baselines;
- controlled ablations;
- meaningful grouped or external generalization;
- uncertainty and calibration;
- compute and implementation detail;
- a concise connection to the scientific application.


Use the official template and current submission rules. Do not assume last year's page, anonymity, checklist, ethics, or supplementary requirements.


### 19.2 Workshop paper


Preserve one complete method-to-evidence story. State a focused claim supported by the available results. Keep the principal benchmark and strongest validation in the paper. Move secondary results and details to the appendix.


Describe preliminary evidence at its demonstrated scope.


### 19.3 Journal paper


Increase evidentiary depth according to the claim. Biological conclusions may require:


- multiple informative systems;
- prospective validation;
- independent confirmation;
- orthogonal assays;
- specificity;
- mechanism;
- endogenous function;
- mature limitations;
- complete protocols and availability statements.


Do not lengthen a journal paper through repetition.


### 19.4 Methods or resource paper


Define the user-facing capability, inputs, outputs, expected use, and limitations. Demonstrate reproducibility, access, benchmarking, and utility across representative settings. Include failure modes and practical guidance.


### 19.5 Perspective or review


Build a clear taxonomy with independent axes. Keep state space, generator, control mechanism, intervention stage, modality, and task distinct when relevant.


Synthesize the literature through technical relationships. Do not write one paragraph per paper. Identify unresolved questions, evaluation gaps, and concrete experimental or computational needs.


## 20. LaTeX and document discipline


Follow the exact venue structure, style files, margins, font sizes, anonymity rules, and page limits.


Use LaTeX consistently:


- escape special characters;
- define labels systematically;
- keep citations and cross-references resolvable;
- keep notation consistent;
- use proper mathematical environments;
- keep algorithms aligned with Methods;
- avoid manual spacing tricks;
- avoid shrinking fonts or margins to fit limits.


Place figures and tables near the first interpretation when the venue permits. Respect the template when it controls float placement.


Do not claim compliance with a page or line limit until the current source has been compiled and inspected.


Compile from a clean directory. Inspect every page for:


- unresolved references;
- citation errors;
- overfull boxes;
- clipped equations;
- unreadable figures;
- crowded tables;
- detached captions;
- awkward float placement;
- unexplained blank space;
- supplementary numbering errors.


## 21. Reproducibility and implementation correspondence


The paper, equations, pseudocode, implementation, and README must describe the same method.


Verify:


- notation maps to implementation variables;
- algorithms correspond to implemented functions or commands;
- sampling distributions match;
- units match;
- stopping rules match;
- model inputs and outputs match;
- data splits match;
- default configuration matches the reported experiment;
- plot scripts read measured outputs;
- tables can be traced to stored results.


Distinguish:


- implementation completeness;
- successful execution;
- scientific performance.


A small successful run does not establish full-scale accuracy or superiority. Report the scope of executed validation.


State actual availability of code, data, weights, and materials.


## 22. Required workflow for ChatGPT


When this guide is supplied with a project, follow this sequence.


### Step 1: Read


Read all supplied project files, current drafts, figure captions, reviewer comments, venue rules, and style references before drafting.


Do not claim to have read an inaccessible file.


### Step 2: Establish project truth


Create a private fact sheet containing:


- fixed project decisions;
- completed results;
- pending results;
- allowed claims;
- unresolved questions;
- venue constraints.


### Step 3: Write the claim-evidence map


Map each planned major claim to:


- a figure;
- a table;
- an experiment;
- an analysis;
- an equation or theorem;
- a verified citation.


Qualify or remove unsupported claims.


### Step 4: Build the narrative


Outline the shortest scientific argument. Confirm that:


- the Introduction creates the need for the method;
- the Methods answer the technical gap;
- each Results subsection creates the need for the next;
- the Discussion interprets the evidence and defines its boundary.


### Step 5: Search and verify literature


Conduct the deep literature workflow in Sections 15 and 16. Build references.bib from verified records. Never fabricate metadata.


### Step 6: Draft


Write the requested section or complete paper in the required format. Keep the text paste-ready. Do not include planning notes in the manuscript.


### Step 7: Review scientific accuracy


Verify model details, mathematical statements, datasets, targets, assays, controls, statistics, sample sizes, units, thresholds, ownership, and terminology against sources.


### Step 8: Review language


Remove:


- em dashes;
- rhetorical questions;
- fragments;
- contrastive scaffolding;
- personification;
- generic AI vocabulary;
- empty intensifiers;
- repetitive first-person openings;
- reviewer-conversation framing;
- unexplained abstractions.


### Step 9: Review figures and citations


Check ordering, labels, captions, numerical agreement, supplementary references, and claim-level citation support.


### Step 10: Compile and inspect


Compile the latest source and inspect that exact output. Do not report visual or page-limit compliance from source text alone.


## 23. Final review passes


Perform these passes separately.


### Pass 1: Central claim


Does every major section support the same claim? Does the title match the highest completed level of evidence?


### Pass 2: Evidence


Can every conclusion be traced to a measurement, analysis, figure, table, theorem, or citation?


### Pass 3: Narrative


Does each paragraph develop one relationship? Does each major result motivate the next question?


### Pass 4: Technical accuracy


Are the model, data, biology, assays, mathematics, and ownership described correctly?


### Pass 5: Agency and tense


Are author actions active? Are completed experiments in past tense? Are definitions and established facts in present tense? Is future tense limited to planned work?


### Pass 6: Quantitative rigor


Are all values, sample sizes, units, uncertainty estimates, statistical tests, thresholds, and comparison conditions correct?


### Pass 7: Baseline fairness


Are inputs, budgets, tuning, data, and evaluation settings stated and matched where possible?


### Pass 8: Figures and supplement


Are all items cited in order, placed appropriately, labeled consistently, and matched to captions and source data?


### Pass 9: Literature and bibliography


Does every citation support the nearby claim? Is every BibTeX entry verified, current, unique, and correctly formatted?


### Pass 10: Compression


Has repetition been removed without damaging the causal and evidentiary chain?


### Pass 11: Compilation


Has the latest source been compiled and visually inspected after the final edit?


## 24. Final paper checklist


Before delivery, verify all of the following:


- The title matches the strongest supported claim.
- The abstract contains the problem, gap, method, evidence, and consequence.
- The Introduction ends with the exact missing capability and the paper's response.
- Related Work represents prior methods accurately and fairly.
- Methods define inputs, outputs, data, notation, objectives, training, and inference.
- Every mathematical symbol is defined.
- Data splits reflect the independent scientific unit.
- Leakage controls are explicit.
- Each Results subsection contains motivation, experiment or analysis, result, interpretation, and connection.
- Completed experiments and observations use past tense.
- Every quantitative claim is traceable.
- Baseline inputs and budgets are clear.
- Ablations test stated contributions.
- Uncertainty is reported at the correct independent unit.
- Predictions are distinguished from biological measurements.
- Screening is distinguished from confirmation.
- Controls test specific alternative explanations.
- Biological and technical replicates are distinct.
- Statistical tests and effect sizes are reported appropriately.
- Figures, tables, and supplementary items appear in order.
- Captions are self-contained and factual.
- Discussion interprets the evidence without repeating or overstating it.
- Limitations define where the claim stops.
- Future work follows from named limitations.
- Code, data, weights, and materials statements are accurate.
- Every citation is verified.
- references.bib compiles without missing keys or duplicates.
- No em dashes remain.
- No rhetorical questions remain unless explicitly approved.
- No contrastive scaffolding remains.
- No method, model, figure, dataset, or assay receives human intention.
- No generic AI filler remains.
- No unsupported result, citation, or claim remains.
- The final compiled paper has been visually inspected.


## 25. Compact bad-writing diagnostic


Rewrite a sentence or paragraph if it contains any of the following:


- a short dramatic setup followed by a list;
- a rhetorical question;
- a slogan after a colon;
- an em dash;
- “not X, but Y”;
- “rather than” used as contribution framing;
- “we treat,” “we frame,” “we view,” or “we formulate”;
- “Figure X shows”;
- a method that thinks, decides, reveals, or proves;
- “leveraging,” “unlocking,” or “paving the way”;
- “robust,” “generalizable,” or “transformative” without evidence;
- several adjacent sentences beginning with “We”;
- several adjacent fragments;
- a long methods inventory without scientific purpose;
- a citation cluster detached from the claims;
- a result written in present tense;
- an interpretation stronger than the assay;
- a therapeutic claim without the required evidence ladder;
- a baseline comparison with unequal information or budget left unstated;
- an unverified citation or BibTeX record;
- a paragraph that could be moved anywhere in the paper without changing the logic.


## 26. Reusable launch prompt for students


Copy this guide into ChatGPT together with the manuscript materials, then use a request such as:


    Follow the Chatterjee Lab Scientific Writing Master Guide in full. Read every supplied project file before drafting. Establish the project facts, completed results, pending results, central claim, claim-evidence map, and venue requirements. Conduct a deep, current, and verified literature search for the closest prior work, foundational sources, datasets, mathematical foundations, and appropriate baselines. Verify every citation against an official primary record and produce a clean references.bib. Write or revise the requested manuscript in paste-ready LaTeX. Preserve all verified scientific details, numbers, citations, labels, equations, and figure references. Do not invent results or citations. Apply the house style across the title, abstract, introduction, related work, methods, results, discussion, limitations, captions, supplement, and rebuttal. Compile and inspect the final document when the tools permit it. Report any unresolved scientific decision or unavailable verification plainly.


For a section-specific request, add:


    Work only on [SECTION]. Keep the section consistent with the full paper's central claim and evidence. Preserve the surrounding notation, citations, labels, and scientific decisions. Return only the paste-ready replacement unless I ask for commentary.


For a revision request, add:


    Preserve all substantive scientific detail. Repair logic, agency, tense, claim strength, paragraph flow, and citation placement before compressing. Do not remove controls, assumptions, caveats, or negative results that affect interpretation. Provide a short list of material scientific changes after the revised text only if requested.


For a literature request, add:


    Search broadly across the field and adjacent terminology. Use primary sources and official records. Verify publication status, DOI, authors, title, venue, year, and claim support. Return references.bib and a concise citation review identifying the claim supported by each source. Do not include a source that you could not verify.


## 27. Final instruction to ChatGPT


Apply these rules through judgment, not mechanical phrase substitution. Preserve natural cadence and distinctive scientific authorship. Every sentence should clarify the problem, method, evidence, or consequence. Every claim should stop where the evidence stops.


## 28. Complete prohibition index


This section consolidates the explicit dislikes recorded across the source guides. Apply it to manuscript prose, captions, supplementary text, rebuttals, cover letters, README prose, and technical documentation. Quoted terms appear here only so they can be identified and removed from final prose.


### 28.1 Prohibited words and stock phrases


Do not use “audit,” “legitimate,” “legitimately,” “precisely,” “genuine,” “genuinely,” “silent,” or “silently,” including variants and misspellings, in final scientific prose. Avoid “delve,” “leverage,” “foster,” and related inflated vocabulary.


Remove these stock constructions:


- “Below is a detailed overview.”
- “Below is...”
- “The following section...”
- “It is important to note...”
- “Needless to say...”
- “At its core...”
- “The key insight is simple.”
- “The problem is clear.”
- “In today's rapidly evolving landscape...”
- “This point is critical.”
- “This is transformative.”
- “This distinguishes X from Y:” followed by a slogan.
- “Success would create...” when the sentence can state the capability established.
- “opens new avenues”;
- “unlocks broad applications”;
- “explores possibility spaces”;
- “paves the way”;
- “change everything.”


Do not use “innovative,” “novel,” “powerful,” “transformative,” “robust,” “generalizable,” “scalable,” “comprehensive,” “integrated,” “synergistic,” “platform,” “pipeline,” “framework,” or “ecosystem” as unearned praise. Use a term only when the sentence states its technical definition or presents direct evidence.


### 28.2 Prohibited rhetorical structures


Remove:


- em dashes;
- three hyphens used as a dash;
- semicolons or colons used to manufacture a dramatic pause;
- rhetorical questions;
- short dramatic fragments;
- several short declarative sentences used as a buildup;
- a broad claim followed by a rapid list of examples and a slogan;
- applause lines;
- empty emphasis;
- advertising language;
- speech-like cadence;
- theatrical inversions;
- unsupported claims of broad applicability;
- long introductory clauses that bury the main action;
- more than one long list in a paragraph;
- several consecutive paragraphs with the same opening construction.


Do not use “not X, but Y,” “not only X, but also Y,” “this is not about X,” “rather than” as contribution framing, “we do X instead of Y,” or “while X, Y” when it merely stages a contrast.


Do not write “we treat X as Y,” “we frame X as Y,” “we view X as Y,” “we formulate X as Y,” “we conceptualize X as Y,” or “we model optimization as transport.” State the definition or operation directly.


Avoid declarations that data, variables, or examples do or do not “enter” the method. Name what is used for fitting, validation, calibration, conditioning, selection, and evaluation.


Do not write a separate justification for every sentence or design choice. Explain each consequential scientific choice once, where it becomes necessary.


Avoid a mechanical five-sentence Results template and repeated endings such as “therefore, we next....” Preserve the motivation, experiment, result, interpretation, and connection through natural paragraph flow.


### 28.3 Prohibited agency


Do not assign human reasoning, intention, or rhetoric to methods, models, datasets, equations, controls, representations, molecules, figures, tables, assays, or results.


Avoid constructions such as:


- “The model exposes...”
- “The method supplies...”
- “This supplies...”
- “This provides...”
- “This gives us...” when an equation, distribution, transformation, model, dataset, or figure is assigned vague narrative agency;
- “The framework defines...” when the authors made the definition;
- “The data speak to...”
- “The figure proves...”
- “The dataset reveals...” when the authors inferred the result;
- “The assay understands...”;
- “The model decides...”;
- “The screen identifies...” when the team used the screen to identify candidates;
- “The method provides a prediction...” when “we predicted” is clearer;
- “The project demands...”;
- “The molecules provide controls....”


Physical properties, computations, and observed readouts may have nonhuman subjects. Preserve this distinction to avoid awkward prose.


Do not replace every first-person statement with passive voice or an abstract object. Do not begin several consecutive sentences with “We” or “We will.” Vary sentence structure while keeping agency clear.


Avoid generic actors such as “researchers” and “investigators” when the authors performed the action.


### 28.4 Prohibited scientific framing


Do not:


- organize the paper by project chronology;
- place related facts next to one another and leave their relationship implicit;
- introduce a method because it is impressive or available;
- introduce a new method category without explaining the unresolved limitation;
- infer architecture, objective, mechanism, or capability from a method name;
- begin Methods with a layer inventory;
- force a formulation to fit a pun or acronym;
- defend whether inputs are “legitimate”;
- repeat explanations about the absence of external examples at every stage;
- repeatedly claim that a choice is sensible, necessary, appropriate, or important;
- present several targets as disconnected demonstrations;
- let computational work end with benchmark performance when the claim concerns scientific use;
- let experimental work become an assay inventory;
- add a wet-lab component solely to imitate a biological paper;
- add a modality, assay, target, animal model, collaborator, or experiment merely to broaden scope;
- label an unimplemented comparison as completed;
- describe a small successful run as evidence of full-scale accuracy or superiority;
- represent approximate search over a subset as exhaustive catalog search;
- describe a predictor as a generator;
- describe a language model as a mechanistic simulator;
- describe an assay as measuring something it does not measure.


### 28.5 Prohibited result and evidence practices


Do not:


- invent numerical results, significance claims, successful outcomes, or citations;
- write anticipated improvements in the past tense;
- draw fabricated performance curves;
- convert nonsignificant trends into findings;
- use “significant” without a stated statistical meaning;
- use “robust” without a defined stress test or replication basis;
- omit mixed or unfavorable results that affect the conclusion;
- report technical replicates as the independent sample size;
- claim binding from docking;
- claim cellular activity from a language-model score;
- claim delivery from transfected expression;
- claim direct binding from an abundance change;
- claim mechanism from a reporter alone;
- claim therapeutic impact from computational scores, binding, degradation, or cellular phenotype alone;
- overstate what a cited experiment established;
- describe a preprint, accepted manuscript, or forthcoming paper as published without verification.


### 28.6 Prohibited figure, table, and placeholder practices


Do not write:


- “Figure 2 shows...”
- “Table 1 demonstrates...”
- a sentence that announces that a figure proves the claim.


Do not:


- place a table or figure far from the paragraph that interprets it when the venue permits local placement;
- use a top-only float setting when it separates the item from its prose, unless the venue requires it;
- force float placement that produces large blank regions or detaches a result;
- repeat the entire Results argument in a caption;
- claim more in a caption than the assay supports;
- use repeated titled cards, prose-filled boxes, decorative icons, or a sequence of numbered panels for a unified method overview;
- expose gray instruction boxes, result-slot commands, placeholder explanations, or instructions to a future author in the rendered paper;
- leave fabricated curves or invented table values in placeholders.


Keep completion reminders in source comments. Pending tables may use consistent empty cells. Pending figures may specify axes, comparisons, and captions without invented data.


### 28.7 Prohibited citation practices


Do not:


- invent a citation from memory;
- cite a search-result snippet as evidence;
- use an abstract alone for a claim that depends on methods or quantitative results;
- attach unrelated references to the end of a long paragraph;
- cite a review for a precise technical or mechanistic claim when the primary paper is available;
- retain duplicate preprint and published entries accidentally;
- cite a source without checking whether it supports the exact nearby claim;
- leave foundational biological claims uncited;
- leave unresolved citation keys in the final document.


### 28.8 Prohibited compression and formatting practices


Do not cut:


- the causal connection;
- the central hypothesis;
- the defining equation;
- the decisive experiment;
- the essential control;
- the quantitative result;
- the supported interpretation;
- the biological consequence.


Do not fit page limits by shrinking fonts, margins, or spacing. Do not use underlining for long text that may wrap. Do not claim page, line, or visual compliance before compiling the latest source.


Use one main.tex for manuscript prose, tables, algorithms, and appendix content when the project follows the standard lab workflow. Do not split the manuscript into many section or table files unless the venue or an explicit project instruction requires it.


Do not duplicate the main paper in the appendix. Do not repeat the Results section metric by metric in the Discussion. Do not append a generic list of fashionable future technologies.


### 28.9 Full-paragraph review requirement


A final word search is necessary and insufficient. Read every complete paragraph to detect:


- personification;
- defensive framing;
- contrastive scaffolding;
- rhetorical punctuation;
- artificial cadence;
- abrupt method introductions;
- unclear pronouns;
- repeated openings;
- adjacent sentences with the same noun or verb;
- a paragraph that lacks one connected scientific relationship;
- a paragraph that could move anywhere in the paper without changing the logic.


Revise until the writing remains direct, natural, quantitative, and specific.