# Reviewer feedback on the experimental proposal (September 2026)

Synthesis of the comments and tracked edits from Patrick Reed (20 comments, 2 edits,
2026-09-20/21) and Lillian Lau (31 comments, 302 edits, 2026-09-10/18) on
`Amestoy_NYC_Reoptimization_experimental_proposal.docx`, with the disposition of each
item in the v2 revision (`..._v2.docx`, tracked changes under the author label
"Revision"). Neither reviewer had access to the code, so every method question was
checked against the repository before a disposition was assigned.

## 1. What future versions must do better

1. **Tie every literature paragraph to the stochastic search focus.** Pat read the
   literature review as topically scattered because each study was summarized on its
   own terms rather than by what it did to the search ensemble. Each cited study now
   answers one question: what conditions did the search see, and what did that change?
2. **Say "stochastic simulation and optimization" and mean it.** The opening conflated
   deterministic historical search, stochastic search, and direct policy search. The
   proposal now names the stochastic form as its subject and separates the overfitting
   evidence from flexible policy approximators from the fixed FFMP rule structure
   optimized here.
3. **Engage DU optimization explicitly.** MORDM, multi-scenario MORDM, and sampled
   DU factors in the search are the frame a Reed-group reader expects. The proposal now
   states that those studies vary the deeply uncertain factors the search sees, whereas
   this study holds deep uncertainty out of the search and varies which realizations of
   the well-characterized natural variability the search sees.
4. **Describe, name, then justify.** Both reviewers lost the thread between the
   description of the method, its name, and the benchmark. The contribution paragraph
   now runs description, name, benchmark, matching, re-evaluation in that order.
5. **Define before use.** Deep uncertainty, SOW, scenario design, Decree Parties, and
   function-evaluation budget are defined at first use. "Scenario" is no longer used for
   a re-evaluation point (SOW) or a realization.
6. **Give numbers where a design choice is asserted.** N = 300, L = 10, the hazard-axis
   correlations, the tolerance values, and the re-evaluation envelope are stated with the
   diagnostic results behind them rather than as bare assertions.
7. **Consistency across sections.** The document mixed "multi-objective" and
   "many-objective", "SOW" and "scenario", "reductions" for decision variables that are
   multipliers, and "min releases fixed" alongside a conservation-release decision
   variable. All are reconciled.

## 2. Comments by category and disposition

### 2.1 Writing clarity and structure

| Reviewer | Comment | Disposition |
|---|---|---|
| Lillian C23 | Disconnect between the 1960s drought paragraph and the FFMP paragraph | Adopted. Bridging clause added at the start of the FFMP paragraph. |
| Lillian C30 | Define who the Decree Parties are | Adopted. Defined at first use in 1.1 (the four basin states and NYC). |
| Lillian C62 | Condense the tree-ring passage to two sentences | Adopted. |
| Lillian C77 | Bridge from synthetic ensembles to MOEAs; introduce how ensembles are used before MOEAs | Adopted. New opening sentence for the MOEA paragraph. |
| Lillian C129 | Define deep uncertainty at first mention | Adopted. Defined in 1.2; the repeated definition in 3.5.1 removed. |
| Lillian C140 | SOW versus scenario switching | Adopted. SOW defined once in 1.2 and used for every re-evaluation point; "scenario design" defined in 1.3. |
| Lillian C148 | How does a problem formulation differ from a candidate scenario? | Adopted. The sentence now says the selected subset depends on the system model, the objectives, and the policy used to score it. |
| Lillian C195 | Scenario design or streamflow selection? | Adopted. "Scenario design" defined as the rule for constructing the search ensemble. |
| Lillian C200 | Define function-evaluation budget before use | Adopted. |
| Lillian C207 | Rephrase RQ2 as "How does ..." | Adapted. RQ2 is now open-ended and includes regret (Trevor's own comment). Her "moment-based statistical measures" contrast was not adopted because the MC design does not rely on moments. |
| Lillian C220 | Justify satisficing and regret together; name the regret reference | Adopted in the Section 3 overview. |
| Lillian C222 | Explain "climate perturbation" | Adopted. Rephrased as the generator's fitted statistics not being perturbed to represent climate change. |
| Lillian C243 | "and jointly" incomplete | Adopted. |
| Lillian C264 | "robustness out of sample" unclear | Adopted. Rewritten in plain terms. |
| Lillian C265 | Cite Trindade and Herman work for the search-conditions hypothesis | Adopted (Trindade et al., 2017 added). |
| Lillian C268 | "Distortion" may be misread | Adopted. Rewritten as over-representation of extreme conditions and a departure by design from the generator's distributions. |
| Lillian C275 | "depth" of drought conditions: severity or intensity? | Not adopted. "Severity" and "intensity" are reserved for the SSI metric definitions (`docs/terminology.md`). "Depth" is used consistently. |
| Lillian C298 | Does "converged" mean the distributions are uniform? | Adopted. Rewritten as the empirical distribution of each metric, including its 1st and 99th percentile bounds, being stable to further generation. |
| Lillian C336 | How will the tolerance be determined? | Adopted. Method and adopted values from `regret_tolerance_diagnostics.md` stated. |
| Lillian C337 | Is the regret frequency computed per objective? | Adopted. Joint across objectives, with per-objective frequencies also reported. |
| Lillian C6 | "saltwater intrusion" versus "salinity intrusion" | Not adopted. "Salinity intrusion" is the DRBC term. |
| Pat C0, C2 | Clarify the stochastic simulation-optimization focus; ensemble evaluation is not universal | Adopted. Opening paragraph distinguishes deterministic and stochastic search and names the stochastic form as the subject. |
| Pat C7, C8, C9, C10 | Literature review is vague and jumps around; tie each study to the search focus; pull the contribution forward | Adopted. Section 1.2 reorganized into three paragraphs (overfitting evidence and policy structure; the MORDM and DU-optimization lineage; ensemble construction methods), each opening with its relation to the search ensemble. |
| Pat C11, C12 | "Not all hazards, you're focusing on droughts"; go directly to the HF design then its rationale | Adopted in wording (drought and flood hazards named at every mention). The premise is off-base for the code: two of the six selection axes are flood metrics. |
| Pat C13, C14 | Unclear whether MC is the benchmark | Adopted. "We benchmark the HF design against the standard practice of stochastic MOEA search, a Monte Carlo (MC) design ..." |
| Pat C19 | Scott and Vivek will not follow the 10-year argument | Adopted. Rewritten in plain terms with the reason a long realization would blur the event-based metrics. |
| Pat C25 | Some text is repetitive | Adopted. Duplicated deep-uncertainty definition and demand-growth statements consolidated. |
| Lillian (edits) | 302 tracked edits | Adopted where they improved flow or fixed errors (sentence splits, "best known", "re-optimization", the 1960s sentence repair, "(Table 2)" reference, "The current FFMP policy therefore serves as a fixed reference"). Not adopted where they introduced unsupported claims ("tensions ... previously concealed", "unanimous consent has made it challenging to assess", "Gaussian historical probability", "hazard generator", "first known application ... comprehensive Monte Carlo hazard sampling generator", "mandated" for the Trenton target, which is not binding on NYC) or placeholders. |

### 2.2 Method questions and concerns

| Reviewer | Comment | Disposition |
|---|---|---|
| Pat C3, C4 | Conflating EMODPS-style policy search with the fixed FFMP parameterization; the FFMP's adaptive structure ties to robustness | Adopted. Section 1.2 now separates the overfitting evidence (direct policy search, forecast-informed policies) from the fixed, storage-conditioned rule structure optimized here, and states why the search ensemble still governs the rule parameters returned. |
| Pat C6 | Not engaging DU optimization explicitly; how the work relates given one uncertainty | Adopted. See item 3 above. The scope statement is exact: the sampled deep uncertainty is streamflow change along three forcing dimensions, re-evaluation only. |
| Pat C17 | Justification of N = 300 and L = 10 is limited | Adopted. Section 3.1 now states the diagnostic ladder (N = 50 to 500), the criterion (sampling noise of paired policy differences at or below half the epsilon precision), and the length argument (one drought of record plus onset and recovery fits in 10 years; a common (N, L) is required to avoid confounding the construction with record length). |
| Pat C18 | Include climate change in the search ensembles | Not adopted in this round (Trevor's decision; reasons in the project presentation's supplemental slides: a clean split between search and re-evaluation conditions, the HF design as a complement to DU optimization rather than a member of it, and the diagnostic showing that most re-evaluation hazard conditions lie within the stationary candidate ensemble's range). Logged in `TODO.md` for a later decision. |
| Pat C21, C22 | Connection to risk-averse objective formulations (Quinn et al., 2017) and to MORDM's original well-characterized search | Adopted. One sentence in 3.2.1 on the objective-formulation counterpart; MORDM's search under the well-characterized baseline stated in 1.2. |
| Pat C25 | Not sold on CMIP6 as the DU re-evaluation; it may not expand the envelope of severe droughts | Partly adopted. Section 3.5.1 now states the sampled envelope in numbers (the full CMIP6 range widened by one quarter at each end, annual volume from about 13 percent below to 32 percent above the historical mean). The drought-side question is logged in `TODO.md` as a point of investigation (the existing hazard-support decomposition shows the re-evaluation ensemble's excursions beyond the stationary candidate ensemble are mostly on the flood side). |
| Pat C20 | Bring the DRB context forward; a five-page document plus slides | Deferred (Trevor's decision). The order is unchanged in this round; shortening is planned as a later pass with a supplement. |
| Lillian C113 | MORDM is out of place; state its limitation like the others | Adopted. |
| Lillian C297 | Examples of how storage-conditioned rules respond to onset and recovery differences | Adopted. One sentence of physical interpretation added. |
| Lillian C304 | Save compute by evaluating only summer weeks | Not adopted. The daily simulation must run in full regardless; the CVaR is computed after simulation at no simulation cost. |
| Lillian C306 | Why no Trenton flow deficit objective; a regional minimax for Trenton and Montague | Partly adopted. A rationale sentence is added (the Trenton target is a Water Code target supported by DRBC-directed lower-basin reservoirs outside the FFMP rules, and the depth of Trenton shortfalls varies little across candidate FFMP policies in initial diagnostics). No written rationale existed in the repository; the minimax suggestion is logged in `TODO.md`. |
| Lillian C301 | Explain why each decision-variable group has its count | Adopted. Table 3 descriptions now give the arithmetic (six rule curves, three drought stages, two gauges, three reservoirs, two sub-zones, four seasons). |
| Lillian C291 | Add a source column to Table 2 | Adapted. Symbols added to the metric names and sources added to the descriptions (no new column). |
| Lillian C292, C312 | Move SSI and CMIP6 details to Supporting Information | Deferred to the shortening pass. |

### 2.3 Figures requested

| Reviewer | Request | Disposition |
|---|---|---|
| Lillian C244 | The hazard-space figure shown at group meeting | Adopted. The ensemble-composition figure (`figures/manuscript/fig04_ensemble_composition.png`) is inserted as Figure 3 after the HF construction steps. Its legend still reads "PS ensemble" and "SSI-months" from the N = 100 diagnostic; a comment flags the re-render (`TODO.md` item 5). |
| Lillian C300 | A figure alongside Table 2 | Adopted. The hazard-examples figure (`outputs/supplemental/hazard_examples/figures/hazard_examples_parallel_hazfill_stat_abs_10yr_n40_d0.png`) is inserted as Figure 2 after the flood-metric paragraph. It is an illustrative 40-member HF ensemble; a comment flags the re-render on the campaign ensemble. |
| Lillian C301 | A figure for the decision variables | Not embedded. The existing policy-rules figure carries stale zone labels and a candidate policy; a comment notes the planned FFMP operating-rules figure. |
| Trevor C1 | Figure 1 is outdated (N = 100, PS, 25 realizations) | Unchanged; the editable source is `figures/manuscript/methods_diagram/methods_diagram_draft.pptx`. |

### 2.4 Code and method changes surfaced by the review

Logged in `TODO.md` (Section 6). In brief: the re-evaluation ensemble size in the
proposal (1,000 SOWs, 50 realizations) versus the code (500 re-evaluated SOWs, 25
realizations, with R = 50 already planned); the drought-side question on the
re-evaluation envelope; a written rationale or a formulation decision for the Trenton
deficit; the regional minimax suggestion; the climate-change-in-search decision; the
downstream-flow correlation of the hazard axes, which the proposal previously asserted
and which has not been computed; and re-rendering of the three figures with current
labels and campaign sizes.

## 2.5 Second round: stand-in reviewer findings folded into v2

Two independent reviewer agents (Opus 5.5) read the first v2 draft as stand-ins for Pat
and Lillian, with the original comments and `docs/terminology.md`. Both returned REVISE.
Their must-fix items and dispositions:

| Stand-in | Finding | Disposition |
|---|---|---|
| Lillian | "500,000 NFE per design" is per seed trial; "one re-optimization" conflicts with two seeds | Fixed (Section 3.4). |
| Lillian | Trenton rationale ("lower-basin reservoirs lie outside the FFMP rules") conflicts with the Trenton flow-target decision variables | Rewritten: Trenton flow is supported by DRBC-directed lower-basin releases as well as NYC releases, and Trenton shortfall depth varies little across policies. |
| Lillian | "Base conservation release rates fixed" conflicts with the seasonal-scale decision variables | Rewritten: maximum discharges fixed; schedules varied only through the seasonal scales. |
| Lillian | HIST design never introduced in 1.3; "only the construction is varied" contradicts HIST | HIST introduced in 1.3; overview sentence restricted to the matched designs. |
| Lillian | 78 versus 79 years; "1964 through 1967" versus the terminology file's 1961 | 78-year realization dated December 1945 through November 2023; the drought dates are left for Trevor (the repository is inconsistent). |
| Lillian | Six rule curves and the spill mitigation sub-zones never introduced before Table 3 | Introduced in Section 2.2. |
| Lillian | Regret reference named inconsistently; RQ1 omits the NJ diversion | "Current FFMP policy" used for every regret reference; NJ diversion added. |
| Lillian | Figure 1 caption thin; Figure 3 cited before Figure 2; Table 4 "3 failure weeks" versus "more than k" | Caption expanded; figures renumbered to citation order; failure definition aligned with the code ("k or more weeks"). |
| Pat | HF implicitly reweights the frequency-based objectives | Stated in 3.2.1 as an intended risk-averse framing, tied to the tail statistics and to Quinn et al. (2017). |
| Pat | Isolation argument (3.1) versus hazard-overlap hypothesis (3.2.1) | Isolation restated as "no design samples the deeply uncertain factors"; commitment added to compute the six hazard metrics on the re-evaluation ensemble and report the overlap. |
| Pat | Seed count unjustified; between-design differences not separated from seed and draw variability | Precedent cited; differences judged against the seed spread and a paired bootstrap over SOWs; draw sensitivity via two further draws (already in the campaign plan). |
| Pat | Re-evaluation envelope dry end | Natural variability within SOWs and the drier SOWs stated (supported by the hazard-support decomposition); the widening decision stays in `TODO.md`. |
| Pat | Literature exactness: Bartholomew and Kwakkel mis-slotted; Herman 2016 and Hilbers 2019 not search studies; Zatarain Salazar purpose and finding missing; Lehner and Deser over-cited; Zaniolo 2023 cited for a re-evaluation it did not do | All corrected from the verified summaries in `docs/study_motivation.md`. |
| Pat | N = 300 not justified for the HF design; L = 10 ignores initial storage | HF construction replicates described; initial storage (80 percent of capacity) and the six-month exclusion stated. |
| Pat | "Widest", "uniform", "distinct dimensions" overclaims | Softened, with the target-displacement report and the nested candidate-ensemble check named. |
| Pat | Cost estimate for the re-evaluation; restructuring; deeper drought envelope | Not adopted in this round (cost is not stated as motivation; structure and envelope are deferred decisions). |

A third pass on the round-two draft (both stand-ins, "minor revise") produced the last set of
targeted edits: the constraints paragraph no longer says the decision variables alter "only"
the drought-stage reductions; the spill mitigation zone is described as three sub-zones with
releases in the upper two; Figure 1f is cited at the HF construction steps; the worst
first-percentile year is stated to reduce to the worst year for the 77-year HIST design; the
seed-trial wording no longer implies a single run; the draw-sensitivity check is described as
a test of the set's performance under other draws rather than of the set itself; the
hazard-overlap comparison is specified on consecutive 10-year windows of the re-evaluation
realizations; Savage (1951) is dropped in favor of the status-quo form of regret (Herman et
al., 2015; McPhail et al., 2018) and its reference entry removed; Bartholomew and Kwakkel
(2020) and Eker and Kwakkel (2018) are described with their comparators; Lehner and Deser
(2023) is cited only for the general statement; the regret tolerance is stated as the level
that limits spurious regret to 5 percent of SOWs; and the remaining "uniformly" claims are
qualified. Both stand-ins approved the resulting draft in a confirmation round; their last
optional polish (the antecedent of "overlap" in 3.5.1, naming the two compounding effects in
3.2.1, and aligning how design differences are judged between 3.1 and 3.4) was applied. Open
for Trevor: the 1960s drought dates (1964 through 1967 in the text; the terminology file and
the Hogarty summary say 1961 through 1967), and the document is about 2,400 words longer than
v1, so the deferred shortening pass with a supplement remains.

## 3. Code facts corrected in v2

| Proposal v1 | Repository | v2 |
|---|---|---|
| 1,000 SOWs, 50 realizations of 50 years | 1,000 SOWs generated with 25 realizations; 500 re-evaluated; R = 50 planned (`src/etest.py`, `campaign_design.md`, `TODO.md`) | 500 SOWs, 50 realizations of 50 years, with a comment on the pending regeneration |
| Two seeds at 500,000 NFE per design | Seed 1 runs 187,500 per island; the equal-budget snapshot at 125,000 per island is merged with seed 2 (`src/moea_config.py`, `campaign_design.md`) | Stated as the convergence check and the equal-budget merge |
| "Minimum daily releases ... fixed" | Base conservation-release rates are fixed; seasonal scales are decision variables (`src/formulations/ffmp.py`) | Corrected |
| Flow-target "reductions" | Multipliers on the drought-stage factors, capped at the normal-zone target (`ffmp.py`) | Corrected |
| Rule-curve shifts "of each drought-stage curve" | Lower plateau of six curves plus upper plateau of three drought-stage curves (`ffmp.py`) | Corrected |
| "No pair of axes is strongly correlated" | Largest between-metric rank correlation among the six selection axes is 0.61 (`hazard_selector_diagnostics.md`) | Number stated |
| "Downstream flows are strongly correlated with upper basin flows" | Not computed (planned diagnostic) | Replaced by the supported rationale (combined storage governs every FFMP operation; NYC releases are the principal regulated contribution to the Montague flow) |
| N = 300 "sampling noise smaller than the objective precision" | Criterion is paired-difference standard error at or below half the epsilon precision, on a ladder from N = 50 to 500 (`ensemble_size_diagnostics.md`) | Stated precisely |


## 4. Re-evaluation ensemble decisions after v2 (2026-09-30)

Two items deferred in v2 were settled from evidence and carried into v3
(`..._v3.docx`, tracked under the same "Revision" author; Section 3.5.1 only).

| Item | Decision | Evidence |
|---|---|---|
| Realizations per SOW (proposal 50, code 25) | 25 retained; v3 states 1,000 SOWs generated, 500 re-evaluated, 25 realizations of 50 years | Cross-SOW standard error 2.2 percentage points at 500 SOWs is independent of the count; the paired per-SOW noise floors sit at 0.12 to 0.60 of the epsilon precisions and the largest misclassified share at the adopted criteria is 1.1 percent of SOWs; doubling adds 66,500 SU; published re-evaluations use 1 to 1,000 realizations per SOW and none tests the count at a fixed SOW count (`campaign_design.md` §5, SI Table S1) |
| Dry end of the envelope (Pat C25, second round) | Annual-volume lower bound extended from 0.866 to 0.80; v3 states 20 percent below to 32 percent above | Paired generation test: the driest CMIP6 run leaves the stationary drought envelope in 1.7 percent of decades, the widened bound in 10 percent, the adopted bound in 17 percent (48 percent on summed magnitude); 0.80 is the dry limit of the Research Triangle inflow multiplier (Herman et al., 2014; Trindade et al., 2017) (`forcing_parameterization.md`, SI Figure S18) |
