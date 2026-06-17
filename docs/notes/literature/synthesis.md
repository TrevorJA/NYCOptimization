# Synthesis — running cross-paper takeaways

**Last updated:** 2026-04-30 (after batch 3 — Lin et al. preprint added: 25 papers).

> **Status note (2026-06-11).** Still valid as cross-paper synthesis of the per-paper notes. The open question in §1 (DU-in-search vs DU-only-in-re-evaluation) has since been resolved by `../experimental_design.md`: the study compares six scenario designs during search, including the Trindade-style resampled probabilistic design and the proposed hazard-filling design, with cross-design comparison only via re-evaluation on a held-out deeply uncertain test ensemble. Newer and more specific scenario-design literature is organized as per-subtopic annotated-bibliography notes indexed in `scenario_design.md` (with `scenario_design_tables.md` and `scenario_design_taxonomy.md`), which take precedence where they overlap.

This file accumulates *cross-paper* synthesis. Per-paper findings stay in `notes/`; this file captures what the literature collectively says, where papers agree or disagree, and what gaps emerge for the NYCOptimization manuscript. **Editing rule:** new claims should cite the per-paper note(s) by slug.

---

## 1. MORDM framework — mature canon, our methodological spine

The MORDM framework (Kasprzyk 2013) sits at the intersection of three traditions:
- **Many-objective evolutionary optimization** for alternative generation (Reed 2013, Hadka & Reed 2015) — our optimizer is MM-Borg, run for the canonical 4-axis diagnostic suite (effectiveness, efficiency, reliability, controllability).
- **Robust Decision Making** for re-evaluation under deep uncertainty (Marchau 2019 Ch. 2, Lempert tradition).
- **Interactive visual analytics + scenario discovery** for decision support (Kasprzyk 2013, Bryant & Lempert via Hadjimichael 2023 references).

Across the 21-paper corpus, MORDM appears as the consensus canonical workflow for water-resources MOEA + robustness studies. Our manuscript instantiates it directly. Key extensions of MORDM in the recent literature — all citable when relevant:

- **Per-stakeholder decomposition** (Hadjimichael 2020 — Defining Robustness; Trindade 2017; Herman 2014). Not a single basin-wide robustness; each user/utility/decree-party gets their own.
- **Multi-metric framings** (Herman 2015, McPhail 2018, Maier 2016). No single robustness metric is "right"; run multiple and examine ranking sensitivity.
- **DU-in-search** (Trindade 2017): include deep uncertainty in the optimization search phase, not just re-evaluation. Improves discovered Pareto front under deep uncertainty.
- **Narrative storyline classification** (Hadjimichael 2023 FRNSIC). Hierarchical classification of consequential scenarios → narrative drought storylines.

**Synthesis question still open:** does our manuscript do DU-in-search (à la Trindade) or DU-only-in-re-evaluation (classical Kasprzyk 2013 MORDM)? Trade-off is computational cost vs. discovered Pareto-front quality.

## 2. Borg / MM-Borg — algorithm choices justified

Reed et al. (2013) establishes the diagnostic protocol; Hadka & Reed (2015) introduces the MM-Borg parallelization that we are running. Together they justify:
- Auto-adaptive multi-operator search (SBX, DE, PCX, SPX, UNDX, UM) — controllability matters.
- ε-dominance archiving — bounded archive, decision-maker-controllable resolution.
- ε-progress restart — escape local optima.
- Multi-master island parallelization — reliability gains via independent island exploration; scales to 16k+ cores.
- 50-seed × NFE × 75% attainment threshold for reliability (Reed 2013).

**Open methodological question:** how many islands × seeds × NFE for our problem class? Hadka 2015 benchmarks LRGV (6-obj water portfolio); our problem is 7–8 obj reservoir-operations — different but comparable difficulty.

## 3. Robustness metrics — multi-framing recommendation

Three papers in our corpus speak directly to metric choice (Herman 2015, McPhail 2018, Maier 2016) plus Hadjimichael 2020 (Defining Robustness) on per-user thresholds. Convergent recommendations:

- **No single metric is "right"** — choice reflects decision-maker risk philosophy and decision context.
- **Run multiple metrics** (Herman 2015 explicit recommendation): satisficing + regret + variance-based, then check ranking stability.
- **Satisficing + multivariate is the recommended default** for deep uncertainty without probability rankings (Herman 2015, McPhail 2018). Our manuscript should default to multivariate satisficing per stakeholder, with multi-framing sensitivity check.
- **McPhail 2018 framework** decomposes any metric as T1 (performance transformation) × T2 (scenario subset) × T3 (statistical moment). Useful implementation scaffold; signals lineage to Hadjimichael 2020.
- **Herman 2014** demonstrates multi-stakeholder robustness tradeoffs in the Research Triangle case (cooperative water supply): efforts to improve one utility's robustness can degrade others'. The same pattern likely applies to NYC vs. NJ vs. Trenton ecological flows in our DRB case.

**For our manuscript:** multivariate satisficing per decree party / sector, with multi-framing sensitivity (also report regret-based and an aggregate metric); cite Herman 2015 + McPhail 2018 + Hadjimichael 2020 (Defining Robustness) together.

## 4. Multi-stakeholder robustness — institutional structure matters

Three papers establish the methodological state-of-the-art for multi-stakeholder robustness: **Herman 2014** (Research Triangle, 4 utilities), **Trindade 2017** (Research Triangle extended, DU-in-search), **Hadjimichael 2020 — Defining Robustness** (UCRB, hundreds of users under prior appropriation). Pattern:

- Aggregate basin-wide robustness *masks* user-level heterogeneity. Senior-rights users are robust; junior-rights or environmental users are fragile (Hadjimichael 2020).
- **"Robustness conflicts"** (Trindade 2017) — improvement for one party degrades another. Quantifiable as a tradeoff metric.
- **Institutional structure** (water rights, decree parties) is the dominant determinant of who gets hurt — *not* hydrology alone (Hadjimichael 2020).
- **Moore 2021** documents how the DRB's decree-party governance + unanimous-consent rule means our optimized policies must be politically viable across all 5 parties (NY, NYC, NJ, PA, DE). Cooperative policies discovered under MORDM (Trindade 2017) are an existence-proof.

**For our manuscript:** decompose robustness across (1) NYC supply (2) NJ delivery (3) Montague flow (4) Trenton flow (5) flood mitigation (6) storage resilience (7) salinity. Each gets its own satisficing threshold spectrum. Add a "robustness conflicts" metric à la Trindade. Frame the result against Moore 2021's institutional-viability test.

## 5. Scenario discovery — modern ML-based methods (PRIM is rejected)

**Decision (2026-04-30):** PRIM is **not** to be used in the manuscript — the user judges it outdated and inflexible. CART has the same flexibility issues and is similarly deprioritized. Instead, the manuscript scenario discovery uses **modern ML-based methods**.

Three papers in our corpus drive this:
- **Hadjimichael 2020 (Defining Robustness)** uses logistic regression on per-user shortage outcomes against the 14 deeply-uncertain factors. Per-user / per-stakeholder regression maps directly onto our DRB decree-party decomposition.
- **Hadjimichael 2023 (FRNSIC)** uses hierarchical classification + dynamic-pattern clustering for narrative storylines. Explicitly designed to layer on top of standard scenario-discovery outputs and produce actionable narratives.
- **Hadjimichael 2020 (Predator-Prey)** demonstrates that multi-objective robustness shifts tradeoff topology (not just performance) — relevant evidence that the *structure* of consequential scenarios shifts across objective framings.

The brainstorm Idea 3 (SHAP/LIME-based scenario discovery, in `decisions/brainstorm_methodological_contributions.md`) is the *preferred direction* for our manuscript — not a "PRIM alternative" but the primary methodology, complemented by gradient-boosted classifiers and per-stakeholder logistic regression.

**For our manuscript:**
- **§Methods §scenario discovery:** logistic regression per stakeholder (à la Hadjimichael 2020 — Defining Robustness), with SHAP-based feature attribution for interpretability. Optionally extend to FRNSIC-style hierarchical classification if our results show clear cross-actor heterogeneity worth narrating.
- **§Background §why not PRIM:** brief mention citing Bryant & Lempert (2010) as the foundational PRIM reference (good to cite per Trevor's note), then explicitly flag that more recent ML-based methods are more flexible and better-suited to correlated, high-dimensional uncertainty spaces.
- **§Discussion — narrative storylines:** if the analysis surfaces clear actor-specific consequential patterns, FRNSIC-style storyline construction is a natural follow-on per Hadjimichael 2023.

## 6. Stochastic streamflow generation — DU-in-search via drought-spread ensemble; SynHydro now, MOEA-FIND later

Decision 2026-04-30: **DU-in-search using a structured drought-spread ensemble during MM-Borg search + a larger re-evaluation ensemble** (`decisions/2026-04-30_inflow_and_du_search.md`). The pipeline is **ensemble-generator-agnostic** so the underlying generator can be swapped without code rewrites.

### Generator: SynHydro (Kirsch–Nowak) → MOEA-FIND (when finalized)

- **Default (SynHydro):** Sibling repo [`SynHydro/`](../../../SynHydro/) provides Kirsch's mFGN single-site generator, Nowak's multi-site disaggregation, plus several other generator families (Matalas, HMM, WARM, copula-based). Used in [`StochasticExploratoryExperiment/`](../../../StochasticExploratoryExperiment/) and [`MOEA-FIND/`](../../../MOEA-FIND/). `Kirsch (2013)` is the citation for mFGN; Nowak (2010) is the multi-site companion (deferred from our literature wishlist per `feedback_lit_canon.md` since SynHydro implements both).
- **Future swap (MOEA-FIND):** Sibling repo [`MOEA-FIND/`](../../../MOEA-FIND/) — Borg-driven multi-objective construction over Kirsch–Nowak bootstrap indices. Objectives: drought frequency, intensity, duration (relative to historical) + an L1 anti-ideal Manhattan-norm objective. Pareto front *is* the ensemble — near-uniform coverage of drought-characteristic space. Cites Borgomeo 2015 and Zaniolo 2023 as single-objective predecessors. Currently under development (per OQ-2 resolution).

### Ensemble-size guidance (Bonham 2024)

**Bonham et al. (2024)** provides direct methodological guidance on **how many stochastic realizations are needed** for robustness rankings to converge:
- Satisficing metrics converge faster (~50–300 scenarios) than regret-from-best metrics (~400–500).
- MSTmean (mean-of-MST-edges) is the best space-filling predictor of ranking convergence.
- Method is post-hoc (subsample existing baseline) — no extra simulation needed.

For us, Bonham 2024's framework applies primarily to the *re-evaluation* ensemble. The *search* ensemble is small (O(10)) by design — drought-characteristic-space coverage rather than statistical convergence is the relevant criterion for the search ensemble.

### How the DU-in-search story plays out in the manuscript

- **§Methods §3.5 (optimization):** DU-in-search per `Trindade (2017)` philosophy. Each NFE evaluates the candidate policy across the search-time ensemble; per-objective metric used by Borg's archive is satisficing rate / robustness aggregate.
- **§Methods §3.6 (ensemble design):** Two ensembles. Search (small, drought-spread, currently SynHydro K-N with stratified post-hoc selection → MOEA-FIND when finalized). Re-evaluation (large, Bonham-2024-sized, K-N).
- **§Discussion §5.4 (methodological generalizability):** if MOEA-FIND finalizes in time, claim C6 — first DU-optimization application of the MOEA-FIND drought-spread ensemble. Otherwise SynHydro K-N is fine; manuscript story remains intact, just a smaller methodological step.

### Open question: stationarity

The mFGN stationarity assumption may be challenged by post-1970 precipitation increase documented in `McCabe and Wolock (2020)`. Options: (i) accept the bias as a documented limitation; (ii) condition the generator on pre-/post-1970 separately; (iii) leave for future work given climate-perturbed scenarios are out of scope per OQ-5.

## 7. Direct policy search & policy-formulation comparison — our 3-policy ladder fits the literature

Five papers in our corpus speak directly to direct policy search and policy formulation comparison: **Giuliani et al. (2016)** (EMODPS canonical), **Quinn et al. (2017)** (rival framings), **Quinn et al. (2019)** (black-box sensitivity), **Herman et al. (2020)** (climate-adaptation control framing), and **Lin et al. (preprint, 2026)** (state-aware EMODPS for DRB thermal control — direct in-basin precedent by our group). Together they form a tightly integrated methodological lineage that our work plugs into directly.

**Lin et al. (preprint) is especially load-bearing:** it is the first study to define DRB thermal releases as state-aware EMODPS policies with thermal-mitigation-bank constraints, using 4 RBFs / 28 DVs / 3 objectives (`J_rel`, `J_add`, `J_tbur`). Our manuscript inherits the same Pywr-DRB v2 + LSTM coupling infrastructure but holds the T/S system as observe-only objectives and varies the *broader* policy architecture (FFMP rule re-tuning → FFMP_VR sweep → ANN structure-free) across 7-8 objectives — complementary studies addressing different methodological questions in the same simulation environment.

- **Giuliani 2016** formalizes EMODPS = DPS + nonlinear approximating networks (RBF or ANN) + multi-objective evolutionary algorithm (Borg). Demonstrates EMODPS dominates SDP even on simple cases SDP should win. Recommends RBF over ANN empirically for reservoir-ops problems. Our ANN sits in this framework.
- **Quinn 2017 (rival framings)** is our closest published analog. Where we vary policy *architecture* along a structural-flexibility axis (FFMP → FFMP_VR → ANN), Quinn 2017 varies problem *formulation* along an objective-quantification axis (expectation → percentile → min-max). Both are systematic explorations of formulation space beyond "pick one and optimize." Strong companion citation.
- **Quinn 2019** provides the **time-varying sensitivity-analysis framework** for "opening the black box" of nonlinear policies. Directly applicable to interpreting our ANN. Anchors a Methods §interpretability or Results §policy-explanation subsection.
- **Herman 2020** positions climate adaptation as a control problem and reviews the parametric-rules → DPS → MPC → RL spectrum. Our 3-policy ladder fits this spectrum.

**Together, the manuscript can:**
- **§Background — methods spectrum:** cite Herman 2020 to position our 3 architectures as instantiating the parametric → DPS → learned-policy spectrum.
- **§Background — formulation uncertainty:** cite Quinn 2017 to legitimize our rival-architectures approach as a structural extension of the rival-framings concept.
- **§Methods §3.2 (3-policy ladder):** cite Giuliani 2016 + Quinn 2017 jointly as the methodological foundation. Acknowledge that we hold problem formulation roughly fixed and vary architecture, complementary to Quinn 2017's choice.
- **§Methods §interpretability:** Quinn 2019 sensitivity analysis applied to our ANN gives reviewers a tractable explanation of how the network uses its 5 state inputs.
- **§Discussion — perverse-expectation warning:** Quinn 2017 finds that minimizing *expected* flood damage perversely *increases* catastrophic-flood risk. Our flood objective is frequency-based (days-exceeding-threshold), not expectation-based — but worth disclosing the choice and its implication.
- **§Discussion — RBF vs. ANN:** Giuliani 2016 recommends RBF over ANN for typical reservoir-ops; we chose ANN. Either defend the choice or flag it as a sensitivity check for future work.

## 8. DRB context — institutional, hydroclimatic, economic, and the Pywr-DRB simulation infrastructure

Six DRB-specific papers in our corpus:
- **Hamilton 2024 (Pywr-DRB v1)** — original Pywr-DRB simulation engine.
- **Lin et al. (preprint, 2026) — Pywr-DRB v2 + LSTM coupling.** Documents the modular extension we use (Pywr-DRB v2) and the Pywr-DRB-ML plug-in (TempLSTM + SalinityLSTM). **Trevor is third author.** Establishes the three coupling modes (sync / hybrid / async) — our `salinity_async = False` choice corresponds to Lin's "synchronized for salt-front control" recommendation.
- **Kolesar & Serio 2011** — pre-FFMP DRB OR analysis; rare quantitative DRB-ops paper. Useful Background context: there's almost no peer-reviewed MORDM-on-DRB literature, which our work fills.
- **McCabe & Wolock 2020** — DRB hydroclimate, drought regime characterization. Informs Phase 3 ensemble design (1960s drought-of-record framing; recent precipitation increase).
- **Moore 2021** — DRB governance politics; the institutional viability of any optimized policy depends on multi-party negotiation.
- **Kauffman 2016** — DRB economics. $22B annual activity, $21B ecosystem services, 600k jobs — quantifies the stakes of getting reservoir operations right.
- **Hogarty 1970** — institutional report on the 1960s drought emergency; *not yet readable* (scanned PDF without OCR). Pending OCR.

**For our manuscript:**
- **§Methods §3.1 (simulation environment):** cite Hamilton 2024 (Pywr-DRB v1) **and** Lin et al. (preprint) (Pywr-DRB v2 + LSTM plug-in). The simulation we run on is Pywr-DRB v2 with the LSTM coupling.
- **§Methods §3.4 (salt-front parameterization):** cite Lin et al. (preprint) for the SalinityLSTM (5 features: Q_T, Q_S, day-of-year, plus 7-day moving averages). Document the synchronized coupling-mode choice per Lin's recommendation.
- **§Introduction motivation:** Kauffman + Moore (stakes + governance).
- **§Background DRB context:** Hamilton + Lin + Kolesar (+ Hogarty once readable). Lin et al. provides the most current methodological framing for *why* coupled water-quantity-quality modeling is needed in the DRB.
- **§Discussion — institutional viability:** Moore + decree-party-decomposed robustness (Hadjimichael 2020).
- **§Discussion — thermal/salinity decoupling:** cite Lin et al.'s system-scale finding that thermal releases from Cannonsville have negligible influence on the salt front; useful to frame our salinity-as-objective decision.

### Sister paper: Amestoy et al. (2026) reconstructed inflows

Lin et al. uses **Amestoy, Hamilton & Reed (2026)** reconstructed DRB streamflows (1945–2023) — published in *Environmental Modelling & Software* 195, 106756. **Trevor is first author.** This product is the recommended Pywr-DRB v2 input. If we use the reconstructed inflows for our optimization period (which we should, given they cover 1945–2023 and resolve the 1960s drought-of-record), we will need to cite Amestoy 2026 in Methods as the streamflow source. Worth bringing the PDF into our corpus.

## 9. Deep-uncertainty foundations

The Marchau 2019 edited volume is the canonical synthesis of DMDU approaches (RDM, DAP, DAPP, Info-Gap, EOA). Maier 2016 provides the vocabulary clarification (deep uncertainty / scenarios / robustness / adaptation). Together they ground our deep-uncertainty framing without requiring readers to reach for older RAND-RDM source material.

**For our manuscript:** §Background's deep-uncertainty paragraph should cite Marchau 2019 (Ch. 1, Lempert et al. 2003 definition reproduced there) + Maier 2016 (vocabulary). Then transition to MORDM via Kasprzyk 2013.

---

## What's missing — none blocking (post Lin et al. addition 2026-04-30)

The lit review is **complete** for this manuscript. The 25-paper corpus covers all methods, framing, and DRB context needed.

- **Lin et al. (preprint)** has now arrived — the LSTM source paper Trevor was waiting for is in the corpus.
- **Amestoy et al. (2026) reconstructed streamflows** — sister paper by Trevor; not yet in our corpus but methodologically central if we use reconstructed inflows. Decision pending whether to bring into our corpus or treat as external (Trevor's own paper, possibly already fully internalized).

Other previously-listed wishlist items (Hadka 2013, Nowak 2010, Bryant & Lempert 2010, etc.) were formally deferred per Trevor's 2026-04-30 review (see [reading_list.md](reading_list.md)).

**The manuscript-scaffolding gate (per the 2026-04-30 sequencing decision) is now passed.** The next step is to stand up `local_notes/manuscript/` (outline, motivation, target venue, figure plan, open methodological questions).

## Notes on note-quality (transparency)

During the lit-review process, **two agent-written notes (Herman 2015, McPhail 2018) were detected as having fabricated quotes and a wrong author list**, respectively, and were rewritten from direct PDF extraction.

A subsequent **systematic audit on all remaining 9 agent-written notes** (Quinn 2019, Herman 2020, Bonham 2024, Moore 2021, Kauffman 2016, Herman 2014, Maier 2016, Kolesar 2011, McCabe 2020) was performed: each PDF was extracted via pypdf, and every quoted claim in each note was searched in the source PDF text. **All 31 quoted claims across the 9 notes were verified** as appearing verbatim or with strong anchor matches in their source PDFs; all 9 citations were verified against PDF first pages. **No further fabrications detected.**

The Hogarty 1970 note is incomplete because the PDF is a scanned image without an OCR text layer; explicitly flagged as deferred pending OCR.

The 3 batch-2 notes (Giuliani 2016, Kirsch 2013, Quinn 2017) were written directly by Claude from pypdf-extracted source text.

**Confidence summary:**
- Notes I wrote myself from extracted text (15 papers, including Lin et al. preprint): high confidence.
- Notes written by agents and audit-passed (9 papers): high confidence.
- Hogarty 1970: low confidence pending OCR — explicitly disclosed.
