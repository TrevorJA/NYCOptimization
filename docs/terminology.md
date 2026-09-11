# Terminology Reference

Single source of truth for terminology in the NYC re-optimization committee proposal
and, going forward, the manuscript. Every term is used identically in text, tables,
captions, and the Supporting Information. Each entry cites its support: the official
FFMP document (Decree Parties, 2017, cited by page), the Pywr-DRB, reconstruction, and
stochastic DRB papers (Hamilton et al., 2024; Amestoy et al., 2026; the sibling
stochastic-exploratory manuscript, cited as "sibling manuscript"), or the Reed-group
methods literature (verbatim usage verified against the full texts of Quinn et al.,
2017; Zatarain Salazar et al., 2017; Trindade et al., 2017; Hadjimichael et al., 2020;
Hadka and Reed, 2015; Gold et al., 2023; all in Zotero collection `ISYGLK35`). Acronym
discipline follows `docs/manuscript/acronyms.md` (each acronym defined exactly once at
first use). The literature claims that motivate the study are collected, with verified
summaries, in `docs/study_motivation.md`.

Status legend
- **Adopted.** Locked for use; supported by the cited source and/or Trev's comments.
- **Avoid.** Do not use; the replacement is listed.

---

## A. Institutions and current operations

| Status | Term | Use and support |
|---|---|---|
| Adopted | **current operations** | Umbrella for the FFMP, the 1954 Decree, and DRBC regulation as collectively implemented (sibling manuscript, Section 2.1: "we use the term 'current operations' to refer to the combined framework of the FFMP, the 1954 Supreme Court decree, the DRBC Water Code, and associated DRBC regulations"). |
| Adopted | **current FFMP policy** | The simulated 2017 FFMP policy that is the reference for all regret comparisons. Literature anchor is the "status quo (prespecified) solution" of Herman et al. (2015); "status quo policy" is an acceptable variant. Matches the existing figure legends ("Current FFMP policy"). |
| Adopted | **Flexible Flow Management Program (FFMP)** | The program's official name (FFMP 2017 Operations Plan, p. 1). Never "Plan". Refers only to the FFMP-defined rules, never the broader institutional umbrella. |
| Adopted | **operating rules** | The FFMP's rules as negotiated by the Decree Parties: the seasonally varying rule curves, the zone-conditional diversion and flow-target reductions, and the release schedules. The decision variables **parameterize the FFMP's operating rules** while preserving the rule structure (register precedent: Cohen et al., 2021, parameterize existing operating rules with 34 decision variables). Where the numerical values themselves are meant, write "the values that define the operating rules" or "rule parameters". Replaces "negotiated quantities" (Trev, 2026-09-02). |
| Adopted | **FFMP rule structure** | The program's structural logic: storage zones delineated by seasonally varying rule curves, zone-conditional operations, running-average diversion accounting. |
| Adopted | **adaptive operating rules / zone-dependent** | The FFMP is an adaptive policy in the sense that the allowable NYC diversion, the NJ diversion, and the Montague and Trenton flow targets are **reduced with each successive drought stage** (sibling manuscript, Section 2.1: "declining reservoir storage triggers successively more restrictive operating zones that reduce the allowable NYC diversions and the Montague flow target"). Write "reduced with each successive drought stage" or "successively more restrictive"; never "deepen", "deeper stages", or "waive". |
| Adopted | **rule curves** | The seasonally varying storage curves that delineate the FFMP's storage zones. Official term ("the three drought management rule curves," FFMP p. 4) and the standard water-resources term. Never "zone boundary curves." |
| Adopted | **storage zones** | The FFMP delineates five zones of combined NYC storage (FFMP p. 4): the spill mitigation zone (L1), the normal zone (L2), and the drought stages Drought Watch (L3), Drought Warning (L4), and Drought Emergency (L5). Prose uses the full zone names, with the L labels given once at first definition and retained only in tables. Zone names are capitalized as the FFMP writes them (Drought Watch, Drought Warning, Drought Emergency). The reservoir-specific spill mitigation sub-zones (L1-a, L1-b, L1-c; FFMP pp. 13, 21) appear **only in the Supporting Information** of the proposal; they are not needed to understand the experimental design (Trev, 2026-09-02). |
| Adopted | **conservation releases** | The scheduled ecological releases of the FFMP release schedules (Tables 4a–4g), official category name (FFMP p. 2). Enhanced releases above the base schedule are selected by Forecast-based Available Water (FAW) assessments. Mentioned briefly; the proposal's emphasis is on the NYC diversion and the directed releases to Montague, which are the large release volumes. |
| Adopted | **directed releases** | NYC releases directed by the Delaware River Master to meet the Montague flow target ("as directed by the River Master," FFMP p. 2). Long form at first use: "directed NYC releases to Montague" (sibling manuscript, Section 3.4). |
| Adopted | **spill mitigation releases** | The enhanced releases made in the spill mitigation zone to lower storage, capture storm runoff, and reduce spills and downstream flooding (FFMP p. 21). "Spill" alone is reserved for uncontrolled spillway flow. Not a focus of the proposal main text. |
| Adopted | **NYC diversion / NJ diversion** | The transbasin diversions (800 MGD and 100 MGD allocations). "Transbasin diversion" as the generic term at first definition (Hamilton et al., 2024; sibling manuscript). **"Diversion" is the only noun for delivered water**, in prose, objective names ("NYC diversion reliability", "NYC diversion deficit", "NJ diversion reliability"), tables, and captions (Trev, 2026-09-02). Decree accounting limits the running average of diversion, not any single day: NYC's annual running average resets June 1, NJ's is a monthly average (FFMP pp. 1, 3). |
| Adopted | **Montague flow target** | The Decree-mandated minimum flow of 1,750 cfs at Montague, NJ. The official instrument name, the Montague Flow Objective (FFMP p. 2), appears once parenthetically at first definition; prose thereafter uses "Montague flow target", reserving "objective" for the optimization formulation. |
| Adopted | **Trenton flow target** | The 3,000 cfs flow target at Trenton, NJ, established under the DRBC Water Code and not legally binding on NYC (Amestoy et al., 2026). Official instrument name, the Trenton Equivalent Flow Objective (FFMP p. 2), once parenthetically at first definition. Never "Trenton objective" in prose. |
| Adopted | **combined NYC storage** | Aggregate usable storage in the three NYC Delaware Basin reservoirs (Cannonsville, Pepacton, Neversink). Long form at first use; always with "combined" (sibling manuscript). |
| Adopted | **salt front** | The 250 mg/L isochlor in the Delaware Estuary (FFMP p. 7). Its 1960s position: within 12.9 km of Philadelphia's drinking water intake (Kolesar and Serio, 2011; Amestoy et al., 2026). |
| Adopted | **1954 Decree / Decree Parties** | The 1954 Amended Decree of the U.S. Supreme Court in *New Jersey v. New York* and its five parties, the four basin states and NYC (FFMP p. 1). Statements about the Decree are exact: it **sets** the NYC diversion allocation and the Montague flow target and is **not renegotiated**; the optimization **maintains the Decree's constraints** (the 800 MGD and 100 MGD allocations, the Montague flow target) and the physical release limits. Capitalization follows the FFMP document ("Decree Parties"); the sibling manuscript writes "decree parties" and "1954 decree", a difference flagged for the author to settle across the two papers. |
| Adopted | **1960s drought of record** | The 1961 through 1967 drought that forced emergency negotiations among the Decree Parties when storage could not support both the decreed diversion and the Montague releases, coupling diversion curtailments with flow-target reductions (Hogarty, 1969; Kolesar and Serio, 2011; Amestoy et al., 2026). Cite this triplet for operational deterioration during the drought. Hogarty's Inter-University Case Program report is dated 1969 in Zotero and 1970 in the DRBC library filename; the author settles the year. |
| Adopted | **FFMP-year** | The June 1 through May 31 year on which the FFMP's diversion accounting and seasonal rules reset (FFMP pp. 1, 3). Hyphenated, matching the sibling manuscript ("Storage-zone and directed-release metrics are reported on the FFMP-year, defined as June 1 through May 31"). Defined descriptively at first use, in its own sentence, never mid-sentence inside another definition. |

### Avoid

| Avoid | Replacement |
|---|---|
| "incumbent" (any usage) | current FFMP policy / current operations |
| "baseline policy" / "baseline operations" | current FFMP policy ("baseline" collides with the baseline-SOW usage of Herman et al., 2015) |
| "status quo" as a bare noun | current operations |
| "negotiated quantities" / "negotiated constants" / "constants" | operating rules; rule parameters for the numerical values |
| "delivery" / "deliveries" / "delivery reliability" | diversion / diversions / diversion reliability |
| "deepen" / "deeper drought stages" / "waive" | reduced with each successive drought stage; successively more restrictive; a zero reduction at a stage |
| "ancestors" (of today's rules) or any lineage metaphor | "the emergency measures negotiated then were formalized over the following decades into the storage-conditioned drought rules" |
| "discharge mitigation releases" (project prose) | spill mitigation releases |
| "zone boundary curves" / "boundary curves" | rule curves |
| "Trenton objective" (operations prose) | Trenton flow target |
| FFMP as umbrella for current operations | current operations |
| "mandated release" | directed releases |
| "renegotiation of the Decree" | the Decree is not renegotiated; write "the optimization maintains the Decree's constraints" |
| L1-a/L1-b/L1-c sub-zone detail in main text | Supporting Information only |

## B. The optimization study

| Status | Term | Use and support |
|---|---|---|
| Adopted | **many-objective** | Problems with four or more objectives (Reed et al., 2013). Hyphenated as an adjective. "Multiobjective" only inside proper names (multiobjective evolutionary algorithm). |
| Adopted | **search ensemble** | The fixed ensemble of streamflow realizations on which every candidate policy is evaluated during the optimization search. Reed-group forms are "synthetic streamflow ensembles" and "the streamflows over which they were optimized" (Quinn et al., 2017); "search ensemble" is this study's compound, defined at first use ("the ensemble of streamflow realizations over which candidate policies are evaluated during the optimization search, hereafter the search ensemble"). Never hyphenated; adjectival compounds are restructured ("the construction of the search ensemble"). |
| Adopted | **constructing / construction of the search ensemble** | The single verb and noun for how a search ensemble is built. "In this study, we present an alternative method for constructing the streamflow ensembles used during many-objective optimization of water resources systems operations." Never "composing / composition" (Trev, 2026-09-02). |
| Adopted | **candidate ensemble** | The large ensemble of streamflow realizations generated independently from the stationary generator, from which the HF design selects. Defined at first use as "a large ensemble of candidate streamflow realizations". Replaces "candidate pool" ("pool" is informal). Its members are "streamflow realizations", never "real candidates". |
| Adopted | **realization** | One synthetic or observed streamflow sequence. "Ensemble member" is the accepted synonym (Quinn et al., 2017); "realization" is the Trindade et al. (2017) and Hadjimichael et al. (2020) form. "Trace" and "scenario" are never used for a streamflow record. |
| Adopted | **scenario design** | A rule for constructing the search ensemble. The three designs: Historical (HIST), Monte Carlo Sampling (MC), Hazard Filling (HF). Each acronym is defined at its first use and used exclusively thereafter ("the HF design", "the MC design"); never bare "hazard filling" or "Hazard Filling" after the acronym exists, and never bare "HF" without "design" when the design is meant. "The matched designs" collectively for MC and HF. With three designs, write "each design", never "both" (Trev, 2026-09-02). |
| Adopted | **naming the method after describing it** | The proposed method is described before it is named: state what the construction does (selects realizations from a large candidate ensemble so that the selected members span and uniformly cover a multi-dimensional space of hydrologic hazard characteristics), then "We refer to this construction as the Hazard Filling (HF) design." Never open with "we present hazard filling" (Trev, 2026-09-02). |
| Adopted | **size notation** | "$N$ = 300 realizations of $L$ = 10 years" (Quinn et al., 2017: "N = 50 ensemble members of length T = 20 years"); "3,000 years of streamflow per function evaluation". The unit noun follows the numeral. |
| Adopted | **design choices stated as design choices** | $L$ = 10 years, the epsilon-dominance precisions, and $N$ = 300 are design choices with a stated rationale, written plainly: "We use 10-year realizations, a design choice that keeps the event-specific drought and flood metrics of Section 3.2 interpretable"; "Initial diagnostics determined that $N$ = 300 is a sufficiently large ensemble size that the sampling noise of every objective is smaller than the objective precision specified for the search" (Trev, 2026-09-02). |
| Adopted | **candidate policy / policy** | One complete assignment of decision-variable values. Never "solution" or "alternative" unqualified. |
| Adopted | **decision variables** | The 36 parameters of the FFMP operating rules searched by the optimization. |
| Adopted | **during the optimization search** | The phrase for the search phase. Never "the optimizer" as an agent (Trev, 2026-09-02). |
| Adopted | **Pareto-approximate set** | The nondominated policies from a search (Quinn et al., 2017: "best known approximations of the Pareto optimal sets"). Seed archives are "combined and re-sorted" (Quinn et al., 2017) into a single best known Pareto-approximate set per design. |
| Adopted | **function evaluations (NFE)** | Spelled out at first use with the abbreviation ("number of function evaluations, NFE"; Zatarain Salazar et al., 2017). Budgets state both the per-island and total counts ("500,000 function evaluations per design, 125,000 per island") because Hadka and Reed (2015) count total NFE. NFE "is the key controlling parameter" of Borg (Hadka and Reed, 2015); search terminates at the specified number of function evaluations (Gold et al., 2023). Never "termination criterion" or "stopping rule". |
| Adopted | **MM Borg component names** | "The multi-master Borg multiobjective evolutionary algorithm (MM Borg; Hadka and Reed, 2013, 2015)". Components in the authors' words: ε-dominance archive; ε-progress triggered restarts; auto-adaptive multi-operator search (adaptive operator selection); adaptive population sizing. Architecture: a hierarchical parallelization in which multiple **islands**, each a **master-worker** instance of the Borg MOEA, co-evolve through a **controller** that maintains a **global ε-dominance archive** and global operator probabilities and provides **guidance** to islands whose local restarts fail to escape stagnation (Hadka and Reed, 2015; Quinn et al., 2017: "multiple master-worker implementations of the Borg MOEA, called islands, which coevolve through the aid of a controller that keeps a global archive"). "Master-worker" throughout (Gold et al., 2023), never mixed with "master-slave". |
| Adopted | **epsilon-dominance precision** | Per-objective "levels of precision for each objective below which they are indifferent to differences in performance" (Quinn et al., 2017); "epsilon (or significant precisions)". Governs the archive's resolution between policies. |
| Adopted | **random seed trials / seeds** | Independent trials "to account for variability in their initial populations and operator probabilities" (Zatarain Salazar et al., 2017). The seed count is justified on evaluation cost with the precedent of Bartholomew and Kwakkel (2020) and Gold et al. (2023), and seeds are presented as contributors of search diversity, not as a basis for a variance estimate. |
| Adopted | **runtime diagnostics** | "Runtime hypervolume" relative to each design's own best known set, with ε-progress tracked as a function of NFE (Hadka and Reed, 2013 for the definition of ε-progress; Bartholomew and Kwakkel, 2020, and McPhail et al., 2020, for its use as convergence evidence). Convergence language: "progress had reached an asymptote of diminishing returns" (Quinn et al., 2017). Diagnostics are computed within each design and never compared across designs. |
| Adopted | **two-stage aggregation (Φ, Ψ)** | Every objective composes a within-year temporal aggregator Φ and an across-year summary statistic Ψ over the pooled FFMP-years of all realizations, following the Φ/Ψ operator formalism of Quinn et al. (2017, 2018), where Ψ "is a statistic used to filter the noise across ensemble members". |
| Adopted | **out-of-sample** | For evaluation on data not used during search ("reevaluating the optimized policies on an out-of-sample set of stochastic inputs to ensure that they generalize well," Quinn et al., 2017). "Generalize" is likewise supported. |

### Avoid

| Avoid | Replacement |
|---|---|
| "composing" / "composition" (of the ensemble) | constructing / construction |
| "the optimizer" | during the optimization search; the search |
| "both" (of the designs) | each design; the two matched designs (only when MC and HF alone are meant) |
| "hazard filling" bare, "Hazard Filling" bare after first use | the Hazard Filling (HF) design at first use; the HF design thereafter |
| "training ensemble" / "training scenarios" | search ensemble (the "training" form is Cohen et al.'s (2021) usage, referenced only when describing that study) |
| "trace" / "scenario" (for a streamflow record) | realization |
| "real candidates" | streamflow realizations |
| "solution" / "alternative" (unqualified) | candidate policy / policy |
| "termination criterion" / "tuned termination" | the specified number of function evaluations; NFE as the key controlling parameter |
| "master-slave" | master-worker |
| "controlled contrast" | state the matching plainly (the designs are matched in ensemble size, record length, generator, and budget) |
| "exact statistical control" as a slogan | state the property once, plainly, or omit |
| "held-out" | out-of-sample; or "never used during any search" |
| "selection rule" | the construction of the search ensemble; how realizations are chosen. Attribute policy differences to differences in the streamflow conditions the ensembles present during evaluation |
| "candidate pool" / "pool" | candidate ensemble |
| "design-time" | rephrase precisely (e.g., "during search", "fixed before the search begins") |
| Obvious statements ("HIST cannot be matched in $N$ and $L$"; "hazard characteristics emerge only once a sequence exists") | omit |

## C. Hazard space and hazard metrics

| Status | Term | Use and support |
|---|---|---|
| Adopted | **aggregate NYC inflow** | Daily sum of inflows to the three NYC reservoirs; long form "aggregate inflow to the three NYC reservoirs" at first definition (sibling manuscript). |
| Adopted | **hazard metrics** | The quantities computed on each realization before any system simulation (sequence-level ensemble characterization per Salehabadi et al., 2024). Never "descriptors," "indicators," or "candidate metrics." |
| Adopted | **hydrologic hazard** | The drought and flood conditions expressed in a specific streamflow sequence, within the risk framing in which risk arises from the interaction of hazards with the exposure and vulnerability of the affected system (Simpson et al., 2021). The **motivating property** is independence from the managed system: hazards are characterized on the streamflow sequence alone, independently of the system's operations, response, or outcomes, so the operational consequences of each hazard are resolved by the simulation rather than embedded in the characterization (sibling manuscript, Section 3.3; Hadjimichael et al., 2020; AghaKouchak et al., 2021). The absence of any simulation is a consequence of this property, stated once at most, never as the motivation (Trev, 2026-09-02). |
| Adopted | **hazard characteristics** | The quantified hazard properties of one realization (its hazard metric values). "Properties of specific streamflow sequences", never "properties of the hydrology" (Trev, 2026-09-02). Replaces "hazard coordinates" everywhere. |
| Adopted | **water system outcomes** | The objective values a policy attains under a realization, which require simulation of the water system. Never "impacts" (Trev, 2026-09-02). |
| Adopted | **hazard space** | The multi-dimensional space with the six selection axes as dimensions, in which each realization occupies a position given by its hazard characteristics. |
| Adopted | **diversity / range / coverage** | The HF design's goal vocabulary, defined at first use and reused thereafter. **Diversity**: variety among the hazard characteristics of the ensemble members, so that redundant streamflow hazard conditions are not repeatedly evaluated during the search. **Range**: the span of each hazard characteristic from benign conditions to droughts and floods more extreme than any observed event. **Coverage**: uniform filling of the hazard space, per axis and jointly. Contrast stated precisely: the stationary generator produces marginal and joint hazard distributions reflective of the historical record, so the MC ensemble concentrates near the central mass of the joint hazard distribution, whereas the HF design seeks uniform coverage across the hazard space (Trev, 2026-09-02). |
| Adopted | **probability and frequency language** | An independently sampled ensemble represents hydrologic conditions in proportion to their probability under the fitted generator; the HF design over-represents rare conditions relative to their frequency under the generator and, by construction, distorts the marginal and joint distributions of the hazard characteristics. Use "probability" for the generator's distribution and "frequency" for how often conditions appear in a finite ensemble; never "likelihood" in this context. |
| Adopted | **selection axes** | The six hazard metrics that enter the selection distance (drought magnitude, severity, onset rate, recovery rate; peak discharge, pulse duration). |
| Adopted | **target hazard characteristics** | The Latin hypercube sample points in the scaled hazard space to which streamflow realizations are assigned. Replaces "anchor" and "anchor points" (Trev, 2026-09-02). |
| Adopted | **SSI-6** | The six-month Standardized Streamflow Index on aggregate NYC inflow (Vicente-Serrano et al., 2012), with gamma distributions fitted per calendar month to the historical record and held fixed. The accumulation window is a design choice: the six-month window aligns with the semi-annual-to-annual drawdown and refill cycle of combined NYC storage (sibling manuscript, Section 3.3). Register for the event definition follows the sibling manuscript: "a threshold-based definition" with a "retention threshold" and a "termination requirement" (Fleig et al., 2006; Van Loon, 2015). Never "run theory". |
| Adopted | **largest drought event** | The drought event scored for a realization when more than one qualifies: the event with the largest drought magnitude, that is, the largest accumulated SSI-6 deficit, stated explicitly at first use. Replaces "controlling event" and "controlling drought" everywhere (Trev, 2026-09-02). |
| Adopted | **drought magnitude, M (deficit-months)** | Accumulated SSI-6 deficit over the event's deficit months, "in units of deficit-months" (sibling manuscript, Eq. 3; McKee et al., 1993; Van Loon, 2015). Replaces "SSI-months"; figure axes are to be relabeled. |
| Adopted | **drought severity, S** | "The absolute value of the minimum SSI reached during the event, so that larger severities indicate deeper deficits" (sibling manuscript, Eq. 2; McKee et al., 1993; Van Loon, 2015). Never "intensity" or "peak deficit". |
| Adopted | **onset rate / recovery rate** | This study's normalizations of severity by the months from event onset to the minimum index value and from the minimum to event termination. They have no precedent in the sibling manuscript and are defined explicitly as this study's metrics of the development and termination phases of an event. |
| Adopted | **peaks-over-threshold** | The flood-metric framework; threshold fixed at the historical 95th percentile of daily flow. Metrics: **peak discharge, D** (maximum daily flow normalized by the historical mean daily flow) and **pulse duration, $T_P$** (days of the exceedance run containing the maximum). Citation for the framework is settled in `docs/study_motivation.md`. |
| Adopted | **"extreme"** as the plain adjective | For conditions beyond the observed record ("droughts and floods more extreme than any observed event"). |

### Avoid

| Avoid | Replacement |
|---|---|
| "severe" / "severity" outside the SSI severity metric | extreme, deep, prolonged, demanding (severity is a defined metric, Trev: "NEVER USE 'severe' OUTSIDE OF SSI DROUGHT METRIC DEFINITIONS") |
| "magnitude" / "intensity" outside their SSI definitions | plain words; "intensity" is never used |
| "controlling event" / "controlling drought" / "controlling flood" | the largest drought event (by accumulated SSI-6 deficit); the flood pulse containing the maximum daily flow |
| "anchor" / "anchor points" | target hazard characteristics |
| "SSI-months" | deficit-months |
| "run theory" | threshold-based event definition (Fleig et al., 2006; Van Loon, 2015) |
| "hazard coordinates" | hazard characteristics |
| "properties of the hydrology" | properties of specific streamflow sequences |
| "impacts" (for objective values) | water system outcomes |
| "informative conditions" and other undefined qualifiers | the defined vocabulary: diverse, extreme, range, coverage |
| "defined prior to simulation", "at negligible cost", "millions of candidates are practical" as motivation | characterized independently of system operations, response, or outcome |
| "input space" | do not coin; describe the generator and the forcing space directly |
| "stress space" (for the hazard space) | reserved for the climate-stress-testing concept in the literature review (Fowler et al., 2024) |
| "forcing" (for aggregate NYC inflow) | inflow; "forcing" is reserved for the climate-forcing context of Section D |

## D. Deep uncertainty re-evaluation and robustness

| Status | Term | Use and support |
|---|---|---|
| Adopted | **deep uncertainty / deeply uncertain** | Factors "for which expert opinion cannot know or cannot agree on the full set of outcomes and their associated likelihoods" (Hadjimichael et al., 2020; Lempert et al., 2003; Walker et al., 2013). Defined once. |
| Adopted | **well-characterized uncertainty (WCU)** | Stochastic variability with reliable historical data or known distributions (Trindade et al., 2017). In this study, natural hydroclimatic variability is well characterized; the magnitude and seasonal structure of future streamflow change is deeply uncertain and is sampled along **three forcing dimensions** (annual flow volume, seasonal amplitude, and the shape of the snowmelt shoulder). Scope restrictions are stated as "demand growth is not included as an uncertainty" and "no financial factors are considered" (Trev, 2026-09-02). |
| Adopted | **many-objective robust decision making (MORDM)** | The framework of Kasprzyk et al. (2013) whose re-evaluation step this study follows; named at first use of the re-evaluation approach. |
| Adopted | **state of the world (SOW)** | "A fully specified world, comprised of one fully specified sampled vector of deeply uncertain factors and one streamflow time series" (Trindade et al., 2017). Here, one Latin hypercube point in the forcing space crossed with its stochastic realizations (structure as in Hadjimichael et al., 2020). Open, lowercase, abbreviated SOWs after first use. The unit of every robustness and regret fraction ("per-SOW"). |
| Adopted | **deep uncertainty re-evaluation (DU re-evaluation)** | The evaluation of every Pareto-approximate policy on the common re-evaluation ensemble (Trindade et al., 2017). "Re-evaluate" is the verb of the Reed-group papers (AGU journals close it to "reevaluate"); "stress test" only as a verb and sparingly (Gold et al., 2023). |
| Adopted | **re-evaluation ensemble** | The common ensemble of SOWs and realizations on which every design's policies are compared. It is never used during any search. |
| Adopted | **forcing space** | The three-dimensional harmonic amplitude space of the re-evaluation design. "Forcing" is used only in this climate context, and "forcing" here refers to the imposed monthly change factors on the generator's fitted moments, not to streamflow itself. |
| Adopted | **generalization test** | The re-evaluation tests whether the policies produced under each scenario design generalize to forcing conditions absent from every search ensemble. Section 3.5 ends on this statement (Trev, 2026-09-02). |
| Adopted | **satisficing robustness / domain criterion** | The fraction of sampled SOWs in which a policy meets all performance requirements (Starr, 1962; Herman et al., 2015: "multivariate domain criterion satisficing measure"; Hadjimichael et al., 2020: "the domain criterion satisficing metric (Starr, 1962)"). "Satisficing criteria" and "performance requirements" are the compound forms. "Satisficing" appears only in this re-evaluation context. Criteria placements are explained, with each criterion's source stated (Decree or FFMP quantity, or the current FFMP policy's attainment). |
| Adopted | **regret** | Deviation from a reference, per the taxonomy of Herman et al. (2015) and McPhail et al. (2018). This study's reference is the current FFMP policy evaluated in the same SOW; regret in a SOW means the Decree Parties would have fared better in that future by retaining current operations. Reported per objective in natural units, never aggregated across objectives. |
| Adopted | **low-regret SOW / low-regret frequency** | A SOW in which every objective performs as well as or better than the current FFMP policy, within a stated tolerance, is a low-regret SOW; the fraction of such SOWs is the low-regret frequency. The regret framing is used throughout; the inverse "no-harm" vocabulary is retired from prose, and figures using it are to be relabeled (Trev, 2026-09-02). |
| Adopted | **robustness conflicts** | Where robustness gains for one actor degrade another (Trindade et al., 2017). Actor-level results are never aggregated away (Hadjimichael et al., 2020; Sunkara et al., 2023). |
| Adopted | **scenario discovery / factor mapping** | The classification of the SOW ensemble into success and failure regions of the forcing space (Trindade et al., 2017; Hadjimichael et al., 2020). Open, lowercase. Products are "factor maps"; identified regions are "consequential scenarios". |

### Avoid

| Avoid | Replacement |
|---|---|
| "scenario" (for a re-evaluation point) | SOW |
| "test ensemble" / "validation set" | re-evaluation ensemble |
| "enters the analysis" / "enters the study" (any variant) | "is not included as an uncertainty"; "is sampled only in the re-evaluation ensemble" |
| "no-harm" / "harm frequency" / "zero harm" | low-regret SOW / low-regret frequency / regret frequency |
| "No robustness value reported in this study is an expectation" and similar | omit; state only that every sampled SOW counts equally |
| "stress test" as a noun | re-evaluation; DU re-evaluation |
| "comprehensive" DU claims | the sampled deep uncertainty is hydroclimatic forcing only; scope stated |
| "the single deeply uncertain dimension" | three deeply uncertain forcing dimensions |

## E. Variability and uncertainty language

| Status | Term | Use |
|---|---|---|
| Adopted | **natural variability** | The basin's intrinsic hydroclimatic variability sampled by the stationary generator. Never "internal variability" (GCM initial-condition meaning). |
| Adopted | **stationary stochastic generator** | "The stochastic generator is held fixed" or "the generator is stationary". Never "the generating process" (confusing; Trev, 2026-09-02). "Stationary" modifies the generator only, never the basin's climate. |
| Adopted | **synthetic / stochastic** | "Synthetic" modifies generated flows ("synthetic streamflow ensembles", "synthetically generated streamflows"); "stochastic" modifies the generator and the inputs seen by the policy ("stochastic streamflows"; Reed-group usage across the four verified papers). |
| Adopted | **plausible** | Describing conditions the ensembles produce. |
| Adopted | **statistically consistent with** | Comparing ensemble outputs to the historical record. |

## F. Quantitative and formatting conventions

| Status | Convention |
|---|---|
| Adopted | Annual aggregation unit is the FFMP-year (June 1 through May 31), defined descriptively at first use in its own sentence. |
| Adopted | Simulated record lengths in plain years, with the partition into SOWs and realizations stated explicitly ("500 SOWs, each with 50 realizations of 50 years, 25,000 realizations and 1,250,000 years of simulated streamflow"). |
| Adopted | Fractions on 0 to 1 scales unless expressly in percentage points. |
| Adopted | Parallel computing language limited to nodes, cores, islands, and service units; no code-level identifiers in text or figures. |
| Adopted | Figure references: "Figure 5" whole figure, "Figures 5a–5c" panel sets, "Figure 5a" single panel. |
| Adopted | Acronyms per `docs/manuscript/acronyms.md`: defined exactly once at first use, never redefined, never opening a sentence. |
| Adopted | Units: MGD and cfs as the FFMP writes them, defined at first use. |

## G. Style reservations (cross-cutting)

- **"objective"** — only the eight formal optimization objectives. Operational
  instruments are flow targets (Section A).
- **"trade-off"** — only for conflicts among objective values in optimization results,
  never for methodological tensions.
- **"satisficing"** — only the re-evaluation robustness family (Section D).
- **"forcing"** — only the climate-forcing context of the re-evaluation (Section D).
- **"robust / robustness"** — only the robustness analysis; percentile scaling bounds
  are "1st and 99th percentile bounds," not "robust ranges."
- **"Monte Carlo"** — the MC design's proper name and sampling-theory statements only.
- **"severe"** — only the SSI severity metric (Section C).
- **Cost and convenience are never motivations.** The motivation for hazard
  characterization is independence from system operations, response, and outcome; the
  motivation for the search ensemble's construction is the diversity, range, and coverage
  of streamflow hazard conditions presented during the search.
- No invented compound jargon ("controlled contrast," "external anchor," "signed
  advantage," "natural stress index," "probability-faithful," "design-time," "hazard
  filling" as a bare verb phrase); write the underlying statement in plain terms.
- No metaphor in technical description ("lens," "anchor," "ancestors," "companion,"
  "live," "goalposts").
- No vague or informal paragraph openers. The first sentence of every paragraph states
  the paragraph's substantive point in terms the reader already knows ("The scenarios
  used during the optimization search shape the characteristics of the resulting
  policies", not "Characterizing what an ensemble contains requires a precise
  vocabulary").
- No obvious statements and no self-undermining literature summaries (a prior study
  is characterized exactly, with its nuance; see `docs/study_motivation.md`).
- Gap statements are general ("To the best of our knowledge, no study has..."), never
  built on one citation's call for future work.
- Statements touching the basin's legal arrangements are exact and cited.
- Prose contains no colons, semicolons, or dashes inside sentences (citation
  semicolons excepted); sentences are split instead.

## H. Reed-group register conventions (verbatim-verified)

- **Introduction architecture.** Stakes with citation density; methodological state of
  the art; the specific limitation ("However...", "Yet none of these studies..."); the
  contribution paragraph ("This study contributes / advances ... by demonstrating ...");
  research questions; roadmap sentence.
- **Citation-summary register.** Author-year as the subject plus a finite verb stating
  what the study did and found ("Cohen et al. (2021) found that...", "As reviewed by
  Herman et al. (2015), ...", "Following Kasprzyk et al. (2013), we ..."). One sentence
  per study, exact about what was compared and what was concluded.
- **Priority claims.** "To our knowledge, this is the first study to ..." with a narrow,
  verifiable X (Zatarain Salazar et al., 2017); "we believe this to be the first study
  to ..." (Hadjimichael et al., 2020).
- **Hedges.** "may promote", "at least when", "for the ... test case", "it cannot be
  inferred from these results that ... will always". Results verbs: "Our results show /
  indicate / highlight", "We find that".
- **MOEA paragraph order.** Algorithm with both citations and a prior-diagnostics
  justification (Reed et al., 2013); components in the authors' words; parallel
  architecture citing Hadka and Reed (2015); platform, nodes, islands, seeds with reason,
  default parameterization, NFE per island and total, epsilons with the indifference
  rationale; merged best known Pareto-approximate set; convergence evidence.
- **Synthetic streamflow paragraph order.** Purpose (extremes beyond the record);
  generator citations (Kirsch et al., 2013; Nowak et al., 2010); one-paragraph mechanical
  summary; validation pointer to the Supporting Information; $N$ × $L$ notation.
- **Robustness paragraph order.** Taxonomy cite (Herman et al., 2015; McPhail et al.,
  2018); the chosen measure named and justified; prose definition; display equation with
  an indicator function; criteria listed with their sources.
- **Voice and tense.** First-person plural for choices; passive present for procedure;
  present tense for framework definitions; past tense for what was run.

## I. Resolutions of previously open items

1. **Trenton.** "Trenton flow target" in all prose; the official Trenton Equivalent
   Flow Objective named once parenthetically at first definition.
2. **Zone labels.** FFMP labels with FFMP names, five combined-storage zones, L labels
   once at first definition; reservoir-specific sub-zones confined to the SI.
3. **Aggregation unit.** "FFMP-year", hyphenated as in the sibling manuscript.
4. **Drought magnitude units.** "deficit-months", as in the sibling manuscript; figure
   axes to be relabeled from "SSI-months".
5. **"Current operations" umbrella.** Adopted; "incumbent" is retired everywhere.
6. **Release-category names.** Official FFMP terms: conservation releases, directed
   releases, spill mitigation releases.
7. **Hazard vocabulary.** "Hydrologic hazard" motivated by independence from the
   managed system; "hazard characteristics" replaces "hazard coordinates"; goal
   vocabulary is diversity, range, and coverage as defined in Section C.
8. **Operating rules.** "Operating rules" replaces "negotiated quantities"; the
   decision variables parameterize the FFMP's operating rules.
9. **Diversion.** "Diversion" replaces "delivery" everywhere, including objective names.
10. **Largest drought event.** Replaces "controlling event".
11. **Target hazard characteristics.** Replaces "anchor points".
12. **Regret framing.** "Low-regret frequency" replaces "no-harm frequency".
13. **Method naming.** The HF design is named only after its construction is described;
    acronyms HIST, MC, HF are used exclusively after first definition.
14. **MM Borg language.** Component and architecture names per Hadka and Reed (2013,
    2015) and Gold et al. (2023); "master-worker"; NFE as the key controlling parameter;
    no "termination criterion".
