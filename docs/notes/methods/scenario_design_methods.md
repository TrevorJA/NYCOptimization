# Scenario-Design Generation Methods

*Construction recipe for every streamflow scenario set in the comparison. Terminology per `docs/notes/terminology.md`; the experiment itself is in `experimental_design.md`; objective formulations are in `objective_definitions.md`. Citations author-year, resolved via `docs/notes/literature/`. The manuscript draft (`docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md`) is the authoritative specification.*

---

## 1. Scope and the central asymmetry

Three **evaluation ensembles** (one per scenario design) and one held-out **test ensemble** for re-evaluation. All three designs are drawn from a single stationary population. The methodological contribution is **hazard filling**: an in-the-loop evaluation ensemble built by space-filling coverage of the empirical hazard manifold of a candidate pool, contrasted against random sampling from the same stationary generator.

Each design is constructed by its own recipe with its own seed stream. No design draws from another's data. This is what lets `monte_carlo` and `hazard_filling` each honestly represent the practice it stands for while sharing a population law.

Hazard filling is the only design that needs a candidate pool, and this is intrinsic rather than incidental. Hazard coordinates are emergent properties of a realized flow sequence, so no generator can be asked to produce a realization at a prescribed drought severity. A hazard-space design must therefore **select from** a finite pool (LHS anchors plus a nearest-neighbour snap), whereas the Monte Carlo design **generates** its realizations directly. The nearest-neighbour step is a consequence of this fact, not an approximation of a preferable procedure.

Deep uncertainty enters the study only through the test ensemble E_test (§5). Both search designs are stationary.

**Out of scope** (decided elsewhere): objective formulations and across-scenario aggregation (`objective_definitions.md`); MOEA internals (the `MOEAConfig` axis); robustness-metric mathematics.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $Q_{\text{obs}}$ | Observed/reconstructed daily multi-site historical record (`pub_nhmv10_BC_withObsScaled`); the simulated trace spans Dec 1945 – Nov 2023. |
| $r$ | A **realization**: one generator output sequence, seed-deterministic given a realization index. |
| $h(r) \in \mathbb{R}^{m}$ | **Hazard-metric vector**: $m$ drought and flood descriptors (§3.3). |
| $C$ | The **candidate pool**: $P$ i.i.d. realizations owned by the `hazard_filling` design. Never simulated in full — only its hazard image is stored. |
| $\mathcal{H}$ | The $P \times m$ **empirical hazard manifold** — the image of $C$ in hazard space. The object hazard filling covers. |
| $E_d$ | **Evaluation ensemble** for design $d$: size $N$, length $L$. |
| $E_{\text{test}}$ | The held-out **test ensemble** (§5): an **LHS** over the full DU forcing range, $N_{\theta,\text{test}}$ points × $R_{\text{test}}$ realizations each. The largest ensemble in the study. Never the source of any $E_d$. |
| $\theta_j$ (SOW) | One LHS point of $E_{\text{test}}$ — a **state of the world**, a deeply uncertain forcing vector. Its $R_{\text{test}}$ realizations sample natural variability within it. The unit of robustness; precision is governed by $N_\theta$, not $N_{\text{test}}$. |
| $K$ | Independent **ensemble draws** per design (a draw = the design's construction re-run from scratch with a fresh seed). |
| $S$ | MOEA random **seeds** per (design, draw). |

**Spaces:** **input space** (the DU forcing factors, entering only through $E_{\text{test}}$); **hazard space** (the empirical hazard manifold of the pool — strata are defined on realized marginals, not an a-priori domain); **outcome space** (simulated objectives).

---

## 3. Shared upstream machinery

One generator, one stationary population, used at every call site. The designs differ only in **whether a hazard image is streamed and used to select**.

### 3.1 The stationary population

The Kirsch–Nowak pipeline fitted to $Q_{\text{obs}}$, generating without climate perturbation. Cholesky-based monthly generation preserves cross-site and lag correlations (Kirsch et al. 2013) and a nearest-neighbour scheme disaggregates monthly to daily flows (Nowak et al. 2010). This is the flow model of the prevailing water-supply optimization literature (the Trindade/Gold/Zeff/Zatarain lineage). Hazard variation within it comes entirely from **natural variability**: some 10-year windows contain severe multi-year droughts and most do not.

Restricting the study to this one population is deliberate. It isolates RQ2 — whether the rule that selects realizations from a fixed generating process matters — without confounding that rule with a change in the generating process itself. The deeply uncertain climate-forcing space enters only in $E_{\text{test}}$ (§5), where it makes the re-evaluation a test of generalization to conditions absent from search. Prior work supports hazard-oriented selection within a stationary population: Zatarain Salazar et al. (2017) stratify a stationary pool by a realized-flow statistic, and Herman et al. (2016) amplify drought severity with a stationary weighted bootstrap.

### 3.2 The candidate pool must be sampled i.i.d., not LHS

The candidate pool is drawn **i.i.d.** from the stationary generator, never by LHS or any other structured design.

This is load-bearing. A uniform random size-$N$ subset of an i.i.d. pool has exactly the joint law of $N$ fresh i.i.d. draws. That is what makes `monte_carlo` the **exact statistical control** for `hazard_filling`: the two differ only in the selection rule applied to the same population law, and any difference in re-evaluated robustness is attributable to that rule alone. A random subset of an LHS design is not i.i.d., so an LHS-sampled pool would silently void the control. An invariant test enforces the i.i.d. condition, because nothing else in the pipeline would fail if it were broken.

The pool owns $P = 10^6$ realizations (§6) and is never simulated. Only each realization's hazard coordinates and generation seed are stored, and the $N$ selected members are regenerated exactly on demand from the deterministic, globally indexed random-stream architecture (§3.4). This storage arrangement is what makes a $10^6$-member pool tractable.

### 3.3 Realizations, windows, and hazard metrics

**Windows.** Disjoint, non-overlapping $L$-year blocks, December-aligned: every realization spans December 1 of year 0 through November 30 of year $L$ (`ENSEMBLE_START_DATE`). The generator synthesizes calendar-year (January-anchored) sequences — that structure cannot be rotated — so generation produces $L+1$ calendar years and trims the monthly frames to the epoch window before disaggregation; every stamp stays true by construction. Each scenario starts from fixed initial storage (`INITIAL_VOLUME_FRAC = 0.80`) — a terminating-simulation design. The metric window of every scenario opens six months after its start: SSI-6 requires six months of accumulation before its first defined value, and on the December epoch the cut lands exactly on June 1 of year 1 — the FFMP operating-year boundary. The annual-unit set keeps whole FFMP years (June 1 – May 31), the trailing June–November fragment of year $L$ is discarded, and the hazard-selection metrics trim the same trailing partial, so an $L$-year realization yields $L-1$ units spanning June 1 year 1 – May 31 year $L$ and selection and evaluation score the IDENTICAL window. The cut is applied by date from each window's own calendar, so leap years need no special case. A drought event straddling a window boundary is split; this is bounded by the $L$-vs-design-drought check (§6) and by long $L_{\text{test}}$, where rankings are decided.

**Hazard axes.** Computed on each realized sequence, before any system simulation, from the aggregate NYC inflow (Cannonsville + Pepacton + Neversink). The candidate set is drought metrics from SSI-6 run theory on the controlling event (magnitude, duration, severity, onset rate, recovery rate) and flood metrics from peaks-over-threshold on daily flow (peak discharge, pulse duration, rise rate). **Event models, pinned** (`scengen.hazard_metrics`): a drought event is a run of SSI-6 < 0 reaching ≤ −1, terminated by 3 consecutive non-negative months; the **controlling** event (largest cumulative deficit — the operationally binding event under fixed initial storage) is scored, and all dry descriptors are zero when a window has no qualifying event. The POT threshold is the p95 of the $Q_{\text{obs}}$ daily flow; the **critical pulse** is the exceedance run containing the window's maximum daily flow (one pulse per window; no declustering parameter); peak discharge = window max / $Q_{\text{obs}}$ mean daily (always defined), pulse duration = days above threshold in that run, rise rate = max 1-day rise on the rising limb / $Q_{\text{obs}}$ mean. The SSI distribution fit, the POT threshold, and the normalizing mean are made **once** on $Q_{\text{obs}}$ and reused for every realization, so hazard coordinates are comparable across the pool and the historic reference.

**Scope, disclosed:** the hazard space is NYC-aggregate-inflow-only (downstream objective sites include unregulated tributary flow the image never sees) and single-event-per-window (no frequency, timing, or inter-event descriptor enters the selection; event multiplicity is an explicit extension).

**Redundancy handling.** The screen on the pool's hazard image drops degenerate descriptors (near-zero spread) and prunes near-duplicate groups — $\lvert\rho_S\rvert \ge 0.95$ — to one canonical member, so a single hazard concept cannot enter the snap distance twice under two names. The screen performs no further reduction: correlated-but-distinct descriptors are deliberately kept as computed descriptors, because the design's aim is coverage of the full hydrologic diversity the descriptor set measures, and the selector's coverage guarantee is marginal rather than joint (§4.3). The screen retains all eight candidates at every measured pool size ($2{\times}10^3$–$10^6$): no degenerate axes, largest pairwise $\lvert\rho_S\rvert$ 0.87–0.88 (drought magnitude–duration), below the cut and insensitive to tightening it to 0.90. The campaign **selection**-axis subset is a separate, caller-level restriction fixed below. The Spearman rank-correlation structure (matrix and $1-\lvert\rho_S\rvert$ cluster tree, Olden & Poff 2003) is reported as a redundancy **diagnostic** alongside the selection, never used to reduce it further; the implicit weighting that correlated axes induce in the snap distance is characterized by a selection-invariance diagnostic (overlap of the selected members as individual axes are added or removed), reported as SI.

**Campaign selection-axis set.** The campaign **selection** axis set is $m = 6$: magnitude, severity, onset rate, recovery rate (dry) and peak discharge, pulse duration (wet). The full retained set's per-axis tail enrichment is dimension-limited at any affordable pool size (minimum per-axis share above the pool P90 ~0.22 at $P = 10^6$; improvement exponent ~0.04, far below $P^{-1/8}$; `hazard_selector_diagnostics.md` §5), while this set reaches 0.27–0.29 on the regenerated $P = 10^6$ pools, about three times the 0.10 share of an i.i.d. selection (recorded per production draw). `drought_duration` and `flood_rise_rate` remain computed in every hazard image and reported post-hoc; they simply do not enter the snap distance. Single source: `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`), consumed by the step-03 selection.

### 3.4 Determinism and seed separation

Realization $k$ is fully determined by a child RNG stream keyed to its **global** index, invariant to MPI or batch partition, driving the Kirsch monthly step, the Nowak daily step, and the KDE downstream fill identically regardless of how the index range is split. `regenerate_realization(root_seed, k)` reproduces any single realization bit-for-bit. This is what makes a $10^5$–$10^6$ pool tractable: only the hazard image is persisted, and the few hundred *selected* realizations are materialized on demand.

**Seed domains are disjoint by construction.** Each generated artifact draws its seed from a namespaced stream — `stat_pool`, `fixed`, `du_pool`, `resample_pool`, `input_strat`, `hazard_select_stat`, `hazard_select_du`, `etest:kn`, `etest:hmm` — so no two designs, and no design and $E_{\text{test}}$, ever share realizations. A shared seed would produce correlated realizations across designs, reintroducing the confound the architecture removes. The disjointness is asserted at import, and a search/test seed-domain collision is a hard error (selection-bias guard, Bonham et al. 2024).

---

## 4. The three designs

Both matched designs run at **$N = 300$, $L = 10$ yr**. `historic` is an unmatched reference.

**4.1 `historic`.** The observed record as one continuous trace; $N=1$, one 78-yr trace (Dec 1945 – Nov 2023) scored as 77 FFMP-year units. The reference for prevailing applied practice (Giuliani et al. 2016; Herman et al. 2020). $K=1$: composition variance is zero by construction. Cannot be size-matched, and is not part of the controlled contrast.

**4.2 `monte_carlo`.** Generate $N$ realizations of length $L$ i.i.d. from the stationary generator; freeze for the search. $K$ draws × $S$ seeds (draw = ensemble-sampling variance; seed = MOEA variance). Precedent: Quinn et al. (2017); Zatarain Salazar et al. (2017). This is the discipline's random-sampling default, the reference against which designed selection is judged, and — by §3.2 — the exact statistical control for `hazard_filling`.

**4.3 `hazard_filling` (registry key `hazard_filling_stationary`) — novel.** Select $E_d \subset C$, $\lvert E_d \rvert = N$, whose hazard coordinates cover the empirical hazard manifold.

**Selector.** Scale each retained axis to $[0,1]$ by its **robust pool range** — the per-axis p1 and p99 percentiles, clipping members beyond them to the box faces — so distances are in **absolute, range-scaled magnitude** units and no single axis dominates while spacing within each axis stays proportional to physical magnitude. Draw $N$ Latin hypercube anchors in the scaled hazard box and snap each to the nearest not-yet-selected pool member in Euclidean distance. The selector is deterministic given its anchor seed. It involves no iterative optimization of a coverage criterion and no tuning parameters. It is a deterministic LHS + nearest-neighbour snap and is never an annealing search.

**Why the bounds are robust percentiles, not the sample extremes.** The sample min/max of a right-skewed hazard metric are extreme order statistics: they do not converge as $P$ grows, so a full-range box would (i) tie the strength of the tail distortion to the nuisance sizing parameter $P$ — a larger pool would widen the box and mechanically strengthen the intervention; (ii) differ materially across the $K$ pool re-rolls, breaking draw commensurability; and (iii) let a single outlier compress the bulk of the pool into a corner of the box. The p1/p99 quantiles are central order statistics, root-$P$-consistent, so the selection geometry converges to a fixed population functional and is common in law across draws. On the zero-inflated dry event axes p1 collapses to the natural zero automatically. The clipped mass per axis face is bounded (≤ 1% + 1%) and is reported as build-QC (per-axis bounds and clipped fractions are persisted with the staged ensemble); clipped members sit on the box faces and remain selectable. The bounds choice is swept as a supplemental diagnostic (`hazard_selector_diagnostics.md`).

**Why absolute range scaling is the operative choice.** The hazard marginals of the generator are strongly right-skewed, so most pool members cluster near the center of the hazard space and few occupy the severe corners. Filling the range uniformly draws selected members from the sparse corners far more often than their pool frequency, so severe drought and flood conditions are over-represented in the search ensemble relative to their probability under the generator. This is the deliberate distribution shift RQ2 tests. An alternative that transforms each axis by its empirical cumulative distribution before filling would instead reproduce the pool's marginal frequencies and distort only the joint dependence among axes. That ECDF/rank-space variant is registered as a **non-campaign sensitivity** — it isolates how much of any effect is attributable to tail over-representation specifically — but it is not the campaign selector. The campaign selector is the absolute, range-scaled version.

Because the selector does not optimize a discrepancy objective, coverage statistics (L2-star discrepancy, MST edge statistics, snap distance) remain an *independent* diagnostic of what the selector achieved rather than the quantity being minimized. They are build-QC / method verification that the intervention was administered at strength, not a comparison result (§6). $K$ draws vary the pool and the anchor seed together and therefore measure the design's construction variance.

Selection operates on the stored hazard image alone; the pool's daily traces are never loaded. Only the $N$ selected realizations are materialized.

### 4.4 The controlled contrast

| Contrast | Held fixed | Question |
|---|---|---|
| `monte_carlo` → `hazard_filling` | generator, population law, $N$, $L$ | Does hazard-space coverage change robustness relative to random sampling? |
| `historic` | — | Prevailing-practice reference, unmatched in size and budget. |

The contrast varies exactly one thing (the selection rule) within a single population, and the i.i.d. pool gives it an exact random-selection control (§3.2).

### 4.5 Probability distortion

Hazard filling deliberately distorts scenario probabilities toward uniform hazard coverage: rare-but-severe corners are over-represented relative to their frequency in the pool. The rationale for uniform *coverage* of a condition space comes from the bottom-up / decision-scaling tradition (Brown et al. 2012; Culley et al. 2016; Herman et al. 2016; Fowler et al. 2024) — taken as a coverage argument for constructing the ensemble, **not** as a precedent for biasing a reported robustness number.

The distortion biases the **search trajectory**, not only a reported number: any objective aggregated over the search ensemble is computed under the distorted measure and biases *which policies the optimizer selects*. This is exactly why it is a design intervention worth testing and why search objectives are never compared across designs.

Re-evaluation removes the **evaluation** bias of scoring each design on its own ensemble — both designs are measured with the same instrument. It does **not** remove **selection** bias, and it does **not** "restore the true measure": there is no true measure to restore, because $E_{\text{test}}$ is itself a *designed* LHS exploration of a deeply uncertain box, not a probability sample (§5.1). What makes the comparison valid is that $E_{\text{test}}$ is **identical across both designs**, not that it is probability-faithful. Importance-sampling reweighting is rejected on its own terms: the snapped-selection rule induces no tractable density, and estimator variance would explode as coverage → uniform, precisely in the corners the design targets. Search objectives are therefore reported as coverage-weighted quantities, never as estimates of an expectation, and cross-design comparison rests entirely on the common re-evaluation.

---

## 5. Test ensemble ($E_{\text{test}}$)

**Role.** The single, common, held-out basis of cross-design comparison, and the sole carrier of deep uncertainty in the study. Never used during search and **never the source of any search ensemble**. Every design's final Pareto-approximate set is re-simulated on $E_{\text{test}}$ with the trimmed model, exactly as in search. The non-NYC STARFIT releases are policy-independent, so the presim pass over $E_{\text{test}}$ is computed once per realization and reused for every Pareto set.

### 5.1 Construction

Single source: `src/etest.py` (constants and the SIZING docstring); campaign values in `campaign_design.md` §5.

- **Latin hypercube over the full range of the deeply uncertain forcing factors**, the CMIP6 harmonic forcing amplitudes $[m, r_1, r_2]$ (the CV axis is off; `forcing_parameterization.md`). The envelope is deliberately wider than any variation the search ensembles contain.
- **Many realizations per LHS point.** Each $\theta$ is a state of the world (SOW); its $R_{\text{test}}$ realizations sample natural variability within it. $N_{\text{test}} = N_{\theta,\text{test}} \times R_{\text{test}} \gg N = 300$.
- $L_{\text{test}} = 50 \gg L = 10$. Long records test sustained operation (storage carryover across consecutive droughts, matured delivery-entitlement banking), which the terminating 10-yr search windows reset away. This asymmetry with the search framing is identical across designs and adds system-state continuity, not hydrologic persistence.
- **Independent seed domains** (`etest:kn`, `etest:hmm`), disjoint from every search ensemble (§3.4).
- **One construction.** Kirsch–Nowak over the wide DU box. A structurally different second construction (a multi-site HMM, `etest_hmm_*`) is registered as an optional variant and is not part of the campaign.

This is the standard construction of the re-evaluation lineage (Trindade et al. 2017; Gold et al. 2022; Kasprzyk et al. 2013; Bartholomew & Kwakkel 2020; Quinn et al. 2020).

**The re-evaluation is a generalization test.** The search ensembles are drawn from the unperturbed stationary generator while $E_{\text{test}}$ spans a forced climate envelope no search ensemble contains, so the re-evaluation measures whether hazard-space coverage of the natural-variability manifold produces policies that generalize to conditions never presented during search. Whether the design ranking depends on the region of the test space emphasized is measured by re-scoring subsets of the persisted matrix (the hazard-support decomposition, `hazard_support_decomposition.md`), not assumed.

**$E_{\text{test}}$ is sampled by LHS, not i.i.d.** The i.i.d. requirement of §3.2 belongs only to the candidate pool, where it underwrites the exact control. $E_{\text{test}}$ is never a control. It is the measuring stick, and it should cover the deeply uncertain space rather than sample it in proportion to a measure. Consequently robustness on $E_{\text{test}}$ is not an expectation. Under deep uncertainty there is no probability measure over the forcing space (Lamontagne et al. 2018), so a satisficing fraction over an LHS-designed box is a coverage-weighted count and is reported as such. Cross-design comparison is commensurable because $E_{\text{test}}$ is identical across designs, not because it is probability-faithful.

**Not scenario-neutral, a stated limitation.** No experimental design is truly neutral (Quinn et al. 2020), so all scenario-design rankings are conditional on $E_{\text{test}}$, and with a single construction this conditioning is declared, not bounded. The Kirsch bootstrap does not reproduce the record's interannual wet/dry persistence (water-year ρ₁ ≈ 0.10 in the generator vs 0.26 on the fitted record; `persistence_axis_diagnostics.md`), so multi-year persistence is stressed in neither the search population nor the test ensemble, and claims are scoped accordingly (§6).

### 5.2 The re-evaluation matrix is persisted in full

Re-evaluation persists the **(solution × SOW × objective)** matrix of per-SOW annual-unit objective values in natural units (`reeval_raw` plus a self-describing `reeval_raw_meta.json`). Each SOW's realizations' unit-years are pooled through the objective's own unit operator (`objective_definitions.md` §2), so the SOW is the only counting unit. Every robustness and regret metric is scored offline from that matrix (`src/robustness.py`), so a new metric never requires re-simulating. This is the McPhail et al. (2018) T1×T2×T3 substrate. Precision is governed by $N_\theta$, not $N_{\text{test}}$, because realizations within a SOW are not independent.

### 5.3 Reference-set precision

Recomputing one nondominated set from re-evaluated values pooled across designs induces a self-reference bias (a design contributes points to the frontier it is scored against). Dominance-based summaries over the pooled set are a supplement with that caveat stated, never the primary comparison (Zatarain Salazar et al. 2017 §5.3).

### 5.4 Sizing

Generated: $N_{\theta,\text{test}} = 1{,}000$, $R_{\text{test}} = 25$, $L_{\text{test}} = 50$ yr, in 50 staged chunks of 500 realizations (`etest_kn_50yr_n25000`). Re-evaluated: the leading 500 SOWs, the first 25 chunks (`etest_kn_50yr_n25000_first25ch`, a metadata-only prefix subset; LHS rows are randomly ordered, so the prefix is an unbiased half of the design). The derivation (cross-SOW precision from the literature by factor-space dimension, worst-case satisficing SE $0.5/\sqrt{500}$ = 2.2 pp, 1,225 pooled annual units per SOW against the ε floors, the measured re-evaluation cost, and the θ- and R-subsample stability checks) is stated once in `campaign_design.md` §5 and the `src/etest.py` SIZING docstring. Hazard-space coordinates of $E_{\text{test}}$ are computed on disjoint, December-aligned 10-yr sub-windows of each realization, scored on the pool's exact metric span, so they are commensurable with the pool's window convention. Trimmed-vs-full objective agreement is demonstrated on the historical trace (SI Text S1); because the boundary releases are policy-independent by construction, no per-ensemble re-check is performed.

---

## 6. Sizing, budget, and diagnostics

### Budget

Both matched designs run at $N = 300$, $L = 10$ yr → **3,000 scenario-years per evaluation, at equal NFE**.

Because $N$ and $L$ are common, per-evaluation simulation cost, scenario-years, and wall-clock are **identical**. Equal-NFE and equal-scenario-years coincide, so there is one budget condition and no confound between ensemble composition and search effort. This is a consequence of the sizing choice, not a control that has to be imposed.

**Why $N$ and $L$ are what they are.** The selection comparison requires a *common* $(N, L)$ — if $L$ differed, the selection rule would be confounded with record length. Coverage rests on the LHS anchors' stratified marginals: every selection axis is stratified into $N$ bins regardless of $m$, so the severe tail of each descriptor is over-represented at any dimension, and joint coverage is reported descriptively against a random design of the same $(N, m)$ rather than claimed as uniform filling. $N = 300$ is set by estimator precision (`ensemble_size_diagnostics.md` §7.3): it is the smallest size on the measured ladder at which the i.i.d. control meets the pre-registered paired-precision criterion (standard error of a paired difference ≤ ε/2) on every objective; the reliability and flood objectives meet it from $N = 75$ in both designs, and the three pooled-percentile operators reach 0.40–0.45 ε at 300. Nothing at the selection level penalizes the larger $N$: on the production $P = 10^6$ pools the minimum per-axis tail share is flat in $N$ (0.28–0.29 from $N = 50$ to 500; the pool holds ~$10^5$ members above P90 per axis, so selection never depletes the supply) and declines with $N$ only at $P' \le 2\times10^4$ (§7.1 of that note; the six-axis share is 0.27–0.29 on the regenerated December-epoch pools, §7.1a). Per-evaluation cost scales as $N^{0.95}$ (§6 below). Long records are not viable here: at a fixed per-evaluation budget, $L = 50$ forces $N \approx 60$, too few anchors for meaningful stratification. $L = 10$ also exceeds the 1960s DRB drought of record (~4–5 yr) plus onset and recovery, so a design-basis event fits inside a window, and it keeps duration-type hazard axes from being truncation-limited. The per-evaluation ensemble of 300 realizations is smaller than the 1,000-realization convention of the Zeff–Herman–Trindade lineage; under the annual-unit pooling of `objective_definitions.md` §2 it yields 2,700 metric-bearing unit-years per evaluation, of which 1,840–2,700 act as independent samples (measured $n_{\text{eff}}$, §7.3 of the sizing note), the same order as the 1,000 one-year realizations of that lineage.

`historic` ($N=1$, one 78-yr trace, 77 annual units) cannot be matched and is reported as a reference.

**NFE.** The reporting budget is **500,000 NFE per search** (125,000 per island), within the range used for comparable reservoir control-policy problems (Quinn et al. 2017; Bartholomew & Kwakkel 2020). The first seed of every design is continued to 750,000 NFE; its runtime archive at 125,000 per island is the equal-NFE result, and its runtime hypervolume beyond the 500k snapshot is the convergence evidence. The runtime archive records every intermediate level, so the attained budget is justified against observed convergence after the fact.

**Campaign geometry.** Each search runs Multi-Master Borg on 12 Anvil wholenode nodes (1,536 cores): 4 islands × 382 workers + 4 island masters + 1 controller = **1,533 MPI ranks** at 128 per node, with each evaluation simulating the 300 realizations in two batches of 150 (`NYCOPT_SEARCH_REALIZATION_BATCH=150`; the unbatched resident set projects above the node's safe capacity, `config.search_node_rss_gb`). Runtime snapshots every 2,500 island-NFE. Island partitioning is throughput-free at fixed slot count (measured), so 4 islands is a search-reliability choice that keeps per-island trajectories long. Searches are NFE-bounded — no Borg maxTime cap, which could truncate NFE unequally across designs — and there is no resume of a killed job, so every search is sized to finish inside one 96 h job (twelve nodes rather than eight because the 750k-NFE seed projects to ~99 h on eight). The `production` entry of `src/moea_config.py` is the single source of these numbers; `campaign_design.md` §3 states them with their provenance.

**Cost and budget.** The measured cost basis (SU per search at $N = 100$, its scaling in $N$ and NFE, the batching penalty, the node-scaling factor) and the full campaign budget table are in `campaign_design.md` §6 and SI Text S8. Per-evaluation cost scales as $(N/100)^{0.951}$, which is the basis of the $N$-vs-$L$ trade-off above. The `ffmp_N` sweep runs only on whatever SU remains at the end of the campaign.

**Pool-size headroom.** Candidate-pool generation is Kirsch–Nowak sampling plus the streamed hazard image, with no Pywr-DRB simulation, so its cost is under 2% of the campaign even at $P = 10^6$. $P$ is therefore not budget-constrained. The binding considerations are generation wall-time, pool storage, and selector behavior, which is why $P$ is set by the saturation diagnostic rather than by budget.

### Replication

A **draw** is the design's construction re-run from scratch with a fresh seed — one definition for every design, re-rolling *everything* that is random about building the ensemble. For `monte_carlo` that is a fresh i.i.d. sample; for `hazard_filling`, **a fresh candidate pool *and* a fresh LHS anchor plan**.

The pool must be re-drawn per draw, and this is load-bearing. Generating the pool *is* part of a hazard-filling design's construction. If the pool were pinned across draws, a hazard-filling draw would vary only its anchor plan while a `monte_carlo` draw re-rolls its entire sample — the two between-draw variances would not be commensurable, and hazard filling would appear more stable **by construction** rather than as a finding.

Three draws are staged per matched design (d0–d2). The campaign **searches one draw** (d0) with $S = 2$ seeds; `historic` has structural-zero composition variance and runs the same two seeds. The unit of analysis for between-design comparison is the **seed**, and the comparison is conditional on the one draw per design. Draws d1 and d2 serve the SI draw-sensitivity re-evaluation: each design's final Pareto set is re-simulated on its own other two draws and the paired per-policy shifts are reported against ε (`experimental_design.md`, Replication; `campaign_design.md` §5).

### Ensemble-quality diagnostics

**Build-QC.** Scenario redundancy (the §3.3 rank-correlation diagnostic re-run on $E_d$ and reported alongside the pool's — a diagnostic, not a gate). Statistical fidelity to $Q_{\text{obs}}$ (monthly moments, lag-1 and cross-site correlation, flow-duration curve) is a **within-`monte_carlo` check only**; hazard filling distorts marginals by design and is never ranked on fidelity — only checked that each selected member is a valid generator output.

**Coverage is method verification, not a comparison result.** L2-star discrepancy and minimum-spanning-tree edge statistics on normalized hazard coordinates, plus the snap-distance distribution, are reported **against the expected discrepancy of a random design at the same $(N, m)$** so the $m$-vs-$N$ tension is visible rather than asserted. Because the LHS + nearest-neighbour selector does not optimize discrepancy, this is an independent measurement that the selector administered the intervention at strength — that the `hazard_filling` ensemble is compositionally shifted relative to `monte_carlo`. It is build-QC, not an endpoint.

**Outcome hypotheses** (falsifiable, may be null). The primary cross-design comparison is the **multivariate Starr satisficing fraction** on $E_{\text{test}}$, with univariate satisficing, Laplace, and maximin as secondary anchors (`objective_definitions.md` §3). Incumbent-relative regret (per-objective regret and gain magnitudes, harm frequencies, and the no-harm frequency against the default FFMP policy in the same SOW) is the co-primary family and answers RQ1. No set-relative (best-in-set) regret and no perfect-foresight (Cohen-style) regret are computed.

Scenario discovery operates in the **DU factor space** of $E_{\text{test}}$ after re-evaluation (`objective_definitions.md` §4) as a supporting analysis, not the primary comparison. The one hazard-space inference is the step-11 coverage-deficit mechanism test (`experimental_design.md`, Evaluation).

### Open parameters

| Parameter | Status |
|---|---|
| Candidate pool size $P$ | **$P = 10^6$**, the pool size at which the campaign $m = 6$ tail share has saturated (0.268 → 0.282 from $10^5$ to $10^6$; `ensemble_size_diagnostics.md` §7.1). Pools are generated in shards (`workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`). |
| Ensemble draws | **1 searched (d0); 3 staged** (d1, d2 for the SI draw-sensitivity re-evaluation). Fixed before generation — draws are independent *generations*. |
| Ensemble size $N$ | **$N = 300$** (`campaign_design.md`): the smallest ladder size at which the i.i.d. control meets the paired-precision criterion on every objective (`ensemble_size_diagnostics.md` §7.3; fitted crossings 210–283, the binding Montague-deficit crossing 283 [249, 321]); hazard filling meets it on every objective but NYC deficit P99, a shelf-valued operator, at no $N \le 400$, a disclosed residual. The ε floors are re-verified on the $N = 300$ ensembles before the searches. |
| $S$ seeds | 2 (a floor; a third seed does not fit the balance, `campaign_design.md` §1). |
| Hazard-axis count $m$ | **$m = 6$ selection set** (§3.3). Single source: `config.HAZARD_SELECTION_AXES`. |
| $E_{\text{test}}$ envelope width | Wider than the search forcing box, so $E_{\text{test}}$ is not a subset of it. |

### Flagged methodological uncertainties

- **Estimator precision.** The two designs differ in the variance of their fitness estimates (a frozen i.i.d. ensemble and a frozen coverage-designed ensemble present different sampling structures to the optimizer). Matching $N$ and $L$ removes the compute confound but not this one; it is disclosed, not removed — and measured (`ensemble_size_diagnostics.md` §7.3): at $N = 300$ every objective resolves to $\le 0.45\,\varepsilon$ (worst-pair paired SE) for the i.i.d. control, while the hazard-filling construction reproduces NYC deficit P99 across anchor plans only to a few $\varepsilon$ at every $N \le 400$ because that operator moves in discrete shelves that the selected extreme members decide. The residual is disclosed, and its draw-dependence is measured by the SI draw-sensitivity re-evaluation.
- **$E_{\text{test}}$ conditioning.** Rankings are conditional on the test-ensemble design; the optional second construction would bound but not eliminate this.
- **$m$-vs-$N$ sparsity.** Joint filling at moderate $N$ is not claimed; the intervention rests on stratified marginals and per-axis tail enrichment, and the coverage diagnostics must demonstrate, not assert, that it was administered at strength. Snap concentration and per-axis tail enrichment degrade with $m$ at fixed $P$ — the measured basis of the decided $m = 6$ (§3.3). Raising $N$ at fixed $P$ lowers enrichment only when the pool's tail supply is finite relative to $N$ ($P' \le 2\times10^4$); at $P = 10^6$ enrichment is flat in $N$ to 500 (`ensemble_size_diagnostics.md` §7.1). The realized minimum per-axis share above the pool P90 is 0.27–0.29 at $P = 10^6$, about three times the 0.10 i.i.d. share, and is recorded per production draw.
- **Generator stationarity under perturbation.** The Kirsch correlation structure is fit on history and reused under shifted moments in $E_{\text{test}}$. The DU forcing space spans volume, seasonality, and variability, but not multi-year drought persistence — and the generator's i.i.d. year-bootstrap under-expresses even the record's own persistence (water-year ρ₁ ≈ 0.10 vs 0.26 on the record, which is itself epochal: 0.51 pre-1980, 0.06 after; the record sits in an unprecedented ~43-yr pluvial, Pederson et al. 2013). There is no persistence DU axis (`persistence_axis_diagnostics.md`: the CMIP6 Δρ₁ anchor's upper tail is 39-yr sampling noise, and NE projections point to flash, not multi-year, drought). Because all scenario designs and $E_{\text{test}}$ share the generator, the design comparison is unaffected; **absolute** robustness claims are scoped to "under historical-level (pluvial-era) interannual persistence", with the $E_{\text{test}}$ hazard overlay quantifying the multi-year-dry stress actually expressed.
- **Partial-event truncation.** Bounded but not eliminated by disjoint windows and the $L$-vs-design-drought check; flagged wherever an event-based hazard axis is used.
