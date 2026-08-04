# Scenario-Design Generation Methods

*Construction recipe for every streamflow scenario set in the comparison. Terminology per `docs/notes/terminology.md`; the experiment itself is in `experimental_design.md`; objective formulations are in `objective_definitions.md`. Citations author-year, resolved via `docs/notes/literature/`. The manuscript draft (`docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md`) is the authoritative specification.*

---

## 1. Scope and the central asymmetry

Three **evaluation ensembles** (one per scenario design) and one held-out **test ensemble** for re-evaluation. All three designs are drawn from a single stationary population. The methodological contribution is **hazard filling**: an in-the-loop evaluation ensemble built by space-filling coverage of the empirical hazard manifold of a candidate pool, contrasted against random sampling from the same stationary generator.

Each design is constructed by its own recipe with its own seed stream. No design draws from another's data. This is what lets `fixed_probabilistic` and `hazard_filling` each honestly represent the practice it stands for while sharing a population law.

Hazard filling is the only design that needs a candidate pool, and this is intrinsic rather than incidental. Hazard coordinates are emergent properties of a realized flow sequence, so no generator can be asked to produce a realization at a prescribed drought severity. A hazard-space design must therefore **select from** a finite pool (LHS anchors plus a nearest-neighbour snap), whereas a probabilistic design **generates** its realizations directly. The nearest-neighbour step is a consequence of this fact, not an approximation of a preferable procedure.

Deep uncertainty enters the study only through the test ensemble E_test (§5). Both search designs are stationary.

**Out of scope** (decided elsewhere): objective formulations and across-scenario aggregation (`objective_definitions.md`); MOEA internals (the `MOEAConfig` axis); robustness-metric mathematics.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $Q_{\text{obs}}$ | Observed/reconstructed daily multi-site historical record (`pub_nhmv10_BC_withObsScaled`, ~1945–2022). |
| $r$ | A **realization**: one generator output sequence, seed-deterministic given a realization index. |
| $h(r) \in \mathbb{R}^{m}$ | **Hazard-metric vector**: $m$ drought / low-flow / high-flow descriptors (§3.3). |
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

Restricting the study to this one population is deliberate. It isolates RQ1 — whether the rule that selects realizations from a fixed generating process matters — without confounding that rule with a change in the generating process itself. The deeply uncertain climate-forcing space enters only in $E_{\text{test}}$ (§5), where it makes the re-evaluation a test of generalization to conditions absent from search. Prior work supports hazard-oriented selection within a stationary population: Zatarain Salazar et al. (2017) stratify a stationary pool by a realized-flow statistic, and Herman et al. (2016) amplify drought severity with a stationary weighted bootstrap.

### 3.2 The candidate pool must be sampled i.i.d., not LHS

The candidate pool is drawn **i.i.d.** from the stationary generator by plain Monte Carlo.

This is load-bearing. A uniform random size-$N$ subset of an i.i.d. pool has exactly the joint law of $N$ fresh i.i.d. draws. That is what makes `fixed_probabilistic` the **exact statistical control** for `hazard_filling`: the two differ only in the selection rule applied to the same population law, and any difference in re-evaluated robustness is attributable to that rule alone. A random subset of an LHS design is not i.i.d., so an LHS-sampled pool would silently void the control. An invariant test enforces the i.i.d. condition, because nothing else in the pipeline would fail if it were broken.

The pool owns $P = 10^6$ realizations (decided 2026-07-30; §6) and is never simulated. Only each realization's hazard coordinates and generation seed are stored, and the $N$ selected members are regenerated exactly on demand from the deterministic, globally indexed random-stream architecture (§3.4). This storage arrangement is what makes a $10^6$-member pool tractable.

### 3.3 Realizations, windows, and hazard metrics

**Windows.** Disjoint, non-overlapping $L$-year blocks, October-aligned to the water year. Each scenario starts from fixed initial storage (`INITIAL_VOLUME_FRAC = 0.80`) — a terminating-simulation design. The metric window of every scenario opens six months after its start: SSI-6 requires six months of accumulation before its first defined value, so both the hazard-selection metrics and the objective metrics ignore that interval, and selection and evaluation therefore score the identical effective window. The cut is applied by date from each window's own calendar, so leap years need no special case. Because the remainder begins April 1 of the first water year, the annual-unit set is unchanged — whole water years only, first unit WY2, $L-1$ units per realization — while the daily and weekly metrics start April 1 of WY1. A drought event straddling a window boundary is split; this is bounded by the $L$-vs-design-drought check (§6) and by long $L_{\text{test}}$, where rankings are decided.

**Hazard axes.** Computed on each realized sequence, before any system simulation, from the aggregate NYC inflow (Cannonsville + Pepacton + Neversink). The candidate set is drought metrics from SSI-6 run theory on the controlling event (deficit volume, duration, peak depth, onset rate, recovery rate) and flood metrics from peaks-over-threshold on daily flow (peak magnitude, pulse duration, rise rate). **Event models, pinned** (`scengen.hazard_metrics`): a drought event is a run of SSI-6 < 0 reaching ≤ −1, terminated by 3 consecutive non-negative months; the **controlling** event (largest cumulative deficit — the operationally binding event under fixed initial storage) is scored, and all dry descriptors are zero when a window has no qualifying event. The POT threshold is the p95 of the $Q_{\text{obs}}$ daily flow; the **critical pulse** is the exceedance run containing the window's maximum daily flow (one pulse per window; no declustering parameter); peak magnitude = window max / $Q_{\text{obs}}$ mean daily (always defined), pulse duration = days above threshold in that run, rise rate = max 1-day rise on the rising limb / $Q_{\text{obs}}$ mean. The SSI distribution fit, the POT threshold, and the normalizing mean are made **once** on $Q_{\text{obs}}$ and reused for every realization, so hazard coordinates are comparable across the pool and the historic reference.

**Scope, disclosed:** the hazard space is NYC-aggregate-inflow-only (downstream objective sites include unregulated tributary flow the image never sees) and single-event-per-window (no frequency, timing, or inter-event descriptor enters the selection; event multiplicity is an explicit extension).

**Redundancy handling.** The screen on the pool's hazard image drops degenerate descriptors (near-zero spread) and prunes near-duplicate groups — $\lvert\rho_S\rvert \ge 0.95$ — to one canonical member, so a single hazard concept cannot enter the snap distance twice under two names. The screen performs no further reduction: correlated-but-distinct descriptors are deliberately kept as computed descriptors, because the design's aim is coverage of the full hydrologic diversity the descriptor set measures, and the selector's coverage guarantee is marginal rather than joint (§4.3). The screen retains all eight candidates at every measured pool size ($2{\times}10^3$–$10^6$): no degenerate axes, largest pairwise $\lvert\rho_S\rvert$ 0.87–0.88 (deficit volume–duration), below the cut and insensitive to tightening it to 0.90. The campaign **selection**-axis subset is a separate, caller-level restriction fixed below. The Spearman rank-correlation structure (matrix and $1-\lvert\rho_S\rvert$ cluster tree, Olden & Poff 2003) is reported as a redundancy **diagnostic** alongside the selection, never used to reduce it further; the implicit weighting that correlated axes induce in the snap distance is characterized by a selection-invariance diagnostic (overlap of the selected members as individual axes are added or removed), reported as SI.

**Campaign selection-axis decision (2026-07-30).** The nested-P saturation diagnostic (`hazard_selector_diagnostics.md`; results in `outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md`) showed the full retained set cannot meet the per-axis adequacy gate at any affordable pool size (min per-axis tail share 0.256 at $P = 10^6$; improvement exponent ~0.04, far below $P^{-1/8}$ — the snap is dimension-limited), activating the sparsity clause above. The campaign **selection** axis set is fixed to $m = 6$: deficit volume, peak depth, onset rate, recovery rate (dry) and peak magnitude, pulse duration (wet) — dropping `drought_duration` (quasi-discrete; $\lvert\rho_S\rvert = 0.87$ with deficit volume, the most redundant retained pair) and `flood_rise_rate` (flood-group-entangled, $\lvert\rho_S\rvert = 0.60$ with peak magnitude). This set passes the gate at $P = 10^6$ (min per-axis tail share 0.311; measured alternatives fail — `m6_axis_set_assessment.json`). Both dropped descriptors remain computed in every hazard image and reported post-hoc; they simply do not enter the snap distance. Single source: `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`), consumed by the step-03 selection.

### 3.4 Determinism and seed separation

Realization $k$ is fully determined by a child RNG stream keyed to its **global** index, invariant to MPI or batch partition, driving the Kirsch monthly step, the Nowak daily step, and the KDE downstream fill identically regardless of how the index range is split. `regenerate_realization(root_seed, k)` reproduces any single realization bit-for-bit. This is what makes a $10^5$–$10^6$ pool tractable: only the hazard image is persisted, and the few hundred *selected* realizations are materialized on demand.

**Seed domains are disjoint by construction.** Each generated artifact draws its seed from a namespaced stream — `stat_pool`, `fixed`, `du_pool`, `resample_pool`, `input_strat`, `hazard_select_stat`, `hazard_select_du`, `etest:kn`, `etest:hmm` — so no two designs, and no design and $E_{\text{test}}$, ever share realizations. A shared seed would produce correlated realizations across designs, reintroducing the confound the architecture removes. The disjointness is asserted at import, and a search/test seed-domain collision is a hard error (selection-bias guard, Bonham et al. 2024).

---

## 4. The three designs

Both matched designs run at **$N = 100$, $L = 10$ yr**. `historic` is an unmatched reference.

**4.1 `historic`.** The observed record as one continuous trace; $N=1$, full window (~77 yr). The reference for prevailing applied practice (Giuliani et al. 2016; Herman et al. 2020). $K=1$: composition variance is zero by construction. Cannot be size-matched, and is not part of the controlled contrast.

**4.2 `fixed_probabilistic`.** Generate $N$ realizations of length $L$ i.i.d. from the stationary generator; freeze for the search. $K$ draws × $S$ seeds (draw = ensemble-sampling variance; seed = MOEA variance). Precedent: Quinn et al. (2017); Zatarain Salazar et al. (2017). This is the discipline's random-sampling default, the reference against which designed selection is judged, and — by §3.2 — the exact statistical control for `hazard_filling`.

**4.3 `hazard_filling` (registry key `hazard_filling_stationary`) — novel.** Select $E_d \subset C$, $\lvert E_d \rvert = N$, whose hazard coordinates cover the empirical hazard manifold.

**Selector.** Scale each retained axis to $[0,1]$ by its **robust pool range** — the per-axis p1 and p99 percentiles, clipping members beyond them to the box faces — so distances are in **absolute, range-scaled magnitude** units and no single axis dominates while spacing within each axis stays proportional to physical magnitude. Draw $N$ Latin hypercube anchors in the scaled hazard box and snap each to the nearest not-yet-selected pool member in Euclidean distance. The selector is deterministic given its anchor seed. It involves no iterative optimization of a coverage criterion and no tuning parameters. It is a deterministic LHS + nearest-neighbour snap and is never an annealing search.

**Why the bounds are robust percentiles, not the sample extremes.** The sample min/max of a right-skewed hazard metric are extreme order statistics: they do not converge as $P$ grows, so a full-range box would (i) tie the strength of the tail distortion to the nuisance sizing parameter $P$ — a larger pool would widen the box and mechanically strengthen the intervention; (ii) differ materially across the $K$ pool re-rolls, breaking draw commensurability; and (iii) let a single outlier compress the bulk of the pool into a corner of the box. The p1/p99 quantiles are central order statistics, root-$P$-consistent, so the selection geometry converges to a fixed population functional and is common in law across draws. On the zero-inflated dry event axes p1 collapses to the natural zero automatically. The clipped mass per axis face is bounded (≤ 1% + 1%) and is reported as build-QC (per-axis bounds and clipped fractions are persisted with the staged ensemble); clipped members sit on the box faces and remain selectable. The bounds choice is swept as a supplemental diagnostic (`hazard_selector_diagnostics.md`).

**Why absolute range scaling is the operative choice.** The hazard marginals of the generator are strongly right-skewed, so most pool members cluster near the center of the hazard space and few occupy the severe corners. Filling the range uniformly draws selected members from the sparse corners far more often than their pool frequency, so severe drought and flood conditions are over-represented in the search ensemble relative to their probability under the generator. This is the deliberate distribution shift RQ1 tests. An alternative that transforms each axis by its empirical cumulative distribution before filling would instead reproduce the pool's marginal frequencies and distort only the joint dependence among axes. That ECDF/rank-space variant is registered as a **non-campaign sensitivity** — it isolates how much of any effect is attributable to tail over-representation specifically — but it is not the campaign selector. The campaign selector is the absolute, range-scaled version.

Because the selector does not optimize a discrepancy objective, coverage statistics (L2-star discrepancy, MST edge statistics, snap distance) remain an *independent* diagnostic of what the selector achieved rather than the quantity being minimized. They are build-QC / method verification that the intervention was administered at strength, not a comparison result (§6). $K$ draws vary the pool and the anchor seed together and therefore measure the design's construction variance.

Selection operates on the stored hazard image alone; the pool's daily traces are never loaded. Only the $N$ selected realizations are materialized.

### 4.4 The controlled contrast

| Contrast | Held fixed | Question |
|---|---|---|
| `fixed_probabilistic` → `hazard_filling` | generator, population law, $N$, $L$ | Does hazard-space coverage change robustness relative to random sampling? |
| `historic` | — | Prevailing-practice reference, unmatched in size and budget. |

The contrast varies exactly one thing (the selection rule) within a single population, and the i.i.d. pool gives it an exact random-selection control (§3.2).

### 4.5 Probability distortion

Hazard filling deliberately distorts scenario probabilities toward uniform hazard coverage: rare-but-severe corners are over-represented relative to their frequency in the pool. The rationale for uniform *coverage* of a condition space comes from the bottom-up / decision-scaling tradition (Brown et al. 2012; Culley et al. 2016; Herman et al. 2016; Fowler et al. 2024) — taken as a coverage argument for constructing the ensemble, **not** as a precedent for biasing a reported robustness number.

The distortion biases the **search trajectory**, not only a reported number: any objective aggregated over the search ensemble is computed under the distorted measure and biases *which policies the optimizer selects*. This is exactly why it is a design intervention worth testing and why search objectives are never compared across designs.

Re-evaluation removes the **evaluation** bias of scoring each design on its own ensemble — both designs are measured with the same instrument. It does **not** remove **selection** bias, and it does **not** "restore the true measure": there is no true measure to restore, because $E_{\text{test}}$ is itself a *designed* LHS exploration of a deeply uncertain box, not a probability sample (§5.1). What makes the comparison valid is that $E_{\text{test}}$ is **identical across both designs**, not that it is probability-faithful. Importance-sampling reweighting is rejected on its own terms: the snapped-selection rule induces no tractable density, and estimator variance would explode as coverage → uniform, precisely in the corners the design targets. Search objectives are therefore reported as coverage-weighted quantities, never as estimates of an expectation, and cross-design comparison rests entirely on the common re-evaluation.

---

## 5. Test ensemble ($E_{\text{test}}$)

**Role.** The single, common, held-out basis of cross-design comparison, and the sole carrier of deep uncertainty in the study. Never used during search; **never the source of any search ensemble**. Every design's final Pareto-approximate set is re-simulated on $E_{\text{test}}$ with the trimmed model, exactly as in search: the non-NYC STARFIT releases are policy-independent, so the step-04 presim pass over $E_{\text{test}}$ is computed once per realization and reused for every Pareto set (§5.4).

### 5.1 Construction: LHS over the full DU space × many realizations per point

$E_{\text{test}}$ is **the largest ensemble in the study, by a wide margin**, and is designed to be maximally *uncertainty-encompassing*, not a probability sample.

- **Latin hypercube over the FULL range of the deeply uncertain forcing factors** — the CMIP6 harmonic forcing amplitudes $[m, r_1, r_2]$ (the CV axis is off for the campaign — measured basis in `forcing_parameterization.md`). The envelope is deliberately **wider than any variation the search ensembles contain**, so $E_{\text{test}}$ is not a subset of the space any design searched.
- **Many realizations per LHS point**: $R_{\text{test}} \gg 1$. Each $\theta$ is a **state of the world (SOW)**; its $R_{\text{test}}$ realizations sample the natural variability *within* that SOW. $N_{\text{test}} = N_{\theta,\text{test}} \times R_{\text{test}} \gg N = 100$.
- $L_{\text{test}} = 50 \gg L = 10$: long records test *sustained* operation — storage carryover across consecutive droughts and matured delivery-entitlement banking — which the terminating 10-yr search windows reset away (§5.4).
- **Independent seed domains** (`etest:kn`, `etest:hmm`), disjoint from every search ensemble (§3.4). A search/test seed-domain collision is a hard error.

This is the standard construction of the re-evaluation lineage: Trindade et al. (2017) cross 10,000 LHS DU samples with 1,000 flow realizations; Gold et al. (2022) use $10^6$ SOWs; Kasprzyk et al. (2013) and Bartholomew & Kwakkel (2020) use 10,000 LHS SOWs; Quinn et al. (2020) LHS the generator parameters.

**The re-evaluation is a generalization test.** Because the search ensembles are drawn from the unperturbed stationary generator while $E_{\text{test}}$ spans a forced climate envelope no search ensemble contains, the re-evaluation measures whether hazard-space coverage of the natural-variability manifold produces policies that generalize to conditions never presented during search. This keeps the test instrument structurally distinct from both designs, so $E_{\text{test}}$ favors neither.

**$E_{\text{test}}$ is sampled by LHS, not i.i.d., and the distinction is deliberate.** The i.i.d. requirement of §3.2 belongs *only* to the candidate pool, because a random subset of an i.i.d. pool is distributionally identical to i.i.d. draws — that is what makes `fixed_probabilistic` the exact control for `hazard_filling`. $E_{\text{test}}$ is never subsampled and is never a control; it is the measuring stick, and it should *cover* the deeply uncertain space as evenly as possible rather than sample it in proportion to a measure.

**Consequence, and it is load-bearing: robustness on $E_{\text{test}}$ is not an expectation.** Under deep uncertainty there is no probability measure over the forcing space — that is what "deep" means (Lamontagne et al. 2018). A satisficing fraction over an LHS-designed DU box is a coverage-weighted count over a designed exploration, and must be reported as such, never as an estimate of $\mathbb{E}[\cdot]$. Cross-design comparison is commensurable because $E_{\text{test}}$ is **identical across both designs**, not because it is probability-faithful.

**Construction (the campaign default).** One $E_{\text{test}}$: **Kirsch–Nowak over the wide DU box**, LHS × $R_{\text{test}}$. This is what the campaign requires and what every comparison is computed on.

**Not scenario-neutral — a stated limitation.** Following Quinn et al. (2020), no experimental design is truly neutral: $E_{\text{test}}$ is one deliberately broad, design-conditional reference, and **all scenario-design rankings are conditional on it**. With a single construction this conditioning is *declared, not bounded*. The Kirsch bootstrap does not reproduce the record's interannual wet/dry persistence (measured water-year ρ₁ ≈ 0.10 in the generator vs 0.26 on the fitted record; `persistence_axis_diagnostics.md`), so multi-year persistence is stressed in neither the search population nor the test ensemble; claims are scoped accordingly (§6).

A **second, structurally different construction** (a multi-site HMM — a different dependence model, though the DRB-fitted transition matrix is near-memoryless, so it should not be assumed to add persistence) is **registered as an optional variant** (`etest_hmm_*`, seed domain `etest:hmm`) but is **not part of the default campaign**. Standing it up would let ranking stability across test-ensemble constructions be *measured* (Kendall's $\tau_b$) rather than assumed. The persisted re-evaluation matrix (§5.2) additionally supports zero-simulation-cost composition-sensitivity checks, in which hazard-restricted and envelope-restricted subsets of the single $E_{\text{test}}$ are re-scored to probe whether the ranking depends on the region of the test space emphasized. It is a scope decision, not a technical blocker.

### 5.2 The re-evaluation matrix is persisted in full

Unlike search — which keeps only objective scores — re-evaluation persists the **entire $(\text{solution} \times \text{realization} \times \text{objective})$ matrix in natural units** (`reeval_raw.parquet` + a self-describing `reeval_raw_meta.json`). Every robustness metric is then scored *offline* from that matrix (`src/robustness.py`), so a new metric never requires re-simulating. This is the McPhail et al. (2018) T1×T2×T3 substrate: persisting natural units preserves enough to recompute any performance-value transformation (T1), any scenario subset (T2), and any aggregation (T3) later.

The matrix also records each realization's **SOW id** (its $\theta$ / LHS-point index), so both units of robustness are computable offline without re-simulation:

- **SOW unit** (the MORDM lineage standard: Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023). Stage 1: collapse each $\theta$'s $R_{\text{test}}$ realizations into one performance vector (within-SOW aggregator — mean is risk-neutral, worst is risk-averse; the choice is reported). Stage 2: Starr domain criterion across the $N_{\theta}$ SOWs.
- **Realization unit**: the Starr criterion across all $N_{\text{test}}$ realizations directly.

**Precision is governed by $N_{\theta}$, not by $N_{\text{test}}$.** Realizations within a $\theta$ are not independent, so uncertainty on ensemble-level quantities is assessed at the SOW level; adding more realizations per SOW sharpens the within-SOW estimate but does not buy additional independent information about the DU space.

### 5.3 Reference-set precision

Recomputing one nondominated set from re-evaluated values pooled across designs induces a self-reference bias — a design contributes points to the frontier it is scored against. Dominance-based summaries over the pooled set are therefore a supplement, read with that caveat stated up front, and never the primary comparison (§6, and Zatarain Salazar et al. 2017 §5.3).

### 5.4 Sizing (decided 2026-07-30)

$N_{\theta,\text{test}} = 1{,}000$, $R_{\text{test}} = 25$, $L_{\text{test}} = 50$ yr — 25,000 realizations, 1.25M scenario-years, declared as single-source constants in `src/etest.py` and hardcoded nowhere else. The chunked staging/re-evaluation path (`src/chunk_reeval.py`, workflow step 09) exists for exactly this scale. The derivation:

- **Cross-SOW precision governs $N_\theta$** (§5.2). The worst-case Monte Carlo standard error of a satisficing fraction is $0.5/\sqrt{N_\theta}$ = ±1.6 pp at 1,000 — below the resolution at which the criterion-sweep invariance claim (§6) is made — and $E_{\text{test}}$ sits in the $10^3$–$10^4$-SOW class of the re-evaluation lineage (Kasprzyk et al. 2013; Herman et al. 2014; Bartholomew & Kwakkel 2020).
- **Within-SOW sample matches the study's own unit-count standard.** $R_{\text{test}} \times (L_{\text{test}}-1) = 1{,}225$ metric-bearing annual units per SOW — the same ~900–1,000 unit-year sample §6 argues stabilizes this lineage's tail operators for one search evaluation, so the stage-1 (within-SOW) estimate is resolved at least as well as an entire search evaluation. Closest structural precedent: Quinn et al. (2020), ~1,050 realization-years per generator parameterization.
- **Long records test sustained operation.** $L_{\text{test}} = 50$ removes the terminating-window reset of search: consecutive droughts are experienced on carried-over storage, and the running-average entitlement banking matures. This asymmetry with the search framing is deliberate, identical across designs, and adds *system-state* continuity, not hydrologic persistence (the §5.1 persistence limitation is unchanged). Hazard-space coordinates of $E_{\text{test}}$ (the overlay figure and the hazard-restricted composition subsets) are computed on disjoint, October-aligned 10-yr sub-windows of each realization, keeping them commensurable with the pool's window convention.
- **Trimmed model, priced at the measured rate.** Re-evaluation cost is $n_{\text{policies}} \times N_{\theta,\text{test}} \times R_{\text{test}} \times L_{\text{test}}$ scenario-years at the measured trimmed cost (§6): ~80,000 SU at the ~1,000–1,200 campaign policies implied by the epsilon-calibrated archive sizes (~300–400 per set, merged across seeds per design). The one-time full-model presim pass over $E_{\text{test}}$ adds ~70 SU. Trimmed-vs-full objective agreement is demonstrated on the historical trace (all policy × objective pairs agree to $\leq 2.5 \times 10^{-13}$ relative) and reported in SI (S1); because the boundary releases are policy-independent by construction, the historic validation suffices and no per-ensemble re-check is performed.
- **Adequacy is verified, not assumed.** $\theta$-subsample (500 vs 1,000) and $R$-subsample (5/10/25) ranking-stability curves are scored offline from the persisted matrix (§5.2) at zero simulation cost — the same empirical-convergence move as Trindade et al.'s (2017) 100–5,000 ensemble-size sweep and Quinn et al.'s (2020) 100-vs-1,000 SOW check.

---

## 6. Sizing, budget, and diagnostics

### Budget

Both matched designs run at $N = 100$, $L = 10$ yr → **1,000 scenario-years per evaluation, at equal NFE**.

Because $N$ and $L$ are common, per-evaluation simulation cost, scenario-years, and wall-clock are **identical**. Equal-NFE and equal-scenario-years coincide, so there is one budget condition and no confound between ensemble composition and search effort. This is a consequence of the sizing choice, not a control that has to be imposed.

**Why $N$ and $L$ are what they are.** The selection comparison requires a *common* $(N, L)$ — if $L$ differed, the selection rule would be confounded with record length. Coverage rests on the LHS anchors' stratified marginals: every selection axis is stratified into $N$ bins regardless of $m$, so the severe tail of each descriptor is over-represented at any dimension, and joint coverage is reported descriptively against a random design of the same $(N, m)$ rather than claimed as uniform filling. $N = 100$ is decided (2026-07-30, jointly with $m$ and $P$): the N-sweep shows per-axis tail enrichment is flat-to-declining in $N$ at fixed $P$ — the pool holds only ~$P/10$ members above P90 per axis, so additional selections draw from the bulk — and per-evaluation cost scales linearly with $N$ (the budget below is priced at $N = 100$). Long records are not viable here: at a fixed per-evaluation budget, $L = 50$ forces $N \approx 20$, too few anchors for meaningful stratification. $L = 10$ also exceeds the 1960s DRB drought of record (~4–5 yr) plus onset and recovery, so a design-basis event fits inside a window, and it keeps duration-type hazard axes from being truncation-limited. The per-evaluation ensemble of 100 realizations is smaller than the 1,000-realization convention of the Zeff–Herman–Trindade lineage; the ground for it is the annual-unit pooling of `objective_definitions.md` §2, which yields ~900 metric-bearing unit-years per evaluation, comparable to the sample on which that lineage stabilizes the same tail operators.

`historic` ($N=1$, $L \approx 77$) cannot be matched and is reported as a reference.

**NFE.** The function-evaluation budget is **500,000 NFE per search**, within the range used for comparable reservoir control-policy problems (Quinn et al. 2017; Bartholomew & Kwakkel 2020). The runtime archive records intermediate NFE levels, so the attained budget is justified against observed convergence after the fact, and the design comparison can be recomputed at two or three earlier budgets at re-evaluation cost only.

**Campaign geometry.** Each search runs Multi-Master Borg on 8 Anvil wholenode nodes (1,024 cores): 4 islands × 254 workers + 4 island masters + 1 controller = **1,021 MPI ranks**, 125,000 NFE per island, runtime snapshots every 2,500 island-NFE (50 per island). Island partitioning is throughput-free at fixed slot count (measured), so 4 islands is a search-reliability choice that keeps per-island trajectories long. Searches are NFE-bounded — no Borg maxTime cap, which could truncate NFE unequally across designs — with the SLURM wall (~40 h) as the safety net; a killed run resumes from the last snapshot. SU cost is nearly flat in node count, so 8 nodes (vs 4) halves wall time at no cost penalty. The `production` entry of `src/moea_config.py` is the single source of these numbers.

**Measured cost and budget.** At 128 evaluator ranks per node on Purdue Anvil, one function evaluation over the campaign ensemble (100 realizations of 10 years, trimmed model) takes a median of **173.8 s** (the full model — used only for the presim passes and the single-trace historic baseline — is **1.16×** as expensive). With the measured 0.729 strong-scaling efficiency, one 500,000-NFE search runs ~32.6 h and costs **~33,400 SU**; the `historic` reference (~32 s per evaluation on its single 77-yr trace) costs ~6,100 SU per seed at the same NFE.

| Item | Basis | SU |
|---|---|---|
| `fixed_probabilistic` searches | 3 draws × 2 seeds × ~33,400 | ~200,300 |
| `hazard_filling_stationary` searches | 3 draws × 2 seeds × ~33,400 | ~200,300 |
| `historic` reference searches | 2 seeds × ~6,100 | ~12,300 |
| Ensemble generation | K = 3 pools (re-rolled per draw) + `fixed_probabilistic` draws + E_test; allowance, no Pywr simulation involved | ≤ 10,000 |
| Re-evaluation on E_test | trimmed model, ~1,000–1,200 merged policies × $N_\theta{=}1{,}000$ × $R{=}25$ × $L{=}50$ (§5.4), incl. the one-time full-model presim pass | ~80,000 |
| **Campaign total** | | **~503,000 (67% of the 750,000-SU allocation)** |
| **Reserve** | | **~247,000** |

Exchange rates for spending the reserve: one additional matched search ≈ 33,400 SU; one additional draw for both matched designs (4 searches) ≈ 134,000 SU; a variable-resolution `ffmp_N` sweep at one design × one draw × two seeds × three N values (6 searches) ≈ 200,000 SU. The `ffmp_N` sweep is deprioritized — it runs only on whatever SU remains at the end of the campaign — so the reserve's first call is an additional draw, its second the sweep.

**Pool-size headroom.** Candidate-pool generation is Kirsch–Nowak sampling plus the streamed hazard image — no Pywr-DRB simulation — so its SU cost is bounded by roughly a full-node day (~3,000 SU) per pool even at $P = 10^6$, or ≤ ~9,000 SU across the K = 3 re-rolled pools: under 2% of the campaign. $P$ is therefore not SU-constrained; the binding considerations are generation wall-time, pool storage, and selector behavior, which is why $P$ is set by the saturation diagnostic rather than by budget.

### Replication

A **draw** is the design's construction re-run from scratch with a fresh seed — one definition for every design, re-rolling *everything* that is random about building the ensemble. For `fixed_probabilistic` that is a fresh i.i.d. sample; for `hazard_filling`, **a fresh candidate pool *and* a fresh LHS anchor plan**.

The pool must be re-drawn per draw, and this is load-bearing. Generating the pool *is* part of a hazard-filling design's construction. If the pool were pinned across draws, a hazard-filling draw would vary only its anchor plan while a `fixed_probabilistic` draw re-rolls its entire sample — the two between-draw variances would not be commensurable, and hazard filling would appear more stable **by construction** rather than as a finding. The cost is that step-02 generation scales with $K$; this is disclosed rather than optimized away.

`historic` has $K = 1$ (structural-zero composition variance); each matched design has $K$ draws × $S$ seeds. The unit of analysis for between-design comparison is the **draw**; seeds within a draw are pseudoreplicates, and draw- and seed-level results are reported transparently. Target $K = 3$, $S = 2$, set against the compute allocation.

### Ensemble-quality diagnostics

**Build-QC.** Scenario redundancy (the §3.3 rank-correlation diagnostic re-run on $E_d$ and reported alongside the pool's — a diagnostic, not a gate). Statistical fidelity to $Q_{\text{obs}}$ (monthly moments, lag-1 and cross-site correlation, flow-duration curve) is a **within-`fixed_probabilistic` check only**; hazard filling distorts marginals by design and is never ranked on fidelity — only checked that each selected member is a valid generator output.

**Coverage is method verification, not a comparison result.** L2-star discrepancy and minimum-spanning-tree edge statistics on normalized hazard coordinates, plus the snap-distance distribution, are reported **against the expected discrepancy of a random design at the same $(N, m)$** so the $m$-vs-$N$ tension is visible rather than asserted. Because the LHS + nearest-neighbour selector does not optimize discrepancy, this is an independent measurement that the selector administered the intervention at strength — that the `hazard_filling` ensemble is compositionally shifted relative to `fixed_probabilistic`. It is build-QC, not an endpoint.

**Outcome hypotheses** (falsifiable, may be null). The primary cross-design comparison is the **multivariate Starr satisficing fraction** on $E_{\text{test}}$, with univariate satisficing, Laplace, maximin, and signed improvement-over-status-quo as secondary anchors (`objective_definitions.md` §3). The only regret-type quantity computed is the fixed-reference, design-independent signed improvement over the status quo; no set-relative (best-in-set) regret and no perfect-foresight (Cohen-style) regret are computed.

Scenario discovery, where it is run, operates in the **DU factor space** of $E_{\text{test}}$ after re-evaluation (`objective_definitions.md` §4) as an **optional supporting analysis**, not the primary comparison and not a falsification result; no scenario discovery is performed in hazard space. A search-minus-test overfitting gap is **not** used: its in-sample term is coverage-weighted for hazard filling, so the difference is never an expectation-vs-expectation quantity.

### Open parameters

| Parameter | Status |
|---|---|
| Candidate pool size $P$ | **Decided (2026-07-30): $P = 10^6$.** The nested-P saturation diagnostic showed the adequacy gate (minimum per-axis tail share $\ge$ ~0.30 at $N = 100$) is unreachable at the full $m = 8$ set at any affordable $P$ and passes for the campaign $m = 6$ set only at $P = 10^6$ (0.311; thin margin — re-confirm per production draw). Sharded generation ≈ 600 allocated core-hours per pool (`workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`), so K = 3 re-rolls ≈ 2,000 SU — within the §6 allowance. |
| $K$ ensemble draws | Target 3, set against the compute allocation. Must be set before generation — draws are independent *generations*. |
| $S$ seeds | Target 2. |
| Hazard-axis count $m$ | **Decided (2026-07-30): $m = 6$ selection set** (§3.3 campaign decision): deficit volume, peak depth, onset rate, recovery rate, peak magnitude, pulse duration; `drought_duration` and `flood_rise_rate` computed and reported but excluded from the snap. Chosen via the nested-P saturation diagnostic after the full set failed the adequacy gate at every affordable $P$ (sparsity clause). Single source: `config.HAZARD_SELECTION_AXES`. |
| $E_{\text{test}}$ envelope width | Wider than the search forcing box, so $E_{\text{test}}$ is not a subset of it. |

### Flagged methodological uncertainties

- **Estimator precision.** The two designs differ in the variance of their fitness estimates (a frozen i.i.d. ensemble and a frozen coverage-designed ensemble present different sampling structures to the optimizer). Matching $N$ and $L$ removes the compute confound but not this one; it is disclosed, not removed.
- **$E_{\text{test}}$ conditioning.** Rankings are conditional on the test-ensemble design; the optional second construction would bound but not eliminate this.
- **$m$-vs-$N$ sparsity.** Joint filling at moderate $N$ is not claimed; the intervention rests on stratified marginals and per-axis tail enrichment, and the coverage diagnostics must demonstrate, not assert, that it was administered at strength. Snap concentration and per-axis tail enrichment degrade with $m$ at fixed $P$, and raising $N$ at fixed $P$ *lowers* enrichment (finite tail supply) — the measured basis of the decided $(m = 6, N = 100, P = 10^6)$ configuration (§3.3, §6). Residual: the realized per-axis enrichment margin at $P = 10^6$ is thin and is re-confirmed on every production draw.
- **Generator stationarity under perturbation.** The Kirsch correlation structure is fit on history and reused under shifted moments in $E_{\text{test}}$. The DU forcing space spans volume, seasonality, and variability, but not multi-year drought persistence — and the generator's i.i.d. year-bootstrap under-expresses even the record's own persistence (water-year ρ₁ ≈ 0.10 vs 0.26 on the record, which is itself epochal: 0.51 pre-1980, 0.06 after; the record sits in an unprecedented ~43-yr pluvial, Pederson et al. 2013). A persistence DU axis was evaluated and decided against — the CMIP6 Δρ₁ anchor's upper tail is 39-yr sampling noise and NE projections point to flash, not multi-year, drought (`persistence_axis_diagnostics.md`). Because all scenario designs and $E_{\text{test}}$ share the generator, the design comparison is unaffected; **absolute** robustness claims are scoped to "under historical-level (pluvial-era) interannual persistence", with the $E_{\text{test}}$ hazard overlay quantifying the multi-year-dry stress actually expressed.
- **Partial-event truncation.** Bounded but not eliminated by disjoint windows and the $L$-vs-design-drought check; flagged wherever an event-based hazard axis is used.
