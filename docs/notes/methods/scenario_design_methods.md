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
| $h(r) \in \mathbb{R}^{m}$ | **Hazard-metric vector**: $m$ low-redundancy drought / low-flow / high-flow indices (§3.3). |
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

The pool owns $P$ realizations and is never simulated. Only each realization's hazard coordinates and generation seed are stored, and the $N$ selected members are regenerated exactly on demand from the deterministic, globally indexed random-stream architecture (§3.4). This storage arrangement is what makes a pool of $10^5$ to $10^6$ members tractable. [Open: $P$ is a production target between 100,000 and 1,000,000.]

### 3.3 Realizations, windows, and hazard metrics

**Windows.** Disjoint, non-overlapping $L$-year blocks. Each scenario starts from fixed initial storage (`INITIAL_VOLUME_FRAC = 0.80`) with a 365-day warm-up excluded from metric computation — a terminating-simulation design. A drought event straddling a window boundary is split; this is bounded by the $L$-vs-design-drought check (§6) and by long $L_{\text{test}}$, where rankings are decided.

**Hazard axes.** Computed on each realized sequence, before any system simulation, from the aggregate NYC inflow (Cannonsville + Pepacton + Neversink). The candidate set is drought metrics from SSI-6 run theory on the controlling event (deficit volume, duration, peak depth, onset rate, recovery rate) and flood metrics from peaks-over-threshold on daily flow (peak magnitude, pulse duration, rise rate). The SSI distribution fit is made **once** on $Q_{\text{obs}}$ and reused for every realization, so hazard coordinates are comparable across the pool and the historic reference.

**Redundancy screen.** Run on the pool's hazard image: drop degenerate axes (near-zero spread), cluster on $1-\lvert\rho_S\rvert$ cutting at $\lvert\rho_S\rvert \ge 0.7$ (Olden & Poff 2003), then select a balanced set spanning distinct clusters across the dry and wet concepts. Target $m \in [3, 4]$ — large enough to separate drought from flood concepts, small enough that uniform filling is achievable at $N=100$ (§6). Report retained $m$ and the per-axis redundancy structure.

### 3.4 Determinism and seed separation

Realization $k$ is fully determined by a child RNG stream keyed to its **global** index, invariant to MPI or batch partition, driving the Kirsch monthly step, the Nowak daily step, and the KDE downstream fill identically regardless of how the index range is split. `regenerate_realization(root_seed, k)` reproduces any single realization bit-for-bit. This is what makes a $10^5$–$10^6$ pool tractable: only the hazard image is persisted, and the few hundred *selected* realizations are materialized on demand.

**Seed domains are disjoint by construction.** Each generated artifact draws its seed from a namespaced stream — `stat_pool`, `fixed`, `hazard_select`, `etest:kn`, `etest:hmm` — so no two designs, and no design and $E_{\text{test}}$, ever share realizations. A shared seed would produce correlated realizations across designs, reintroducing the confound the architecture removes. The disjointness is asserted at import, and a search/test seed-domain collision is a hard error (selection-bias guard, Bonham et al. 2024).

---

## 4. The three designs

Both matched designs run at **$N = 100$, $L = 10$ yr**. `historic` is an unmatched reference.

**4.1 `historic`.** The observed record as one continuous trace; $N=1$, full window (~77 yr). The reference for prevailing applied practice (Giuliani et al. 2016; Herman et al. 2020). $K=1$: composition variance is zero by construction. Cannot be size-matched, and is not part of the controlled contrast.

**4.2 `fixed_probabilistic`.** Generate $N$ realizations of length $L$ i.i.d. from the stationary generator; freeze for the search. $K$ draws × $S$ seeds (draw = ensemble-sampling variance; seed = MOEA variance). Precedent: Quinn et al. (2017); Zatarain Salazar et al. (2017). This is the discipline's random-sampling default, the reference against which designed selection is judged, and — by §3.2 — the exact statistical control for `hazard_filling`.

**4.3 `hazard_filling` — novel.** Select $E_d \subset C$, $\lvert E_d \rvert = N$, whose hazard coordinates cover the empirical hazard manifold.

**Selector.** Scale each retained axis to $[0,1]$ by its pool minimum and maximum, so distances are in **absolute, range-scaled magnitude** units and no single axis dominates while spacing within each axis stays proportional to physical magnitude. Draw $N$ Latin hypercube anchors in the scaled hazard box and snap each to the nearest not-yet-selected pool member in Euclidean distance. The selector is deterministic given its anchor seed. It involves no iterative optimization of a coverage criterion and no tuning parameters. It is a deterministic LHS + nearest-neighbour snap and is never an annealing search.

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

**Role.** The single, common, held-out basis of cross-design comparison, and the sole carrier of deep uncertainty in the study. Never used during search; **never the source of any search ensemble**. Every design's final Pareto-approximate set is re-simulated on $E_{\text{test}}$ with the full untrimmed model.

### 5.1 Construction: LHS over the full DU space × many realizations per point

$E_{\text{test}}$ is **the largest ensemble in the study, by a wide margin**, and is designed to be maximally *uncertainty-encompassing*, not a probability sample.

- **Latin hypercube over the FULL range of the deeply uncertain forcing factors** — the CMIP6 harmonic forcing amplitudes $[m, r_1, r_2]$ (+3 with the CV axis). The envelope is deliberately **wider than any variation the search ensembles contain**, so $E_{\text{test}}$ is not a subset of the space any design searched.
- **Many realizations per LHS point**: $R_{\text{test}} \gg 1$. Each $\theta$ is a **state of the world (SOW)**; its $R_{\text{test}}$ realizations sample the natural variability *within* that SOW. $N_{\text{test}} = N_{\theta,\text{test}} \times R_{\text{test}} \gg N = 100$.
- $L_{\text{test}} \ge L$, long enough to contain whole multi-year droughts.
- **Independent seed domains** (`etest:kn`, `etest:hmm`), disjoint from every search ensemble (§3.4). A search/test seed-domain collision is a hard error.

This is the standard construction of the re-evaluation lineage: Trindade et al. (2017) cross 10,000 LHS DU samples with 1,000 flow realizations; Gold et al. (2022) use $10^6$ SOWs; Kasprzyk et al. (2013) and Bartholomew & Kwakkel (2020) use 10,000 LHS SOWs; Quinn et al. (2020) LHS the generator parameters.

**The re-evaluation is a generalization test.** Because the search ensembles are drawn from the unperturbed stationary generator while $E_{\text{test}}$ spans a forced climate envelope no search ensemble contains, the re-evaluation measures whether hazard-space coverage of the natural-variability manifold produces policies that generalize to conditions never presented during search. This keeps the test instrument structurally distinct from both designs, so $E_{\text{test}}$ favors neither.

**$E_{\text{test}}$ is sampled by LHS, not i.i.d., and the distinction is deliberate.** The i.i.d. requirement of §3.2 belongs *only* to the candidate pool, because a random subset of an i.i.d. pool is distributionally identical to i.i.d. draws — that is what makes `fixed_probabilistic` the exact control for `hazard_filling`. $E_{\text{test}}$ is never subsampled and is never a control; it is the measuring stick, and it should *cover* the deeply uncertain space as evenly as possible rather than sample it in proportion to a measure.

**Consequence, and it is load-bearing: robustness on $E_{\text{test}}$ is not an expectation.** Under deep uncertainty there is no probability measure over the forcing space — that is what "deep" means (Lamontagne et al. 2018). A satisficing fraction over an LHS-designed DU box is a coverage-weighted count over a designed exploration, and must be reported as such, never as an estimate of $\mathbb{E}[\cdot]$. Cross-design comparison is commensurable because $E_{\text{test}}$ is **identical across both designs**, not because it is probability-faithful.

**Construction (the campaign default).** One $E_{\text{test}}$: **Kirsch–Nowak over the wide DU box**, LHS × $R_{\text{test}}$. This is what the campaign requires and what every comparison is computed on.

**Not scenario-neutral — a stated limitation.** Following Quinn et al. (2020), no experimental design is truly neutral: $E_{\text{test}}$ is one deliberately broad, design-conditional reference, and **all scenario-design rankings are conditional on it**. With a single construction this conditioning is *declared, not bounded*. The Kirsch structure inherits its interannual wet/dry persistence from the historical record, so persistence is stressed in neither the search population nor the test ensemble; claims are scoped accordingly (§6).

A **second, structurally different construction** (a multi-site HMM, which differs precisely in that persistence) is **registered as an optional variant** (`etest_hmm_*`, seed domain `etest:hmm`) but is **not part of the default campaign**. Standing it up would let ranking stability across test-ensemble constructions be *measured* (Kendall's $\tau_b$) rather than assumed. The persisted re-evaluation matrix (§5.2) additionally supports zero-simulation-cost composition-sensitivity checks, in which hazard-restricted and envelope-restricted subsets of the single $E_{\text{test}}$ are re-scored to probe whether the ranking depends on the region of the test space emphasized. It is a scope decision, not a technical blocker.

### 5.2 The re-evaluation matrix is persisted in full

Unlike search — which keeps only objective scores — re-evaluation persists the **entire $(\text{solution} \times \text{realization} \times \text{objective})$ matrix in natural units** (`reeval_raw.parquet` + a self-describing `reeval_raw_meta.json`). Every robustness metric is then scored *offline* from that matrix (`src/robustness.py`), so a new metric never requires re-simulating. This is the McPhail et al. (2018) T1×T2×T3 substrate: persisting natural units preserves enough to recompute any performance-value transformation (T1), any scenario subset (T2), and any aggregation (T3) later.

The matrix also records each realization's **SOW id** (its $\theta$ / LHS-point index), so both units of robustness are computable offline without re-simulation:

- **SOW unit** (the MORDM lineage standard: Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023). Stage 1: collapse each $\theta$'s $R_{\text{test}}$ realizations into one performance vector (within-SOW aggregator — mean is risk-neutral, worst is risk-averse; the choice is reported). Stage 2: Starr domain criterion across the $N_{\theta}$ SOWs.
- **Realization unit**: the Starr criterion across all $N_{\text{test}}$ realizations directly.

**Precision is governed by $N_{\theta}$, not by $N_{\text{test}}$.** Realizations within a $\theta$ are not independent, so standard errors are clustered by $\theta$; adding more realizations per SOW sharpens the within-SOW estimate but does not buy additional independent information about the DU space.

### 5.3 Reference-set precision

Recomputing one nondominated set from re-evaluated values pooled across designs induces a self-reference bias — a design contributes points to the frontier it is scored against. Report against **both** the pooled reference set **and** a design-leave-one-out reference (the gap is the self-reference inflation). This is a supplement, not the primary comparison (§6, and Zatarain Salazar et al. 2017 §5.3).

### 5.4 Sizing (open — set against the compute budget)

Re-evaluation cost is $n_{\text{policies}} \times N_{\theta,\text{test}} \times R_{\text{test}} \times L_{\text{test}}$ scenario-years per (design, draw), and re-evaluation uses the **full** model rather than the trimmed one. The full model is only 1.16× the trimmed model's per-evaluation cost (measured on Anvil), so re-evaluation is a small fraction of the campaign and $E_{\text{test}}$ can be sized generously. $N_{\theta,\text{test}}$, $R_{\text{test}}$, and $L_{\text{test}}$ are fixed against the SU allocation and declared as single-source constants; the chunked re-evaluation path (`src/chunk_reeval.py`, workflow step 09) exists for exactly this scale.

---

## 6. Sizing, budget, and diagnostics

### Budget

Both matched designs run at $N = 100$, $L = 10$ yr → **1,000 scenario-years per evaluation, at equal NFE**.

Because $N$ and $L$ are common, per-evaluation simulation cost, warm-up, scenario-years, and wall-clock are **identical**. Equal-NFE and equal-scenario-years coincide, so there is one budget condition and no confound between ensemble composition and search effort. This is a consequence of the sizing choice, not a control that has to be imposed.

**Why $N$ and $L$ are what they are.** The selection comparison requires a *common* $(N, L)$ — if $L$ differed, the selection rule would be confounded with record length. $N$ is then bounded below by the fill requirement: at $m = 4$ hazard axes, $N = 100$ gives ~3.2 points per dimension (~4.6 at $m = 3$), the smallest defensible fill. Long records are not viable here: at a fixed per-evaluation budget, $L = 50$ forces $N \approx 20$, and space-filling in 4-D with 20 points is noise, not filling. $L = 10$ also exceeds the 1960s DRB drought of record (~4–5 yr) plus onset and recovery, so a design-basis event fits inside a window, and it keeps duration-type hazard axes from being truncation-limited. The per-evaluation ensemble of 100 realizations is smaller than the 1,000-realization convention of the Zeff–Herman–Trindade lineage; the ground for it is the annual-unit pooling of `objective_definitions.md` §2, which yields ~900 metric-bearing unit-years per evaluation, comparable to the sample on which that lineage stabilizes the same tail operators.

`historic` ($N=1$, $L \approx 77$) cannot be matched and is reported as a reference.

**NFE.** The function-evaluation budget targets **500,000 NFE per search**, within the range used for comparable reservoir control-policy problems (Quinn et al. 2017; Bartholomew & Kwakkel 2020). This is a target, revisable once initial searches reveal the convergence behavior of the DRB problem at the campaign ensemble size; the runtime archive records intermediate NFE levels so the attained budget can be justified after the fact and a lower budget adopted if search has plateaued.

**Measured cost.** At 128 evaluator ranks per node on Purdue Anvil, one function evaluation over the campaign ensemble (100 realizations of 10 years, trimmed model) takes a median of **173.8 s**, and the full model used at re-evaluation is **1.16×** as expensive. Combined with the measured strong-scaling efficiency, one 500,000-NFE search costs approximately **33,300 SU**. The full matched campaign of two designs at three draws and two seeds each, plus the cheap `historic` reference and the re-evaluation phase, is approximately **415,000 SU of the 750,000-SU Anvil allocation**, leaving reserve for the variable-resolution sweep, any additional draws a power calculation indicates, and the optional second test-ensemble construction. Re-evaluation is nearly free relative to search.

### Replication

A **draw** is the design's construction re-run from scratch with a fresh seed — one definition for every design, re-rolling *everything* that is random about building the ensemble. For `fixed_probabilistic` that is a fresh i.i.d. sample; for `hazard_filling`, **a fresh candidate pool *and* a fresh LHS anchor plan**.

The pool must be re-drawn per draw, and this is load-bearing. Generating the pool *is* part of a hazard-filling design's construction. If the pool were pinned across draws, a hazard-filling draw would vary only its anchor plan while a `fixed_probabilistic` draw re-rolls its entire sample — the two between-draw variances would not be commensurable, and hazard filling would appear more stable **by construction** rather than as a finding. The cost is that step-02 generation scales with $K$; this is disclosed rather than optimized away.

`historic` has $K = 1$ (structural-zero composition variance); each matched design has $K$ draws × $S$ seeds. The unit of analysis for between-design tests is the **draw**; seeds within a draw are pseudoreplicates, so effective $n \approx K$. Model: outcome ~ design (fixed) + draw(design) (random) + seed(draw) (random). Target $K = 3$, $S = 2$, revisable from a pilot minimum-detectable-effect calculation and the estimated seed-versus-draw variance ratio.

### Ensemble-quality diagnostics

**Build-QC.** Scenario redundancy (§3.3 clustering re-run on $E_d$; pass = spans ≥ $m$ clusters). Statistical fidelity to $Q_{\text{obs}}$ (monthly moments, lag-1 and cross-site correlation, flow-duration curve) is a **within-`fixed_probabilistic` check only**; hazard filling distorts marginals by design and is never ranked on fidelity — only checked that each selected member is a valid generator output.

**Coverage is method verification, not a comparison result.** L2-star discrepancy and minimum-spanning-tree edge statistics on normalized hazard coordinates, plus the snap-distance distribution, are reported **against the expected discrepancy of a random design at the same $(N, m)$** so the $m$-vs-$N$ tension is visible rather than asserted. Because the LHS + nearest-neighbour selector does not optimize discrepancy, this is an independent measurement that the selector administered the intervention at strength — that the `hazard_filling` ensemble is compositionally shifted relative to `fixed_probabilistic`. It is build-QC, not an endpoint.

**Outcome hypotheses** (falsifiable, may be null). The primary cross-design comparison is the **multivariate Starr satisficing fraction** on $E_{\text{test}}$, with univariate satisficing, Laplace, maximin, and signed improvement-over-status-quo as secondary anchors (`objective_definitions.md` §3). The only regret-type quantity computed is the fixed-reference, design-independent signed improvement over the status quo; no set-relative (best-in-set) regret and no perfect-foresight (Cohen-style) regret are computed.

The mechanism hypothesis — that a design's policies fail on $E_{\text{test}}$ in the hazard region that design *under-covered* — is an **optional supporting analysis** via hazard-space scenario discovery (`objective_definitions.md` §4), not the primary comparison and not a falsification result. A search-minus-test overfitting gap is **not** used: its in-sample term is coverage-weighted for hazard filling, so the difference is never an expectation-vs-expectation quantity.

**Few-clusters inference.** $\theta$-clustered standard errors are unreliable with few clusters. If a pool or $E_{\text{test}}$ has fewer than ~40 distinct $\theta$ draws, use a hierarchical bootstrap by $\theta$ rather than asymptotic cluster-robust SEs.

### Open parameters

| Parameter | Status |
|---|---|
| Candidate pool size $P$ | $P \gg N$; dense enough that anchors snap to near neighbours. Report the snap-distance distribution. Production target $10^5$–$10^6$. |
| $K$ ensemble draws | Target 3 (revisable from a pilot MDE calc). Must be set before generation — draws are independent *generations*. |
| $S$ seeds | Target 2. |
| Hazard-axis count $m$ | From the redundancy screen on the production pool. Determines whether $N = 100$ fills adequately. |
| NFE budget | Target 500,000 per search; revisable once convergence behavior is observed. |
| $E_{\text{test}}$ envelope width | Wider than the search forcing box, so $E_{\text{test}}$ is not a subset of it. |

### Flagged methodological uncertainties

- **Estimator precision.** The two designs differ in the variance of their fitness estimates (a frozen i.i.d. ensemble and a frozen coverage-designed ensemble present different sampling structures to the optimizer). Matching $N$ and $L$ removes the compute confound but not this one; it is disclosed, not removed.
- **$E_{\text{test}}$ conditioning.** Rankings are conditional on the test-ensemble design; the optional second construction would bound but not eliminate this.
- **$m$-vs-$N$ tension.** Uniform filling at $m \ge 4$ and $N = 100$ is sparse. The coverage-in-context diagnostic must demonstrate, not assert, that the intervention was administered at strength.
- **Generator stationarity under perturbation.** The Kirsch correlation structure is fit on history and reused under shifted moments in $E_{\text{test}}$. Historical interannual persistence is retained by design: the DU forcing space spans volume, seasonality, and variability, but not multi-year drought persistence. Claims are scoped to "under historical persistence" — and because the default $E_{\text{test}}$ is also Kirsch–Nowak, persistence is **not stressed in the test ensemble either**. The optional structurally different $E_{\text{test}}$ variant (§5.1) is what would stress it.
- **Partial-event truncation.** Bounded but not eliminated by disjoint windows and the $L$-vs-design-drought check; flagged wherever an event-based hazard axis is used.
