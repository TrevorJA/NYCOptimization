# Scenario-Design Generation Methods

*Construction recipe for every streamflow scenario set in the comparison. Terminology per `docs/notes/terminology.md`; the experiment itself is in `experimental_design.md`; objective formulations are in `objective_definitions.md`. Citations author-year, resolved via `docs/notes/literature/`.*

---

## 1. Scope and the central asymmetry

Six **evaluation ensembles** (one per scenario design) and one held-out **test ensemble** for re-evaluation. The methodological contribution is **hazard-filling**: an in-the-loop evaluation ensemble built by space-filling subsampling over the empirical hazard manifold of a candidate pool, contrasted against prevailing probabilistic and input-space designs.

**Each design is constructed by its own published recipe, with its own seed stream.** No design draws from another's data. This is what lets each design honestly represent the practice it stands for.

**Hazard-filling is the only design that needs a pool, and this is not an implementation detail.** Hazard coordinates are *emergent* properties of a realized flow sequence — no generator can be asked to produce a realization at a prescribed drought severity. Forcing parameters $\theta$, by contrast, *are* a knob. So:

- input-space designs **generate to** their design points (LHS alone, nothing to snap to);
- hazard-space designs must **select from** a finite pool (LHS anchors + nearest-neighbour snap).

The nearest-neighbour step is intrinsic to hazard-space design, not an approximation of something better. The asymmetry is part of the argument.

**Out of scope** (decided elsewhere): objective formulations and across-scenario aggregation (`objective_definitions.md`); MOEA internals (the `MOEAConfig` axis); robustness-metric mathematics.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $Q_{\text{obs}}$ | Observed/reconstructed daily multi-site historical record (`pub_nhmv10_BC_withObsScaled`, ~1945–2022). |
| $\theta$ | Forcing parameter vector: the intrinsic harmonic amplitudes $[m, r_1, r_2]$ (+3 for the CV axis) of the monthly change-factor profile (§3.1). |
| $\theta_{\text{hist}}$ | The historic fit — no perturbation. Defines the **stationary population**. |
| $r$ | A **realization**: one generator output sequence, seed-deterministic given $\theta$ and a realization index. |
| $h(r) \in \mathbb{R}^{m}$ | **Hazard-metric vector**: $m$ low-redundancy drought / low-flow / high-flow indices (§3.3). |
| $C$ | A **candidate pool**: $P$ i.i.d. realizations owned by one hazard-filling design. Two exist (stationary and DU-forced). Never simulated in full — only its hazard image is stored. |
| $\mathcal{H}$ | The $P \times m$ **empirical hazard manifold** — the image of $C$ in hazard space. The object hazard-filling fills. |
| $E_d$ | **Evaluation ensemble** for design $d$: size $N$, length $L$. |
| $E_{\text{test}}$ | The held-out **test ensemble** (§5): an **LHS** over the full DU forcing range, $N_{\theta,\text{test}}$ points × $R_{\text{test}}$ realizations each. The largest ensemble in the study. Never the source of any $E_d$. |
| $\theta_j$ (SOW) | One LHS point of $E_{\text{test}}$ — a **state of the world**. Its $R_{\text{test}}$ realizations sample natural variability within it. The unit of robustness; precision is governed by $N_\theta$, not $N_{\text{test}}$. |
| $K$ | Independent **ensemble draws** per design (a draw = the design's construction re-run from scratch with a fresh seed). |
| $S$ | MOEA random **seeds** per (design, draw). |

**Spaces:** **input space** ($\Theta$); **hazard space** (the empirical hazard manifold of the pool — strata are defined on realized marginals, not an a-priori domain); **outcome space** (simulated objectives).

---

## 3. Shared upstream machinery

One generator, six call sites. The designs differ only in **which population they sample**, **which $\theta$ the generator is handed**, and **whether a hazard image is streamed**.

### 3.1 Populations

**Stationary** ($\theta = \theta_{\text{hist}}$). The Kirsch–Nowak pipeline fitted to $Q_{\text{obs}}$, generating without climate perturbation. This is the flow model of the prevailing water-supply optimization literature (the Trindade/Gold/Zeff/Zatarain lineage). Hazard variation here comes entirely from **natural variability**: some 10-year windows contain severe multi-year droughts, most do not.

**DU-forced** ($\theta \sim$ hypercube). The deeply-uncertain forcing space is the set of plausible monthly climate perturbations applied to the Kirsch generator's log-space statistics. CMIP6 multimodel change-factor profiles are decomposed into a low-order harmonic series with **phases held fixed** at the canonical CMIP6 shape; only the amplitudes — annual-mean level $m$, annual $r_1$, semiannual $r_2$ — are sampled, bounded by the empirical 90% range of the anchors. Full parameterization in `forcing_parameterization.md`. Climate adjustment is applied to the fitted generator's `mean_period`/`std_period` per Kirsch et al. (2013, eqs. 10–11) before generation; this requires fitting **two** Kirsch generators (one on the baseline period supplying the reference moments, one on the full record that generates). Hazard variation here comes from **natural variability × forcing uncertainty**.

The two populations are not ranked. They are the two settings in which the selection question is asked, and running the method in both is what gives every comparison an exact within-population control (§4).

### 3.2 Pools must be sampled i.i.d., not LHS

Both candidate pools are drawn **i.i.d.** — the stationary pool trivially (there is no $\theta$ to sample), the DU pool by plain Monte Carlo over the hypercube.

This is load-bearing. A uniform random size-$N$ subset of an i.i.d. pool has exactly the joint law of $N$ fresh i.i.d. draws. That is what makes `fixed_probabilistic` the **exact statistical control** for `hazard_filling_stationary`: the two differ only in the selection rule applied to the same population law. **A random subset of an LHS design is not i.i.d.**, so an LHS-sampled pool would silently void the control. Only `input_stratified` uses LHS, and it uses it to *generate*, never to build a pool. An invariant test enforces this, because nothing else would fail if it were broken.

Consequently the DU pool's hazard image is the honest empirical hazard manifold of the DU population, rather than an artifact of a design imposed on $\Theta$.

### 3.3 Realizations, windows, and hazard metrics

**Windows.** Disjoint, non-overlapping $L$-year blocks, so scenarios are i.i.d. *conditional on* $\theta$ (they remain positively correlated marginally under a shared $\theta$; standard errors on ensemble-level quantities are clustered by $\theta$). Each scenario starts from fixed initial storage (`INITIAL_VOLUME_FRAC = 0.80`) with a 365-day warm-up excluded from metric computation — a terminating-simulation design. A drought event straddling a window boundary is split; this is bounded by the $L$-vs-design-drought check (§6) and by long $L_{\text{test}}$, where rankings are decided.

**Hazard axes.** Computed on each realized sequence, before any system simulation, from the aggregate NYC inflow (Cannonsville + Pepacton + Neversink). The candidate set is drought metrics from SSI-6 run theory on the controlling event (deficit volume, duration, peak depth, onset rate, recovery rate) and flood metrics from peaks-over-threshold on daily flow (peak magnitude, pulse duration, rise rate). The SSI gamma fit is made **once** on $Q_{\text{obs}}$ and reused, so hazard coordinates are comparable across populations.

**Redundancy screen.** Run on each pool's hazard image: drop degenerate axes (near-zero spread), cluster on $1-\lvert\rho_S\rvert$ cutting at $\lvert\rho_S\rvert \ge 0.7$ (Olden & Poff 2003), then select a balanced set spanning distinct clusters across the dry and wet concepts. Target $m \in [3, 4]$ — large enough to separate drought from flood concepts, small enough that uniform filling is achievable at $N=100$ (§6). Report retained $m$ and the per-axis redundancy structure.

### 3.4 Determinism and seed separation

Realization $k$ is fully determined by a child RNG stream keyed to its **global** index, invariant to MPI or batch partition, driving the Kirsch monthly step, the Nowak daily step, and the KDE downstream fill identically regardless of how the index range is split. `regenerate_realization(root_seed, k)` reproduces any single realization bit-for-bit. This is what makes a $10^5$–$10^6$ pool tractable: only the hazard image is persisted, and the few hundred *selected* realizations are materialized on demand.

**Seed domains are disjoint by construction.** Each generated artifact draws its seed from a namespaced stream — `stat_pool`, `du_pool`, `fixed`, `resample_pool`, `input_strat`, `hazard_select_stat`, `hazard_select_du`, `etest:kn`, `etest:hmm` — so no two designs, and no design and $E_{\text{test}}$, ever share realizations. Before per-design generation this did not matter (designs merely selected indices from shared data); now that every design *generates*, a shared seed would produce correlated realizations across designs, reintroducing the confound the architecture removes. The disjointness is asserted at import, and a search/test seed-domain collision is a hard error (selection-bias guard, Bonham et al. 2024).

---

## 4. The six designs

All of designs 2–6 run at **$N = 100$, $L = 10$ yr**. `historic` is an unmatched reference.

### Stationary population

**4.1 `historic`.** The observed record as one continuous trace; $N=1$, full window (~77 yr). The reference for prevailing applied practice (Giuliani et al. 2016; Herman et al. 2020). $K=1$: composition variance is zero by construction. Cannot be size-matched, and is not a controlled comparison.

**4.2 `fixed_probabilistic`.** Generate $N$ realizations of length $L$ i.i.d. from the stationary generator; freeze for the search. $K$ draws × $S$ seeds (draw = ensemble-sampling variance; seed = MOEA variance). Precedent: Quinn et al. (2017); Zatarain Salazar et al. (2017). This is the reference against which designed selection is judged, and — by §3.2 — the exact statistical control for `hazard_filling_stationary`.

**4.3 `resampled_probabilistic`.** Generate a stationary pool of $P$ realizations; at every function evaluation redraw $N$ of them using an evaluation-indexed RNG stream, so the draw is reproducible but differs each evaluation. No fixed ensemble exists; the per-evaluation draw variance folds into fitness noise, and in-search objective values are not comparable across evaluations — which is why re-evaluation is the comparison point.

**Precedent and declared deviation.** The design tests whether *freezing* the search ensemble causes overfitting; its primary anchor is Brodeur et al. (2020), which imports bagging and cross-validation into reservoir control-policy search. Trindade et al. (2017, 2019) and Gold et al. (2022, 2023) are cited **only for the principle** that the search ensemble is re-randomized across evaluations. **They are not cited as the mechanism.** Trindade evaluates *all* 1,000 realizations every evaluation and re-randomizes the flow ↔ DU-vector *pairing*; our $\theta$ is fused into the realization at generation, so there is no pairing to re-randomize. Ours is index resampling of $N$ from a pre-staged pool. This deviation is stated in the manuscript, not footnoted.

**4.4 `hazard_filling_stationary` — novel.** Select $E_d \subset C_{\text{stat}}$, $\lvert E_d \rvert = N$, whose hazard coordinates are approximately uniform and well-separated over the empirical hazard manifold.

**Selector.** Normalize $\mathcal{H}$ per axis to $[0,1]$ by its empirical CDF, so each axis is uniform and "uniform in hazard space" is well-defined under skewed marginals. Draw $N$ Latin hypercube anchors in $[0,1]^m$ and snap each to the nearest not-yet-selected pool member. Deterministic given the anchor seed; no annealing, no tuning.

The selector is deliberately the simplest defensible space-filling design. Because it does **not** optimize a discrepancy objective, L2-star discrepancy remains an *independent* diagnostic of the achieved design rather than the quantity being minimized — so coverage can be reported as a result, not merely as a build-verification gate. $K$ draws vary the anchor seed and therefore measure **anchor-placement variance**, the construction variance of the design.

Selection operates on the stored hazard image alone; the pool's daily traces are never loaded. Only the $N$ selected realizations are materialized.

### DU-forced population

**4.5 `input_stratified`.** Latin hypercube sample over the **intrinsic harmonic forcing parameters** — $[m, r_1, r_2]$, plus three more if the CV axis is active — and **generate** $R$ realizations at each design point ($N = N_\theta \times R$). Frozen for the search; $K$ draws × $S$ seeds.

LHS **alone**: there is no pool and no snapping, because $\theta$ is a knob on the generator. Stratifying the derived 12-dimensional monthly change-factor vector would be a mistake — those twelve numbers are a deterministic function of the three amplitudes, so LHS there stratifies a space of intrinsic dimension 3.

$R > 1$ separates forcing uncertainty from natural variability within a forcing (as in Quinn et al. 2018). At fixed $N$ this is a real tradeoff: $R = 1$ maximizes input coverage, while $R > 1$ represents within-$\theta$ variability so that a severe $\theta$ is not represented by a single benign draw. The $N_\theta / R$ split is set before generation (§6).

Precedent: Quinn et al. (2020); Bartholomew & Kwakkel (2020); Eker & Kwakkel (2018); Watson & Kasprzyk (2017).

**4.6 `hazard_filling_du` — novel.** Identical selector and hazard axes to §4.4, applied to the DU candidate pool $C_{\text{du}}$. Contrasting this with `input_stratified` isolates the central claim: uniform coverage in *input* space need not produce uniform coverage in *hazard* space, because distinct $\theta$ often yield hydrologically redundant realizations (Quinn et al. 2020; Guo et al. 2018).

### 4.7 The controlled contrasts

| Contrast | Held fixed | Question |
|---|---|---|
| `fixed_probabilistic` → `hazard_filling_stationary` | generator, population law, $N$, $L$ | Does hazard coverage beat random sampling? |
| `input_stratified` → `hazard_filling_du` | forcing space, $N$, $L$ | Does hazard coverage beat **input** coverage? *(the central claim)* |
| `hazard_filling_stationary` → `hazard_filling_du` | selection rule, $N$, $L$ | What does the DU forcing space add? |
| `historic`, `resampled_probabilistic` | — | Prevailing-practice anchor; overfitting probe. |

Each contrast varies one thing and maps onto a real published design. Running hazard-filling in both populations is what makes this possible: a stationary-only pool would leave `input_stratified` with no input space to stratify, and a DU-only pool would leave hazard-filling with no exact random-selection control.

### 4.8 Probability distortion

Hazard-filling deliberately distorts scenario probabilities toward uniform hazard coverage: rare-but-severe corners are over-represented relative to their frequency in the pool. The rationale for uniform *coverage* of the response surface comes from the bottom-up / decision-scaling tradition (Culley et al. 2016; Herman et al. 2016) — cited as a sampling/coverage argument, **not** as a precedent for biasing an optimization objective.

The distortion biases the **search trajectory**, not only the reported number: any objective aggregated over scenarios is computed under the distorted measure and biases *which policies the optimizer selects*.

Re-evaluation removes the **evaluation** bias of scoring each design on its own ensemble — every design is measured with the same instrument. It does **not** remove **selection** bias, and it does **not** "restore the true measure": there is no true measure to restore, because $E_{\text{test}}$ is itself a *designed* LHS exploration of a deeply-uncertain box, not a probability sample (§5.1). What makes the comparison valid is that $E_{\text{test}}$ is **identical across all designs**, not that it is probability-faithful.

Importance-sampling reweighting is rejected on its own terms: the estimator variance explodes as coverage → uniform, precisely in the corners the design targets. Search objectives are therefore reported as **coverage-weighted quantities, never as estimates of an expectation** under any measure, and cross-design comparison rests entirely on the common re-evaluation.

---

## 5. Test ensemble ($E_{\text{test}}$)

**Role.** The single, common, held-out basis of cross-design comparison. Never used during search; **never the source of any search ensemble**. Every design's final Pareto-approximate set is re-simulated on $E_{\text{test}}$.

### 5.1 Construction: LHS over the full DU space × many realizations per point

$E_{\text{test}}$ is **the largest ensemble in the study, by a wide margin**, and it is designed to be maximally *uncertainty-encompassing* — not to be a probability sample.

- **Latin hypercube over the FULL range of the deeply-uncertain factors** — the harmonic forcing parameters $[m, r_1, r_2]$ (+3 with the CV axis). The envelope is deliberately **wider than the search forcing box**, so $E_{\text{test}}$ is not a subset of the space any design searched.
- **Many realizations per LHS point**: $R_{\text{test}} \gg 1$. Each $\theta$ is a **state of the world (SOW)**; its $R_{\text{test}}$ realizations sample the natural variability *within* that SOW. $N_{\text{test}} = N_{\theta,\text{test}} \times R_{\text{test}} \gg N = 100$.
- $L_{\text{test}} \ge L$, long enough to contain whole multi-year droughts.
- **Independent seed domains** (`etest:kn`, `etest:hmm`), disjoint from every search ensemble (§3.4). A search/test seed-domain collision is a hard error.

This is the standard construction of the lineage: Trindade et al. (2017) cross 10,000 LHS DU samples with 1,000 flow realizations; Gold et al. (2022) use $10^6$ SOWs; Kasprzyk et al. (2013) and Bartholomew & Kwakkel (2020) use 10,000 LHS SOWs; Quinn et al. (2020) LHS the generator parameters.

**$E_{\text{test}}$ is sampled by LHS, not i.i.d., and the distinction is deliberate.** The i.i.d. requirement of §3.2 belongs *only* to the candidate pools, because a random subset of an i.i.d. pool is distributionally identical to i.i.d. draws — that is what makes `fixed_probabilistic` the exact control for `hazard_filling_stationary`. $E_{\text{test}}$ is never subsampled and is never a control; it is the measuring stick, and it should *cover* the deeply-uncertain space as evenly as possible rather than sample it in proportion to a measure.

**Consequence, and it is load-bearing: robustness on $E_{\text{test}}$ is not an expectation.** Under deep uncertainty there is no probability measure over $\Theta$ — that is what "deep" means (Lamontagne et al. 2018: *"our imperfect perceptions of what is probable do not limit our exploration of what is possible"*). A satisficing fraction over an LHS-designed DU box is a **coverage-weighted count over a designed exploration**, and must be reported as such — never as an estimate of $\mathbb{E}[\cdot]$. Cross-design comparison is commensurable because $E_{\text{test}}$ is **identical across all designs**, *not* because it is probability-faithful.

**Construction (the campaign default).** One $E_{\text{test}}$: **Kirsch–Nowak over the wide DU box**, LHS × $R_{\text{test}}$. This is what the campaign requires and what every comparison is computed on.

**Not scenario-neutral — a stated limitation.** Following Quinn et al. (2020), no experimental design is truly neutral: $E_{\text{test}}$ is one deliberately-broad, design-conditional reference, and **all scenario-design rankings are conditional on it**. With a single construction this conditioning is *declared, not bounded*. In particular, the Kirsch structure inherits its interannual wet/dry persistence from the historical record, so persistence is not stressed in the test ensemble either — the same scoping caveat already applied to the forcing space (§6).

A **second, structurally different construction** (a multi-site HMM, which differs precisely in that persistence) is **registered as an optional variant** (`etest_hmm_*`, seed domain `etest:hmm`) but is **not part of the default campaign**. Standing it up would let ranking stability across test-ensemble constructions be *measured* (Kendall's $\tau_b$) rather than assumed, which is Quinn et al. (2020)'s recommendation and the natural reviewer question. It is a scope decision, not a technical blocker.

### 5.2 The re-evaluation matrix is persisted in full

Unlike search — which keeps only objective scores — re-evaluation persists the **entire $(\text{solution} \times \text{realization} \times \text{objective})$ matrix in natural units** (`reeval_raw.parquet` + a self-describing `reeval_raw_meta.json`). Every robustness metric is then scored *offline* from that matrix (`src/robustness.py`), so a new metric never requires re-simulating. This is the McPhail et al. (2018) T1×T2×T3 substrate: persisting natural units preserves enough to recompute any performance-value transformation (T1), any scenario subset (T2), and any aggregation (T3) later.

The matrix also records each realization's **SOW id** (its $\theta$ / LHS-point index), so both units of robustness are computable offline without re-simulation:

- **SOW unit** (the Triangle-lineage standard: Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023). Stage 1: collapse each $\theta$'s $R_{\text{test}}$ realizations into one performance vector (within-SOW aggregator — mean is risk-neutral, worst is risk-averse; the choice is reported). Stage 2: Starr domain criterion across the $N_{\theta}$ SOWs.
- **Realization unit**: the Starr criterion across all $N_{\text{test}}$ realizations directly.

**Precision is governed by $N_{\theta}$, not by $N_{\text{test}}$.** Realizations within a $\theta$ are not independent, so standard errors are clustered by $\theta$; adding more realizations per SOW sharpens the within-SOW estimate but does not buy additional independent information about the DU space.

### 5.3 Reference-set precision

Recomputing one nondominated set from re-evaluated values pooled across designs induces a self-reference bias — a design contributes points to the frontier it is scored against. Report against **both** the pooled reference set **and** a design-leave-one-out reference (the gap is the self-reference inflation). This is a supplement, not the primary comparison (§6, and Zatarain Salazar et al. 2017 §5.3).

### 5.4 Sizing (open — set against the compute budget)

Re-evaluation cost is $n_{\text{policies}} \times N_{\theta,\text{test}} \times R_{\text{test}} \times L_{\text{test}}$ scenario-years per (design, draw), and re-evaluation uses the **full** model rather than the trimmed one. At the Anvil-measured floor of ~0.15 core-seconds per scenario-year, a large $E_{\text{test}}$ is comparable in cost to the entire MOEA search campaign. $N_{\theta,\text{test}}$, $R_{\text{test}}$, and $L_{\text{test}}$ are therefore fixed against the SU allocation and are declared as single-source constants; the chunked re-evaluation path (`src/chunk_reeval.py`, workflow step 09) exists for exactly this scale.

---

## 6. Sizing, budget, and diagnostics

### Budget

All matched designs run at $N = 100$, $L = 10$ yr → **1,000 scenario-years per evaluation, at equal NFE**.

Because $N$ and $L$ are common across designs, per-evaluation simulation cost, warm-up, scenario-years, and wall-clock are **identical**. Equal-NFE and equal-scenario-years coincide, so there is one budget condition rather than two arms, and no confound between ensemble composition and search effort. This is a consequence of the sizing choice, not a control that has to be imposed.

**Why $N$ and $L$ are what they are.** The selection comparison requires a *common* $(N, L)$ — if $L$ differed across designs, the selection rule would be confounded with record length. $N$ is then bounded below by the fill requirement: at $m = 4$ hazard axes, $N = 100$ gives ~3.2 points per dimension (~4.6 at $m = 3$), the smallest defensible fill. This is why long records are not viable here: at a fixed per-evaluation budget, $L = 50$ forces $N \approx 20$, and space-filling in 4-D with 20 points is noise, not filling. $L = 10$ also exceeds the 1960s DRB drought of record (~4–5 yr) plus onset and recovery, so a design-basis event fits inside a window, and it keeps duration-type hazard axes from being truncation-limited.

`historic` ($N=1$, $L \approx 77$) cannot be matched and is reported as a reference, not a controlled arm.

### Replication

A **draw** is the design's construction re-run from scratch with a fresh seed — one definition for every design, and it re-rolls *everything* that is random about building the ensemble. For `fixed_probabilistic` that is a fresh i.i.d. sample; for `input_stratified`, a fresh LHS design; for `resampled_probabilistic`, a fresh pool; for the hazard-filling designs, **a fresh candidate pool *and* a fresh LHS anchor plan**.

The pool must be re-drawn per draw, and this is load-bearing rather than incidental. Generating the pool *is* part of a hazard-filling design's construction. If the pool were pinned across draws, a hazard-filling draw would vary only its anchor plan while a `fixed_probabilistic` draw re-rolls its entire sample — the two between-draw variances would not be commensurable, and hazard-filling would appear more stable **by construction** rather than as a finding. The cost is that step-02 generation scales with $K$ for every design, pools included; this is disclosed rather than optimized away.

`historic` has $K = 1$ (structural-zero composition variance); every other design has $K$ draws × $S$ seeds. The unit of analysis for between-design tests is the **draw**; seeds within a draw are pseudoreplicates, so effective $n \approx K$. Model: outcome ~ design (fixed) + draw(design) (random) + seed(draw) (random). Target $K = 10$, $S = 2$–3 — more draws, fewer seeds.

### Ensemble-quality diagnostics

**Build-QC.** Scenario redundancy (§3.3 clustering re-run on $E_d$; pass = spans ≥ $m$ clusters). Statistical fidelity to $Q_{\text{obs}}$ (monthly moments, lag-1 and cross-site correlation, flow-duration curve) is a **within-probabilistic-design check only**; hazard-filling distorts marginals by design and is never ranked on fidelity — only checked that each selected member is a valid generator output.

**Coverage (a result, not a gate).** L2-star discrepancy and minimum-spanning-tree edge statistics on normalized hazard coordinates, reported **against the expected discrepancy of a random design at the same $(N, m)$** so the $m$-vs-$N$ tension is visible rather than asserted. Because the LHS+NN selector does not optimize discrepancy, this is an independent measurement.

**Outcome hypotheses** (falsifiable, may be null). The primary cross-design comparison is the **multivariate Starr satisficing fraction** on $E_{\text{test}}$, with Laplace, maximin, and signed improvement-over-status-quo as secondary anchors (`objective_definitions.md` §3). **No regret is computed** — best-in-set regret is design-coupled, and Cohen-style baseline regret would need a perfect-foresight optimization per scenario.

The mechanism hypothesis — that a design's policies fail on $E_{\text{test}}$ in the hazard region that design *under-covered* — is tested directly by hazard-space scenario discovery (`objective_definitions.md` §4), not by a correlational coverage-vs-robustness regression. A search-minus-test overfitting gap is **not** used: its in-sample term is coverage-weighted for hazard-filling, so the gap is never expectation-vs-expectation.

**Few-clusters inference.** $\theta$-clustered standard errors are unreliable with few clusters. If a pool or $E_{\text{test}}$ has fewer than ~40 distinct $\theta$ draws, use a hierarchical bootstrap by $\theta$ rather than asymptotic cluster-robust SEs.

### Open parameters

| Parameter | Status |
|---|---|
| $N_\theta / R$ split for `input_stratified` at $N = N_\theta R = 100$ | To set before generation. |
| Candidate pool sizes $P$ (stationary, DU) | $P \gg N$; dense enough that anchors snap to near neighbours. Report the snap-distance distribution. |
| $K$ ensemble draws | Target 10. Must be set before generation — draws are now independent *generations*. |
| Hazard-axis count $m$ | From the redundancy screen on the production pools. Determines whether $N = 100$ fills adequately. |
| $E_{\text{test}}$ envelope width | Wider than the search forcing box, so $E_{\text{test}}$ is not a subset of it. |

### Flagged methodological uncertainties

- **Estimator precision.** Designs differ in the variance of their fitness estimates (a frozen ensemble gives low-variance/biased fitness; `resampled_probabilistic` gives unbiased/noisy fitness). Matching $N$ and $L$ removes the compute confound but not this one; it is disclosed, not removed.
- **$E_{\text{test}}$ conditioning.** Rankings are conditional on the test-ensemble design; the ≥2-construction sensitivity bounds but does not eliminate this.
- **$m$-vs-$N$ tension.** Uniform filling at $m \ge 4$ and $N = 100$ is sparse. The coverage-in-context diagnostic must demonstrate, not assert, an advantage.
- **Generator stationarity under perturbation.** The Kirsch correlation structure is fit on history and reused under shifted moments. Historical interannual persistence is retained by design: the DU forcing space spans volume, seasonality, and variability, but not multi-year drought persistence. Claims are scoped to "under historical persistence" — and because the default $E_{\text{test}}$ is also Kirsch–Nowak, persistence is **not stressed in the test ensemble either**. The optional structurally-different $E_{\text{test}}$ variant (§5.1) is what would stress it, and standing it up is the natural way to close this caveat.
- **Partial-event truncation.** Bounded but not eliminated by disjoint windows and the $L$-vs-design-drought check; flagged wherever an event-based hazard axis is used.
