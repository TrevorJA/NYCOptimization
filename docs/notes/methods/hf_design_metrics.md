# Hazard Filling design metrics (supplemental diagnostic)

*Definition of the Hazard Filling (HF) design's measurable properties and of the diagnostic that computes them on hazard images. Machinery: `scengen/subsample.py` (`lhs_nn_assignment`, the pairing-preserving selection), `scengen/selector_diagnostics.py` (`knn_min_sum_assignment`, the certified exact assignment), `scripts/supplemental/hf_design_metrics_run.py` and `hf_design_metrics_figures.py`; wrapper `workflow/supplemental/hf_design_metrics.sh`; settings `supplemental_config.py` (`HFM_*`). Outputs: `outputs/supplemental/hf_design_metrics/{tables,figures}`. Selection recipe: `scenario_design_methods.md` §4.3; measure statement: §4.5. Terminology: `docs/terminology.md` §C.*

---

## 1. Purpose

The HF design is described by three goal terms (diversity, range, coverage) and by the statement that the ensemble represents a probability measure other than the generator's. Each term is given one defining metric from the design-of-experiments and scenario-reduction literature, the selection itself is stated as an optimization problem of which the wired rule is a heuristic, and every quantity is measured on the constructed ensembles. The diagnostic therefore replaces four assertions with measurements: "uniform coverage", "widest range", the size of the gap between the sequential rule and the optimal assignment, and the reweighting of frequency-type objectives. It also produces the two records the experimental proposal promises, the target-displacement report and the stability of the p1/p99 bounds over nested candidate ensembles. Coverage statistics remain method verification, never a comparison result (`scenario_design_methods.md` §6).

## 2. Notation and geometry

$\mathcal{C}$ is the candidate ensemble of $P$ realizations; $\mathbf h(x) \in \mathbb{R}^m$ the hazard characteristics of realization $x$ on the $m = 6$ selection axes (`config.HAZARD_SELECTION_AXES`); $\tilde{\mathbf h}(x) \in [0,1]^m$ the scaled characteristics (each axis scaled by the candidate ensemble's p1 and p99, clipped to the box; manuscript Eq. 6; `subsample.minmax_normalize`). $\mathbf u_1, \dots, \mathbf u_N$ are the target hazard characteristics (a scrambled, unoptimized Latin hypercube sample from `scipy.stats.qmc.LatinHypercube`, seeded by `ScenarioDesign.selector_seed(draw)`). $E \subset \mathcal{C}$, $|E| = N$, is a search ensemble. Distances are Euclidean in the scaled space. Every point-set statistic depends on $N$, so all sets are scored at the common $N$ and, where the statistic is not bounded by construction, as a ratio to the mean over random $N$-subsets of the candidate ensemble (never random points in the cube, which the correlated candidate set cannot attain).

## 3. The selection as an assignment problem

The construction seeks an injective assignment $\sigma:\{1,\dots,N\}\to\mathcal{C}$ minimizing the total target displacement

$$D(\sigma) = \sum_{i=1}^{N} \lVert \mathbf u_i - \tilde{\mathbf h}(\sigma(i)) \rVert_2 ,$$

a linear assignment problem (Kuhn 1955). The wired rule (manuscript Eq. 7) is its sequential nearest-unused heuristic: targets are visited in the sampler's emission order and each takes its nearest not-yet-used candidate (`lhs_nn_assignment`, which returns the pairing; the selectors return `np.sort(rows)`, so production output is unchanged). The exact optimum is solved by min-weight full bipartite matching on the graph joining each target to its $k$ nearest candidates, $k$ grown along `HFM_KNN_LADDER` until linear-programming duality certifies the solution: with dual potentials $u_i$ (targets) and $v_j \le 0$ (candidates in the graph, zero outside it), every excluded edge costs at least the $k$-th neighbour radius $r_i(k)$, so $u_i \le r_i(k)$ for all $i$ makes the sparse dual feasible for the complete problem and proves the sparse optimum global (`knn_min_sum_assignment`; the manifest records `certified`, the slack $\max_i (u_i - r_i(k))$, and the rung). Three quantities bracket the design per draw:

- $\bar d_{\text{free}}$, each target's nearest candidate ignoring injectivity (the lower bound);
- $\bar d^\star$, the certified exact assignment;
- $\bar d_{\text{greedy}}$, the sequential rule (Eq. 7),

with the relative gap $(D_{\text{greedy}} - D^\star)/D^\star$, the maximum displacement of each, and the Jaccard overlap of the two selected sets (`hfm_summary.csv`; `hfm_points.csv` per target).

## 4. Design metrics

| Term (`docs/terminology.md` §C) | Metric | Definition | Reference |
|---|---|---|---|
| Coverage, joint | minimax distance relative to the candidate ensemble | $\phi_{mM}(E) = \max_{x \in \mathcal C} \min_{y \in E} \lVert \tilde{\mathbf h}(x) - \tilde{\mathbf h}(y) \rVert_2$, with the mean and the 50th, 90th, and 99th percentiles of the same candidate-to-nearest-member distances | Johnson, Moore & Ylvisaker (1990); Pronzato & Müller (2012), who evaluate it on a discretization of the region, here the candidate ensemble itself |
| Coverage, per axis | Kolmogorov–Smirnov distance to uniform | $K_a(E) = \sup_t \lvert F_{E,a}(t) - t \rvert$ on the scaled axis, the one-dimensional star discrepancy (Pronzato & Müller 2012, Eq. 1); the exact Latin hypercube attains $K_a \le 1/N$ | McKay, Beckman & Conover (1979) for the stratification property |
| Diversity | minimum-spanning-tree edge lengths | mean edge length as the summary; minimum edge length is the closest pair, the maximin distance | Franco et al. (2009); Damblin, Couplet & Iooss (2013); Bonham et al. (2024); Johnson et al. (1990) |
| Range | span | $\rho_a(E) = \max_{y\in E} \tilde h_a(y) - \min_{y \in E} \tilde h_a(y)$ (1 = the full p1–p99 interval), and the count of members beyond the historical windows' maximum on axis $a$ | McSweeney & Jones (2016) for the range fraction as a subset criterion |

Not adopted, and why: L2-star, centered, wrap-around, and mixture discrepancies (their baseline is uniformity on the cube, unattainable for a correlated candidate set, so they favour box-filling rules; wrap-around periodicity has no meaning on hazard axes; L2-star is not reflection-invariant; the older selector-comparison battery keeps its L2-star for the rule comparison only, `hazard_selector_diagnostics.md`); the closest pair alone (it is the MST minimum edge, and Damblin et al. 2013 show the single-pair statistic is a less robust diagnosis than the edge distribution); coverage λ and mesh ratio (coefficients of variation of the same nearest-neighbour distances); MaxPro and two-dimensional subprojection criteria (no interpretable scale for a diagnostic; projections are shown in the composition figure's pair panels); the cLHS objective terms (Minasny & McBratney 2006; the stratum count duplicates the per-axis KS distance, the correlation term is a partial summary); energy distance and optimal-transport distances (the measure statement of §5 already carries the identity); kernel-density importance weights (bandwidth-dependent, whereas the Voronoi partition of §5 has no tuning); support points and kernel herding (constructions aimed at representativeness, the opposite intent).

## 5. The probability measure the HF ensemble represents

Let $\mu$ be the generator's pushforward measure on hazard space, $\hat\mu_P$ the candidate ensemble's empirical measure, and

$$\nu_E = \frac{1}{N} \sum_{y \in E} \delta_{\tilde{\mathbf h}(y)}$$

the equal-weight empirical measure of a search ensemble, the measure under which every objective's across-year statistic is computed. For the MC design $\nu_E$ is the empirical measure of an i.i.d. sample from $\mu$; for the HF design it is not. Two quantities state the departure:

- the nearest-member redistribution $w_i = \hat\mu_P(V_i)$, where $V_i$ is the set of candidates whose nearest member is $y_i$. This is the optimal redistribution of Dupačová, Gröwe-Kuska & Römisch (2003, Theorem 2: the deleted scenarios' mass goes to the nearest kept scenario) and Rujeerapaiboon et al. (2022, p. 228), and the mean of the coverage distances of §4 equals the Kantorovich (Wasserstein-1) distance between $\hat\mu_P$ and the closest measure supported on $E$. $N w_i$ is the discrete likelihood ratio (Owen 2013, Eq. 9.1; Homem-de-Mello & Bayraksan 2014, §7.4) between the redistributed generator measure and $\nu_E$ on the Voronoi partition;
- the effective sample size $n_{\text{eff}}(E) = 1 / \sum_i w_i^2$ (Kish 1965; Owen 2013, Eq. 9.13), reported as $n_{\text{eff}}/N$ against MC and the random reference. It states, as a number, the variance cost of reading any HF frequency as a generator-measure probability.

The per-axis tail share above the candidate p90 (i.i.d. expectation 0.10) and the per-axis Kolmogorov–Smirnov distance to the candidate marginal give the marginal form of the same statement. Consequence for the objectives: an objective computed over the HF ensemble is a functional of $\nu_E$, reported as such and never reweighted; reliabilities and the mean flood exceedance are frequencies under $\nu_E$, not probabilities under $\mu$. Applying $w$ to persisted per-realization objective units would give the reweighted value; that is not part of this diagnostic.

## 6. Sets compared and references

Per draw: the candidate ensemble (per-axis rows only); the Latin hypercube targets scored against the candidate ensemble (the ideal design, so coverage and diversity show what the snap costs and what a scrambled, unoptimized Latin hypercube already lacks); the realized HF ensemble; the MC ensemble, scaled with the same candidate bounds (members beyond the box are counted, `n_beyond_box`); `HFM_RANDOM_REPLICATES` random $N$-subsets of the candidate ensemble (mean, sd, and a 5th to 95th percentile band); the historical disjoint 10-year windows (`outputs/supplemental/historic_hazard_windows/hazard_windows_10yr.npz`; range and tail statistics, and per-window nearest-member distances). Bound stability: the per-axis p1/p99 on nested prefixes of the candidate ensemble (`HFM_NP_PREFIXES`), as deviations from the full-$P$ value in span units.

## 7. The diagnostic

Inputs (hazard images only; no simulation): the candidate image `statpool_10yr_n{P}_d{k}/hazard_image.npz` or, if absent, the HF image, which embeds the full candidate $H$; the HF image `hazfill_stat_abs_10yr_n{N}_d{k}/hazard_image.npz` with its `_meta.json` (selector seed; recorded bounds cross-checked against the recomputed ones); the MC image `fixprob_10yr_n{N}_d{k}/hazard_image.npz` (skipped with a notice if absent); the historic windows cache (skipped if absent). The selection is replayed from the recorded seed and must reproduce `selected_rows` exactly, else the run errors.

Tables (`smoke_` prefix under `NYCOPT_HFM_SMOKE=1`): `hfm_summary.csv` (one row per set, draw, replicate: coverage, diversity, and measure scalars, ratios to the random mean, and for the HF rows the assignment bracket, gap, Jaccard, certification); `hfm_axes.csv` (set × draw × replicate × axis: min, max, span, KS to uniform, KS to candidate, tail share, beyond-box and beyond-historic counts, bounds); `hfm_points.csv` (per target: coordinates, greedy, exact, and free rows and displacements; per member: nearest-member weight; per historic window: clipped and unclipped coordinates, nearest distances); `hfm_bound_stability.csv`. JSON: `hfm_manifest.json` (slugs, seeds, bounds, ladder, certification, skipped inputs) and `hfm_distributions.json` (ECDF grids for the figures).

Figures (PNG): `F1_marginals_range` (per-axis scaled ECDFs of every set with the random band, the candidate p90, historical ticks, and span bars); `F2_coverage_diversity` (ECDFs of the coverage distances with the minimax marked and of the MST edge lengths; the headline statistics as ratios to random); `F3_target_displacement` (displacement ECDFs of the sequential rule, the exact assignment, and the free lower bound; greedy against exact per target); `F4_measure_weights` (sorted $N w_i$ with $n_{\text{eff}}/N$; $w_i$ against the drought-magnitude coordinate).

Run: `sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ensemble_size_diagnostics.env workflow/supplemental/hf_design_metrics.sh` on Anvil (about a minute per draw on 8 threads; 16 GB); locally `NYCOPT_HFM_SMOKE=1 python scripts/supplemental/hf_design_metrics_run.py` then `hf_design_metrics_figures.py` on the staged P = 300 / N = 40 images. Tests: `tests/test_hf_design_metrics.py` (pure helpers and the smoke identity) and, in `scengen`, `tests/test_subsample.py` (pinned regression of the selection) and `tests/test_selector_diagnostics.py` (certificate against the dense assignment).

## 8. Results

Production values (P = 10⁶, N = 300, draws d0 and d1) are recorded here after the run on the June 1 window images. Until then the SI and manuscript carry bracketed placeholders for: the assignment bracket and gap per draw; the minimax distance and mean coverage distance of HF, targets, and MC as ratios to random; the MST mean and minimum edge ratios; the per-axis Kolmogorov–Smirnov distances and spans; $n_{\text{eff}}/N$ for HF, targets, MC, and random; the tail shares; the bound-stability deviations.
