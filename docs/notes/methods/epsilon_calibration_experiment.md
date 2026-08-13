# Epsilon-Calibration Experiment

Calibrates the annual-unit (§2) search epsilons — the values Borg's
ε-dominance archive uses (`config.get_epsilons()` →
`src/objectives_ensemble._ANNUAL_REGISTRY_SPEC`) — on the campaign search
measures. An epsilon must (i) resolve *meaningful* policy differences in
interpretable native units, (ii) not resolve *sampling noise* of the ensemble
estimator, and (iii) keep the archive (and therefore the reported Pareto set)
at a tractable cardinality. The published precedent gives the signal-scale
rule (archive resolution ≈ signal scale, Reed et al. 2013) and the
noise-exclusion requirement (epsilon precision set to suppress simulation
noise, Kasprzyk et al. 2013); this experiment measures both floors directly,
per design, and sweeps the archive-size consequence.

Scripts: `scripts/supplemental/epsilon_calibration_{run,figures}.py`;
launcher `workflow/supplemental/epsilon_calibration.sh` (one sbatch job per
design, design selected by `NYCOPT_ENV_FILE` — `workflow/envs/eps_calib_*.env`);
settings in `supplemental_config.py` (`EPS_*`). Outputs under
`outputs/supplemental/epsilon_calibration/`.

## Setup

- **Designs**: each campaign design is calibrated on its own search ensemble —
  `historic` (N = 1 over the trace's consecutive water-year units),
  `fixed_probabilistic` and `hazard_filling_stationary` (N = 100 × L = 10,
  draw 0) — because the search objectives are computed under different
  measures across designs and the archive lives inside each search.
- **Policies**: `EPS_N_POLICIES` = 512 random policies drawn **uniform on the
  constraint-feasible region** (rejection against the two DV-space formal Borg
  constraints — pure DV arithmetic; realized acceptance rate persisted as QC),
  plus the FFMP baseline (id −1). Feasible-only sampling matches the archive's
  population: constraint-dominance keeps infeasible vectors out of the
  archive, so infeasible spread must not inflate the signal scale.
- **Evaluation**: each policy runs once per design through the same batched
  path Borg workers use (`run_simulation_ensemble_batched`, campaign
  realization batch), storing the full stage-(i) cube
  `(n_dv × n_real × n_obj × n_units)` for the **entire annual-unit registry**
  (the 8 active objectives + the flood-P99 diagnostic; the framing-convention
  screens read from this same artifact —
  `framing_convention_diagnostics.md` §0b). Pooling the cube and applying the
  unit operator reproduces the search scalar exactly.

## Per-objective quantities (figures script; post-hoc, no re-simulation)

1. **Signal scale** — IQR of the natural-unit search scalar across the
   feasible random policies; `eps_signal = IQR / 10`.
2. **Noise floor** — bootstrap SD of the unit-operator estimator under
   resampling of the realization axis with replacement
   (`EPS_BOOTSTRAP_B` = 1000, one shared index draw per design); summarized
   as the median (and p90) across policies. The single-trace historic design
   resamples the unit-year axis instead — an i.i.d. approximation that
   understates noise under serial dependence (disclosed, not corrected, per
   the project convention).
3. **Granularity floor** — the failure-frequency objectives take values on a
   1/(N·units) lattice; an epsilon below that step cannot merge adjacent
   attainable values.
4. **Recommendation** — `eps_rec = ceil_to_clean_step(max(1, 2, 3))` in native
   units, with the binding floor and a plain-language interpretation recorded
   per objective.
5. **Archive-size sweep** — ε-nondominated archive size (Borg box convention,
   `src.sensitivity_common.epsilon_nondominated`) of the evaluated policies
   for the recommended vector × `EPS_SCALE_GRID` and for the current registry
   vector, over the active objective subset. Random feasible policies
   under-fill a converged front, so the absolute sizes are a proxy; the
   scaling trend is the decision signal.

## Campaign vector

The Borg problem and the MOEAFramework JARs carry ONE epsilon set for all
designs, so the campaign value per objective is the clean-rounded maximum of
the raw requirement across the **ensemble campaign designs**
(`EPS_CAMPAIGN_DESIGNS` = fixed_probabilistic + hazard_filling_stationary; no
campaign design's archive may resolve below its own noise floor); the binding
design is recorded, and a >4× spread in the raw requirement across the
campaign designs triggers a review warning. The historic single-trace arm is
analyzed and reported alongside but EXCLUDED from the max: it is a
reference for prevailing practice, outside the matched
contrast, and its 76-unit estimator's noise floor would coarsen the shared
vector ~3-4× beyond what the ensemble measures need (reliability ε 0.10
instead of 0.02 on the 0-1 scale). Consequence, disclosed: the historic arm's
archive resolves below its own noise floor on those axes. Adopt the combined
table (`tables/epsilon_recommendation_*.csv`) into `_ANNUAL_REGISTRY_SPEC`
**before** JAR generation (step 00), updating the provenance comment there.

## Outputs

- `cube/unit_cube_{formulation}_{design}_seed{S}_n{N}.h5` — per-unit metric
  cube + DV vectors + acceptance-rate QC.
- `tables/epsilon_diagnostics_{…}.csv` (per design), `archive_sweep_{…}.csv`
  (per design; the campaign vector × `EPS_SCALE_GRID` plus the previous
  provisional vector), `epsilon_recommendation_{…}.csv` (combined).
- `figures/eps_calibration_ladder` (F1, combined: per-design floors vs the
  previous and adopted epsilons, log axis; designs color-keyed, historic in
  reference gray), `scalar_distributions_{design}` (F3: signal spread with
  the adopted and previous epsilon widths), `archive_size_vs_scale` (F2,
  combined: archive cardinality vs scaling of the adopted vector),
  `parallel_axes_{design}` (F4: the evaluated policies on the active
  objectives — shared parallel-coordinates renderer — with the members
  retained by ε-box nondominance under the adopted vector highlighted and
  the FFMP baseline bold; shows the adopted resolution thins the set
  without collapsing any tradeoff axis's span).

## Converged-front re-filter sweep and the adopted vector

A complementary diagnostic re-filters CONVERGED search archives under
candidate epsilon vectors: `scripts/supplemental/epsilon_refilter_sweep.py`
(launcher `workflow/supplemental/epsilon_refilter_sweep.sh`; outputs under
`outputs/supplemental/epsilon_refilter/`), using
`sensitivity_common.epsilon_nondominated`. Random feasible policies
under-fill a converged front, so this sweep prices archive cardinality on
the population the archive actually holds; validation: under the registry
vector the box filter reproduces Borg's C archive membership exactly.

**The adopted campaign vector** is the registry in
`src/objectives_ensemble._ANNUAL_REGISTRY_SPEC`: reliability epsilons paired
at 0.02 (NYC/Montague) with Trenton 0.015 and NJ 0.025, the NYC/Montague
deficit-P99 pair PAIRED at 5.0, flood exceedance 0.3, storage-P01 5.0 (a
5%-of-capacity distinction is significant). Rationale recorded per axis:
the reliability axes sit on the historic trace's 1/76 unit lattice (visible
axis gaps are the lattice, not epsilon coarseness); archive cardinality is
driven by the deficit-P99 pair and the reliability axes; the re-filter
parallel-axes figures confirm every tradeoff axis span is preserved at the
adopted resolution. Two coarser fallback vectors (moderate / at-measured-
floor) remain registered in `epsilon_refilter_sweep.py` if campaign archives
run too large.

**Merger behavior (constraint).** MOEAFramework v5 `ResultFileMerger` merges
by PLAIN Pareto dominance regardless of `--epsilon`, so a bare merge
overstates the front at archive resolution. Step 07 (`run_full_diagnostics`)
therefore ε-box-filters the cross-seed `{slug}_merged.set` in place under
the campaign vector (`src/diagnostics.py::epsilon_box_filter_set`; the plain
union is kept as `*_raw.set`) before computing metrics — and that file is
the first-choice reference set of the step 08/09 re-evaluation
(`src/reevaluate{,_mpi}.py`), so the merge → ε-filter → re-evaluate ordering
holds for every optimization configuration by construction.

## Confirmatory search under the adopted vector (2026-08-05)

The confirmatory 2-seed historic search is DONE and the adoption CONFIRMS.
Job 19688163 (array 1-2, `ffmp_obj8_historic.env`, `mm_full`, 5 nodes x 33
ranks, `--time=02:30:00`); both seeds COMPLETED clean, and both echoed the
adopted vector `[0.02, 5.0, 0.02, 5.0, 0.015, 0.3, 5.0, 0.025]` with
`36 vars, 8 objs, 3 constrs` at pre-flight.

| | seed 1 | seed 2 |
| --- | --- | --- |
| wall | 1:49:51 | 1:55:19 |
| SU (640 cores x wall) | ~1,172 | ~1,230 |
| archive members | 1,546 | 1,384 |
| columns | 44 (36 DV + 8 obj) | 44 (36 DV + 8 obj) |
| min NYC weekly reliability | 0.5000 | 0.5000 |
| members below the 0.5 floor | 0 | 0 |
| members exactly on the floor | 8 | 11 |
| eval exceptions / fatal signatures | 0 / 0 | 0 / 0 |

Total ~2,402 SU, against the ~2.4k SU estimate. The `nyc_reliability_floor`
constraint behaves as an active constraint: the front presses against
0.5000 in both seeds with zero crossings.

**Cardinality: the direction confirms, the magnitude is understated by the
re-filter sweep.** Measured seed archives are ~33% larger than the sweep
predicted, consistently across both seeds, and the cross-seed ε-front is
~10% larger:

| | old ε (shakeout) | re-filter prediction | measured | cut vs old ε |
| --- | --- | --- | --- | --- |
| seed 1 | 2,051 | 1,159 | 1,546 | −25% (−43% predicted) |
| seed 2 | 1,990 | 1,036 | 1,384 | −31% (−48% predicted) |
| cross-seed ε-front | — | 1,599 | 1,761 | — |

Nothing here is near the >~2,000 stale-ε signature, so the vector took
effect. The gap is the expected direction for a live search rather than a
defect in the sweep: Borg's ε-archive steers selection and restarts during
the run, so the search repopulates boxes that a static filter of a finished
archive simply discards. **The re-filter sweep therefore prices a LOWER
BOUND on retained cardinality, not the search's fixed point** — worth
carrying into campaign re-evaluation sizing, where the merged-front count
drives cost.

Step 07 diagnostics: job 19694044 (2:27 on `shared`). ε-box filter logged
`6755 -> 1761 (union kept as *_raw.set)`; per-island runtime metrics scored
against that filtered cross-seed set (8 `.metrics` files) and the three
standard figures rendered under `figures/historic/ffmp_obj8_mm_full/`.
Final-row indicators are finite and cross-seed consistent (hypervolume
0.0338-0.0669, GD 0.0046-0.0093, spacing 0.98-1.13, bands overlapping).
Hypervolume is NOT comparable to the shakeout's 0.0704/0.0672: those were
scored against a different reference set (per-seed, then the 7,544-member
plain union), whereas these are scored against the 1,761-member ε-front.

## Ensemble-front ε re-assessment (2026-08-12, INTERIM — hazfill pending)

The 500k-NFE production searches (draw 0 / seed 1) exposed the 2026-08-05
calibration as under-resolving ENSEMBLE fronts: the fixed_probabilistic
archive returned 6,353 solutions, 100% ε-consistent under the adopted vector
(cross-seed ε-front 6,170), vs 2,685/2,604 for the single-trace historic run
— ensemble-averaged objectives are far smoother, so box occupancy explodes.
Both are far above the ~1,000–1,200 merged-policy re-evaluation sizing.

Diagnostic: `scripts/supplemental/epsilon_ensemble_refilter.py` (jobs
19839023 + 19840678 on `shared`, minutes each) re-filters the production
sets under GROUPED candidate vectors — one ε shared by the four
`*_reliability_annual` axes, one by the two `*_deficit_p99_pct` axes, flood
and storage each their own (Trevor's family-symmetry rule, extending the
2026-08-05 deficit pairing; the grouping raises Trenton 0.015 and NJ 0.025
to the reliability-group value). Substrates: each seed's own `.set` archive
(validation) and the step-07 `*_merged_raw.set` plain union (canonical).
Fast exact reimplementation of the ε-box filter (sum-sorted dominance
sweep), cross-checked in-run against `sensitivity_common.
epsilon_nondominated` and against Borg's own archive membership: exact on
fixed_probabilistic (6,353; union filter reproduces the step-07 6,170
exactly), 2,683/2,685 on historic — both misses are box-boundary
write-precision artifacts (a coordinate exactly on a box edge), tolerated
at ≤0.1%.

Findings on historic + fixed_probabilistic (union sizes; ×1.10 = measured
cross-seed live-search inflation, band 1,000–1,200 on the LARGEST design —
fronts differ ~2.5× across designs, so a per-design band is unsatisfiable):

- The reliability group is the dominant cardinality lever: 0.02 → 0.04
  alone cuts the fixedprob union 6,170 → 2,675 (historic 2,604 → 1,059).
  Grouping alone (Trenton/NJ → 0.02) already cuts 6,170 → 5,384.
- The flood axis is size-inert: production flood values span ~1.6 ε-boxes
  at the adopted 0.3 (axis_structure tables), so flood coarsening barely
  moves cardinality (and is non-monotone via box-boundary placement).
- Deficit and storage coarsening are secondary levers (~35–45% and ~30–40%
  cuts respectively at their measured-recommendation values).
- Band bracketing on fixedprob: rel_x4 (0.08) → 1,205 (adj 1,326, just
  over); rel_x5 (0.10) → 948 (adj 1,043, in band but Trenton collapses to
  2 boxes / 52% span); joint_x1.5 (0.03, 7.5, 0.45, 7.5) → 901 (adj 991, a
  hair under band, with every axis ≥ 87% span retained and Trenton at 8
  boxes). Mixed shapes at rel 0.04–0.05 with def 7.5 / flood 0.5 / storage
  5–7.5 are staged for the next round to fill the 991–1,326 gap with
  better-resolved reliability axes.

Outputs: `outputs/supplemental/epsilon_refilter/{scenario}_ffmp_obj8/`
(axis_structure, grouped_size_sweep, grouped_axis_coverage, two-panel
parallel axes) and `.../combined_ffmp_obj8_grouped/` (recommendation.csv/.md,
size_vs_coarsening.png). NEXT: re-run across all three designs once the
hazard_filling_stationary step-07 merge lands, then recommend a vector for
adoption (part 2: registry edit, JAR rebuild, env re-pin; the production
`.set`/`_merged*.set` artifacts are preserved verbatim — re-filtered copies
get new filenames).

### Final three-design results (2026-08-12, job 19844656)

hazard_filling_stationary production landed (seed archive 5,761; step-07
ε-front 5,544; raw union 32,661) and validates exactly (both reproduction
checks PASS). Full grid (19 candidates), union sizes with the ×1.10
cross-seed inflation, band 1,000–1,200 on the largest design:

| candidate | (rel, def, flood, stor) | historic | fixedprob | hazfill | max adj110 | in band |
| --- | --- | --- | --- | --- | --- | --- |
| adopted | ungrouped | 2,604 | 6,170 | 5,544 | 6,787 | no |
| rel_x5 | 0.10, 5.0, 0.3, 5.0 | 334 | 948 | 734 | 1,043 | **yes** |
| mixed_r2.5f | 0.05, 7.5, 0.5, 5.0 | 352 | 952 | 779 | 1,047 | **yes** |
| mixed_r2f | 0.04, 7.5, 0.5, 5.0 | 487 | 1,173 | 980 | 1,290 | just over |
| mixed_r2fs | 0.04, 7.5, 0.5, 7.5 | 342 | 864 | 753 | 950 | just under |

The two in-band candidates trade resolution very differently
(grouped_axis_coverage tables): rel_x5 concentrates all coarsening on the
reliability family — on fixedprob Trenton falls to 2 boxes / 52% span and
NJ to 6 boxes / 62% span — while mixed_r2.5f keeps reliability at 0.05
(fixedprob: NYC 10 / Montague 15 / Trenton 5 / NJ 16 boxes, every axis
≥ 85% span except Trenton hazfill 73%) by spending the difference on the
deficit pair (7.5% of target) and the size-inert flood axis (0.5 — note
this leaves the historic flood axis single-boxed; it spans < 1 box there
regardless of any ε ≥ 0.5). RECOMMENDED: **mixed_r2.5f = reliabilities
0.05, deficit-P99s 7.5, flood 0.5, storage 5.0** — reliability is the
study's headline objective family and 0.10-step resolution under-serves
it; storage keeps the "5% of capacity is significant" rationale; all four
values are clean and family-symmetric. Adoption pending Trevor's
acceptance (part 2: registry edit, step-00 JAR rebuild, env re-pin,
regret-τ recheck; production artifacts preserved verbatim, re-filtered
copies under new filenames).

### Flood-0.3-preserving round and final recommendation (2026-08-12, job 19845321)

Trevor rejected the flood-0.5 component of mixed_r2.5f, and rightly: the
flood axis spans only 0.35–0.80 ft-days/yr across all three production
fronts (2 / 2 / 4 occupied ε-boxes at the adopted 0.3), so flood-ε
cardinality effects are BOX-BOUNDARY PLACEMENT artifacts, not resolution —
ε=0.6 reproduces the fixedprob base archive exactly while ε=0.5 cuts 34%.
That is fragile across draws/seeds and indefensible in the SI. Third round
kept flood at 0.3 and moved the cut to the deficit group (10.0 = the
measured hazfill eps_rec, externally anchored):

| candidate | (rel, def, flood, stor) | historic | fixedprob | hazfill | max adj110 |
| --- | --- | --- | --- | --- | --- |
| keepf_a | 0.05, 10.0, 0.3, 5.0 | 335 | 991 | 784 | **1,090 — in band** |
| keepf_b | 0.06, 7.5, 0.3, 5.0 | 352 | 1,105 | 816 | 1,216 (just over) |
| keepf_c | 0.05, 7.5, 0.3, 5.0 | 426 | 1,287 | 1,089 | 1,416 |
| keepf_d | 0.06, 10.0, 0.3, 5.0 | 292 | 818 | 632 | 900 |

keepf_a coverage (grouped_axis_coverage): reliabilities 4–14 boxes per
design with span ≥ 0.68 everywhere (NYC 11 boxes / ~1.00 span in all three
designs); deficits 6–9 boxes; flood and storage identical to adopted.
FINAL RECOMMENDATION: **keepf_a — reliabilities 0.05, deficit-P99s 10.0,
flood 0.3, storage 5.0** = full 8-vector
[0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05]. Only two families move
from the 2026-08-05 vector, both to interpretable, externally-anchored
values; flood and storage rationales carry over unchanged.

### Adoption record (2026-08-12)

Trevor ACCEPTED keepf_a. New campaign vector:
**[0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05]**
(reliabilities 0.05 grouped incl. Trenton/NJ; deficit-P99s 10.0 paired;
flood 0.3 and storage 5.0 unchanged from 2026-08-05). Landed same day:

- Registry: `src/objectives_ensemble.py::_ANNUAL_REGISTRY_SPEC` epsilons
  edited (6 entries); spec comment records the revision rationale.
- Regret τ (k·max(ε, floor)): only the two deficit entries move (5.0 →
  10.0, now ε-bound); reliabilities (floors 0.122–0.137), flood (0.927)
  and storage (5.758) stay floor-bound. `NYCOPT_REGRET_TAU` re-pinned in
  all 10 env files carrying it.
- JARs: NO rebuild required for ε-only changes — only DV/objective COUNTS
  reach the problem JARs (`workflow/00_setup_borg_jars.sh` header); ε
  enters Borg at runtime via `config.get_epsilons()` (`src/mmborg.py`) and
  MOEAFramework CLIs via `--epsilon` strings (`src/diagnostics.py`).
  Step 00 re-run anyway (idempotent; JARs byte-identical). This corrects
  the 2026-08-05 note's implication that ε changes need a JAR rebuild.
- Preserved re-filter (job 19845739,
  `scripts/supplemental/write_refiltered_sets.py`): the raw unions
  filtered under the new vector into NEW files
  `outputs/{design}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set` —
  historic 335, fixed_probabilistic 991, hazard_filling_stationary 784,
  matching the sweep's keepf_a row exactly. The 2026-08-05-ε
  `{slug}_merged.set` files and every other production artifact are
  untouched. Step 08/09 should consume the `_eps20260812` refs.
- Tests: the seven ε/τ-touching suites pass against the edited registry
  (173 passed / 2 skipped, same job).

Disclosure carried forward: ε steers the live search, so campaign
archives under the new vector will run ~10–35% larger than these static
re-filter counts (measured 2026-08-05 confirmatory search). The draw-0 /
seed-1 production archives were SEARCHED at the old resolution; only
their re-filtered reference sets are at the new one.
