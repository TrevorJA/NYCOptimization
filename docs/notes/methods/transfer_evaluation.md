# Transfer Evaluation Across Scenario-Design Search Ensembles (supplemental)

*Asks what each scenario design's Pareto-approximate set looks like when it is evaluated under
the **other** designs' search ensembles, and answers it entirely with metrics the study already
uses: Pareto dominance over the FFMP baseline, epsilon-nondominated merged reference sets, and
hypervolume. No new metric is defined. All nine matrix cells are simulated through one
evaluation path - the diagonal too, for the reason in section 7.
Code: `src/transfer_eval.py` (driver library), `src/transfer_stats.py` (pure arithmetic),
`scripts/supplemental/transfer_evaluation_run.py` (stages), `.../transfer_evaluation_figures.py`;
configuration in `supplemental_config.py` (`TEV_*`); run identity in
`workflow/envs/transfer_evaluation.env`; wrappers
`workflow/supplemental/transfer_evaluation_{check,eval_array,analysis}.sh`; tests
`tests/test_transfer_evaluation.py`. Outputs under
`outputs/supplemental/transfer_evaluation/{tables,figures,sets}`, per-unit artifacts in projects
space.*

---

## 1. Question and scope

Each campaign design searched against its own finite scenario ensemble, and each design's
resulting set has so far been scored on exactly two substrates: that same search ensemble, and
the common held-out `E_test`. The cross terms have never been computed. This instrument fills
the matrix:

> Evaluate every design's adopted Pareto-approximate set under every design's search ensemble,
> and ask (a) what fraction of each set dominates the scenario-matched FFMP baseline, (b) if all
> three sets are pooled and merged into one epsilon-nondominated reference set *for a given
> ensemble*, how much of that reference set each optimization contributed, and (c) how each
> set's hypervolume changes across target ensembles.

The expectation on (b) is that an ensemble's merged reference set is built mostly from solutions
optimized under that ensemble. The result of interest is the degree, and any departure from it.

**Scope guards.** This is supplemental and exploratory. Nothing here is routed into
`src.robustness` or the step-10 scorecards: robustness in this study is defined on `E_test`
states of the world, and a transfer cell has no SOW structure and no deep-uncertainty forcing.
No numbered pipeline step is added, and no manuscript figure depends on any of it. The campaign
modules (`src/chunk_reeval.py`, `src/reeval_core.py`, `config.py`, every campaign wrapper) are
read but never modified.

**Claim scoping.** An off-design cell is **not** a generalization test. `monte_carlo` and
`hazard_filling_stationary` draw from the same stationary population, so an off-design cell
estimates the same population quantity from a differently-composed finite sample. The matrix
speaks to how search outcomes transfer between sample *compositions*; it says nothing about
out-of-sample performance against a held-out population, which is `E_test`'s role.

## 2. Substrate, and why it is N = 100

The adopted sets (`outputs/{design}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set`;
335 / 991 / 784 solutions for `historic` / `monte_carlo` / `hazard_filling_stationary`) were
produced 2026-08-11/12. `src/scenario_designs.py` raised the default `SEARCH_ENSEMBLE_N` from
100 to 300 on 2026-08-26 (commit `3890f88`), and no `n300` ensemble is staged. The instrument
therefore pins `NYCOPT_SEARCH_N=100` in its env file so `resolve_search_spec(0)` reproduces the
ensembles the sets were actually searched on, and asserts, per cell before any simulation, that
the resolved target's realization count and length match. Without the pin the run either fails
at stage-in or, if an `n300` ensemble were later staged, would silently score N = 100 diagonal
values against N = 300 off-diagonal values and confound every readout with ensemble size.

Three independent artifacts confirm the N = 100 substrate: the staged directories
(`fixprob_10yr_n100_d0`, `hazfill_stat_abs_10yr_n100_d0`), the `_meta.json` of each
scenario-matched baseline, and the commit date of the default change relative to the set files.

The three target ensembles are `historic_single` (one 78-year observed record, 77 FFMP
unit-years), `fixprob_10yr_n100_d0` and `hazfill_stat_abs_10yr_n100_d0` (N = 100 x L = 10, 900
unit-years each).

## 3. Evaluation path

`src.reeval_core.evaluate_solution_raw` cannot serve a design-to-design transfer, for two
independent reasons. It calls `resolve_reeval()` with no arguments, so it always reads
`config.REEVAL_ENSEMBLE_SPEC` and silently ignores any spec the caller intended; and its
ensemble branch raises whenever `sow_grouping` returns `None`, which it does for every campaign
search ensemble, since `fixprob_*` and `hazfill_stat_abs_*` are `population: "stationary"` with
no `forcing_profiles.npz`. That guard is correct and is not weakened here.

The driver instead composes the two public primitives:

```python
target_spec = get_scenario_design(target).resolve_search_spec(draw)
units, obj_names = evaluate_annual_units(dv, formulation_name="ffmp",
                                         objective_set=obj_set, ensemble_spec=target_spec)
matrix, _labels, survivors = sow_objective_matrix(units, obj_set, [0] * units.shape[0])
```

A single pooled group means every unit-year of every surviving realization passes through the
same stage-(ii) unit operators the search used, so the composed vector is the search-equivalent
one. Passing the target as an explicit `ensemble_spec` value, rather than mutating
`NYCOPT_REEVAL_ENSEMBLE_PRESET`, also leaves
`config.assert_search_test_seed_domains_disjoint` at full strength — which matters, because the
own-draw hazard-filling cells would otherwise trip it (both sides carry
`seed_domain: "stat_pool"`).

**One job serves all three sources.** The active scenario design is set to `historic` so
importing `config` needs no staged search ensemble. It does not need to match a cell's source,
because nothing on the evaluation path reads it once the target is explicit, and because the
three campaign env files are byte-identical on every evaluation-relevant knob (objectives,
demand source, LSTM flags, MOEA config); they differ only in `NYCOPT_SCENARIO_DESIGN` and in
`NYCOPT_SEARCH_REALIZATION_BATCH`, which is a documented results-identical memory knob and is
inert at N = 100. The driver asserts the evaluation-relevant configuration into each cell's
provenance sidecar so a future divergence fails loudly.

## 4. Execution

A work unit is one (cell, solution) pair: 4,220 units over the six off-diagonal cells. Unit cost
is heterogeneous by a factor of five — the scenario-matched baseline sidecars record 149.8 s and
155.4 s for one N = 100 x L = 10 evaluation, against roughly 31 s for the single historic record.

The task farm follows `src/chunk_reeval.py`, the project's tested MPI pattern: ranks pull units
via `O_CREAT|O_EXCL` claim files rather than receiving a contiguous slice, which is what keeps
the five-fold cost spread from stranding ranks on the cheap cells; per-unit writes are atomic
(temp file plus `os.replace`) with a `.failed` sidecar on error; resume state is reconstructed
entirely from the filesystem, so resubmitting the same job *is* the resume; and merge is a
separate stage, so a merge bug can never cost a re-simulation. The work list is ordered
cell-major so a node's ranks re-read one staged target through the page cache and hit the
`src.simulation` model-dict cache.

Memory is not a constraint: `config.search_rank_rss_mb(100, 10)` gives 1.09 GB per rank, so 128
ranks per node is 136 GB against the 218 GB safety line. The step-09 page-cache failure mode
does not apply, because these targets are 178–233 MB rather than 7.3 GB and are shared across
the ranks working one cell.

**Persistence.** Each unit stores the full per-realization annual-unit tensor in long format
alongside its composed objective vector and surviving-realization count, so any alternative
composition, bootstrap or paired shift can be recomputed offline without re-simulation. Each
cell carries a provenance sidecar naming the source design and set file (with its SHA-256 and
row count), the target slug, draw and `_meta.json` identity, the objective names, directions,
epsilons and unit operators, the pinned flow prediction mode and trimmed-model flag, and the git
state. The matrix is two-indexed, so none of this is inferable from a path.

## 5. Readouts

All three work in the orientation each source already uses, converted explicitly at every
boundary. `.set` objective columns are Borg-oriented (every objective minimized, maximize axes
negated); the scenario-matched baseline CSVs are in natural units; and
`src.solution_selection` expects natural units with a directions vector. `src.transfer_stats`
provides `to_borg` / `to_natural` so no call site multiplies by a sign inline.

**Readout 1 — dominance over the FFMP baseline.** Per cell, the fraction of the source set that
Pareto-dominates the baseline *as evaluated on that cell's ensemble*. Reading down a column
compares the three designs on one common substrate; reading across a row shows how a design's
apparent advantage moves with the target ensemble. All three scenario-matched baselines
already exist (`config.baseline_objectives_csv`), so this readout costs no simulation. Because
strict dominance on eight objectives is stringent and may legitimately be zero everywhere — the
current FFMP policy already scores zero on the joint satisficing criterion, with Montague
reliability binding — three companions are reported beside it: the epsilon-dominance fraction at
the adopted vector, the fraction *not dominated by* the baseline, and the distribution of the
number of objectives beaten. The arithmetic is
`src.solution_selection.dominance_mask` / `n_objectives_beaten`, already verified against brute
force in `tests/test_solution_selection.py`.

**Readout 2 — merged reference set per ensemble, and contribution.** For each target ensemble,
all 2,110 solutions from all three optimizations, as evaluated on that ensemble, are pooled and
merged into one epsilon-nondominated reference set at the adopted vector
`[0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05]`. Two attribution numbers are reported, because
the sets differ in size by a factor of three and the two questions have different answers:
*composition share* (contributed / merged-set size) says what the reference set is made of, and
*contribution rate* (contributed / source-set size) says how likely one solution from that
search is to survive the merge. Reporting only the first would let the largest archive look best
by construction.

**Readout 3 — hypervolume.** `CalculateIndicator -i Hypervolume`, every call against one shared
reference set formed from the pooled union of all nine cells. MOEAFramework normalizes indicators
against the reference passed with `--reference` and places the hypervolume reference point at
that set's nadir offset by `hypervolume.delta` (0.01 by default, and
`moeaframework.properties` ships entirely commented out, so every default is in force). Passing
one shared reference is therefore what makes the nine values comparable, with no hand-rolled
normalization. Exact 8-dimensional hypervolume proved tractable here — the three campaign sets
against a 947-member reference returned in about 7 s — so the Monte Carlo fallback that
`TEV_INDICATOR_TIMEOUT_S` guards was not needed.

## 6. Tooling: three measured corrections

MOEAFramework 5.0 is the authority for the merge and the indicators, per the project convention.
Three things about this installation were measured rather than assumed, and each would have
produced a wrong answer silently.

1. **`ResultFileMerger` does apply `--epsilon`.** `src/diagnostics.py` states that it "merges by
   PLAIN Pareto dominance and ignores `--epsilon` for archiving (measured: identical output under
   two vectors)". For `.set` inputs on this installation that is not the case: merging the three
   campaign sets yields 1,330 rows with no `--epsilon`, 1,248 under the adopted vector, and 144
   under a ten-fold vector. The merger applies epsilon dominance in MOEAFramework's own
   formulation, which is not Borg's box convention. This instrument therefore calls the merger
   **without** `--epsilon`, taking the plain-Pareto union (1,330), and applies the epsilon
   archive exactly once with `src.sensitivity_common.epsilon_nondominated`, the Borg convention
   that `src/diagnostics.py` itself documents as validated to reproduce Borg's own seed-archive
   membership. One epsilon definition, the project's own, in one place. The union reduces to 947
   members and the filter is idempotent. (The campaign path is unaffected: filtering an
   already-epsilon-filtered union with the Borg convention still yields a valid Borg archive.
   Only the stated reason in that docstring is wrong, not its result.)
2. **`SetContribution` is an indicator, not a missing tool.** The standalone 3.x tool is absent
   from the 5.0 CLI, but `CalculateIndicator -i Contribution` provides it and, with `--epsilon`,
   matches at epsilon-box resolution. It is used as the cross-check against a Python attribution
   that matches merged-set members back to their source by the 36-value decision vector — exact,
   unique across independent searches, and immune to two policies sharing an objective vector.
   On the format probe the two agreed to all six printed digits (0.353749 / 0.646251).
3. **Argument order changes the result, silently.** With `--reference` placed before
   `--epsilon`, `CalculateIndicator` writes no output and exits 0. A wrong order therefore reads
   back as "no values" rather than as an error. The driver emits `--epsilon` before
   `--reference`, and raises if the CLI exits 0 having produced nothing parseable, so this class
   of failure cannot be mistaken for an absent result.

A related format hazard: two incompatible `.set` dialects coexist in `outputs/`. Files this
instrument writes for MOEAFramework carry the v5 header (`# Version=5`, `# Problem=...`, the
variable and objective definitions), full `repr` precision, and the trailing lone `#` entry
terminator. Without the version line the reader assumes the v4 layout and expects a constraint
column; without the terminator the file parses as zero entries; and at `%.6e` precision
`Contribution`'s value matching fails. None of the three errors — all report zero.

## 7. Why the diagonal is simulated: the metric-window change

The instrument was designed to read its three on-design cells from the objective columns
already stored in each `.set` file, at no compute cost. The path-consistency check was built to
bound the difference between that stored path and this driver, on the expectation that the two
compute an identical aggregation and would agree at pywrdrb's linear-program jitter floor.

They do not. Ten solutions per design, re-evaluated through the driver on that design's **own**
d0 ensemble (job 20576426, 2026-09-11), differ from the stored columns by **1 to 4 epsilon** -
worst 3.91 eps on `montague_flow_reliability_annual` for `hazard_filling_stationary`, with
medians as high as 3.38 eps. The offsets are systematic and signed, not scattered, so they are
not run-to-run jitter.

The cause is a change to the metric window that landed after the production searches:

| | at search time (`dc7e70b`, 2026-08-11/12) | now |
|---|---|---|
| `config.START_DATE` | `1945-10-01` | `1945-12-01` |
| `config.END_DATE` | `2022-09-30` | `2023-11-30` |
| `ENSEMBLE_START_DATE` | did not exist | `1945-12-01` |

Commit `a1e88bd`, 2026-08-18, "align metrics on June 1". Two signatures confirm it. The
`historic` stored reliabilities are exact multiples of 1/76 where the driver produces multiples
of 1/77 - the shifted window admits one more complete FFMP year. The ensemble designs keep the
same nine unit-years per realization, because the December epoch and the six-month exclusion
still leave `L - 1` complete FFMP years, but those nine years now cover a two-month-shifted
slice of each realization, which is enough to move pooled percentiles and failure frequencies by
the observed amounts.

**Consequence for this instrument.** The stored columns are not on the same substrate as
anything the current code produces, so they cannot be used as data. Readout 2 in particular
places all three designs' values into a single nondominated sort; a systematic one-to-four-box
displacement of whichever design supplied stored values would decide which solutions enter the
merged reference set. Every cell is therefore simulated through this driver, diagonal included:
6,330 units rather than 4,220, about 200 core-hours rather than 122. One evaluation path, one
metric window, no mixing.

The path-consistency table
(`outputs/supplemental/transfer_evaluation/tables/tev_path_consistency.csv`) is retained, no
longer as a licence to mix paths but as the measurement of the window change itself.

**Consequence beyond this instrument**, recorded in `TODO.md` rather than acted on here: every
stored `.set` objective column in the campaign predates `a1e88bd`. Any comparison that reads
those columns alongside freshly computed values mixes two metric windows, and the adopted
epsilon vector and the re-filtered set cardinalities were derived on the old one. The `E_test`
re-evaluation path is unaffected, because it simulates rather than reading stored columns.

## 8. Results (2026-09-11)

All 6,330 units completed with zero failures and zero missing units; recomposing objectives
from the persisted per-realization tensor reproduced the per-unit composed vectors exactly
(`recompose_max_dev = 0.000e+00` in every cell). MOEAFramework's own `Contribution` indicator
and the Python decision-vector attribution agree to four decimals in eight of nine cells and to
0.006 in the ninth, the difference being one solution's epsilon-box equivalence.

**Readout 1, dominance over the FFMP baseline.** Strict dominance on all eight objectives is
essentially empty: 1.2 % for the `historic` set on the historic record and 0.0 % in every other
cell. This matches the campaign's finding that the current FFMP policy's joint
satisficing score is zero, and it confirms that eight-objective strict dominance is too
stringent to discriminate. The companions carry the signal: epsilon-dominance runs 25.7 % for
`historic` on the historic record against 3.3 % and 3.6 % on the two ensembles, and the mean
number of objectives beaten is 5.0-5.6 for `historic`, 4.2-4.4 for `hazard_filling_stationary`
and 3.5-3.9 for `monte_carlo`, in every column.

**Readout 2, merged reference set composition.** Each ensemble's merged
epsilon-nondominated reference set is not built mostly from the design optimized on that
ensemble:

| merged set for | from `historic` | from `monte_carlo` | from `hazard_filling` | size |
|---|---|---|---|---|
| historic record | 98.4 % | 0.5 % | 1.1 % | 188 |
| Monte Carlo ensemble | 52.1 % | **34.4 %** | 13.5 % | 163 |
| hazard-filling ensemble | 57.5 % | 21.5 % | **21.0 %** | 186 |

Bold marks the on-design contribution. Three diagnostics were run before this was interpreted,
and they change what it means.

*Enrichment against pool share.* A composition share is not interpretable on its own, because a
source holding 47 % of the pooled input supplies 47 % of the merged set under no advantage
whatsoever. Dividing by pool share gives a null of 1.0 regardless of the unequal archive sizes
(335 / 991 / 784). Under the **plain-Pareto** archive the enrichments are 1.23 / 1.00 / 0.90 on
the Monte Carlo ensemble and 1.31 / 0.90 / 0.99 on the hazard-filling ensemble — that is,
essentially no design effect, and 1,702 of 2,110 pooled solutions are mutually nondominated.
The `historic` advantage appears only after epsilon thinning: 3.28x and 3.62x.

*Leave-one-out.* Rebuilding the epsilon archive with each objective removed in turn localises
the effect to **a single axis**. Dropping `nyc_storage_min_p01_pct` collapses `historic` from
3.28x to **0.98x** on the Monte Carlo ensemble and from 3.62x to 1.95x on the hazard-filling
ensemble. Every other leave-one-out leaves it between 3.06x and 4.52x.

*Box occupancy.* Spread is not the mechanism: `historic` occupies 91.3 distinct epsilon boxes
per 100 solutions against 83.9 and 82.5 for the other two, a ratio of 1.09 that cannot produce a
3.3x enrichment.

So the correct statement is narrow and specific. **The merged-set result is a statement about
one objective.** Policies from the `historic` search hold substantially more minimum NYC
storage — median 22.5 % against 13.2 % and 15.6 % on the Monte Carlo ensemble, and 14.2 %
against 3.0 % and 6.2 % on the hazard-filling ensemble. At the adopted epsilon of 5 % that is
roughly two boxes, enough for those solutions to be epsilon-nondominated whatever they do on the
other seven axes. It is not evidence that `historic` produced better policies overall; at
finer-than-epsilon resolution the three sets are close to indistinguishable in nondominance
terms.

**Readout 3, hypervolume.** On one shared reference set the `historic` set has the largest
hypervolume in every column (0.0859 against 0.0630 for `monte_carlo` under the Monte Carlo
ensemble; 0.0135 against 0.0060 for `hazard_filling_stationary` under the hazard-filling
ensemble). Hypervolume rewards the same storage-axis extent, so this is corroboration of the
same one-axis difference rather than independent evidence.

**Epsilon sensitivity.** The ordering is stable across 0.5x, 1x and 2x the adopted vector but
its magnitude is not: on the Monte Carlo ensemble the `historic` and `monte_carlo` shares are
42.5 % and 41.9 % at half epsilon, 52.1 % / 34.4 % at the adopted vector, and 70.0 % / 23.3 % at
twice it. Coarser boxes reward occupying a distinct region, which is consistent with the
leave-one-out result.

**Reading.** What the matrix shows is that the three designs produce policies that sit in
different regions of objective space, and that the separation is concentrated on minimum NYC
storage: on these substrates the `historic` search preserves a storage buffer that the two
ensemble searches trade away. Under the hazard-filling ensemble the two ensemble-optimized
designs run median minimum storage down to 3.0 % and 6.2 % of capacity, while the
historic-optimized policies retain 14.2 %.

That is an observation about where designs place policies, not a ranking of the designs, not a
generalization claim, and not a statement about `E_test`, which remains the only substrate on
which this study compares designs. Section 9 bounds it further.

No mechanism for that separation is asserted here. Every number in this section is conditioned on
the currently staged N = 100 draw-0 ensembles and the adopted archives, and is provisional
pending the full-scale re-run.

**Validity under the metric-window change.** The archives analysed here were themselves
selected by an epsilon re-filter applied to objective columns computed under the
pre-`a1e88bd` window, while every value analysed is computed under the current one. Two
measurements bound what that costs.

Re-applying the adopted epsilon vector to the current on-design values retains only 55.2 % of
the `historic` archive, 15.8 % of `monte_carlo` and 16.7 % of `hazard_filling_stationary`
(`tev_archive_validity.csv`). The input sets are therefore **not** clean epsilon archives of the
values analysed, and the retention differs by design, though a large part of that difference is
set size rather than window adaptation: an archive of 991 members crowds into occupied epsilon
boxes far more than one of 335. The consequence is that the exact composition percentages should
be read as approximate; they are conditioned on a selection performed on a different substrate.

The qualitative finding does not rest on that selection. In the **raw pre-refilter unions**
(23,273 / 40,277 / 32,661 members, whose stored columns share one window across all three
designs and are therefore mutually comparable) the median minimum-storage values are 10.95 %,
8.98 % and 5.15 % — the same ordering, before any epsilon filtering and under the old window.
The re-filter sharpens the separation (13.93 / 10.49 / 5.19) but does not create it, and the
current-window matrix reproduces it on a common substrate. `tev_raw_vs_adopted.csv` carries the
comparison.

**Mechanism.** The difference is visible in the decision variables. Median
`nyc_allocation_reduction_L3` is 0.195 for the `historic` archive against 0.009 for
`monte_carlo` and 0.005 for `hazard_filling_stationary`, so historic-optimized policies impose
close to the maximum permitted diversion cut at drought level 3 while the ensemble-optimized
policies impose essentially none; median `mrf_profile_scale_spring` is 0.809 against 1.384 and
1.588, so the ensemble designs release substantially more water downstream in spring; and the
drought-zone temporal shifts differ by roughly fifty days at level 3. These are the operational
levers a reservoir system uses to preserve carryover storage, so the decision variables locate
the difference; they do not explain why the searches settled there.
Consistent with a trade rather than an advantage, the `historic` archive is about ten percentage
points **worse** on median NYC delivery deficit (38.6 % against 28.7 %) on the same ensemble.

**Figures.** `tev_parallel_axes_by_design` draws all 2,110 policies on the eight objective axes
per target ensemble, coloured by the optimization that produced them, with a bold per-design
median; the storage separation is directly visible there. `tev_parallel_axes_merged_set` draws
only the merged-set members on the same axes. `tev_enrichment_diagnostic` carries the enrichment
and leave-one-out panels that make the one-axis attribution explicit.

## 9. Scope limits to carry into any reading of the matrix

- Off-design cells are differently-composed finite samples of the same stationary population
  (for `monte_carlo` and `hazard_filling_stationary`), not held-out populations. Nothing here
  licenses a generalization claim.
- `historic` is a single 78-year observed record: 77 annual units against 900 for the ensembles.
  Reliabilities are multiples of 1/77 on one substrate and 1/900 on the others, and the
  percentile objectives (deficit P99, storage P01) are different order statistics on the two
  supports. Comparisons that cross this boundary are directional, not quantitative.
- Merged-set shares depend on the epsilon vector and on source-set cardinality. Both are
  reported: the epsilon sensitivity sweep (`TEV_EPS_SCALES`) shows which shares survive a
  neighbouring epsilon, and the contribution rate is the size-invariant companion to the
  composition share. A share that reorders under a neighbouring epsilon is not a finding.
- **Contributing to a merged reference set rewards spread as well as quality.** A set whose
  members occupy many distinct epsilon boxes contributes more than an equally good set whose
  members cluster, because nondominated membership is about occupying the frontier, not about
  being better on average. The measured epsilon sensitivity shows this is not hypothetical
  here: the `historic` advantage under the Monte Carlo ensemble is 0.6 points at half epsilon
  and 46.7 points at twice it. Readout 2 should be read as "spans more of the frontier under
  this ensemble", not as "is a better set of policies".
- Set cardinality also reflects convergence, not only quality. A single MOEA seed per design at
  equal NFE produced 335 / 991 / 784 members; the differing archive sizes are themselves an
  outcome, and this instrument does not separate convergence state from transfer behaviour.
- Epsilon-box dominance is sensitive to where a value falls relative to a box boundary; a
  baseline sitting exactly on one puts any improvement into the next box. This is a property of
  epsilon dominance, already visible in this project's own epsilon calibration, not an artifact
  of the instrument.
- One MOEA seed per design (`seed_01`, four island files), so no seed-level variability is
  available and none is claimed.
- The on-design (diagonal) values are produced by a search that selected on them - on the old
  metric window, but selected on them nonetheless. Re-simulating removes the window mismatch; it
  does not remove the selection. A design's own cell is therefore still the optimistic end of its
  range by construction, and is reported as such rather than corrected.

## 10. Run sequence

```
# 1. path-consistency check (shared, 30 ranks)     workflow/supplemental/transfer_evaluation_check.sh
# 2. the nine-cell matrix (shared, array 0-39)     workflow/supplemental/transfer_evaluation_eval_array.sh
# 3. merge, analyze, figures (shared)              workflow/supplemental/transfer_evaluation_analysis.sh
#
# Every wrapper takes --export=ALL,NYCOPT_ENV_FILE=workflow/envs/transfer_evaluation.env.
# Smoke: add NYCOPT_TEV_SMOKE=1 (artifacts carry a smoke_ prefix).
# Own-draw arm (SI draw sensitivity): add NYCOPT_TEV_INCLUDE_DRAWS=1 to step 2.
# Resume: resubmit step 2 unchanged - completed units are read back from the
# unit files and skipped, so a fresh array picks up exactly what is left.
```

Step 2 is deliberately many small array tasks rather than one large allocation. The work is
embarrassingly parallel, so SU is flat in rank count and the only question is what the scheduler
will actually start: under the backlog at the time of writing (22,264 jobs pending on `shared`,
1,192 on `wholenode`) a four-node `wholenode` request was estimated nine hours out and a 96-core
`shared` job nineteen hours out, while 16-core half-hour tasks began dispatching within minutes.
All tasks of one array share a single claim space keyed by `SLURM_ARRAY_JOB_ID` and stride the
global work list by global rank, so they draw from one pool without duplicating a unit, and a
task that starts late simply finds less left to do.

Cost: 6,330 units (nine cells), 4,220 at about 155 s and 2,110 at about 31 s, roughly 200
core-hours, about 220 SU on four `wholenode` nodes. Because the farm is embarrassingly parallel,
SU is flat in rank count and node count is purely a wall-time choice. The own-draw arm would add
3,550 units and about 155 core-hours.

Scheduling note: ranks claim units in two passes - a strided first pass, so claims are disjoint
and collide essentially never, then a straggler sweep over what is genuinely unfinished. A naive
"every rank scans the whole list" costs one filesystem claim attempt per (rank, unit), which is
3.2 million Lustre metadata operations at 6,330 units on 512 ranks and would dominate the job.
