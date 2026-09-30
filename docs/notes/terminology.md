# Code-Facing Glossary

*Maps the prose vocabulary of `docs/terminology.md`, which governs the manuscript, the notes and every figure, to the registry keys, slugs, environment variables and column names the code uses. An entry exists only where a code-side name exists; the definitions live in the authority.*

---

## Designs and ensembles

**Scenario design.** Registry `src/scenario_designs.py::SCENARIO_DESIGNS`, selected by `NYCOPT_SCENARIO_DESIGN`; the design name is the top-level `outputs/{scenario}/` directory (`config.active_scenario_name()`), the only sense in which the code says scenario. The Historical (HIST) design is `historic`, the Monte Carlo Sampling (MC) design is `monte_carlo` and the Hazard Filling (HF) design is `hazard_filling_stationary`; only these three carry `campaign=True`.

**Search ensemble.** The staged ensemble every candidate policy is evaluated on during search: `config.SEARCH_ENSEMBLE_SPEC`, slug `fixprob_10yr_n{N}_d{k}` for the MC design and `hazfill_stat_abs_10yr_n{N}_d{k}` for the HF design, staged by workflow steps 02–04.

**Candidate ensemble.** The $P$ i.i.d. realizations the HF design selects from: slug `statpool_10yr_n{P}_d{k}`, size `NYCOPT_CANDIDATE_POOL_N`, built by `workflow/supplemental/gen_pool_*.sh`. Only its hazard image (`hazard_image.npz`) and seeds are stored, and the selected members are regenerated on demand. Slugs, variables and script names keep the word pool.

**Realization.** One generated or observed streamflow sequence, addressed by its global realization index (`regenerate_realization(root_seed, k)`); the unit of independence in every bootstrap and subsample. A **draw** `d{k}` is a design's construction re-run with a fresh seed, and a **seed** (`--array` of step 06) is one MM Borg trial on a fixed draw.

**Re-evaluation ensemble.** `E_test`, preset `etest_kn_50yr_n25000` (`src/etest.py`: `E_TEST_N_THETA` = 1,000 SOWs × `E_TEST_R` = 25 realizations × `E_TEST_YEARS` = 50), re-evaluated on its leading `E_TEST_REEVAL_N_THETA` = 500 SOWs, the preset `etest_kn_50yr_n25000_first25ch` returned by `campaign_reeval_preset()` and passed as `NYCOPT_REEVAL_ENSEMBLE_PRESET` on every re-evaluation submission. Identifiers keep the word test (`E_test`, `etest_*`, `generate_test_ensemble`).

**State of the world (SOW).** One Latin hypercube point of `E_test` with its realizations: column `sow_id` of `reeval_raw.csv.gz`, labels `sow_labels` in `reeval_raw_meta.json`, substrate `sow_annual_unit`.

**Forcing space.** The CMIP6 harmonic change-factor box of `forcing_parameterization.md` (`src/etest.py` bounds and margin), sampled only by `E_test`; the generator is stationary in every search design.

## Hazard space

**Hazard metrics and selection axes.** `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`): `drought_magnitude`, `drought_severity`, `drought_development_rate`, `drought_termination_rate`, `flood_peak_discharge`, `flood_pulse_duration`. The supplement descriptors of every hazard image are reported only. A realization's hazard characteristics are its row of `hazard_image.npz`.

**Target hazard characteristics.** The Latin hypercube sample of `scengen.subsample.lhs_nn_assignment`, seeded by `ScenarioDesign.selector_seed(draw)`; the pairing of each target with its nearest unused candidate is the selection step 03 stages.

**Target displacement.** The target-to-member distances of that pairing, per target in `hfm_points.csv` and summarized in `hfm_summary.csv` (`hf_design_metrics.md`); the selector-diagnostic columns keep the name `snap`.

**Largest drought event.** The event `scengen.hazard_metrics` scores per window, the qualifying SSI-6 run with the largest accumulated deficit.

**Effective sample size.** Of the nearest-member weights only: `ess` and `ess_over_n` in `hfm_summary.csv` (`hf_design_metrics_run.effective_sample_size`) and `ess_over_n` in the selector diagnostic. The serial-dependence statistic of the sizing diagnostic is the **effective number of independent annual units**: `n_eff_ratio` in `n_eff.csv` (`src/ensemble_size_stats.n_eff_ratio`).

## Objectives and robustness

**Per-SOW objective value** $J_i(x,\theta)$. Each SOW's realizations' unit-years pooled through the objective's own unit operator: `reeval_core.sow_objective_matrix`, persisted as `reeval_raw.csv.gz` (`solution_id`, `sow_id`, `objective`, `value`). Every robustness and regret column of `src/robustness.py` is a transformation of it.

**Current FFMP policy.** `get_baseline_values("ffmp")`; its step-05 re-evaluation cube sits under `baseline/` beside each run. Identifiers keep `incumbent` (`incumbent_advantage`, `incumbent_spread`, `include_incumbent`).

**Regret** (against the current FFMP policy in the same SOW). Columns `regret_mean__`, `regret_q90__`, `regret_cond__` and `gain_mean__` (`robustness.regret_magnitudes`), in natural units; the **regret frequencies** `harm_freq__` and `party_harm_freq__` and the **low-regret frequency** $\Pi_\tau$, `no_harm_freq_tau` (`robustness.regret_frequencies`), with $\tau_i = k \cdot \max(\epsilon_i, \tau_i^{\mathrm{floor}})$ pinned as `NYCOPT_REGRET_TAU` in the production env files.

**Satisficing robustness.** `sat_multivariate_sow` (every criterion of the set jointly) and `sat_uni_sow__`; criterion sets in `src/satisficing_criteria.py` (`CriterionSet`, variant `DEFAULT_CRITERIA_VARIANT`, all-axes reference `reference_all8`).

**Laplace and maximin.** `laplace__` and `maximin__`, the mean and the worst per-SOW value.

**Epsilon-dominance precision.** `config.get_epsilons()`, the annual-unit ε vector pinned in the env files.

## Decision variables

**Allocation reduction.** A diversion decision variable (`{nyc,nj}_allocation_reduction_*`): the additional fractional reduction of the party's Decree allocation applied on entry to a drought stage. Stage-wise increments, not absolute factors; the effective delivery factor at a stage is 1 minus the running sum of reductions, so monotone curtailment across stages holds by construction.

**Delivery factor.** The absolute multiplier on the Decree allocation that the Pywr-DRB model consumes per drought level (model parameters `{level}_factor_delivery_{nyc,nj}`). A decoded quantity, never a decision variable: the simulation wrapper converts allocation reductions to delivery factors before handoff.

## Style rules

1. All `_pct` quantities are 0-1 fractions (repo-wide rule).
2. Sequence length is stated in years, and window construction (disjoint windows, initialization of storages, handling of drought events cut by window edges) is specified wherever realizations are introduced.
3. Parallel computing language is nodes, cores, islands and service units; ranks, slugs and environment variables stay out of manuscript text.
