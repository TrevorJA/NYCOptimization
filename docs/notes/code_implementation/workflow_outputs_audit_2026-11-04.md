# Workflow and output organization audit (2026-11-04)

Status: recommendations awaiting approval. Nothing was moved, renamed, or deleted
in this pass. Three read-only passes were run in parallel on Anvil (main workflow
layer; supplemental layer; output and figure trees), followed by a verification
of every concrete defect reported. Approved items are executed in the phases of
Section 7; the move maps in Sections 4 and 5 are the execution specification.

Guiding questions: (1) is the main and supplemental workflow layout clear, are
steps clearly ordered, and is every workflow a self-contained script rather than
a long command line; (2) is the output hierarchy clear, well labelled and
consistent.

## 1. Cross-cutting findings

1. **Step numbers do not encode order.** Step 12 (E_test generation) is a
   prerequisite of 05, 08, 09, 10, 11 and 13, yet sits last; 09b is the only
   lettered step and is a mandatory half of the campaign path; 08 is the
   non-campaign alternative to 09 + 09b but precedes them; 11 is documented as
   exploratory and outside the manuscript plan. Two chain steps have no wrapper
   and no number (`extract_runtime_archive.py` and
   `validate_staged_seasonality.py` are run as bare Python).
2. **Long command lines persist on the campaign path.** Step 09 needs five
   sbatch flags plus four environment overrides (the longest line in the repo,
   `README.md:281-283`, `TODO.md:62`); 03 and 04 pass the design and the pool
   size by `--export` although production env files exist; 05 takes a positional
   `--search-ensemble`; the E_test completion chain is five jobs wired by hand;
   `NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch` is retyped on
   six steps per design. Only step 06 has a submit helper (added today).
3. **Three naming grammars coexist** in `workflow/envs/` (slug grammar with
   abbreviations `mc` and `hazfill_stat`; experiment-name grammar; purpose
   grammar) and in `workflow/supplemental/` (46 scripts, flat, at least six
   prefix idioms, 15 scripts with the `#SBATCH` block before the purpose
   comment). The README is the only way to learn which scripts form a chain and
   which are closed.
4. **Documentation drift.** The step-09 geometry is documented three different
   ways (script header: 8 x 128 dense on `wholenode`; both READMEs, the envs
   README and `campaign_design.md`: `shared` 16 x 8; TODO: open item). The
   replication sequence is narrated in three places (root README, workflow
   README, TODO) that disagree on the already-staged E_test path and on the
   required-env step list.
5. **Output tree: four figure roots and two meanings of "diagnostics".** Code
   writes non-manuscript figures to `figures/{scenario}/{slug}` (what actually
   ran), `outputs/figures/_exploratory` (`config.FIGURES_DIR`),
   `outputs/supplemental/{exp}/figures`, and `figures/{manuscript,si}`.
   `outputs/diagnostics/` holds two pipeline-input caches, while `config.py`
   reserves "diagnostics" for a per-run artifact. The historic baseline is
   special-cased flat; the design-independent incumbent cube is filed inside one
   design's search tree and addressed by literal path.
6. **Stale material in live locations.** 27 run manifests from July and August
   whose searches are archived; `outputs/profiling/`;
   `outputs/supplemental/hazfill_d0_restore_check/` (44 MB presim copy);
   `anvil_scaling_experiment/packing/stale_pre_merge_7cdf527/`; a hand-placed
   spreadsheet; 14 `first10ch` and `smoke_*` files in
   `hazard_support_decomposition`; three unreferenced `_moderate` env files and
   `smoke_hazfill.env`; three Python scripts referenced nowhere.

## 2. Defects to fix regardless of the layout decisions

All verified on disk on 2026-11-04. None affects a running job.

| # | Defect | Location | Fix |
|---|---|---|---|
| F1 | `_common.sh` sourced at the wrong depth (`/..` instead of `/../..`); masked by `SLURM_SUBMIT_DIR` when submitted from the repo root, fails on local runs | `workflow/supplemental/epsilon_ensemble_refilter.sh:22`, `epsilon_refilter_sweep.sh:21` | one-token edit each |
| F2 | TODO names a script that does not exist | `TODO.md:380` (`transfer_evaluation_eval.sh`; the file is `transfer_evaluation_eval_array.sh`) | text |
| F3 | Two login-node submit helpers import project Python on the login node (`import config`, `supplemental_config`) | `anvil_scaling_borg_submit.sh:28,42`, `ensemble_cost_stage_submit.sh:32` | rewrite as pure bash reading a committed table, or move the resolution into the job's pre-flight (both experiments are closed; hygiene only) |
| F4 | pytest executed on Anvil inside a job | `nestedp_smoke_calibrate.sh:33,39` | remove the two pytest blocks or mark the script local-only |
| F5 | Stale script name in code comments | `src/simulation.py:896`, `scripts/main/run_baseline.py:55` (`00_generate_presim.sh`; the step is 01) | text |
| F6 | Non-standard scratch root | `predicted_inflow_bitcheck.py:109` writes `outputs/tmp_bitcheck/` | route to `outputs/supplemental/predicted_inflow_bitcheck/` |
| F7 | Stale headers that send a reader to archived inputs | `hazard_examples.sh:21-22` and `hf_design_metrics.sh` (name `hazfill_stat_abs_10yr_n300_d{0,1}`, archived 2026-10-01), `robustness_threshold_diagnostics.sh` (archived `ffmp_obj8_mm_full` baseline path), `staged_hazard_image.sh:9` ("8-axis", retired), `07_run_diagnostics.sh:19-21` (default targets include the VR slugs that do not exist; README says active slug only), `_common.sh:21-23` (required-env list omits 09b and 11), `05_run_baseline.sh` header (no `--search-ensemble`, example env is the pre-campaign file) | text |
| F8 | Fixed log names overwrite on rerun | `01_generate_presim.sh:15` (`logs/presim.out`), `05_run_baseline.sh:18` (`logs/baseline.out`) | add `%j` |
| F9 | Experiment-name leak: the production pool slug variable is `NYCOPT_NESTEDP_POOL_SLUG` | `gen_pool_merge.sh:36`, `pool_verify.sh:44`, `nestedp_ladder.sh` | rename to `NYCOPT_POOL_SLUG` with a default |
| F10 | Dead or misleading constants | `config.OUTPUT_REFERENCE_SETS_DIR` (config.py:148; fallback only in `src/reevaluate.py:81`, `reevaluate_mpi.py:84`), `supplemental_config.TEV_UNITS_ROOT` default points at an archived projects-space path (:1976), `config.py:959` docstring says `figures/{scenario}`, `config.py:150-151` comment on "diagnostics", `ANVIL_NOTES.md:70` records a symlink target that no longer exists | code and text |

## 3. Main workflow layer

### 3.1 Submission ergonomics (what must be typed beyond `sbatch --export=ALL,NYCOPT_ENV_FILE=<env> workflow/NN.sh`)

| Step | Extra sbatch flags | Overrides beyond the env file | Helper |
|---|---|---|---|
| 02 | `--array=0-1` | none | no |
| 03 | none | `NYCOPT_SCENARIO_DESIGN`, `NYCOPT_CANDIDATE_POOL_N` (no env file used) | no |
| 04 | `--array=0-1` | `NYCOPT_SCENARIO_DESIGN` (no env file used) | no |
| 05 | positional `--search-ensemble` for the ensemble designs | `NYCOPT_REEVAL_ENSEMBLE_PRESET`, `NYCOPT_BASELINE_SKIP_REEVAL` | no |
| 06 | nodes, ntasks-per-node, time, array | `DRAW` | yes (`submit_production_search.sh`) |
| 08 | nodes, ntasks-per-node | `NYCOPT_REEVAL_ENSEMBLE_PRESET`, `NYCOPT_REEVAL_SCORE` | no |
| 09 | partition, nodes, ntasks, cpus-per-task, time | `NYCOPT_REEVAL_ENSEMBLE_PRESET`, `NYCOPT_CHUNK_POLICIES`, `NYCOPT_CHUNK_MERGE`, `NYCOPT_SEARCH_REALIZATION_BATCH` | no |
| 09b | none | `NYCOPT_REEVAL_ENSEMBLE_PRESET`, `NYCOPT_CHUNK_POLICIES` (must equal the 09 values) | no |
| 10, 11 | none | `NYCOPT_REEVAL_ENSEMBLE_PRESET` | no |
| 12 | none | `NYCOPT_ETEST_VARIANT` | no |

Recommended helpers, all on the `submit_production_search.sh` pattern (pure bash,
positional identifiers, `--dry-run`, pass-through flags):

- `submit_reeval_chunks.sh <design> [--merge]`: step 09 with the adopted
  geometry, the preset, the derived `NYCOPT_CHUNK_POLICIES` path and the batch,
  then 09b with the same identity. Written only after the step-09 geometry item
  in TODO is settled by the one-node smoke, so that one record exists.
- `submit_search_staging.sh <design>`: 02 (array over K draws), 03 where the
  design has a pool, 04 (array), chained with `--dependency` and `--parsable`.
- `submit_etest_staging.sh`: the completion chain (demand chunks, subset stage,
  hazard image shards and merge, incumbent cube), chained the same way.
- Wrappers for the two bare-Python chain steps: `extract_runtime_archive.py
  --seed N` and `--merge --install` (one wrapper, mode by argument) and
  `validate_staged_seasonality.py`.

Env-file keys rather than overrides: pin `NYCOPT_CANDIDATE_POOL_N=1000000` in
`ffmp_obj8_hazfill_stat_production.env` (as `ensemble_size_diagnostics.env:14`
already does) and let 03 and 04 take `NYCOPT_ENV_FILE` instead of
`--export=NYCOPT_SCENARIO_DESIGN=`.

### 3.2 Step renumbering (decision D1)

| Now | Proposed | Reason |
|---|---|---|
| 00 setup_borg_jars | 00 | unchanged |
| 01 generate_presim | 01 | unchanged |
| 02, 03, 04 (search-ensemble staging) | 02, 03, 04 | unchanged |
| 12 generate_test_ensemble | 05 | E_test precedes everything downstream |
| 05 run_baseline | 06 | needs E_test |
| 06 run_mmborg | 07 | |
| (new) extract_reference_set | 08 | currently bare Python |
| 07 run_diagnostics | 09 | |
| 09 simulate_test_chunks | 10 | campaign path |
| 09b merge_test_chunks | 11 | mandatory; promoted from the letter suffix |
| 10 compare_designs | 12 | |
| 13 main_figures, 14 results_figures | 13, 14 | unchanged |
| 08 reevaluate (unchunked) | `supplemental/reevaluate_unchunked.sh` | documented non-campaign alternative |
| 11 scenario_discovery | `supplemental/scenario_discovery.sh` | exploratory, outside the manuscript plan |

Reference load: step names appear in `src/` (simulation, runtime_archive,
diagnostics, chunk_reeval, transfer_eval, two plotting modules), `config.py:691,830`,
eight `scripts/main/` files, three `scripts/supplemental/` files, both READMEs, the
envs README, every env-file header, the three `submit_*.sh`, nine supplemental
wrappers that name 02 to 04 or 09, `tests/test_ensemble_simulation.py:380`, and
`#SBATCH --output` log stems. About 40 edits, all prose or log names; the only
executable coupling is `submit_*.sh` to the step-06 launcher. `--dependency`
chains carry job ids, not names. A pending sbatch job holds its own copy of the
script, so renaming while jobs are queued is safe.

Minimal alternative if a full renumber is declined: move 12 to 05 and promote
09b; leave the rest.

### 3.3 Env files (decisions D3 and D8)

| File | References outside `envs/` | Status |
|---|---|---|
| `ffmp_obj8_{historic,mc,hazfill_stat}_production.env` | many | live campaign identities |
| `ffmp_obj8_historic.env` (`mm_full`, 165 ranks) | 12 scripts use it as the default or header example (00, 05, 08, nine supplemental E_test and simulation scripts) | de facto "historic identity, no MOEA" env; its own header calls it pre-campaign |
| `ffmp_obj8_{historic,mc,hazfill_stat}_moderate.env` | none | pre-campaign; outputs archived |
| `smoke_hazfill.env` | none | only its own `--wrap` line |
| `ffmp_vr_obj8.env` | 06 header, both READMEs | conditional SI extension |
| `eps_calib_*`, `ensemble_size_diagnostics`, `ensemble_cost`, `anvil_scaling_*`, `transfer_evaluation` | their wrappers | live for closed or pending experiments; `transfer_evaluation.env` pins N = 100 sets that are now archived |

Recommendations: (a) rename the two abbreviated production files to the design
names (`ffmp_obj8_monte_carlo_production.env`,
`ffmp_obj8_hazard_filling_stationary_production.env`), which the `eps_calib_*`
files already use, so `submit_production_search.sh` needs no design-to-file map;
or keep the abbreviations and state the table once in the envs README. (b) Move
the four unreferenced files to `workflow/envs/retired/` with a one-line README.
(c) Either rename `ffmp_obj8_historic.env` to say what it is for
(`ffmp_obj8_historic_identity.env`, no MOEA config) or switch the 12 defaults to
the production file, whose non-MOEA keys are identical. (d) Add
`transfer_evaluation.env` to the envs README inventory.

### 3.4 `_common.sh`

Five steps (02, 03, 12, 13, 14) carry the same inline three-line full-thread
block instead of a helper; add `nycopt_full_threads`. The header omits
`nycopt_run_figure_studies`. The Hopper branches (`python/3.11.5` module, TCP MCA
flags) are dead on Anvil; isolate behind one `NYCOPT_CLUSTER` switch. Steps 05,
09 and 09b produce results but write no manifest; add a light manifest write.
Step 11 names the draw variable differently from 06 to 09b.

### 3.5 Documentation

Keep one replication narrative (workflow README) and make the root README and the
TODO point at it. Record the step-09 geometry in exactly one place once settled.
Add the already-staged E_test completion path to wherever the fresh-build path
lives. Add a status column (live, closed, optional, superseded) to the workflow
README's supplemental list and the seven scripts missing from it
(`hazard_axis_selection`, `hazard_selector_diagnostics`, `hf_design_metrics`,
`staged_hazard_image`, three `transfer_evaluation_*`). Docs under
`docs/notes/methods/` cite about 20 `outputs/supplemental/*` paths that are now
archive-only; a one-line "archived 2026-11-04" note per doc stops readers
searching for them.

## 4. Supplemental layer

### 4.1 Status summary (46 scripts)

- Live on the campaign path: the E_test chain (`etest_demand_utilization`,
  `prep_etest_chunks`, `stage_nyc_demand_etest_chunks`, `etest_subset_stage`,
  `etest_hazard_image_shards`, `etest_hazard_image_merge`,
  `stage_etest_subset_baseline`), `stage_nyc_demand`, `staged_hazard_image`,
  `epsilon_ensemble_refilter`, `dry_envelope` (production leg pending),
  `hazard_axis_selection` (Anvil confirmation pending), `hf_design_metrics`
  (rerun pending), `regret_tolerance_diagnostics` and
  `robustness_threshold_diagnostics` (reruns pending on the new cube),
  `si_figures_design`, `si_figures_results`, `sim_selected_policies`.
- Replication path, done: `gen_pool_shards`, `gen_pool_merge`, `pool_verify`,
  `gen_etest_shards`, `gen_etest_merge`.
- Closed experiments, results kept or archived: `anvil_scaling_*`,
  `ensemble_cost_*`, `ensemble_size_*`, `hazard_selector_diagnostics`,
  `hazard_support_decomposition`, `epsilon_calibration`, `satisfaction_factor`,
  `objective_sensitivity`, `transfer_evaluation_*`, `predicted_inflow_bitcheck`,
  `etest_subset_reeval_check`.
- Superseded: `epsilon_refilter_sweep` (by `epsilon_ensemble_refilter`).
- Optional, not run: `nestedp_smoke_calibrate`, `nestedp_ladder`.
- Tied to a dropped experiment: none. The retired 8-axis scoring survives only as
  wording and as the inputs `hazard_support_decomposition`, `dry_envelope`
  production and `hazard_examples` read until the images are rescored.

Scripts that cannot be submitted bare today (2 to 5 required `--export`
variables, or positional arguments after the script path): `gen_pool_shards`,
`gen_pool_merge`, `pool_verify`, `nestedp_ladder`, `hazard_selector_diagnostics`,
`hazard_axis_selection`, `stage_etest_subset_baseline`,
`ensemble_size_library_stage` (array sized by a Python one-liner),
`ensemble_size_library_eval` (smoke needs four flags), `ensemble_cost_sweep`,
`epsilon_ensemble_refilter`, `epsilon_refilter_sweep`, `sim_selected_policies`,
`stage_nyc_demand --preset`. Give each committed defaults
(`: "${NYCOPT_CANDIDATE_POOL_N:=1000000}"`) so a bare `sbatch` reproduces the
production run, and keep overrides for variants.

### 4.2 Python pairing

Nineteen `scripts/supplemental/*.py` have no wrapper. Fifteen are documented as
local runs (framing_convention_analysis, flood_objective_run, objective_dynamics
and its figures, ensemble_objective_dynamics and its figures,
criteria_reanchoring, etest_realization_precision, the two regret_tolerance_passb
scripts, check_objective_determinism, validate_staged_seasonality,
write_refiltered_sets, the persistence trio). Three are referenced nowhere
(`policy_dynamics_figures.py`, `policy_rules_compact_figures.py`,
`simulation_api_check.py`). Under `scripts/main/`, `compute_etest_hazard_image.py`
and `plot_etest_hazard_overlay.py` are invoked only by supplemental wrappers;
`compute_historic_hazard_windows.py`, `explore_results.py`, `plot_run_results.py`
have no wrapper and no doc reference; `factor_mapping_run.py` has one doc
reference. Recommendation: tag each local-only script `[local only]` in its
docstring and the README; move the two E_test hazard-image scripts to
`scripts/supplemental/`; leave the three unreferenced scripts in place (no
deletion in this pass) but list them in the README as unreferenced.

### 4.3 Proposed layout (decision D2): subfolders by purpose

| Old (`workflow/supplemental/`) | New | Rename |
|---|---|---|
| `gen_pool_shards.sh`, `gen_pool_merge.sh`, `pool_verify.sh` | `staging/pool_shards.sh`, `staging/pool_merge.sh`, `staging/pool_verify.sh` | drop `gen_` so the three sort together |
| `gen_etest_shards.sh`, `gen_etest_merge.sh` | `staging/etest_shards.sh`, `staging/etest_merge.sh` | drop `gen_` |
| `etest_demand_utilization.sh`, `prep_etest_chunks.sh`, `stage_nyc_demand_etest_chunks.sh`, `etest_subset_stage.sh`, `etest_hazard_image_shards.sh`, `etest_hazard_image_merge.sh`, `stage_etest_subset_baseline.sh` | `staging/etest_demand_utilization.sh`, `staging/etest_prep_chunks.sh`, `staging/etest_stage_nyc_demand.sh`, `staging/etest_subset_stage.sh`, `staging/etest_hazard_image_shards.sh`, `staging/etest_hazard_image_merge.sh`, `staging/etest_subset_baseline.sh` | `etest_` first on all seven, so `ls staging/etest_*` lists the chain |
| `stage_nyc_demand.sh`, `staged_hazard_image.sh` | `staging/ensemble_stage_nyc_demand.sh`, `staging/ensemble_hazard_image.sh` | `ensemble_` = acts on a staged search ensemble |
| `hazard_axis_selection`, `hazard_selector_diagnostics`, `hf_design_metrics`, `hazard_support_decomposition`, `hazard_examples`, `dry_envelope`, `nestedp_smoke_calibrate`, `nestedp_ladder`, `ensemble_size_library_stage`, `ensemble_size_library_eval`, `ensemble_size_hazard`, `ensemble_size_analysis` | `design_diagnostics/<same>` | none |
| `epsilon_calibration`, `epsilon_ensemble_refilter`, `epsilon_refilter_sweep`, `satisfaction_factor`, `objective_sensitivity`, `regret_tolerance_diagnostics`, `robustness_threshold_diagnostics`, `transfer_evaluation_check`, `transfer_evaluation_eval_array`, `transfer_evaluation_analysis` | `results_diagnostics/<same>` | none |
| `anvil_scaling_packing`, `anvil_scaling_borg`, `anvil_scaling_borg_submit`, `ensemble_cost_sweep`, `ensemble_cost_stage_submit` | `benchmarks/<same>` | none |
| `predicted_inflow_bitcheck`, `etest_subset_reeval_check` | `checks/<same>` | none |
| `si_figures_design`, `si_figures_results`, `sim_selected_policies` | `figures/<same>` | none |
| (from the main layer, if D1 is approved) `08_reevaluate.sh`, `11_scenario_discovery.sh` | `results_diagnostics/reevaluate_unchunked.sh`, `results_diagnostics/scenario_discovery.sh` | |

Edits each move requires:

1. In every moved job script the `_common.sh` line becomes
   `.../dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/workflow/_common.sh`;
   the two helpers' `cd "$(dirname ...)/../.."` likewise.
2. Each script's own header "Submit:" line and `#SBATCH --output` stem.
3. The eight cross-script path references: `02_generate_ensemble.sh:16`,
   `12_generate_test_ensemble.sh:8`, `anvil_scaling_borg.sh:9`,
   `anvil_scaling_borg_submit.sh:66`, `ensemble_cost_sweep.sh:91`,
   `si_figures_design.sh:19,27`, `si_figures_results.sh:29`.
4. Docs by name: `README.md`, `workflow/README.md`, `workflow/envs/README.md`,
   `TODO.md`, `docs/notes/methods/*.md`, `ANVIL_NOTES.md`, and five one-line
   comments in `supplemental_config.py`. `tests/` references no script name;
   `tests/test_terminology.py:214-222` globs `workflow/**/*.sh`, so subfolders
   are linted automatically.
5. The `bash -n` glob in the workflow README becomes `workflow/supplemental/*/*.sh`.

Alternative (b), a prefix scheme on the flat directory, needs the same doc edits,
renames every file, and still leaves 46 files in one listing; not recommended.
Alternative (c), leave flat and add a status table to the README, is the
no-move fallback.

Also: normalise header layout (purpose comment first, then `#SBATCH`) in the 15
SBATCH-first scripts; give `satisfaction_factor.sh` its own env file instead of
borrowing `eps_calib_*.env`; add a table of contents to `supplemental_config.py`
mapping section to wrapper to output root, ordered to match the README groups.

## 5. Output organization

### 5.1 Current tree (condensed; C = campaign-current, R = retained, S = stale)

```
outputs/
  presim/                                  C   step 01
  synthetic_ensembles/
    etest_kn_50yr_n25000[__chunk000-049] -> projects space   C   (jobs 21950437, 21950524 writing now)
    fixprob_10yr_n100_d{0,1,2}, hazfill_stat_abs_10yr_n100_d{0,1,2}   C (d2 unused)
    fixprob_10yr_n300_d0                   R   step-02 files only
    statpool_10yr_n1000000_d{0,1}          C   P = 1e6 pools;  statpool_10yr_n2000_d{0,1}  R smoke
    kn_50yr_n5                             R   local-fixture mirror
    esd_lib_10yr_p1000000_d0[__chunk000-008] -> projects space   R
  diagnostics/{demand_utilization, forcing_parameterization}   C/R   two pipeline-input caches, no config constant
  comparison/ffmp_obj8/historic_single/figure_tables/demand_utilization/   C   tag misleading (S12 needs no cube)
  run_manifests/  27 flat dirs, all July/August, searches archived            S
  profiling/baseline_x4.ref                                                   S   unreferenced
  supplemental/
    anvil_scaling_experiment/{borg,tables,manifests,packing}   R;  packing/stale_pre_merge_7cdf527/  S;  HPC_cost_calculator.xlsx  S
    ensemble_cost_experiment/{cells,tables,manifests}          R
    ensemble_size_diagnostics/{figures,tables}                 R   Layer A
    hazard_selector_diagnostics/statpool_10yr_n1000000_d{0,1}/ R
    hazard_support_decomposition/{figures,tables}              R;  14 first10ch/smoke files  S
    hazfill_d0_restore_check/                                  S   44 MB presim copy, unreferenced
    historic_hazard_windows/, cv_axis_footprint/, persistence_axis/   R   flat, script-local roots
figures/manuscript (tracked), figures/si (S12 only)
logs/  614 files; 400 are October pool-shard logs
projects space: archive/ (6.9 GB) and synthetic_ensembles/ (333 GB); every entry symlinked at the same relative path; no orphans
```

Defined in code but absent: `outputs/reference_sets/`, `outputs/figures/`,
`outputs/baseline/`, 17 experiment roots (archived or local-only).

### 5.2 Proposed target hierarchy

```
outputs/
  reference/            pipeline-wide derived inputs: presim/, forcing_fdc/, demand_utilization/
                        (today: presim/ and diagnostics/; presim moves only after the queued jobs and the HIST search finish)
  synthetic_ensembles/  unchanged, plus README.md index (slug family -> design, purpose, location, writer)
  baseline/             {scenario}/{f}_baseline_objectives.csv (historic no longer special-cased)
                        + reeval/{preset}/ for the design-independent incumbent cubes
  {scenario}/{slug}/{sets, runtime, runtime_coordinator, metrics, reeval/{preset}, timeseries/{tag}, diagnostics}   unchanged
  comparison/{slug}/{tag}/figure_tables/{kind}   unchanged; cube-less design-support tables under comparison/{slug}/design/
  figures/              the one root for every non-manuscript figure: {scenario}/{slug}/{kind} and _exploratory/
  supplemental/{exp}/{tables, figures, manifests, ...}   unchanged; the three flat experiments gain tables/
  run_manifests/{scenario}/{slug}/seed{N}_{jobid}/
figures/{manuscript, si}   unchanged (tracked)
archive/<date>_<reason>/   for everything retired; nothing deleted
```

Ensemble slugs (`fixprob_*`, `hazfill_stat_abs_*`, `statpool_*`, `etest_kn_*`,
`esd_lib_*`) and the flat `slug__chunkNNN` layout are a stable contract
(`chunk_index.json`, `_meta.json`, every manifest, `ensemble_generation.py:885,1164`).
Renaming or nesting them is high risk and not recommended; an index file is the
fix.

### 5.3 Move table (decisions D4 to D7)

| # | Change | Code | Docs naming the old path | Risk |
|---|---|---|---|---|
| O1 | One figure root: point `plot_run_results.py:341`, `src/plotting/search_diagnostics.py:227`, `policy_dynamics_figures.py:63`, `simulate_selected_policies.py:42` at `config.figure_dir_for` | those four files; `.gitignore:21` comment | `README.md:326-332`, `TODO.md:388`, `archive/README.md` | low (nothing writes there now) |
| O2 | `outputs/diagnostics/` to `outputs/reference/` (`forcing_fdc/`, `demand_utilization/`); `presim/` joins later | `src/demand_utilization.py:70`, `forcing_fdc_cache.py:49`, `figures_forcing_parameterization.py:59`, `etest_demand_utilization.sh` | workflow README, TODO, `campaign_design.md`, `forcing_parameterization.md` | low; presim must not move while 21950437, 21950524 and the HIST search run |
| O3 | Incumbent cube to `outputs/baseline/reeval/{preset}/`; drop the historic special case in `baseline_objectives_csv` (`config.py:174-202`) | `config.py`, `stage_etest_subset_baseline.py:38`, `supplemental_config.RTD_REEVAL_BASELINE_DIR`, step-09 `baseline` policy routing in `simulate_test_chunks.py`, `run_baseline.py` | TODO rerun chain items 3 and 4, `robustness_threshold_diagnostics.md`, `ANVIL_NOTES.md`, `archive/README.md` | low now (items 3 and 4 not yet run), medium once cubes exist |
| O4 | `run_manifests/{scenario}/{slug}/seed{N}_{jobid}/`; move the 27 existing dirs to `archive/2026-11-04_pre_demand_update/outputs/run_manifests/` | `workflow/_common.sh:296-297` only | `README.md:251`, `workflow/README.md:23`, envs README, TODO, `archive/README.md` | low; do before the HIST submit |
| O5 | Stale items to `archive/2026-11-05_output_audit/`: `profiling/`, `supplemental/hazfill_d0_restore_check/`, `anvil_scaling_experiment/packing/stale_pre_merge_7cdf527/`, `HPC_cost_calculator.xlsx` (or to `docs/`), the 14 `first10ch` and `smoke_*` hazard_support files, the two non-pattern logs | none | `archive/README.md` (new section) | low |
| O6 | Route cube-less figure specs (no `cube` or `scorecard` in `needs`) to `comparison/{slug}/design/figure_tables/{kind}`; move today's two CSVs | `src/figures/registry.py:90-94` | none | low |
| O7 | Bring the three flat supplemental writers into `supplemental_config` with `tables/` (`cv_axis_footprint`, `persistence_axis`, `historic_hazard_windows`) and the `epsilon_refilter` literals into an `EPSREF_OUTPUT_ROOT` | six scripts (`diagnose_cv_axis_footprint.py:73`, `diagnose_persistence_axis.py:68`, `check_cmip6_rho1_anchor_noise.py:69`, `verify_kirsch_baseline_persistence.py:51`, `compute_historic_hazard_windows.py:59`, `epsilon_ensemble_refilter.py:470`, `epsilon_refilter_sweep.py:286`) | `persistence_axis_diagnostics.md`, `forcing_parameterization.md`, `scenario_design_methods.md`, `hf_design_metrics.md` | low |
| O8 | `logs/`: one stem rule (`{script_stem}_%j.out`, `_%A_%a.out` for arrays; fix 01, 05 and the `%x` stem of 06) and a pruning rule in `ANVIL_NOTES.md` (tar logs older than the last archive event into `archive/<date>/logs/`) | `#SBATCH` headers of 01, 05, 06 | workflow README | low |
| O9 | `outputs/synthetic_ensembles/README.md` index | new file | none | none |

## 6. Decisions requested

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Renumber the main steps per 3.2 (full), minimal (12 to 05, promote 09b), or leave | full renumber, now, before campaign outputs and logs accumulate under the old names |
| D2 | Supplemental layout: subfolders by purpose (4.3), prefix scheme, or flat with a status table | subfolders |
| D3 | Env naming: full design names, or keep `mc` and `hazfill_stat` with a stated table | full design names |
| D4 | `outputs/diagnostics/` to `outputs/reference/` (with presim joining later) | yes |
| D5 | Incumbent cube and historic baseline to `outputs/baseline/` (O3) | yes, before TODO rerun-chain items 3 and 4 run |
| D6 | `run_manifests` nested by scenario and slug, old manifests archived (O4) | yes, before the HIST submit |
| D7 | One figure root under `outputs/figures/` for every non-manuscript figure (O1) | yes |
| D8 | Retire-to-folder for unreferenced envs (`envs/retired/`) and stale outputs (O5) | yes; nothing deleted |
| D9 | Submit helpers for 09 + 09b, the staging chains, and wrappers for the two bare-Python chain steps (3.1) | yes; the 09 helper waits for the geometry smoke |

## 7. Execution phases and sequencing constraints

Constraints: jobs 21950437 (demand-chunk staging) and 21950524 (subset stage)
write under `outputs/synthetic_ensembles/etest_kn_50yr_n25000*` and read
`outputs/presim/`; the HIST search, once submitted, reads `presim/` and writes
`outputs/historic/ffmp_obj8/` and a run manifest. A queued sbatch job holds its
own copy of the script, so script moves do not affect it; path changes in
`config.py` or `_common.sh` do, so those wait for the affected jobs.

- **Phase 0, before the HIST submit (no decision needed except D6):** F1 to F10;
  O4 (manifest layout, so the first campaign manifest lands in the new layout);
  O8 log stems; O9 index. One commit.
- **Phase 1, text and new files, any time:** the three submit helpers and two
  wrappers (D9; the 09 helper after the geometry smoke); one replication
  narrative; supplemental README status column; archive notes in the methods
  docs; `_common.sh` cleanups (3.4).
- **Phase 2, structural moves, after the queued E_test jobs finish:** D1
  renumber, D2 subfolders, D3 env renames, `envs/retired/`, O1, O2 (without
  presim), O3, O5, O6, O7. One commit per layer (workflow, envs, outputs), each
  verified by `bash -n workflow/**/*.sh`, a grep for every old name, and a
  `--dry-run` of each submit helper. `presim/` moves under `reference/` only after
  the HIST search has finished.
- **Local tests after each phase** on Trevor's machine, never on Anvil.
