# workflow/ — Experiment pipeline

Everything submittable lives here: numbered pipeline steps (`00`–`14`),
shared setup functions (`_common.sh`), per-experiment run configs
(`envs/`), and off-pipeline staging, calibration, and diagnostic drivers
(`supplemental/`). Submit all jobs **from the repo root**. The top-level
[README](../README.md) walks the full replication sequence; this file documents
each step.

Note: `workflow/_common.sh` is unrelated to the repo-root `lib/` directory
(which holds the licensed Borg C sources).

## The env-file contract

A run's identity — **scenario design** (`NYCOPT_SCENARIO_DESIGN`,
`src/scenario_designs.py`) × **MOEA config** (`NYCOPT_MOEA_CONFIG`,
`src/moea_config.py`), plus objectives and physics toggles — comes from one
`KEY=VALUE` env file under [`envs/`](envs/README.md), forwarded via
`sbatch --export=ALL,NYCOPT_ENV_FILE=...`. Scripts pass only identifiers
(`FORMULATION`, `SEED`, `DRAW`); every value comes from the env file + config
registries, and `_common.sh` reads the resolved identity back from `config.py`
so shell and Python agree on a single source of truth. Machine-specific
settings are not identity and come from the cluster profile (below). Steps
`07`/`08`/`10`/`11` (and the supplemental re-evaluation and scenario-discovery
wrappers) require the env file explicitly; the others fall back to `config.py`
defaults.

Outputs land under `outputs/{scenario}/{moea_slug}/{artifact}/`; every MM-Borg
job writes a reproducibility manifest (config + env snapshots, git state) to
`outputs/run_manifests/{scenario}/{moea_slug}/seed{N}_{jobid}/`.

## Pipeline steps

| Step | Script | Allocation (Anvil) | Env file | What it does |
|------|--------|-----------|----------|--------------|
| 00 | `00_setup_borg_jars.sh` | login node (`bash`) | optional | Build one MOEAFramework problem JAR per formulation; rerun after changing the objective set |
| 01 | `01_generate_presim.sh` | `shared`, 1×1, 30 min | optional | Full Pywr-DRB run once; save non-NYC (STARFIT) releases for the trimmed model |
| 02 | `02_generate_ensemble.sh` | `shared`, 8 cpu, 4 h, `--array=0-(K-1)` | optional | Generate the active design's own realizations (or its small pool); array index = ensemble draw (two draws staged, `--array=0-1`: d0 is searched, d1 serves the SI draw-sensitivity re-evaluation). The P = 10⁶ campaign pool is built by `supplemental/staging/pool_shards.sh` → `staging/pool_merge.sh` → `pool_verify.sh` instead |
| 03 | `03_subsample_ensemble.sh` | `shared`, 8 cpu, 1 h | optional (or `NYCOPT_SCENARIO_DESIGN` via `--export`) + `NYCOPT_CANDIDATE_POOL_N=1000000` | Hazard-filling designs only: select N members from the design's own candidate pool, all K draws in one job; other designs generate directly in 02 and skip it |
| 04 | `04_prep_pywrdrb_inputs.sh` | `shared`, 1×33, 1 h, `--array=0-(K-1)` | optional | Format each draw's search ensemble into pywrdrb HDF5 inputs, including the per-realization NYC demand series (MPI across realizations); `--preset NAME` stages an arbitrary ensemble |
| 05 | `05_generate_test_ensemble.sh` | `shared`, 8 cpu, 12 h | optional | Build E_test serially: LHS over the FULL DU box of the mid-century CMIP6 runs (`config.ENSEMBLE_FORCING_PERIOD`) widened 25 % on every axis (annual-volume multiplier 0.899 to 1.28), × R realizations per SOW, chunked, hazard image streamed. `NYCOPT_ETEST_VARIANT=kn` (default) is the campaign's E_test; `hmm` is an opt-in generator sensitivity. The sharded build is `supplemental/staging/etest_shards.sh` → `staging/etest_merge.sh` |
| 06 | `06_run_baseline.sh` | `shared`, 1×1, 30 min | optional | Evaluate the default (unoptimized) FFMP policy, the objective anchor; `--search-ensemble` scores it scenario-matched on an ensemble design's own search ensemble. The incumbent's per-SOW E_test cube comes from step 10 with `NYCOPT_CHUNK_POLICIES=baseline` |
| 07 | `07_run_mmborg.sh` | `wholenode`; header 4×128 (`mm_moderate`); production passes `--nodes=12 --ntasks-per-node=128` (1,533 ranks) and a per-seed `--time` | **required** | MM-Borg MOEA search — ONE launcher for all formulations and scenario designs; `--array` = seed, `DRAW=k` = ensemble draw (default 0; the campaign searches d0 only); config-derived pre-flight refuses already-completed (design, draw, seed) cells |
| 08 | `08_extract_reference_set.sh` | `shared`, 4 cpu, 1 h | **required** | Equal-NFE reference set from the island runtime files: `--seed N` extracts one seed's coordinator-archive snapshot at the reporting NFE; `--merge --install` unions the seeds and installs `{slug}_merged.set`, the step-10 input |
| 09 | `09_run_diagnostics.sh` | `shared`, 8 cpu, 1 h (or `bash`) | optional | MOEAFramework runtime diagnostics on the per-seed coordinator archive (island runtime files merged snapshot by snapshot with the project's epsilon archive; six MOEAFramework indicators vs total NFE; diagnostics reference set `{slug}_diagnostics_reference.set`, never the campaign `{slug}_merged.set`); default target = the env file's active slug at `DRAW=k` (positional literal slugs override) |
| 10 | `10_simulate_test_chunks.sh` | via `submit_reeval_chunks.sh`: `wholenode`, 128 ranks per node, batch 25 (baseline cube: `shared`, 25 ranks × 2 cpus, 3 h) | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`, `NYCOPT_CHUNK_POLICIES`) | Chunked, metrics-only E_test re-evaluation (MPI chunk-and-aggregate, resumable); the campaign path, with `NYCOPT_CHUNK_MERGE=off`; `NYCOPT_CHUNK_POLICIES=baseline` simulates the incumbent cube |
| 11 | `11_merge_test_chunks.sh` | `shared`, 8 cpu, 128 G, 4 h | **required** (same identity as the 10 submission) | Merge the per-(solution, chunk) units into the re-eval cube + robustness scorecards |
| 12 | `12_compare_designs.sh` | `shared`, 4 cpu, 1 h (or `bash`) | optional (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Cross-design comparison of the re-evaluated Pareto sets (criterion sweep, scorecards, ranking stability) |
| 13 | `13_main_figures.sh` | `shared`, 4 cpu, 2 h | optional | Render the manuscript-tier figures from `src/figures/registry.py` |
| 14 | `14_results_figures.sh` | `shared`, 4 cpu, 30 min | optional | Render the SI-tier figures from the same registry |

Anvil notes: the allocation account is hardcoded in every script's header
(`#SBATCH --account=ees260021`); override with `sbatch -A <alloc>` if needed.
96 h is Anvil's `wholenode` per-job maximum. There is no resume: the runtime
files are diagnostic dumps, the Borg checkpoint is disabled, and every search
is sized to finish inside one job (`docs/notes/methods/campaign_design.md`).
`shared` bills per core; `wholenode` bills whole 128-core nodes.

Step order: `01` before `06`/`07`; `02`→`04` (`03` for hazard filling;
`historic` skips them) before `07`; `05` and the `supplemental/staging/etest_*`
chain before `10`; `07`→`08 --seed 1`→`08 --merge --install`→`09` (either
order after the seed-1 extraction)→`10`→`11`→`12`/`13`/`14`. Chain with
`sbatch --dependency=afterok:<jobid>`; the submit helpers below do so for the
staging chains, and the root README walks the sequence with the submit lines.

E_test is generated directly at the registry size (500 SOWs, 25 chunks) and
staged by one chain, `submit_etest_staging.sh all` (sharded generation, merge,
utilization column, per-chunk inputs, realized utilization, hazard image; the
root README lists the steps; `05` is the serial generation alternative). Every
re-evaluation step then takes
`NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500`, and the incumbent cube
must exist for the same tag as the policy-set runs: step `10` with
`NYCOPT_CHUNK_POLICIES=baseline` writes it once, for every design, to
`outputs/baseline/reeval/{tag}/` (under any design env), or the
incumbent-relative regret family is silently skipped.

NYC demand is the time-dynamic extrapolated diversion series in every
simulation; a constant 800 MGD demand is never used and no env file selects a
demand source. Step 04 stages the per-realization series
(`diversion_nyc_extrapolated_mgd.hdf5`); `staging/ensemble_stage_nyc_demand.sh`
adds it to a search ensemble staged before the series was part of the contract
(array index = draw; `--preset NAME` for any staged slug). In E_test each SOW's
series is scaled by `k(u)` of its sampled entitlement utilization
`u ∈ [0.45, 1.0]` before the 800 MGD monthly-mean cap
(`src/demand_utilization.py`; `docs/notes/methods/forcing_parameterization.md`,
"Demand-utilization axis"; the calibration table lives under
`outputs/reference/demand_utilization/`); search ensembles run the unscaled
series (`k = 1`). NJ demand is the constant 100 MGD Decree quantity.

## Building a design's search ensemble (02–04)

Every scenario design **generates its own realizations** from its own namespaced
seed stream (`src/scenario_designs.py`); no design is subsampled from a shared
master. Step 02 dispatches on `design.construction`, so what it builds — and
whether step 03 applies at all — follows from the design alone:

| construction | designs | campaign | 02 builds | 03 | 04 array |
|---|---|---|---|---|---|
| `preset` | `historic` | yes | nothing (static preset) | — | — |
| `direct_iid` | `monte_carlo` | yes | one N×L ensemble **per draw** | — | `0-(K-1)` |
| `hazard_fill` | `hazard_filling_stationary` | yes | one draw-invariant candidate pool + its hazard image | **yes** — all K draws in one job | `0-(K-1)` |
| `hazard_fill` | `hazard_filling_stationary_cdf`, `hazard_filling_du`, `hazard_filling_absolute` | no (retained) | as above | **yes** | `0-(K-1)` |
| `lhs_theta` | `input_stratified` | no (retained) | LHS over forcing params, realizations generated at each design point, **per draw** | — | `0-(K-1)` |
| `pool_resample` | `monte_carlo_resampled` | no (retained) | one draw-invariant pool (redrawn per evaluation in-search) | — | `0` |
| `stationary_kn` | `scaling_stationary` | no (supplemental) | direct Kirsch-Nowak stand-in | — | `0` |

The array index in `02`/`04` is the ensemble-draw index *k*; set `--array=0-(K-1)`
with K = `design.n_ensemble_draws` (= 2 for the matched designs). The campaign
searches draw 0 only; draw 1 is staged for the SI draw-sensitivity
re-evaluation of each design's final set. **Cost:** per-design construction
multiplies step-02 cost by K for `monte_carlo` and `input_stratified` —
each draw is a fresh N×L generation, not a re-index of shared data. Pool-owning
designs pay it once (array tasks k>0 are no-ops), and designs sharing a
population share one pool per draw (the two stationary hazard designs; the two
DU hazard designs). `NYCOPT_ENSEMBLE_FORCE=1` overwrites an already-staged slug.

## Optimization runs are independent jobs

Each optimization is one self-contained multi-day sbatch job — one submission
per (env file × seed), no campaign wrapper. `DRAW=k` selects the staged
ensemble draw (default 0) and the `--array` index is the Borg seed. The
campaign searches draw 0 with S = 2 seeds per design, submitted one seed at a
time (seed 1 runs 750k NFE, seed 2 500k; `docs/notes/methods/campaign_design.md`
§2). Every production search is submitted through `submit_production_search.sh`,
which records the 12 × 128 geometry and the per-seed wall request (matched
designs 96 h / 72 h; historic 12 h / 8 h) and issues the single sbatch line for
one (design, seed):

```bash
# Matched designs: seed 1 prices the campaign; submit seed 2 after it
bash workflow/submit_production_search.sh monte_carlo 1
bash workflow/submit_production_search.sh hazard_filling_stationary 1
# Historic reference (NFE-bounded, single trace)
bash workflow/submit_production_search.sh historic 1
bash workflow/submit_production_search.sh historic 2 --dependency=afterok:<seed-1 jobid>
# --dry-run prints the sbatch line; trailing arguments pass through to sbatch
# Variable-resolution FFMP (non-campaign SI extension, leftover allocation only):
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj8.env,FORMULATION=ffmp_12 \
       --array=1 workflow/07_run_mmborg.sh
```

What has already run is readable from the output tree alone: draw k > 0
appends `_d{k}` to the moea slug (draw 0 is the unsuffixed default), each
completed seed writes `outputs/{scenario}/{moea_slug}/sets/seed_{SS}_{slug}.set`,
and every job records its full identity (including the draw) in
`outputs/run_manifests/`. The step-07 pre-flight refuses to relaunch a
(design, slug, seed) whose `.set` file already exists — pass
`NYCOPT_OVERWRITE=1` to deliberately redo one. A crashed run leaves no `.set`
file and may simply be resubmitted.

## Cluster profiles (Anvil, Hopper)

Everything machine-specific lives in one profile per cluster,
`clusters/anvil.sh` and `clusters/hopper.sh`: partition names, allocation
account, MPI ranks per node, node memory, wall cap, the Python environment and
the OpenMPI transport flags. `cluster.sh` detects the cluster from Slurm's
cluster name and exports `NYCOPT_CLUSTER` plus the profile; `_common.sh`
sources it in every job and the submit helpers source it on the login node.
The cluster is not part of a run's identity, so no env file names it and the
same env file runs on either machine.

- **Anvil**: nothing to do. The `#SBATCH` headers are written for it.
- **Hopper**: run `source workflow/cluster.sh` once in the login shell, so the
  documented `sbatch` lines work unchanged. It exports `SBATCH_PARTITION=normal`
  (an sbatch input variable, which overrides the headers) and wraps `sbatch`
  in a shell function that prepends `NYCOPT_SBATCH_ARGS`, announcing it on
  stderr. Hopper has two node classes with different core counts and a job
  targets one: `epyc` (96 cores, the default) or `xeon` (40 cores), chosen
  with `NYCOPT_NODE_CLASS=xeon source workflow/cluster.sh`; the prepended
  argument is the `--exclude` of the other class. MPI steps whose header
  geometry exceeds the node (`--ntasks-per-node=128`) need
  `--nodes/--ntasks-per-node` on the sbatch line or go through a submit
  helper, which takes them from the profile. Both last for the shell session
  and apply to every sbatch typed in it (`command sbatch` bypasses the wrapper).
- **Searches** are sized by the MOEA config, so each cluster has its own:
  `production` (1,533 ranks) is Anvil's and `submit_production_search.sh`
  refuses to run elsewhere; `hopper_2x96` (189 ranks, env file
  `envs/ffmp_obj8_historic_hopper_2x96.env`) is the Hopper-sized historic
  search and writes under its own slug.
- **A new cluster** is a new `clusters/<slurm cluster name>.sh` defining the
  same variables.

A profile value already set in the environment wins (for example
`NYCOPT_RANKS_PER_NODE=33`). Measured timings and node caveats:
`ANVIL_NOTES.md`, `HOPPER_NOTES.md`.

## Geometry contract and scaling on Anvil

The MPI rank count has a single source per step; the `#SBATCH` lines are only
the container for it:

- **MM-Borg (07)**: ranks = `MOEAConfig.total_ntasks_mpi`
  (`1 + islands × (workers + 1)`, from `src/moea_config.py`). `_common.sh`
  reads it back for `mpirun -np`, and `nycopt_check_allocation` aborts before
  the search starts if the allocation is smaller — printing the exact
  `--nodes/--ntasks-per-node` to resubmit with — and warns when a whole
  node or more would sit idle. `nycopt_check_memory` aborts if the design's
  projected node RSS at this packing exceeds the safety line.
- **All other MPI steps (04, 10, supplemental)**: ranks =
  `SLURM_NTASKS`, the actual allocation — a mismatch is impossible by
  construction; rescale with `sbatch --nodes=N --ntasks-per-node=M` and the
  launch follows.

**Production geometry**: the `production` config is 1 controller + 4 islands ×
(382 workers + 1 master) = 1,533 ranks on 12 nodes
(`--nodes=12 --ntasks-per-node=128`), with
`NYCOPT_SEARCH_REALIZATION_BATCH=150` set in the matched designs' env files so
the N = 300 ensemble fits node memory (`docs/notes/methods/campaign_design.md` §3).
**To scale the search further**: register a larger MOEA config (more
islands/workers) in `src/moea_config.py`, point the env file's
`NYCOPT_MOEA_CONFIG` at it, and submit with
`--nodes=ceil(total_ntasks_mpi / 128) --ntasks-per-node=128` (128/node is the
measured Anvil `wholenode` packing — the node-packing sweep bounds the
eval-time penalty at ~17–21% and it is priced into the cost surface;
`NYCOPT_RANKS_PER_NODE` in the cluster profile). Anvil ceilings: `wholenode`
allows up to 16 nodes (2,048 cores) and 96 h. The `wide` queue reaches 56
nodes but only 12 h. Seeds (`--array`) and experiments (env files) scale
horizontally as fully independent jobs with no cross-job coordination. Shorter
pilots can pass `sbatch --time=...`.

## Submission helpers (login node; they issue sbatch, they are not jobs)

- `submit_production_search.sh <design> <seed> [--dry-run] [sbatch args]` — the
  one submission path for a campaign search (step 07, `production` config);
  design ∈ {historic, monte_carlo, hazard_filling_stationary} (hist/mc/hf),
  seed ∈ {1, 2}. Holds the 12 × 128 geometry and per-seed wall request so no
  production job is ever typed at the prompt.
- `submit_search_staging.sh <design> [--dry-run] [sbatch args]` — steps 02→04
  (`monte_carlo`) or 03→04 from the built pool (`hazard_filling_stationary`),
  arrays over the K = 2 staged draws, chained with `--dependency`.
- `submit_etest_staging.sh <build|hazard|all> [--from STEP] [--after JOBID]
  [--dry-run] [sbatch args]` — the E_test build chain under
  `supplemental/staging/`: sharded generation through the realized-utilization
  merge (`build`), the sub-window hazard image (`hazard`), or both (`all`).
- A helper for steps 10 + 11 follows once the re-evaluation geometry is settled
- `submit_reeval_chunks.sh <baseline|design> [--nodes N] [--no-merge | --merge-only]
  [--dry-run] [sbatch args]` — steps 10 + 11 at the measured geometry
  (`wholenode`, 128 ranks per node, batch 25, step 11 chained `afterok`; the
  one-policy baseline cube on `shared`, one rank per chunk × 2 cpus). Sizes
  the wall from the installed set and the chunk count; a truncated pass is
  finished with `--no-merge` passes, then `--merge-only`.
## Development utilities (not replication)

- `submit_smoke.sh` — one tiny-NFE end-to-end check per formulation
  (`bash workflow/submit_smoke.sh [--dry-run]`; Anvil `debug` queue, 2×40,
  2 h, `smoke` MOEA config + short 2018–2022 window via `envs/smoke.env`).
- `submit_search_memory_smoke.sh` — one-node memory/timing smoke of the batched
  search path (N = 300, realization batch 150, 128 ranks/node via
  `envs/smoke_search_batch.env`); the go criterion is in
  `docs/notes/methods/campaign_design.md` §4.

## `supplemental/` — off-pipeline drivers, by purpose

One subfolder per purpose. Settings live in the root `supplemental_config.py`,
env identities in `envs/`. Status: **live** (campaign path or a rerun pending),
**done** (replication path, outputs kept), **closed** (finished; results kept
or archived), **optional** (never run), **superseded**.

- **`staging/`** — replication path. `pool_shards.sh` → `pool_merge.sh` →
  `pool_verify.sh` (the P = 10⁶ candidate pool, sharded; done);
  `etest_shards.sh` → `etest_merge.sh` → `etest_demand_utilization.sh` →
  `etest_prep_chunks.sh` → `etest_realized_utilization.sh` →
  `etest_hazard_image_shards.sh` → `etest_hazard_image_merge.sh` (the E_test
  build chain, `submit_etest_staging.sh`; live); `etest_stage_nyc_demand.sh`
  (adds the scaled NYC demand series to chunks staged before it was part of
  the contract; not needed by a fresh build);
  `ensemble_stage_nyc_demand.sh` (adds the NYC demand series to a search
  ensemble staged before it was part of the contract; live);
  `ensemble_hazard_image.sh` (post-hoc hazard image of a staged MC ensemble; live).
- **`design_diagnostics/`** — hazard space and ensemble sizing, no policy
  simulation. `hazard_axis_selection.sh` (live; Anvil confirmation pending),
  `hazard_selector_diagnostics.sh` (closed), `hf_design_metrics.sh` (live; rerun
  on the N = 300 restage), `hazard_support_decomposition.sh` (closed),
  `hazard_examples.sh` (live; figure driver on the staged HF ensemble),
  `dry_envelope.sh` (live; production leg pending),
  `ensemble_size_library_stage.sh` → `ensemble_size_library_eval.sh` →
  `ensemble_size_hazard.sh` → `ensemble_size_analysis.sh` (closed;
  `envs/ensemble_size_diagnostics.env`), `nestedp_smoke_calibrate.sh` +
  `nestedp_ladder.sh` (optional).
- **`results_diagnostics/`** — on search sets or re-evaluation cubes.
  `reevaluate_unchunked.sh` (unchunked re-evaluation that also persists
  timeseries; the campaign path is steps 10 + 11), `scenario_discovery.sh`
  (exploratory, outside the manuscript plan), `epsilon_calibration.sh` (closed;
  ε fixed), `epsilon_ensemble_refilter.sh` (live; the post-search re-filter),
  `epsilon_refilter_sweep.sh` (superseded), `satisfaction_factor.sh` and
  `objective_sensitivity.sh` (closed), `regret_tolerance_diagnostics.sh` and
  `robustness_threshold_diagnostics.sh` (live; rerun on the new cube),
  `transfer_evaluation_check.sh` → `transfer_evaluation_eval_array.sh` →
  `transfer_evaluation_analysis.sh` (closed; also the SI draw-sensitivity driver).
- **`benchmarks/`** — closed; tables kept under `outputs/supplemental/`.
  `anvil_scaling_packing.sh`, `anvil_scaling_borg_submit.sh` +
  `anvil_scaling_borg.sh` (node packing and MM Borg strong scaling),
  `ensemble_cost_stage_submit.sh` + `ensemble_cost_sweep.sh` (the
  t_eval(N, L, model) cost surface).
- **`checks/`** — `validate_staged_seasonality.sh` (build QC after any ensemble
  is staged; live), `predicted_inflow_bitcheck.sh` (forecast-kernel acceptance;
  closed), `etest_subset_reeval_check.sh` (chunk-prefix path on a mini fixture;
  closed).
- **`figures/`** — `si_figures_design.sh` (design-support SI figures; live),
  `si_figures_results.sh` (needs campaign outputs; live),
  `sim_selected_policies.sh` (selected policies on the historic trace; live).

Python under `scripts/supplemental/` without a wrapper runs on the local
machine; its docstring says so. `policy_dynamics_figures.py`,
`policy_rules_compact_figures.py` and `simulation_api_check.py` are referenced
nowhere and are kept as is.

## Verifying changes locally (no HPC)

```bash
# Shell syntax
bash -n workflow/*.sh workflow/_common.sh workflow/supplemental/*/*.sh

# Config import + slug per env file
for f in workflow/envs/*.env; do
  (set -a; source "$f"; set +a
   python3 -c "import config; print('$f ->', config.active_scenario_name(), config.derive_slug('ffmp'))")
done

# Smoke submission plan (prints sbatch lines only)
bash workflow/submit_smoke.sh --dry-run
```
