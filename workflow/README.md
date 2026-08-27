# workflow/ — Experiment pipeline

Everything submittable lives here: numbered pipeline steps (`00`–`09`), shared
setup functions (`_common.sh`), per-experiment run configs (`envs/`), and
off-pipeline diagnostics (`supplemental/`). Submit all jobs **from the repo
root**. The top-level [README](../README.md) walks the full replication
sequence; this file documents each step.

Note: `workflow/_common.sh` is unrelated to the repo-root `lib/` directory
(which holds the licensed Borg C sources).

## The env-file contract

A run's identity — **scenario design** (`NYCOPT_SCENARIO_DESIGN`,
`src/scenario_designs.py`) × **MOEA config** (`NYCOPT_MOEA_CONFIG`,
`src/moea_config.py`), plus objectives and physics toggles — comes from one
`KEY=VALUE` env file under [`envs/`](envs/README.md), forwarded via
`sbatch --export=ALL,NYCOPT_ENV_FILE=...`. Scripts pass only identifiers
(`FORMULATION`, `SEED`); every value comes from the env file + config
registries, and `_common.sh` reads the resolved identity back from `config.py`
so shell and Python agree on a single source of truth. Steps `06`/`08`/`09`
require the env file explicitly; the others fall back to `config.py` defaults.

Outputs land under `outputs/{scenario}/{moea_slug}/{artifact}/`; every MM-Borg
job writes a reproducibility manifest (config + env snapshots, git state) to
`outputs/run_manifests/`.

## Pipeline steps

| Step | Script | Allocation (Anvil) | Env file | What it does |
|------|--------|-----------|----------|--------------|
| 00 | `00_setup_borg_jars.sh` | login node (`bash`) | optional | Build one MOEAFramework problem JAR per formulation; rerun after changing the objective set |
| 01 | `01_generate_presim.sh` | `shared`, 1×1, 30 min | optional | Full Pywr-DRB run once; save non-NYC (STARFIT) releases for the trimmed model |
| 02 | `02_generate_ensemble.sh` | `shared`, 8 cpu, 4 h, `--array=0-(K-1)` | optional | Generate the active design's own realizations (or its pool); array index = ensemble draw (three draws staged, `--array=0-2`: d0 is searched, d1–d2 serve the SI draw-sensitivity re-evaluation) |
| 03 | `03_subsample_ensemble.sh` | `shared`, 8 cpu, 1 h | optional (or `NYCOPT_SCENARIO_DESIGN` via `--export`) + `NYCOPT_CANDIDATE_POOL_N=1000000` | Hazard-filling designs only: select N members from the design's own candidate pool, all K draws in one job; other designs generate directly in 02 and skip it |
| 04 | `04_prep_pywrdrb_inputs.sh` | `shared`, 1×33, 1 h, `--array=0-(K-1)` | optional | Format each draw's search ensemble into pywrdrb HDF5 inputs (MPI across realizations); `--preset NAME` stages an arbitrary ensemble (e.g. the held-out re-eval ensemble) |
| 05 | `05_run_baseline.sh` | `shared`, 1×1, 30 min | optional | Evaluate the default (unoptimized) FFMP policy + persist its re-eval matrix for improvement-vs-baseline |
| 06 | `06_run_mmborg.sh` | `wholenode`, 12×128 (1,533 ranks), 96 h | **required** | MM-Borg MOEA search — ONE launcher for all formulations and scenario designs; `--array` = seed replicates, `DRAW=k` = ensemble draw (default 0; the campaign searches d0 only); config-derived pre-flight refuses already-completed (design, draw, seed) cells |
| 07 | `07_run_diagnostics.sh` | `shared`, 8 cpu, 1 h (or `bash`) | optional | MOEAFramework runtime diagnostics (hypervolume, generational distance, reference set); default target = the env file's active slug at `DRAW=k` (positional literal slugs override) |
| 08 | `08_reevaluate.sh` | `wholenode`, 4×16, 8 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Re-evaluate Pareto policies on the common held-out ensemble with the trimmed model (step-04 presim reused across all Pareto sets); opt-in robustness scoring (`NYCOPT_REEVAL_SCORE=1`) |
| 09 | `09_simulate_test_chunks.sh` | `wholenode`, 4×16, 12 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Simulate + score a chunked test ensemble, metrics-only (MPI chunk-and-aggregate) |
| 12 | `12_generate_test_ensemble.sh` | `shared`, 8 cpu, 12 h | optional | Build the held-out test ensemble E_test: LHS over the FULL DU box × R>1 realizations per SOW, chunked, hazard image streamed. `--variant kn` is the campaign's E_test; `hmm` is an opt-in generator sensitivity |

Anvil notes: the allocation account is hardcoded in every script's header
(`#SBATCH --account=x-tamestoy`); override with `sbatch -A <alloc>` if needed. 96 h is Anvil's `wholenode`
per-job maximum. There is no resume: the runtime files are diagnostic dumps, the
Borg checkpoint is disabled, and every search is sized to finish inside one job
(`docs/notes/methods/campaign_design.md`).
`shared` bills per core; `wholenode` bills whole 128-core nodes.

Step order: `01` before `05`/`06`; `02`→`04` before `06` for ensemble scenario
designs (`historic` skips `02`–`04`); `06` before `07`/`08`. Chain with
`sbatch --dependency=afterok:<jobid>`.

`12` builds E_test and is independent of `02`–`07` (it is not a scenario design and
never enters search). It must run before `05`/`08`/`09`/`11`, all of which take
`NYCOPT_REEVAL_ENSEMBLE_PRESET=<its slug>` — and `05` must use the SAME preset as `08`,
or the status-quo baseline lands under a different re-eval tag and the
incumbent-relative regret family is silently skipped.

## Building a design's search ensemble (02–04)

Every scenario design **generates its own realizations** from its own namespaced
seed stream (`src/scenario_designs.py`); no design is subsampled from a shared
master. Step 02 dispatches on `design.construction`, so what it builds — and
whether step 03 applies at all — follows from the design alone:

| construction | designs | 02 builds | 03 | 04 array |
|---|---|---|---|---|
| `preset` | `historic` | nothing (static preset) | — | — |
| `direct_iid` | `fixed_probabilistic` | one N×L ensemble **per draw** | — | `0-(K-1)` |
| `lhs_theta` | `input_stratified` | LHS over forcing params, realizations generated at each design point, **per draw** | — | `0-(K-1)` |
| `pool_resample` | `resampled_probabilistic` — **dropped from the plan; do not run** | one draw-invariant pool (redrawn per evaluation in-search) | — | `0` |
| `hazard_fill` | `hazard_filling_{stationary,du,absolute}` | one draw-invariant candidate pool + its hazard image | **yes** — all K draws in one job | `0-(K-1)` |
| `stationary_kn` | `scaling_stationary` | direct Kirsch-Nowak stand-in (supplemental) | — | `0` |

The array index in `02`/`04` is the ensemble-draw index *k*; set `--array=0-(K-1)`
with K = `design.n_ensemble_draws` (= 3 for the matched designs). The campaign
searches draw 0 only; draws 1–2 are staged for the SI draw-sensitivity
re-evaluation of each design's final set. **Cost:** per-design construction
multiplies step-02 cost by K for `fixed_probabilistic` and `input_stratified` —
each draw is a fresh N×L generation, not a re-index of shared data. Pool-owning designs pay it
once (array tasks k>0 are no-ops), and the two DU hazard designs share one pool.
`NYCOPT_ENSEMBLE_FORCE=1` overwrites an already-staged slug.

## Optimization runs are independent jobs

Each optimization is one self-contained multi-day sbatch job — one submission
per (env file × formulation × ensemble draw), no campaign wrapper. `DRAW=k`
selects the staged ensemble draw (default 0) and the `--array` index is the
Borg seed. The campaign searches draw 0 with S = 2 seeds per design, submitted
one seed at a time (seed 1 runs 750k NFE, seed 2 500k;
`docs/notes/methods/campaign_design.md` §2):

```bash
# Single go/no-go replicate (draw 0, seed 1) before committing the campaign:
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,DRAW=0          --array=1 workflow/06_run_mmborg.sh
# Second seed on the same draw:
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env,DRAW=0          --array=2 workflow/06_run_mmborg.sh
# Variable-resolution FFMP (same launcher, formulation from the identifier):
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj8.env,FORMULATION=ffmp_12   --array=1-10 workflow/06_run_mmborg.sh
```

What has already run is readable from the output tree alone: draw k > 0
appends `_d{k}` to the moea slug (draw 0 is the unsuffixed default), each
completed seed writes `outputs/{scenario}/{moea_slug}/sets/seed_{SS}_{slug}.set`,
and every job records its full identity (including the draw) in
`outputs/run_manifests/`. The step-06 pre-flight refuses to relaunch a
(design, slug, seed) whose `.set` file already exists — pass
`NYCOPT_OVERWRITE=1` to deliberately redo one. A crashed run leaves no `.set`
file and may simply be resubmitted.

## Geometry contract and scaling on Anvil

The MPI rank count has a single source per step; the `#SBATCH` lines are only
the container for it:

- **MM-Borg (06)**: ranks = `MOEAConfig.total_ntasks_mpi`
  (`1 + islands × (workers + 1)`, from `src/moea_config.py`). `_common.sh`
  reads it back for `mpirun -np`, and `nycopt_check_allocation` aborts before
  the search starts if the allocation is smaller — printing the exact
  `--nodes/--ntasks-per-node` to resubmit with — and warns when a whole
  node or more would sit idle.
- **All other MPI steps (04, 08, 09, supplemental)**: ranks =
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
centralized as `NYCOPT_RANKS_PER_NODE` in `_common.sh`, override for other
machines — 33/node was the Hopper-safe packing). Anvil ceilings: `wholenode`
allows up to 16 nodes (2,048 cores) and 96 h. The `wide` queue reaches 56
nodes but only 12 h. Seeds
(`--array`) and experiments (env files) scale horizontally as fully
independent jobs with no cross-job coordination. Shorter pilots can pass
`sbatch --time=...`.

## Development utilities (not replication)

- `submit_smoke.sh` — one tiny-NFE end-to-end check per formulation
  (`bash workflow/submit_smoke.sh [--dry-run]`; Anvil `debug` queue, 2×40,
  2 h, `smoke` MOEA config + short 2018–2022 window via `envs/smoke.env`).
- `submit_search_memory_smoke.sh` — one-node memory/timing smoke of the batched
  search path (N = 300, realization batch 150, 128 ranks/node via
  `envs/smoke_search_batch.env`); the go criterion is in
  `docs/notes/methods/campaign_design.md` §4.
- `supplemental/` — off-pipeline diagnostics: `anvil_scaling_packing.sh`
  (ranks-per-node packing sweep), `ensemble_cost_stage_submit.sh` +
  `ensemble_cost_sweep.sh` (the t_eval(N, L, model) cost surface that prices
  the campaign), `objective_sensitivity.sh` (historic random-DV
  objective-sensitivity sweep), `epsilon_calibration.sh` (per-design epsilon
  calibration) and `satisfaction_factor.sh` (per-design weekly
  satisfaction-factor sweep; all settings in root `supplemental_config.py`).

## Verifying changes locally (no HPC)

```bash
# Shell syntax
bash -n workflow/*.sh workflow/_common.sh workflow/supplemental/*.sh

# Config import + slug per env file
for f in workflow/envs/*.env; do
  (set -a; source "$f"; set +a
   python3 -c "import config; print('$f ->', config.active_scenario_name(), config.derive_slug('ffmp'))")
done

# Smoke submission plan (prints sbatch lines only)
bash workflow/submit_smoke.sh --dry-run
```
