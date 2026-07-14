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
| 02 | `02_generate_ensemble.sh` | `shared`, 8 cpu, 4 h, `--array=0-(K-1)` | optional | Generate the active design's own realizations (or its pool); array index = ensemble draw |
| 03 | `03_subsample_ensemble.sh` | `shared`, 8 cpu, 1 h | optional (or `NYCOPT_SCENARIO_DESIGN` via `--export`) | Hazard-filling designs only: select N members from the design's own candidate pool, all K draws in one job; other designs generate directly in 02 and skip it |
| 04 | `04_prep_pywrdrb_inputs.sh` | `shared`, 1×33, 1 h, `--array=0-(K-1)` | optional | Format each draw's search ensemble into pywrdrb HDF5 inputs (MPI across realizations); `--preset NAME` stages an arbitrary ensemble (e.g. the held-out re-eval ensemble) |
| 05 | `05_run_baseline.sh` | `shared`, 1×1, 30 min | optional | Evaluate the default (unoptimized) FFMP policy + persist its re-eval matrix for regret-from-baseline |
| 06 | `06_run_mmborg.sh` | `wholenode`, 5×33, 96 h | **required** | MM-Borg MOEA search — ONE launcher for all formulations and scenario designs; `--array=1-10` = seed replicates; config-derived pre-flight |
| 07 | `07_run_diagnostics.sh` | `shared`, 8 cpu, 1 h (or `bash`) | — | MOEAFramework runtime diagnostics (hypervolume, generational distance, reference set); positional slug identifiers select targets |
| 08 | `08_reevaluate.sh` | `wholenode`, 4×16, 8 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Re-evaluate Pareto policies on the common held-out ensemble with the full model; opt-in robustness scoring (`NYCOPT_REEVAL_SCORE=1`) |
| 09 | `09_simulate_test_chunks.sh` | `wholenode`, 4×16, 12 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Simulate + score a chunked test ensemble, metrics-only (MPI chunk-and-aggregate) |
| 12 | `12_generate_test_ensemble.sh` | `shared`, 8 cpu, 12 h | optional | Build the held-out test ensemble E_test: LHS over the FULL DU box × R>1 realizations per SOW, chunked, hazard image streamed. `--variant kn` is the campaign's E_test; `hmm` is an opt-in generator sensitivity |

Anvil notes: the allocation account is hardcoded in every script's header
(`#SBATCH --account=x-tamestoy`); override with `sbatch -A <alloc>` if needed. 96 h is Anvil's `wholenode`
per-job maximum (searches needing longer restart from runtime snapshots).
`shared` bills per core; `wholenode` bills whole 128-core nodes.

Step order: `01` before `05`/`06`; `02`→`04` before `06` for ensemble scenario
designs (`historic` skips `02`–`04`); `06` before `07`/`08`. Chain with
`sbatch --dependency=afterok:<jobid>`.

`12` builds E_test and is independent of `02`–`07` (it is not a scenario design and
never enters search). It must run before `05`/`08`/`09`/`11`, all of which take
`NYCOPT_REEVAL_ENSEMBLE_PRESET=<its slug>` — and `05` must use the SAME preset as `08`,
or the status-quo baseline lands under a different re-eval tag and
`improvement_vs_baseline` is silently skipped.

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
| `pool_resample` | `resampled_probabilistic` | one draw-invariant pool (redrawn per evaluation in-search) | — | `0` |
| `hazard_fill` | `hazard_filling_{stationary,du,absolute}` | one draw-invariant candidate pool + its hazard image | **yes** — all K draws in one job | `0-(K-1)` |
| `stationary_kn` | `scaling_stationary` | direct Kirsch-Nowak stand-in (supplemental) | — | `0` |

The array index in `02`/`04` is the ensemble-draw index *k*; set `--array=0-(K-1)`
with K = `design.n_ensemble_draws`. **Cost:** per-design construction multiplies
step-02 cost by K for `fixed_probabilistic` and `input_stratified` — each draw is a
fresh N×L generation, not a re-index of shared data. Pool-owning designs pay it
once (array tasks k>0 are no-ops), and the two DU hazard designs share one pool.
`NYCOPT_ENSEMBLE_FORCE=1` overwrites an already-staged slug.

## Optimization runs are independent jobs

Each optimization is one self-contained multi-day sbatch job — one submission
per (env file × formulation), no campaign wrapper:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env               --array=1-10 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_8   --array=1-10 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_10  --array=1-10 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_12  --array=1-10 workflow/06_run_mmborg.sh
```

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

**To scale the search up**: register a larger MOEA config (more
islands/workers) in `src/moea_config.py`, point the env file's
`NYCOPT_MOEA_CONFIG` at it, and submit with
`--nodes=ceil(total_ntasks_mpi / 33) --ntasks-per-node=33` (33/node is the
memory-bandwidth-safe packing, centralized as `NYCOPT_RANKS_PER_NODE` in
`_common.sh`). Anvil ceilings: `wholenode` allows up to 16 nodes (2,048
cores) and 96 h — at 33 ranks/node that is ~528 ranks (≈ 13 islands × 40
workers); denser packing (override `NYCOPT_RANKS_PER_NODE` after
benchmarking with `supplemental/anvil_scaling_packing.sh`) raises the ceiling
toward 2,048 ranks. The `wide` queue reaches 56 nodes but only 12 h. Seeds
(`--array`) and experiments (env files) scale horizontally as fully
independent jobs with no cross-job coordination. Shorter pilots can pass
`sbatch --time=...`.

## Development utilities (not replication)

- `submit_smoke.sh` — one tiny-NFE end-to-end check per formulation
  (`bash workflow/submit_smoke.sh [--dry-run]`; Anvil `debug` queue, 2×40,
  2 h, `smoke` MOEA config + short 2018–2022 window via `envs/smoke.env`).
- `supplemental/` — off-pipeline diagnostics: `anvil_scaling_packing.sh`
  (ranks-per-node packing sweep), `ensemble_cost_stage_submit.sh` +
  `ensemble_cost_sweep.sh` (the t_eval(N, L, model) cost surface that prices
  the campaign), `objective_sensitivity.sh` and
  `ensemble_objective_sensitivity{,_prep}.sh` (random-DV objective-sensitivity
  sweeps; all settings in root `supplemental_config.py`).

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
