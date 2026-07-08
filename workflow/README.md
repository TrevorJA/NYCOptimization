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
| 02 | `02_generate_ensemble.sh` | `shared`, 8 cpu, 4 h | optional (`ensemble_kn_*.env`) | Generate the stochastic streamflow ensemble / forcing master |
| 03 | `03_subsample_ensemble.sh` | `shared`, 8 cpu, 1 h | optional (or `NYCOPT_SCENARIO_DESIGN` via `--export`) | Hazard-filling designs: subsample the master pool into the reduced search ensemble; no-op for other designs |
| 04 | `04_prep_pywrdrb_inputs.sh` | `shared`, 1×33, 1 h | optional | Format the search ensemble into pywrdrb HDF5 inputs (MPI across realizations); `--preset NAME` stages an arbitrary ensemble (e.g. the held-out re-eval ensemble) |
| 05 | `05_run_baseline.sh` | `shared`, 1×1, 30 min | optional | Evaluate the default (unoptimized) FFMP policy + persist its re-eval matrix for regret-from-baseline |
| 06 | `06_run_mmborg.sh` | `wholenode`, 5×33, 96 h | **required** | MM-Borg MOEA search — ONE launcher for all formulations and scenario designs; `--array=1-10` = seed replicates; config-derived pre-flight |
| 07 | `07_run_diagnostics.sh` | `shared`, 8 cpu, 1 h (or `bash`) | — | MOEAFramework runtime diagnostics (hypervolume, generational distance, reference set); positional slug identifiers select targets |
| 08 | `08_reevaluate.sh` | `wholenode`, 4×16, 8 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Re-evaluate Pareto policies on the common held-out ensemble with the full model; opt-in robustness scoring (`NYCOPT_REEVAL_SCORE=1`) |
| 09 | `09_simulate_master_chunks.sh` | `wholenode`, 4×16, 12 h | **required** (+ `NYCOPT_REEVAL_ENSEMBLE_PRESET`) | Simulate + score a chunked forcing master, metrics-only (MPI chunk-and-aggregate) |

Anvil notes: an allocation account is mandatory — `export SBATCH_ACCOUNT=<allocation>`
once per shell and every submission picks it up. 96 h is Anvil's `wholenode`
per-job maximum (searches needing longer restart from runtime snapshots).
`shared` bills per core; `wholenode` bills whole 128-core nodes.

Step order: `01` before `05`/`06`; `02`→`04` before `06` for ensemble scenario
designs (`historic` skips `02`–`04`); `06` before `07`/`08`. Chain with
`sbatch --dependency=afterok:<jobid>`.

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
benchmarking with `supplemental/bench_ensemble.sh`) raises the ceiling
toward 2,048 ranks. The `wide` queue reaches 56 nodes but only 12 h. Seeds
(`--array`) and experiments (env files) scale horizontally as fully
independent jobs with no cross-job coordination. Shorter pilots can pass
`sbatch --time=...`.

## Development utilities (not replication)

- `submit_smoke.sh` — one tiny-NFE end-to-end check per formulation
  (`bash workflow/submit_smoke.sh [--dry-run]`; Anvil `debug` queue, 2×40,
  2 h, `smoke` MOEA config + short 2018–2022 window via `envs/smoke.env`).
- `supplemental/` — off-pipeline diagnostics: `bench_ensemble.sh` (per-eval
  wall-clock benchmark), `objective_sensitivity.sh` and
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
