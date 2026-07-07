# Env Files

Env files at [`workflow/envs/`](../../../workflow/envs/) are the **artifact of record** for "how was this experiment configured". A workflow script + an env file is everything needed to reproduce a run.

## Why env files (not CLI flags)

CLI flags require remembering the right invocation across sessions and machines. Env files don't. Submitting a run is:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/<file>.env --array=1-10 workflow/06_run_mmborg.sh
```

There is no flag combination to forget. The env file is the same file that gets copied into the run manifest at submission time, so the run's exact configuration is preserved alongside its outputs.

## File format

Plain `KEY=VALUE` shell-sourceable lines. Comments allowed with `#`. No logic.

```bash
# workflow/envs/myexperiment.env
NYCOPT_SCENARIO_DESIGN="historic"
NYCOPT_MOEA_CONFIG="mm_full"
NYCOPT_FORMULATIONS="ffmp"
NYCOPT_OBJECTIVES="nyc_delivery_reliability_weekly,..."
NYCOPT_CLUSTER="anvil"
```

Every key must be a documented `NYCOPT_*` knob — the override table lives at the top of `config.py`.

## Filename grammar

Filename should match the moea slug `derive_slug()` will produce, so `workflow/envs/<slug>.env` and `outputs/{scenario}/<slug>/` line up (the scenario design is the parent directory, not part of the slug; the MOEA config name is appended unless it is the `production` default):

- `ffmp_obj7_sal.env` (mm_full) → outputs at `outputs/historic/ffmp_obj7_sal_mm_full/`
- `ffmp_obj7_hazfill_pilot.env` (pilot) → outputs at `outputs/hazard_filling/ffmp_obj7_pilot/`
- `ffmp_vr_obj7_sal.env` → expands to multiple slugs (one per `FORMULATION=ffmp_N` submission)

For ad-hoc tags, set `RUN_SLUG_TAG=mytag` in the env file; the slug becomes `<auto-derived>_mytag`.

## How env files flow through the pipeline

1. **`sbatch --export=ALL,NYCOPT_ENV_FILE=<file> workflow/06_run_mmborg.sh`** forwards the env-file path into the job. There is no default env file — steps 06/08/09 abort with a listing of `workflow/envs/*.env` when it is unset.
2. **The job script** sources [`workflow/_common.sh`](../../../workflow/_common.sh) and calls its setup functions (`nycopt_setup_env`, `nycopt_source_env_file`, ...).
3. **`nycopt_source_env_file`** sources `${NYCOPT_ENV_FILE}` with `set -a`, exporting the knobs into the Python process env; `nycopt_read_run_identity` then calls one `python3 -c "import config; ..."` read-back to compute `RUN_SLUG`, the scenario name, and the MPI rank count — so shell and Python agree on a single source of truth.
4. **`config.py`** reads `NYCOPT_*` at import time. `derive_slug()` composes the canonical slug from active config.
5. **The Python pipeline** (mmborg_cli, simulation, plotting, reevaluate) writes outputs and figures under the slug-aware paths returned by `run_output_dir()` and `figure_dir_for()`.

## Authoring a new env file

1. Decide what's being varied: scenario design? MOEA config? objectives? LSTM coupling? formulation set?
2. Pick a filename that describes the experiment and matches the slug grammar.
3. Set only the knobs that differ from `config.py` defaults. Leaving a knob out lets the default win.
4. Validate locally:
   ```bash
   set -a; source workflow/envs/<file>.env; set +a
   python3 -c "from config import derive_slug, ACTIVE_OBJECTIVES, INCLUDE_SALINITY_MODEL; \
       print('slug=', derive_slug('ffmp')); print('n_obj=', len(ACTIVE_OBJECTIVES)); \
       print('sal_on=', INCLUDE_SALINITY_MODEL)"
   ```
5. Submit; the MM-Borg pre-flight echoes the resolved identity to the job log and fails fast on inconsistencies before the search starts.

## Don't put logic in env files

If you find yourself wanting an `if` or a computed value, that belongs in either `config.py` (as a derived constant) or in `workflow/_common.sh` (as a shared setup function). Env files stay declarative.
