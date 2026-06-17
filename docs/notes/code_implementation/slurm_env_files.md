# SLURM Env Files

Env files at [`slurm/envs/`](../../slurm/envs/) are the **artifact of record** for "how was this experiment configured". A SLURM script + an env file is everything needed to reproduce a run.

## Why env files (not CLI flags)

CLI flags require remembering the right invocation across sessions and machines. Env files don't. Submitting a campaign is:

```bash
bash slurm/submit_all.sh slurm/envs/manuscript_obj9_ts.env
```

There is no flag combination to forget. The env file is the same file that gets copied into the run manifest at submission time, so the campaign's exact configuration is preserved alongside its outputs.

## File format

Plain `KEY=VALUE` shell-sourceable lines. Comments allowed with `#`. No logic.

```bash
# slurm/envs/myexperiment.env
NYCOPT_FORMULATIONS="ffmp,ann"
NYCOPT_TS_ON=1
NYCOPT_OBJECTIVES="nyc_reliability_weekly,nyc_vulnerability,storage_min_combined_pct,lordville_thermal_exceedance_days"
NYCOPT_CLUSTER="anvil"
```

Every key must be a documented `NYCOPT_*` knob from [knob_reference.md](knob_reference.md).

## Filename grammar

Filename should match the slug `derive_slug()` will produce, so `slurm/envs/<slug>.env` and `outputs/<category>/<slug>/` line up:

- `ffmp_obj7_sal.env` → outputs at `outputs/{cat}/ffmp_obj7_sal/`
- `wcu_obj7_sal_n5.env` → outputs at `outputs/{cat}/ffmp_obj7_sal_wcu5/` (ensemble; `wcu5` slug fragment is appended via `SEARCH_ENSEMBLE_SPEC.slug_fragment`)
- `manuscript_obj7_sal.env` → expands to multiple slugs (one per formulation in `NYCOPT_FORMULATIONS`)

For ad-hoc tags, set `RUN_SLUG_TAG=mytag` in the env file; the slug becomes `<auto-derived>_mytag`.

## How env files flow through the pipeline

1. **`bash slurm/submit_all.sh slurm/envs/<file>.env`** sources the env file in its own shell, reads `NYCOPT_FORMULATIONS`, then submits one or more sbatch jobs with `--export=ALL,NYCOPT_ENV_FILE=<file>`.
2. **Each sbatch'd SLURM script** (e.g. `slurm/mmborg_ffmp.sh`) sources [`_common.sh`](../../slurm/_common.sh).
3. **`_common.sh`** sources `${NYCOPT_ENV_FILE}` again (idempotent — same values), exports the knobs into the Python process env, then calls `python3 -c "from config import derive_slug; print(derive_slug('${FORMULATION}'))"` to compute `RUN_SLUG`.
4. **`config.py`** reads `NYCOPT_*` at import time. `derive_slug()` composes the canonical slug from active config.
5. **The Python pipeline** (mmborg_cli, simulation, plotting, reevaluate) writes outputs and figures under the slug-aware paths returned by `output_dir_for()` and `figure_dir_for()`.

## Authoring a new env file

1. Decide what's being varied: formulation set? objectives? T/S coupling? Cluster?
2. Pick a filename that describes the experiment (e.g., `ann_reduced_state_obj7.env`).
3. Set only the knobs that differ from `config.py` defaults. Leaving a knob out lets the default win.
4. Validate locally:
   ```bash
   set -a; source slurm/envs/<file>.env; set +a
   python3 -c "from config import derive_slug, ACTIVE_OBJECTIVES, INCLUDE_TEMPERATURE_MODEL; \
       print('slug=', derive_slug('ffmp')); print('n_obj=', len(ACTIVE_OBJECTIVES)); \
       print('ts_on=', INCLUDE_TEMPERATURE_MODEL)"
   ```
5. Dry-run the submission:
   ```bash
   bash slurm/submit_all.sh slurm/envs/<file>.env --dry-run
   ```
6. Submit when the printed sbatch lines look right.

## Don't put logic in env files

If you find yourself wanting an `if` or a computed value, that belongs in either `config.py` (as a derived constant) or in `_common.sh` (as cluster-specific dispatch). Env files stay declarative.
