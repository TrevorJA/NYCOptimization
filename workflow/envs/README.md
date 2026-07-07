# workflow/envs/ — Per-experiment environment files

Each `*.env` file pins a run's identity (scenario design, MOEA config, active
objectives, LSTM coupling, formulation set, cluster, re-eval sizing) so that
re-running an experiment is a single `sbatch` invocation with no remembered
CLI flags. These files are the **artifact of record** for "how was this
experiment configured" and are tracked in git; the submitted env file is also
snapshotted into the run manifest (`outputs/run_manifests/`).

**Authoring rule:** files in this directory contain only `KEY=VALUE` lines
(shell-sourceable). Comments allowed with `#`. No logic, no `if` statements —
those belong in the scripts that source these files. Every key is a documented
`NYCOPT_*` knob (see the env-override table at the top of `config.py`).

**No defaults:** steps whose meaning depends on a chosen experiment
(`06_run_mmborg.sh`, `08_reevaluate.sh`, `09_simulate_master_chunks.sh`)
require `NYCOPT_ENV_FILE` explicitly and abort with a listing of this
directory when it is unset. There is deliberately no fallback env file — a
run's identity must be stated at submission.

**Precedence:** `workflow/_common.sh` sources the env file with `set -a`
*after* job setup, so env-file values win over anything pre-exported.
`FORMULATION`, `SEED`, `RUN_SLUG`, and `NTASKS_MPI` are job identifiers, not
`NYCOPT_*` knobs — they are passed via `--export` and never set by env files.

## Usage

```bash
# One optimization = one independent job (array index = Borg seed):
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env \
       --array=1-10 workflow/06_run_mmborg.sh

# Variable-resolution FFMP — same launcher, formulation as an identifier:
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7_sal.env,FORMULATION=ffmp_12 \
       --array=1-10 workflow/06_run_mmborg.sh

# Re-evaluate on the common held-out ensemble:
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 \
       workflow/08_reevaluate.sh
```

## Currently shipped env files

MM-Borg run identities (consumed by steps 05, 06, 08, 09):

- `ffmp_obj7_historic.env` — historic single trace, `mm_full` (50k NFE,
  10 seeds), salinity OFF, 7th objective = Trenton flow reliability.
- `ffmp_obj7_historic_pilot.env` — same, but the 5k-NFE `mm_pilot`
  launch-verification config.
- `ffmp_obj7_sal.env` — historic single trace, `mm_full`, **salinity LSTM
  in-loop** (sync mode), 7th objective = salt-front intrusion.
- `ffmp_vr_obj7_sal.env` — variable-resolution FFMP sweep (N ∈ {8, 10, 12}),
  salinity on; submit once per `FORMULATION=ffmp_N`.
- `ffmp_obj7_hazfill_pilot.env` — hazard-filling scenario design, `pilot`
  config; requires steps 02–04 staged first (pre-flight fails fast otherwise).
- `smoke.env` — dev-only tiny-NFE smoke identity, used by
  `workflow/submit_smoke.sh`. Not for replication.

Ensemble-generation settings (consumed by step 02 only):

- `ensemble_kn_short.env` — Kirsch-Nowak, 5-year traces × 200 realizations.
- `ensemble_kn_long.env` — Kirsch-Nowak, 50-year traces × 1000 realizations.

## Slug grammar

MM-Borg env filenames follow the slug grammar so the env file's name matches
the moea slug its outputs land under (see `derive_slug` in `config.py`):

```
{formulation}_obj{N_OBJ}{ts_suffix}{sfdv_suffix}{moea_config_suffix}{custom_suffix}
```

The scenario design is NOT in the slug — it is the parent directory:
`outputs/{scenario}/{moea_slug}/`. The MOEA config name is appended unless it
is the `production` default. Examples:

- `ffmp_obj7_sal.env` (mm_full) → `outputs/historic/ffmp_obj7_sal_mm_full/`
- `ffmp_obj7_hazfill_pilot.env` (pilot) → `outputs/hazard_filling/ffmp_obj7_pilot/`

For ad-hoc tags, set `RUN_SLUG_TAG=mytag`; the slug becomes
`<auto-derived>_mytag`.
