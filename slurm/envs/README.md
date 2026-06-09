# slurm/envs/ — Per-experiment environment files

Each `*.env` file pins a methodologic configuration (active objectives,
T/S coupling, formulation set, ensemble preset, cluster, re-eval sizing)
so that re-running an experiment is a single `sbatch` invocation with no
remembered CLI flags.

**Authoring rule:** files in this directory contain only `KEY=VALUE`
lines (shell-sourceable). Comments allowed with `#`. No logic, no `if`
statements — those belong in scripts that source these files.

## Usage

```bash
# Submit a campaign described by an env file:
bash slurm/submit_all.sh slurm/envs/ffmp_obj7_sal.env

# Or pin a single-formulation run:
sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_sal.env \
       slurm/mmborg_ffmp.sh

# Re-evaluate using a chosen env file:
sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj7_sal.env \
       workflow/07_reevaluate.sh ffmp 0
```

## Currently shipped env files

- `ffmp_obj7_sal.env` — base FFMP, 7 objectives, salinity on.
- `ffmp_vr_obj7_sal.env` — variable-resolution FFMP sweep (N ∈ {8, 10, 12}).

Per-ensemble env files will be added once the ensemble-design discussion
finalizes preset sizes; expected naming follows the slug grammar (e.g.
`ffmp_obj7_sal_lhs_small.env` → outputs under
`outputs/{cat}/ffmp_obj7_sal_lhs_small/`).

## Slug grammar

Filenames follow the slug grammar so the env file's name matches the slug
its outputs land under (see `derive_slug` in `config.py`):

```
{formulation}_obj{N_OBJ}{ts_suffix}{sfdv_suffix}{ensemble_suffix}{custom_suffix}
```

Examples:
- `ffmp_obj7_sal.env` → `outputs/{cat}/ffmp_obj7_sal/`
- `ffmp_obj7_sal_lhs_small.env` → `outputs/{cat}/ffmp_obj7_sal_lhs_small/`
  (LHS small ensemble preset; slug fragment appended via
  `SEARCH_ENSEMBLE_SPEC.slug_fragment`).
