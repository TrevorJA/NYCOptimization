# slurm/envs/ — Per-experiment environment files

Each `*.env` file pins a methodologic configuration (active objectives,
T/S coupling, formulation set, cluster, re-eval sizing) so that re-running
an experiment is a single `sbatch` invocation with no remembered CLI flags.

**Authoring rule:** files in this directory contain only `KEY=VALUE`
lines (shell-sourceable). Comments allowed with `#`. No logic, no `if`
statements — those belong in scripts that source these files.

## Usage

```bash
# Submit a campaign described by an env file:
bash slurm/submit_all.sh slurm/envs/ffmp_obj7.env

# Or pin a single architecture run:
sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ffmp_obj9_ts.env \
       slurm/mmborg_ffmp.sh

# Re-evaluate using a chosen env file:
sbatch slurm/envs/ann_obj9_ts.env workflow/05_reevaluate.sh
```

## Knob reference

See [local_notes/configuration/knob_reference.md](../../local_notes/configuration/knob_reference.md)
for the full table of `NYCOPT_*` env variables and their defaults.

## Slug grammar

Filenames follow the slug grammar so the env file's name matches the slug
its outputs land under:

- `ffmp_obj7.env` → outputs at `outputs/{cat}/ffmp_obj7/`
- `ffmp_obj9_ts.env` → outputs at `outputs/{cat}/ffmp_obj9_ts/`

See [local_notes/methodology/slug_convention.md](../../local_notes/methodology/slug_convention.md).
