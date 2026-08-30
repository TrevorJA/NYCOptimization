# workflow/envs/ — Per-experiment environment files

Each `*.env` file pins a run's identity (scenario design, MOEA config, active
objectives, LSTM coupling, formulation set, cluster, re-eval sizing, regret
tolerance) so that re-running an experiment is a single `sbatch` invocation
with no remembered CLI flags. These files are the **artifact of record** for
"how was this experiment configured" and are tracked in git; the submitted env
file is also snapshotted into the run manifest (`outputs/run_manifests/`).

**Authoring rule:** files in this directory contain only `KEY=VALUE` lines
(shell-sourceable). Comments allowed with `#`. No logic, no `if` statements —
those belong in the scripts that source these files. Every key is a `NYCOPT_*`
knob read by `config.py`, `src/scenario_designs.py`, or `src/robustness.py`
(grep `os.environ` for the full set; the table at the top of `config.py` lists
the main ones).

**No defaults:** steps whose meaning depends on a chosen experiment (`06`,
`08`, `09`, `09b`, `11`) require `NYCOPT_ENV_FILE` explicitly and abort with a
listing of this directory when it is unset. There is deliberately no fallback
env file — a run's identity must be stated at submission.

**Precedence:** `workflow/_common.sh` sources the env file with `set -a`
*after* job setup, so env-file values win over anything pre-exported.
`FORMULATION`, `SEED`, `DRAW`, `RUN_SLUG`, `NTASKS_MPI` (and the per-step knobs
`DEBUG_SIM`, `MAX_SOLUTIONS`, `DESIGNS`) are job identifiers, not `NYCOPT_*`
knobs — they are passed via `--export` and never set by env files.
`NYCOPT_REEVAL_ENSEMBLE_PRESET` is likewise passed on every re-evaluation
submission and never pinned here, so the common E_test is a recorded choice.

## Usage

```bash
# One optimization = one independent job (array index = Borg seed); production
# needs the 12-node geometry and a per-seed --time (see each file's header):
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env,DRAW=0 \
       --array=1 --nodes=12 --ntasks-per-node=128 --time=96:00:00 workflow/06_run_mmborg.sh

# Chunked re-evaluation on the common held-out ensemble (campaign path, then 09b):
sbatch --partition=shared --nodes=1 --ntasks=16 --cpus-per-task=8 --time=24:00:00 \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_mc_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch,NYCOPT_CHUNK_POLICIES=outputs/monte_carlo/ffmp_obj8/sets/ffmp_obj8_merged.set,NYCOPT_CHUNK_MERGE=off,NYCOPT_SEARCH_REALIZATION_BATCH=50 \
       workflow/09_simulate_test_chunks.sh
```

## Currently shipped env files

MM-Borg run identities (consumed by steps 05, 06, 08, 09, 09b, 11), by family:

- **Production campaign** — `ffmp_obj8_historic_production.env`,
  `ffmp_obj8_mc_production.env`, `ffmp_obj8_hazfill_stat_production.env`:
  the three campaign designs under the `production` MOEA config (1,533 ranks
  on 12 nodes; N = 300 for the matched designs; seed 1 at 750k NFE and seed 2
  at 500k, every seed reported at equal NFE from the 125,000-per-island
  runtime snapshot). The matched files set
  `NYCOPT_SEARCH_REALIZATION_BATCH=150`; otherwise identical except for
  `NYCOPT_SCENARIO_DESIGN` and header comments
  (`docs/notes/methods/campaign_design.md`).
- **Moderate-NFE dev** — `ffmp_obj8_historic_moderate.env`,
  `ffmp_obj8_mc_moderate.env`, `ffmp_obj8_hazfill_stat_moderate.env`:
  the same designs under `mm_moderate` (50k NFE, 511 ranks on 4 nodes).
- **Base** — `ffmp_obj8_historic.env` (`mm_full`, 50k NFE, 165 ranks; the
  pre-campaign historic identity).
- **Variable resolution** — `ffmp_vr_obj8.env` (N ∈ {8, 10, 12}); submit once
  per `FORMULATION=ffmp_N`. A conditional SI extension on leftover allocation,
  not part of the manuscript campaign.
- **Smoke** — `smoke.env`, `smoke_hazfill.env`, `smoke_search_batch.env`:
  dev-only tiny-NFE identities (`workflow/submit_smoke.sh`,
  `workflow/submit_search_memory_smoke.sh`). Not for replication.
- **Supplemental diagnostics** — `eps_calib_historic.env`,
  `eps_calib_monte_carlo.env`, `eps_calib_hazard_filling_stationary.env`
  (epsilon calibration); `ensemble_size_diagnostics.env`
  (`hazard_filling_stationary` identity with `NYCOPT_CANDIDATE_POOL_N` pinned
  to the P = 10⁶ pool; every `workflow/supplemental/ensemble_size_*.sh`);
  `anvil_scaling_borg.env`, `anvil_scaling_packing.env`, `ensemble_cost.env`
  (scaling / cost experiments under `workflow/supplemental/`).

The salinity LSTM is not used (it does not perform well under extreme
droughts); the machinery is dormant. To re-enable it for an experiment, set
`NYCOPT_SALINITY_ON=1` and swap the 5th objective to `salt_front_intrusion_max_rm`
in a new env file (the slug then gains a `_sal` suffix automatically).

**No ensemble-sizing env files.** Every design's sizing (N, L, pool size, K, seed
domain) is a property of the design in `src/scenario_designs.py`, so step 02 needs
only the design name — the same env file that drives the run. There is deliberately
no `ensemble_*.env`: an ensemble whose shape could be overridden at submission would
break the size-matching the cross-design comparison depends on. The one
generation-time knob, `NYCOPT_CANDIDATE_POOL_N`, selects the pool size and is
exported on the pool-building and step-03 lines.

## Slug grammar

MM-Borg env filenames follow the slug grammar so the env file's name matches
the moea slug its outputs land under (see `derive_slug` in `config.py`):

```
{formulation}_obj{N_OBJ}{ts_suffix}{sfdv_suffix}{moea_config_suffix}{custom_suffix}
```

The scenario design is NOT in the slug — it is the parent directory:
`outputs/{scenario}/{moea_slug}/`. The MOEA config name is appended unless it
is the `production` default. Examples:

- `ffmp_obj8_mc_production.env` (production) → `outputs/monte_carlo/ffmp_obj8/`
- `ffmp_obj8_historic_moderate.env` (mm_moderate) → `outputs/historic/ffmp_obj8_mm_moderate/`

For ad-hoc tags, set `RUN_SLUG_TAG=mytag`; the slug becomes
`<auto-derived>_mytag`.
