# NYCOptimization

Multi-objective optimization of NYC reservoir operations (Pywr-DRB + MM-Borg),
focused on the design of the streamflow ensembles used during MOEA evaluation.
Project state: `docs/research_project_summary.md` (entry point); method:
`docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md`; per-step
details: `workflow/README.md`.

## Experimental Replication

> **Note:** This experiment was run on the [Anvil HPC at Purdue](https://docs.rcac.purdue.edu/userguides/anvil/)
> (128-core nodes; production search jobs use 12 nodes × 128 tasks on the
> `wholenode` partition). The instructions below correspond to this HPC, but may be adapted
> for other computing infrastructure. Submit all jobs **from the repo root** —
> the scripts resolve paths from the submission directory.

**Anvil specifics** (already encoded in the scripts' `#SBATCH` headers, listed
here so you know what to expect):

- **Allocation account is mandatory.** It is hardcoded in every sbatch
  script's header (`#SBATCH --account=ees260021`) — no per-shell export
  needed. Override for a different allocation with `sbatch -A <alloc> ...`.
- **Partitions**: small serial steps use `shared` (per-core billing, max 1
  node); multi-node MPI jobs (`07`, `10`, supplemental sweeps) use
  `wholenode` (node-exclusive billing); the smoke test uses `debug` (2 nodes,
  2 h max). These are set in each script's `#SBATCH --partition` line.
- **96-hour wall-time cap.** Anvil's `wholenode` maximum is 96 h per job.
  There is no resume: the island runtime files
  (`outputs/{scenario}/{slug}/runtime/`) are diagnostic dumps and the Borg
  checkpoint is disabled, so every search is sized to finish inside one job
  (`docs/notes/methods/campaign_design.md`).

Everything submittable lives in `workflow/` (numbered steps `00`–`14`). A run's identity — scenario design, MOEA config, objectives, physics
toggles — comes from a single env file under `workflow/envs/`; the scripts take
no value-carrying CLI flags (see [workflow/envs/README.md](workflow/envs/README.md)).
The campaign identities are the three `workflow/envs/ffmp_obj8_{design}_production.env`
files (`historic`, `monte_carlo`, `hazard_filling_stationary`).

### 1.0 Setup

#### 1.1 Clone supporting repos (all repos share one parent folder)

```bash
git clone -b nyc_opt git@github.com:Pywr-DRB/Pywr-DRB.git
git clone git@github.com:TrevorJA/SynHydro.git
git clone git@github.com:TrevorJA/NYCOptimization_scenario_generation.git
git clone git@github.com:TrevorJA/NYCOptimization.git
git clone git@github.com:Pywr-DRB/CMIP6_multimodel_streamflow.git
cd NYCOptimization
```

#### 1.2 Environment

```bash
module load anaconda
conda create -n venv python=3.11.5
conda activate venv
pip install -r requirements.txt
```

Note that the `requirements.txt` installs an 'editable' installation of all
sibling repos (../SynHydro, ../Pywr-DRB, ../NYCOptimization_scenario_generation).

The workflow scripts activate `./venv/bin/activate` if that directory exists.
With a conda environment (as above) there is no `./venv`, so activate the conda
env in your shell before submitting jobs; if a job starts without it,
`workflow/_common.sh` loads `anaconda/2024.02-py311` and activates `venv`
itself (override with `NYCOPT_CONDA_MODULE` / `NYCOPT_CONDA_ENV`).

#### 1.3 BorgMOEA (licensed; not in git) and MOEAFramework

The following BorgMOEA source files must be obtained under license from
[borgmoea.org](http://borgmoea.org/) and manually copied into
`NYCOptimization/lib/borg/`:
- `borg.c`, `borg.h`, `borgmm.c`, `borgmm.h`, `mt19937ar.c`, `borg.py`

`borgmm.c` includes `borg.c` and `borgmm.h`, and `borg.py` loads both the
serial and the multi-master shared libraries (`src/mmborg.py` requires both).
Build them (the serial line is the Borg distribution's own build command; Anvil's
default module set provides `mpicc`, run `module load gcc openmpi` first if it is
missing):

```bash
gcc   -shared -fPIC -O3 -o lib/borg/libborg.so   lib/borg/borg.c   lib/borg/mt19937ar.c -lm
mpicc -shared -fPIC -O3 -o lib/borg/libborgmm.so lib/borg/borgmm.c lib/borg/mt19937ar.c -lm
```

Also, the experiment requires MOEAFramework 5.0, which provides both the `cli`
tool (used by the diagnostics step) and the framework JARs on the compile
classpath. MOEAFramework 5.0 is built for **Java 17+**, which Anvil's `openjdk`
modules do not provide (they cap at Java 11). Install a JDK 17 into the `venv`
conda env so `java`/`javac`/`jar` are on `PATH` whenever the env is active:

```bash
conda install -n venv -c conda-forge openjdk=17
```

Download the MOEAFramework 5.0 release tarball from
[github.com/MOEAFramework/MOEAFramework](https://github.com/MOEAFramework/MOEAFramework/releases),
place it at `NYCOptimization/MOEAFramework-5.0/MOEAFramework-5.0.tar.gz`, then
unpack it in place (the tarball nests everything under a `MOEAFramework-5.0/`
prefix, so `--strip-components=1` lands `lib/`, `cli`, etc. directly in the
existing directory) and make the CLI executable:

```bash
cd MOEAFramework-5.0
tar -xzf MOEAFramework-5.0.tar.gz --strip-components=1
chmod +x cli
cd ..
./MOEAFramework-5.0/cli --version   # should print 5.0
```

Finally, build one MOEAFramework problem JAR per formulation (rerun this after
changing the objective set or formulation list). Requires the Java 17 JDK above
and the `venv` env active:

```bash
bash workflow/00_setup_borg_jars.sh
```

#### 1.4 Pre-simulation

Run the full Pywr-DRB model once to save the non-NYC (STARFIT) reservoir
releases used as boundary conditions by the trimmed optimization model:

```bash
sbatch workflow/01_generate_presim.sh
```

### 2. Scenario Generation, Optimization & Re-evaluation

#### 2.1 Search-ensemble staging (steps 02–04)

The `historic` scenario design uses the single observed trace and **skips this
section**. `monte_carlo` generates its own realizations (step `02`);
`hazard_filling_stationary` selects its search ensemble from its own P = 10⁶
candidate pool (step `03`); each draw is then formatted into pywrdrb HDF5
inputs (step `04`). The array index in `02`/`04` is the ensemble-draw index *k*
(`K = design.n_ensemble_draws` = 2 staged draws: d0 is searched, d1 serves
the SI draw-sensitivity re-evaluation); sizing and seeds come from the design
registry, never from the command line.

```bash
# monte_carlo: one N x L ensemble per draw (step 02, then 04, chained; --dry-run prints the lines)
bash workflow/submit_search_staging.sh monte_carlo

# hazard_filling_stationary: the P = 1e6 pool is built sharded (50 array tasks), merged, and verified
sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50 \
       --array=0-49 workflow/supplemental/staging/pool_shards.sh
sbatch --dependency=afterok:<shards_jobid> \
       --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,NYCOPT_POOL_SLUG=statpool_10yr_n1000000_d0 \
       workflow/supplemental/staging/pool_merge.sh
sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,NYCOPT_ENSEMBLE_DRAW=0,NYCOPT_POOL_SLUG=statpool_10yr_n1000000_d0,NYCOPT_NESTEDP_SMOKE_SLUG=statpool_10yr_n2000_d0 \
       workflow/supplemental/staging/pool_verify.sh
# then select all K draws from the pool in one job (step 03), and prep each draw (04)
bash workflow/submit_search_staging.sh hazard_filling_stationary
```

Step `02` alone builds the small (P = 2,000) pool the step-03 registry default
points at; the helper exports `NYCOPT_CANDIDATE_POOL_N=1000000` to step `03` for
the campaign pool. Run `workflow/supplemental/checks/validate_staged_seasonality.sh`
on each staged ensemble as build QC.

#### 2.2 Test-ensemble staging (step 05 + supplemental)

The held-out re-evaluation ensemble E_test (`etest_kn_50yr_n12500`: 500 LHS
SOWs over the three climate amplitudes and the NYC entitlement utilization `u`,
× 25 realizations × 50 yr, 25 staged chunks) is independent of the
search-ensemble staging and never enters search. Build it sharded, merge, calibrate the demand
scale and append the per-SOW utilization column (metadata only;
`docs/notes/methods/forcing_parameterization.md`), stage the pywrdrb inputs per
chunk (the one-time presim pass over E_test, including each SOW's scaled NYC
demand series), then merge the realized utilization. The campaign E_test is cut
from the larger design generated 2026-08-19 (`etest_kn_50yr_n25000`, 50 chunks)
by `staging/etest_cut.sh` (metadata only; nothing reads the parent afterwards).
`workflow/submit_etest_staging.sh` chains the completion steps after the merge:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/staging/etest_shards.sh                       # or serial: workflow/05_generate_test_ensemble.sh
sbatch --dependency=afterok:<shards_jobid> \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/staging/etest_merge.sh
sbatch --dependency=afterok:<merge_jobid> \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/staging/etest_demand_utilization.sh                 # k(u) table + per-SOW utilization u, metadata only
sbatch --dependency=afterok:<demand_u_jobid> \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
       workflow/supplemental/staging/etest_prep_chunks.sh                        # step-04 inputs + scaled NYC demand per chunk
sbatch --dependency=afterok:<prep_jobid> \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/staging/etest_cut.sh                       # cut etest_kn_50yr_n12500 from the parent + realized u per SOW
bash workflow/submit_etest_staging.sh hazard --dependency=afterok:<cut_jobid>  # sub-window hazard image (shards, then merge)
```

Every re-evaluation step (`10`, `11`, `12`, and the supplemental re-evaluation and scenario-discovery wrappers) takes
`NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500` explicitly. It
is deliberately never defaulted and never set in an env file, so cross-design
comparability is a recorded choice.

#### 2.3 Baseline (step 06)

Evaluate the default (unoptimized) 2017 FFMP policy on the same objective set,
the comparison anchor for every optimized Pareto set. Run it once per campaign
env file, scenario-matched (`--search-ensemble`) for the two ensemble designs:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
       workflow/06_run_baseline.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env \
       workflow/06_run_baseline.sh --search-ensemble
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazard_filling_stationary_production.env \
       workflow/06_run_baseline.sh --search-ensemble
```

The incumbent's per-SOW E_test cube, which the incumbent-relative regret family
scores against, is simulated once for all designs by step `10` with
`NYCOPT_CHUNK_POLICIES=baseline` (section 2.5).

#### 2.4 MM-Borg optimization (step 07)

Each optimization is **one independent sbatch job** per (env file × seed). The
campaign runs S = 2 seeds per design on draw 0 under the `production` MOEA
config (4 islands × 382 workers = 1,533 ranks on 12 nodes); seed 1 runs 750k
NFE and seed 2 500k, and both are reported at 500k
(`docs/notes/methods/campaign_design.md`). The step-07 header is sized for the
smaller `mm_moderate` config, so the production geometry (12 × 128) and the
per-seed `--time` are held by `workflow/submit_production_search.sh`, the one
submission path for every campaign search (its header documents the sbatch
line it issues; `--dry-run` prints it):

```bash
# Matched designs (monte_carlo, hazard_filling_stationary): seed 1 then, once seed 1 has priced the campaign, seed 2
bash workflow/submit_production_search.sh monte_carlo 1               # 96 h
bash workflow/submit_production_search.sh monte_carlo 2               # 72 h
bash workflow/submit_production_search.sh hazard_filling_stationary 1
bash workflow/submit_production_search.sh hazard_filling_stationary 2

# Historic reference: single trace, NFE-bounded, ~15x cheaper
bash workflow/submit_production_search.sh historic 1                  # 12 h
bash workflow/submit_production_search.sh historic 2                  # 8 h
```

Every job starts with a config-derived pre-flight that echoes the resolved run
identity (scenario design, ensemble, MOEA config, objectives) to the job log
and aborts before burning the allocation if the design's ensemble is not
staged, the allocation is too small, or the projected node memory exceeds the
safety line. A full reproducibility manifest is written to
`outputs/run_manifests/`.

The variable-resolution FFMP sweep (`workflow/envs/ffmp_vr_obj8.env`,
`FORMULATION=ffmp_8/10/12`, same launcher) is a conditional SI extension run
only on leftover allocation; it is not part of the manuscript campaign.

#### 2.5 Reference sets, diagnostics, re-evaluation, comparison

After both seeds of a design finish, build the equal-NFE reference set, run the
runtime diagnostics, re-evaluate on E_test (chunked, metrics-only), merge, then
compare designs and render figures. Convergence diagnostics are read from the
MM Borg coordinator archive, one trace per seed on total NFE: the controller
writes no runtime file, so step 09 reconstructs its archive at every snapshot
as the epsilon-box merge of the island runtime files (the project's one
epsilon-archive implementation, `src/sensitivity_common.py`; MOEAFramework's
v5.0 mergers do not honour the epsilon vector) and scores that series with
MOEAFramework's `MetricsEvaluator`. Step 09 writes its own
reference set (`{slug}_diagnostics_reference.set`) and never the campaign
`{slug}_merged.set`, so the two commands below can run in either order:

```bash
# Equal-NFE cross-seed reference set (installs {slug}_merged.set, the step-10 reference)
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env workflow/08_extract_reference_set.sh --seed 1
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env workflow/08_extract_reference_set.sh --merge --install

# MOEAFramework runtime diagnostics on the per-seed coordinator archive (six indicators vs total NFE)
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env workflow/09_run_diagnostics.sh

# Chunked E_test re-evaluation of the reference set (shared partition, 16 ranks x 8 cpus, batch 50), then merge
sbatch --partition=shared --nodes=1 --ntasks=16 --cpus-per-task=8 --time=24:00:00 \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500,NYCOPT_CHUNK_POLICIES=outputs/monte_carlo/ffmp_obj8/sets/ffmp_obj8_merged.set,NYCOPT_CHUNK_MERGE=off,NYCOPT_SEARCH_REALIZATION_BATCH=50 \
       workflow/10_simulate_test_chunks.sh          # resumable: resubmit the same line until every unit is written
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_monte_carlo_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500,NYCOPT_CHUNK_POLICIES=outputs/monte_carlo/ffmp_obj8/sets/ffmp_obj8_merged.set \
       workflow/11_merge_test_chunks.sh
```

Repeat for each campaign env file with the **same** preset
(`supplemental/results_diagnostics/reevaluate_unchunked.sh` is the unchunked
alternative that also persists simulation timeseries). Then:

```bash
sbatch --export=ALL,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500 workflow/12_compare_designs.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazard_filling_stationary_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500 \
       workflow/supplemental/results_diagnostics/scenario_discovery.sh
sbatch workflow/13_main_figures.sh
sbatch workflow/14_results_figures.sh
```

Step order: `01` before `06`/`07`; `02`→`04` (with `03` for hazard filling)
before `07`; `05`→`staging/etest_demand_utilization`→`staging/etest_prep_chunks`→`staging/etest_cut`
before `10`; `07`→`08`→`09`→`10`→`11`→`12`/`13`/`14`. Chain on a cluster with
`sbatch --dependency=afterok:<jobid>`. Outputs land under
`outputs/{scenario}/{moea_slug}/`.

## Run axes

Every run = **scenario design** (`src/scenario_designs.py`, `NYCOPT_SCENARIO_DESIGN`)
× **MOEA config** (`src/moea_config.py`, `NYCOPT_MOEA_CONFIG`), selected via the
env file — no value flags. MM-Borg ranks = `1 + n_islands*(workers+1)` (set by
the MOEA config; `workflow/_common.sh` reads it back so shell and Python agree
on one source of truth).

## Development utilities

Not part of replication. `bash workflow/submit_smoke.sh [--dry-run]` submits a
tiny-NFE end-to-end pipeline check per formulation (Anvil `debug` queue,
2 nodes × 40 tasks, ≤2 h, `smoke` MOEA config, short 2018–2022 window), then
`bash workflow/09_run_diagnostics.sh smoke_ffmp ...` for its diagnostics.
`bash workflow/submit_search_memory_smoke.sh` is the one-node memory/timing
smoke of the batched N = 300 search path. Off-pipeline staging, calibration,
sizing, and diagnostic drivers live in `workflow/supplemental/`.

## Outputs are not tracked in git

Everything under `outputs/` is ignored (`.gitignore`: `outputs/`). Result
CSVs, Borg reference sets, re-evaluation tables, manifests, figures and HDF5
are all machine-local: they are regenerated from the tracked `workflow/`
scripts and `workflow/envs/*.env` files, so a laptop artefact can never
masquerade as an Anvil result. Nothing in `outputs/` is pushed, and `git
status` will never show changes there. The only synced figures are the final
manuscript set in `figures/manuscript/`.

To move a specific result between machines, copy it out of band (e.g.
`rsync -av anvil:.../outputs/comparison/ outputs/comparison/`) rather than
committing it. Large staged ensembles (the E_test chunk directories) live in
Anvil project space and are symlinked into `outputs/synthetic_ensembles/`.

## Staging requirements

The `historic` scenario design runs end-to-end with no staged data. Every
other design resolves once its ensemble is staged: `monte_carlo` via
steps `02`+`04`, and `hazard_filling_stationary` via its P = 10⁶ pool
(`staging/pool_shards.sh` → `staging/pool_merge.sh` → `pool_verify.sh`), then step `03`
with `NYCOPT_CANDIDATE_POOL_N=1000000`, then step `04`. Until staged, the
MM-Borg pre-flight fails fast with a staging message.
