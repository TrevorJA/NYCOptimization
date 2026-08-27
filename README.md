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
  node); multi-node MPI jobs (`06`, `08`, `09`, supplemental sweeps) use
  `wholenode` (node-exclusive billing); the smoke test uses `debug` (2 nodes,
  2 h max). These are set in each script's `#SBATCH --partition` line.
- **96-hour wall-time cap.** Anvil's `wholenode` maximum is 96 h per job.
  There is no resume: the runtime files (`outputs/{scenario}/{slug}/runtime/`)
  are diagnostic dumps and the Borg checkpoint is disabled, so every search is
  sized to finish inside one job (`docs/notes/methods/campaign_design.md`).

Everything submittable lives in `workflow/` (numbered steps `00`–`14`, plus
`09b`). A run's identity — scenario design, MOEA config, objectives, physics
toggles — comes from a single env file under `workflow/envs/`; the scripts take
no value-carrying CLI flags (see [workflow/envs/README.md](workflow/envs/README.md)).
The campaign identities are the three `workflow/envs/ffmp_obj8_*_production.env`
files (`historic`, `fixedprob`, `hazfill_stat`).

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
section**. `fixed_probabilistic` generates its own realizations (step `02`);
`hazard_filling_stationary` selects its search ensemble from its own P = 10⁶
candidate pool (step `03`); each draw is then formatted into pywrdrb HDF5
inputs (step `04`). The array index in `02`/`04` is the ensemble-draw index *k*
(`K = design.n_ensemble_draws` = 3 staged draws: d0 is searched, d1–d2 serve
the SI draw-sensitivity re-evaluation); sizing and seeds come from the design
registry, never from the command line.

```bash
# fixed_probabilistic: one N x L ensemble per draw
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=fixed_probabilistic --array=0-2 workflow/02_generate_ensemble.sh
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=fixed_probabilistic --array=0-2 workflow/04_prep_pywrdrb_inputs.sh

# hazard_filling_stationary: the P = 1e6 pool is built sharded (50 array tasks), merged, and verified
sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50 \
       --array=0-49 workflow/supplemental/gen_pool_shards.sh
sbatch --dependency=afterok:<shards_jobid> \
       --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,NYCOPT_NESTEDP_POOL_SLUG=statpool_10yr_n1000000_d0 \
       workflow/supplemental/gen_pool_merge.sh
sbatch --export=ALL,NYCOPT_CANDIDATE_POOL_N=1000000,NYCOPT_ENSEMBLE_SHARD_COUNT=50,NYCOPT_ENSEMBLE_DRAW=0,NYCOPT_NESTEDP_POOL_SLUG=statpool_10yr_n1000000_d0,NYCOPT_NESTEDP_SMOKE_SLUG=statpool_10yr_n2000_d0 \
       workflow/supplemental/pool_verify.sh
# then select all K draws from the pool in one job, and prep each draw
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary,NYCOPT_CANDIDATE_POOL_N=1000000 \
       workflow/03_subsample_ensemble.sh
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary --array=0-2 workflow/04_prep_pywrdrb_inputs.sh
```

Step `02` alone builds the small (P = 2,000) pool the step-03 registry default
points at; `NYCOPT_CANDIDATE_POOL_N=1000000` must be exported to step `03` for
the campaign pool. Run `scripts/supplemental/validate_staged_seasonality.py`
on each staged ensemble as build QC.

#### 2.2 Test-ensemble staging (step 12 + supplemental)

The held-out re-evaluation ensemble E_test (`etest_kn_50yr_n25000`: 1,000 LHS
SOWs × 25 realizations × 50 yr, 50 staged chunks) is independent of steps
02–07 and never enters search. Build it sharded, merge, stage the pywrdrb
inputs per chunk (the one-time presim pass over E_test), then stage the
500-SOW campaign prefix subset (metadata only, login node):

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/gen_etest_shards.sh                       # or serial: workflow/12_generate_test_ensemble.sh
sbatch --dependency=afterok:<shards_jobid> \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_ETEST_VARIANT=kn \
       workflow/supplemental/gen_etest_merge.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
       workflow/supplemental/prep_etest_chunks.sh
python3 -m scripts.supplemental.make_etest_subset --pool etest_kn_50yr_n25000   # -> etest_kn_50yr_n25000_first25ch
```

Every re-evaluation step (`05`, `08`, `09`, `09b`, `10`, `11`) takes
`NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch` explicitly. It
is deliberately never defaulted and never set in an env file, so cross-design
comparability is a recorded choice.

#### 2.3 Baseline (step 05)

Evaluate the default (unoptimized) 2017 FFMP policy on the same objective set.
This is the comparison anchor for every optimized Pareto set, and the job also
persists the incumbent's per-SOW matrix on E_test so the incumbent-relative
regret family can be scored. Run it once per campaign env file, scenario-matched
(`--search-ensemble`) for the two ensemble designs:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
       workflow/05_run_baseline.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
       workflow/05_run_baseline.sh --search-ensemble
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazfill_stat_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
       workflow/05_run_baseline.sh --search-ensemble
```

To run the baseline before E_test is staged, add `NYCOPT_BASELINE_SKIP_REEVAL=1`
to the `--export` list and rerun step `05` later.

#### 2.4 MM-Borg optimization (step 06)

Each optimization is **one independent sbatch job** per (env file × seed). The
campaign runs S = 2 seeds per design on draw 0 under the `production` MOEA
config (4 islands × 382 workers = 1,533 ranks on 12 nodes); seed 1 runs 750k
NFE and seed 2 500k, and both are reported at 500k
(`docs/notes/methods/campaign_design.md`). The step-06 header is sized for the
smaller `mm_moderate` config, so pass the production geometry and a per-seed
`--time` explicitly (these lines are also in each env file's header):

```bash
# Matched designs (fixedprob, hazfill_stat): seed 1 then, once seed 1 has priced the campaign, seed 2
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,DRAW=0 \
       --array=1 --nodes=12 --ntasks-per-node=128 --time=96:00:00 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,DRAW=0 \
       --array=2 --nodes=12 --ntasks-per-node=128 --time=72:00:00 workflow/06_run_mmborg.sh
# (same two lines with ffmp_obj8_hazfill_stat_production.env)

# Historic reference: single trace, NFE-bounded, ~15x cheaper
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
       --array=1 --nodes=12 --ntasks-per-node=128 --time=12:00:00 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
       --array=2 --nodes=12 --ntasks-per-node=128 --time=08:00:00 workflow/06_run_mmborg.sh
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
compare designs and render figures:

```bash
# Equal-NFE cross-seed reference set (installs {slug}_merged.set, the step-08/09 reference)
set -a; source workflow/envs/ffmp_obj8_fixedprob_production.env; set +a
python3 scripts/main/extract_runtime_archive.py --seed 1
python3 scripts/main/extract_runtime_archive.py --merge --install

# MOEAFramework runtime diagnostics (hypervolume, generational distance), per seed
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env workflow/07_run_diagnostics.sh

# Chunked E_test re-evaluation of the reference set (shared partition, 16 ranks x 8 cpus, batch 50), then merge
sbatch --partition=shared --nodes=1 --ntasks=16 --cpus-per-task=8 --time=24:00:00 \
       --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch,NYCOPT_CHUNK_POLICIES=outputs/fixed_probabilistic/ffmp_obj8/sets/ffmp_obj8_merged.set,NYCOPT_CHUNK_MERGE=off,NYCOPT_SEARCH_REALIZATION_BATCH=50 \
       workflow/09_simulate_test_chunks.sh          # resumable: resubmit the same line until every unit is written
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_fixedprob_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch,NYCOPT_CHUNK_POLICIES=outputs/fixed_probabilistic/ffmp_obj8/sets/ffmp_obj8_merged.set \
       workflow/09b_merge_test_chunks.sh
```

Repeat for each campaign env file with the **same** preset. Step `08`
(`08_reevaluate.sh`, `wholenode`) is the unchunked alternative that also
persists simulation timeseries; the campaign path is `09` + `09b`. Then:

```bash
sbatch --export=ALL,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch workflow/10_compare_designs.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_hazfill_stat_production.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch \
       workflow/11_scenario_discovery.sh
sbatch workflow/13_main_figures.sh
sbatch workflow/14_results_figures.sh
```

Step order: `01` before `05`/`06`; `02`→`04` (with `03` for hazard filling)
before `06`; `12`→`prep_etest_chunks`→`make_etest_subset` before `05`/`08`/`09`/`11`;
`06`→`extract_runtime_archive`→`07`→`09`→`09b`→`10`/`11`/`13`/`14`. Chain on a
cluster with `sbatch --dependency=afterok:<jobid>`. Outputs land under
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
`bash workflow/07_run_diagnostics.sh smoke_ffmp ...` for its diagnostics.
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
other design resolves once its ensemble is staged: `fixed_probabilistic` via
steps `02`+`04`, and `hazard_filling_stationary` via its P = 10⁶ pool
(`gen_pool_shards.sh` → `gen_pool_merge.sh` → `pool_verify.sh`), then step `03`
with `NYCOPT_CANDIDATE_POOL_N=1000000`, then step `04`. Until staged, the
MM-Borg pre-flight fails fast with a staging message.
