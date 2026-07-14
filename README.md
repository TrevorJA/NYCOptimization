# NYCOptimization

Multi-objective optimization of NYC reservoir operations (Pywr-DRB + MM-Borg),
focused on the design of the streamflow ensembles used during MOEA evaluation.
Method + design notes: `docs/notes/methods/experimental_design.md`;
per-step details: `workflow/README.md`.

## Experimental Replication

> **Note:** This experiment was run on the [Anvil HPC at Purdue](https://docs.rcac.purdue.edu/userguides/anvil/)
> (128-core nodes; search jobs use 5 nodes × 33 tasks on the `wholenode`
> partition). The instructions below correspond to this HPC, but may be adapted
> for other computing infrastructure. Submit all jobs **from the repo root** —
> the scripts resolve paths from the submission directory.

**Anvil specifics** (already encoded in the scripts' `#SBATCH` headers, listed
here so you know what to expect):

- **Allocation account is mandatory.** It is hardcoded in every sbatch
  script's header (`#SBATCH --account=x-tamestoy`) — no per-shell export
  needed. Override for a different allocation with `sbatch -A <alloc> ...`.
- **Partitions**: small serial steps use `shared` (per-core billing, max 1
  node); multi-node MPI jobs (`06`, `08`, supplemental sweeps) use
  `wholenode` (node-exclusive billing); the smoke test uses `debug` (2 nodes,
  2 h max). These are set in each script's `#SBATCH --partition` line.
- **96-hour wall-time cap.** Anvil's `wholenode` maximum is 96 h per job, and
  the MM-Borg launcher requests exactly that. A search that needs longer must
  restart from its periodic runtime snapshots
  (`outputs/{scenario}/{slug}/runtime/`) rather than request more time.

Everything submittable lives in `workflow/` (numbered steps `00`–`09`). A run's
identity — scenario design, MOEA config, objectives, physics toggles — comes
from a single env file under `workflow/envs/`; the scripts take no
value-carrying CLI flags (see [workflow/envs/README.md](workflow/envs/README.md)).

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

The workflow scripts activate `./venv/bin/activate` if that directory exists;
with a conda environment (as above) there is no `./venv`, so activate the conda
env in your shell before submitting jobs.

#### 1.3 BorgMOEA (licensed; not in git) and MOEAFramework

The following BorgMOEA source code files must be obtained under license from
[borgmoea.org](http://borgmoea.org/) and manually copied into
`NYCOptimization/lib/borg/`:
- `borgmm.c`, `mt19937ar.c`, `borg.py`

Then, once these are placed in the folder, run the following to compile
(Anvil's default module set provides `mpicc`; run `module load gcc openmpi`
first if it is missing):

```bash
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

#### 1.4 Pre-simulation and baseline

Run the full Pywr-DRB model once to save the non-NYC (STARFIT) reservoir
releases used as boundary conditions by the trimmed optimization model:

```bash
sbatch workflow/01_generate_presim.sh
```

Then evaluate the **baseline**: the default (unoptimized) FFMP policy scored on
the same objective set. This is the comparison anchor for all optimized Pareto
sets, and it also persists the baseline re-evaluation matrix on the common
held-out ensemble so step `08` can compute regret-from-baseline:

```bash
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200 \
       workflow/05_run_baseline.sh
```

The second (re-eval) pass of this job requires the held-out ensemble
(`kn_5yr_n200`) to be staged first — see §2.1. To run the baseline before
staging it, add `NYCOPT_BASELINE_SKIP_REEVAL=1` to the `--export` list and
rerun step `05` later.

### 2. Scenario Generation, Optimization & Re-evaluation

#### 2.1 Ensemble generation and staging

The `historic` scenario design uses the single observed trace and **skips this
section**. Every other design **generates its own realizations** (step `02`),
the hazard-filling designs then select their search ensemble from their own
candidate pool (step `03`), and each draw is formatted into pywrdrb HDF5 inputs
(step `04`). The array index in `02`/`04` is the ensemble-draw index *k*
(`K = design.n_ensemble_draws`); sizing and seeds come from the design registry,
never from the command line:

```bash
# Generate the design's own realizations (or its candidate pool), one array task per draw
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_du \
       workflow/02_generate_ensemble.sh

# Hazard-filling designs only: select N members from the design's own pool (all K draws in one job)
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_du \
       workflow/03_subsample_ensemble.sh

# Format each draw's search ensemble into pywrdrb inputs
sbatch --export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_du \
       --array=0-9 workflow/04_prep_pywrdrb_inputs.sh
```

The held-out re-evaluation ensemble is staged the same way (steps `02` + `04`
with `--preset`); see `workflow/README.md`.

#### 2.2 BorgMOEA optimization of NYC FFMP rules

Each optimization is **one independent sbatch job**: one submission per
(env file × formulation), with `--array=1-10` spawning the 10 seed replicates
as independent array tasks. These are multi-day jobs requesting Anvil's
96-hour `wholenode` maximum; submit whichever experiments you are replicating —
they do not depend on each other:

```bash
# Base FFMP (24 DVs)
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env               --array=1-10 workflow/06_run_mmborg.sh
# Variable-resolution FFMP sweep (same objective set), one job per N
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_8  --array=1-10 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_10 --array=1-10 workflow/06_run_mmborg.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_vr_obj7.env,FORMULATION=ffmp_12 --array=1-10 workflow/06_run_mmborg.sh
```

(This experiment list grows as new env files / scenario designs are added.)
Every job starts with a config-derived pre-flight that echoes the resolved run
identity (scenario design, ensemble, MOEA config, objectives) to the job log
and aborts before burning the allocation if the design's ensemble is not
staged. A full reproducibility manifest is written to
`outputs/run_manifests/`.

#### 2.3 Diagnostics and re-evaluation

```bash
# MOEAFramework runtime diagnostics (hypervolume, generational distance, reference set)
bash workflow/07_run_diagnostics.sh

# Re-evaluate the Pareto policies on the common held-out ensemble + robustness scoring
sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_historic.env,NYCOPT_REEVAL_ENSEMBLE_PRESET=kn_5yr_n200,NYCOPT_REEVAL_SCORE=1 \
       workflow/08_reevaluate.sh
```

`NYCOPT_REEVAL_ENSEMBLE_PRESET` is required explicitly (never defaulted) so
that cross-design comparability is a recorded choice. Repeat `08` once per
env file being compared, with the **same** preset.

Step order: `01` before `05`/`06`; `02`→`04` before `06` for ensemble designs;
`06` before `07`/`08`. Chain on a cluster with
`sbatch --dependency=afterok:<jobid>`. Outputs land under
`outputs/{scenario}/{moea_slug}/`.

## Run axes

Every run = **scenario design** (`src/scenario_designs.py`, `NYCOPT_SCENARIO_DESIGN`)
× **MOEA config** (`src/moea_config.py`, `NYCOPT_MOEA_CONFIG`), selected via the
env file — no value flags. MM-Borg ranks = `1 + n_islands*(workers+1)` (set by
the MOEA config; `workflow/_common.sh` reads it back so shell and Python agree
on one source of truth). Variable-resolution FFMP sweeps use the same launcher
with `FORMULATION=ffmp_8/10/12`.

## Development utilities

Not part of replication. `bash workflow/submit_smoke.sh [--dry-run]` submits a
tiny-NFE end-to-end pipeline check per formulation (Anvil `debug` queue,
2 nodes × 40 tasks, ≤2 h, `smoke` MOEA config, short 2018–2022 window), then
`bash workflow/07_run_diagnostics.sh smoke_ffmp ...` for its diagnostics.
Off-pipeline diagnostics (benchmarks, objective-sensitivity sweeps) live in
`workflow/supplemental/`.

## Pending

The `historic` scenario design runs end-to-end today. The fixed/resampled
probabilistic designs are code-wired and resolve once their Kirsch-Nowak
ensembles are staged (steps `02`+`04`). The forcing-master designs
(`input_stratified`, `hazard_filling`, `hazard_filling_absolute`) are
code-wired but require the `scengen` master-ensemble + subsample staging
(steps `02`–`04`); until staged, the MM-Borg pre-flight fails fast with a
staging message.
