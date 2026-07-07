# NYCOptimization

Multi-objective optimization of NYC reservoir operations (Pywr-DRB + MM-Borg),
focused on the design of the streamflow ensembles used during MOEA evaluation.
Method + design notes: `docs/notes/methods/experimental_design.md`; conventions:
`.claude/CLAUDE.md`.

## Experimental Replication

> **Note:** This experiment was run on the Anvil HPC at Purdue and requires access to N (TBD) compute nodes with M core each.  The instructions below correspond to this HPC, but may be adapted for other computing infrastructure.  For instructions on running smaller experiments (e.g., smoke tests and/or small optimization tests) on the Hopper HPC at Cornell can be found in docs/hopper_experimental_instructions.md.

### 1.0 Setup 

#### 1.1 Clone suporting repos (all repos share one parent folder)

```bash
git clone -b nyc_opt https://github.com/Pywr-DRB/Pywr-DRB.git
git clone https://github.com/TrevorJA/SynHydro.git
git clone https://github.com/TrevorJA/NYCOptimization_scenario_generation.git
git clone https://github.com/TrevorJA/NYCOptimization.git
cd NYCOptimization
```

#### 1.2 Environment

```bash

module load anaconda
conda create -n venv python=3.11.5
conda activate venv
pip install -r requirements.txt
```

Note that the `requirements.txt` installs an 'editable' installation of all sibling repos (../SynHydro, ../Pywr-DRB, ../NYCOptimization_scenario_generation). 


#### 1.3 BorgMOEA (licensed; not in git) and MOEAFramework

The following BorgMOEA source code files must be obtained HERE and manually copied into `NYCOptimization/lib/borg/`:
- `borgmm.c`, `mt19937ar.c`, `borg.py`

Then, once these are placed in the folder, run the following to compile:

```bash
mpicc -shared -fPIC -O3 -o lib/borg/libborgmm.so lib/borg/borgmm.c lib/borg/mt19937ar.c -lm
```

Also, the experiment requires the MOEAFramework 5.0 CLI located at `NYCOptimization/MOEAFramework-5.0/cli`. MOEAFramework can be accessed HERE.

#### 1.4 Run historic baseline simulation




### 2. Scenario Generation, Optimization & Re-evaluation

#### 2.1 Stochastic ensemble generation

```bash
# Preprocessing
sbatch workflow/00_generate_presim.sh

# Generate stochastic flow ensemble 
sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ensemble_kn_short.env workflow/01_generate_stochastic_ensemble.sh

# Prepare Pywr-DRB inputs
srun python scripts/main/prep_pywrdrb_inputs.py --preset $R
```

#### 2.2 BorgMOEA Optimization of NYC FFMP rules

First, we need to specify which configuration/problem formulation we want to use for optimization (objectives, decision variables, etc). 

```bash
E=slurm/envs/ffmp_obj7_historic.env    # mm_full (50k NFE);
R=kn_5yr_n200                          # held-out re-eval ensemble
```

```bash
# Run the 
sbatch --export=ALL,NYCOPT_ENV_FILE=$E,NYCOPT_REEVAL_ENSEMBLE_PRESET=$R workflow/04_run_baseline.sh
```

The first optimization will run the baseline FFMP formulation using 200 realizations of 5-year stochastic streamflows for each function evaluation.

==CONTINUE INSTRUCTIONS AFTER CLEANUP AND REORGANIZATION OF MAIN SCRIPT STEPS.==

```bash
# MM-Borg optimization search
sbatch --export=ALL,NYCOPT_ENV_FILE=$E slurm/main/mmborg_ffmp.sh

# runtime diagnostics
sbatch --export=ALL,NYCOPT_ENV_FILE=$E workflow/06_run_diagnostics.sh

# re-eval on held-out ensemble + robustness
sbatch --export=ALL,NYCOPT_ENV_FILE=$E,NYCOPT_REEVAL_ENSEMBLE_PRESET=$R,NYCOPT_REEVAL_SCORE=1 slurm/main/reevaluate_ensemble.sh ffmp
```

Order: `00` before `05`; `01`→`03` before `04`/`07`. Chain on a cluster with
`sbatch --dependency=afterok:<jobid>`. Outputs land under
`outputs/{scenario}/{moea_slug}/`.

## Run axes

Every run = **scenario design** (`src/scenario_designs.py`, `NYCOPT_SCENARIO_DESIGN`)
× **MOEA config** (`src/moea_config.py`, `NYCOPT_MOEA_CONFIG`), selected via the
env file — no value flags. MM-Borg ranks = `1 + n_islands*(workers+1)` (set by the
MOEA config; `_common.sh` reads it back). Sweep FFMP-VR layer configs (`ffmp_8/10/12`)
and multiple seeds/env files via `slurm/main/submit_all.sh`.

## Pending

Only the `historic` scenario design runs end-to-end today. The five other designs
(`fixed_probabilistic_short/long`, `resampled_probabilistic`, `input_stratified`,
`hazard_filling`) and the LHS-subsample step (`workflow/02`) are not yet wired
(`resolve_search_spec` raises `NotImplementedError`); the forcing/hazard-filling
designs additionally require the `scengen` master-ensemble + subsample steps.
