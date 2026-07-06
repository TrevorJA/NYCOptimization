# NYCOptimization

Multi-objective optimization of NYC reservoir operations (Pywr-DRB + MM-Borg),
focused on the design of the streamflow ensembles used during MOEA evaluation.
Method + design notes: `docs/notes/methods/experimental_design.md`; conventions:
`.claude/CLAUDE.md`.

## Replication (from a fresh HPC directory)

### 1. Clone (all repos share one parent)

```bash
git clone -b nyc_opt https://github.com/Pywr-DRB/Pywr-DRB.git
git clone https://github.com/TrevorJA/SynHydro.git
git clone https://github.com/TrevorJA/NYCOptimization_scenario_generation.git
git clone https://github.com/TrevorJA/NYCOptimization.git
cd NYCOptimization
```

### 2. Environment

```bash
module load python/3.11.5
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # installs the 3 sibling repos editable + deps
```

### 3. Borg (licensed; not in git)

Place `borgmm.c`, `mt19937ar.c`, `borg.py` under `lib/borg/`, then:

```bash
mpicc -shared -fPIC -O3 -o lib/borg/libborgmm.so \
    lib/borg/borgmm.c lib/borg/mt19937ar.c -lm
```

Step 6 additionally needs the MOEAFramework 5.0 CLI at `MOEAFramework-5.0/cli`.

### 4. Run the historic baseline experiment

```bash
E=slurm/envs/ffmp_obj7_historic.env    # mm_full (50k NFE); ffmp_obj7_historic_pilot.env = 5k-NFE shakeout
R=kn_5yr_n200                          # held-out re-eval ensemble (staged in step 1)

# --- staging (once) ---
sbatch workflow/00_generate_presim.sh                                                  # STARFIT releases for trimmed model
sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ensemble_kn_short.env \
       workflow/01_generate_stochastic_ensemble.sh                                     # -> kn_5yr_n200
srun python scripts/main/prep_pywrdrb_inputs.py --preset $R                            # prep re-eval ensemble for trimmed model

# --- experiment ---
sbatch --export=ALL,NYCOPT_ENV_FILE=$E,NYCOPT_REEVAL_ENSEMBLE_PRESET=$R workflow/04_run_baseline.sh
sbatch --export=ALL,NYCOPT_ENV_FILE=$E slurm/main/mmborg_ffmp.sh                       # MM-Borg search
sbatch --export=ALL,NYCOPT_ENV_FILE=$E workflow/06_run_diagnostics.sh                  # runtime diagnostics
sbatch --export=ALL,NYCOPT_ENV_FILE=$E,NYCOPT_REEVAL_ENSEMBLE_PRESET=$R,NYCOPT_REEVAL_SCORE=1 \
       slurm/main/reevaluate_ensemble.sh ffmp                                          # re-eval on held-out ensemble + robustness
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
