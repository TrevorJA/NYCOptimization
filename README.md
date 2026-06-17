# NYCOptimization

Trevor Amestoy — Cornell University, Reed Research Group

## Scope

Multi-objective optimization of NYC reservoir operations implemented in
the Pywr-DRB model, with a methodological focus on the **design of the
streamflow ensembles used during MOEA evaluation**.

**Hypothesis.** Optimizing with a hydrologic-metric **space-filling**
ensemble (LHS-subsampled from a larger stochastic ensemble) yields more
robust MOEA results than a random probabilistic ensemble of the same size.

The planned experiment crosses FFMP-layer configurations (base FFMP +
variable-resolution FFMP at N ∈ {8, 10, 12}) with the six **scenario designs**
for the MOEA evaluation ensemble (`src/scenario_designs.py`, `SCENARIO_DESIGNS`;
see `docs/notes/methods/experimental_design.md`). Each run is specified by two
named identifiers — a scenario design and a MOEA algorithm config
(`src/moea_config.py`) — and outputs land under
`outputs/{scenario}/{moea_slug}/`.

## Setup

```bash
module load python/3.11.5
python3 -m venv venv
source venv/bin/activate
pip install -e ../Pywr-DRB     # nyc_opt branch
pip install -r requirements.txt
```

Compile Borg (place `borgmm.c`, `mt19937ar.c`, and `borg.py` under
`lib/borg/` first):

```bash
mpicc -shared -fPIC -O3 -o lib/borg/libborgmm.so \
    lib/borg/borgmm.c lib/borg/mt19937ar.c -lm
```

## Workflow

Numbered bash scripts in `workflow/` form the pipeline; each wraps a
Python module under `scripts/`.

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `00_generate_presim.sh` | Run full Pywr-DRB once; save STARFIT releases for the trimmed model |
| 1 | `01_generate_stochastic_ensemble.sh` | Generate the large stochastic streamflow ensemble (**stub**) |
| 2 | `02_subsample_lhs_ensemble.sh` | LHS subsample over hydrologic-metric space (**stub**) |
| 3 | `03_prep_pywrdrb_inputs.sh` | Format subsampled ensembles into pywrdrb HDF5 inputs (**stub**) |
| 4 | `04_run_baseline.sh` | Evaluate the default FFMP policy |
| 5 | `05_run_mmborg.sh` | Launch MM-Borg MOEA optimization via MPI |
| 6 | `06_run_diagnostics.sh` | MOEAFramework v5.0 runtime diagnostics |
| 7 | `07_reevaluate.sh` | Re-simulate Pareto solutions with the full model |

For HPC campaigns across all FFMP layer configs and ensemble presets,
use `slurm/main/submit_all.sh` with a per-experiment env file from
`slurm/envs/`.

## MM Borg MPI sizing

`ntasks = 1 + N_ISLANDS * (workers_per_island + 1)`; `maxEvaluations`
is per island. Thread-pinning env vars (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
etc.) are set in `slurm/main/_common.sh` to prevent BLAS contention across ranks.

## Repository structure

```
NYCOptimization/
├── config.py                  # paths, constants, run-axis selection, slug grammar
├── src/
│   ├── simulation.py          # DVs -> objectives
│   ├── objectives.py          # objective registry + ObjectiveSet
│   ├── objectives_ensemble.py # ensemble-aware objectives
│   ├── mmborg.py, mmborg_cli.py
│   ├── diagnostics.py         # MOEAFramework v5.0 pipeline
│   ├── reevaluate.py, reevaluate_mpi.py
│   ├── ensembles.py
│   ├── formulations/          # FFMP + variable-resolution FFMP
│   ├── load/
│   └── plotting/
├── scripts/
│   ├── main/                  # production pipeline scripts (called by workflow/)
│   ├── supplemental/          # benchmarks, diagnostic samplers, smoke tests
│   └── temporary/             # ad-hoc / non-manuscript sandbox
├── workflow/                  # numbered 00..07 pipeline
├── slurm/                     # FFMP and FFMP-VR SLURM templates + envs/
├── tests/                     # simulation/ensemble/objective tests
├── figures/                   # (replanned — empty placeholder)
└── docs/notes/                # literature_review.md
```

Outputs (`outputs/`) and Borg sources (`lib/borg/`) are gitignored.

## Related repositories

- `../Pywr-DRB` (branch `nyc_opt`) — the simulation model. Installed
  editably into the venv.
- `../StochasticExploratoryExperiment` — reference code for stochastic
  streamflow ensemble generation.

## References

Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg
multiobjective evolutionary algorithm to enhance the management of
complex environmental systems. *Environmental Modelling & Software*, 69.

Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An
open-source Python model for water availability and drought risk
assessment. *Environmental Modelling & Software*, 106185.

Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013).
Many objective robust decision making for complex environmental
systems undergoing change. *Environmental Modelling & Software*, 42.
