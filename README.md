

# NYCOptimization

Trevor Amestoy, PhD Candidate
Cornell University, Dept of Civil and Environmental Engineering
Reed Research Group

## Study Scope

This repository performs multi-objective optimization of NYC reservoir operations implemented in the Pywr-DRB model. The study seeks to identify sets of operational policies that outperform the current 2017 Flexible Flow Management Program (FFMP) rules across multiple competing objectives and hydrological conditions.

The study applies Many-Objective Robust Decision Making (MORDM) to: (1) discover Pareto-approximate tradeoff sets under a reference hydrological scenario, (2) compare alternative policy formulations of varying flexibility, and (3) evaluate policy robustness under stochastic streamflow ensembles.

### Policy Formulations

Three nested formulations are considered, each representing increasing operational flexibility:

- **Formulation A (Parameterized FFMP)**: Re-optimize the existing FFMP parameters (MRF baselines, drought factors, storage zone thresholds, flood release limits) within the current rule structure. ~25 decision variables.
- **Formulation B (FFMP + Enhanced Flexibility)**: Extend the FFMP framework with additional degrees of freedom (seasonal drought factors, asymmetric storage curves, dynamic blending weights). ~40-60 decision variables.
- **Formulation C (State-Aware Direct Policy Search)**: Replace rule-based operations with nonlinear policy functions (radial basis functions or multi-layer perceptrons) that map system state to release/diversion decisions. ~100-300 decision variables.


## Workflow

The workflow is organized as numbered bash scripts that should be run in order. Each script is a thin wrapper calling the corresponding Python module.


### Setup venv and dependencies

```bash
module load python/3.11.5
python3 -m venv venv
source venv/bin/activate
pip install -e ../Pywr-DRB # Requires local Pywr-DRB repo nyc_opt branch
```


### Step 0: Generate Pre-Simulated Releases (one-time setup)

Runs the full Pywr-DRB model once and saves non-NYC reservoir releases. These are used by the trimmed model during optimization to avoid re-simulating independent STARFIT reservoirs.

```bash
bash workflow/00_generate_presim.sh
```

Output: `outputs/presim/presimulated_releases_mgd.csv` (~5-10 min)

### Step 1: Evaluate Baseline

Runs the default 2017 FFMP policy (no optimization) with the full model and saves HDF5 output plus objective values. This is the "status quo" reference point.

```bash
bash workflow/01_run_baseline.sh
# Optional: also test the in-memory simulation path
bash workflow/01_run_baseline.sh --test-inmemory
```

Output: `outputs/baseline/ffmp_baseline.hdf5`, `outputs/baseline/ffmp_baseline_objectives.csv`

### Step 2: Run MM Borg Optimization

Launches Multi-Master Borg MOEA optimization using MPI. Uses the trimmed model (requires Step 0) for fast evaluations.

**Local test (4 MPI ranks, 1 island, 1000 NFE):**

```bash
bash workflow/02_run_mmborg.sh --seed 1 --islands 1 --nfe 1000 --np 4
```

**HPC submission (multiple seeds):**

```bash
# Anvil (NSF ACCESS)
for SEED in $(seq 1 10); do
    sbatch --export=ALL,SEED=${SEED} slurm/submit_mmborg.slurm
done

# Hopper (Cornell)
for SEED in $(seq 1 10); do
    sbatch --export=ALL,SEED=${SEED} slurm/submit_mmborg_hopper.slurm
done
```

Output: `outputs/optimization/{formulation}/sets/seed_XX_{formulation}.set` and `outputs/optimization/{formulation}/runtime/seed_XX_{formulation}_%d.runtime`

#### MPI Rank Allocation

MM Borg allocates ranks as: **1 controller + N masters (islands) + remaining workers**.

Formula: `ntasks = 1 + N_ISLANDS * (workers_per_island + 1)`

| Cluster | Nodes | Tasks/Node | Total Ranks | Islands | Workers/Island |
|---------|-------|------------|-------------|---------|----------------|
| Anvil   | 2     | ~65        | 129         | 2       | 63             |
| Hopper  | 2     | 40         | 80          | 2       | 38             |

`maxEvaluations` in the config is **per island**. With 2 islands and 1,000,000 NFE, total evaluations = 2,000,000.

### Step 3: MOEA Diagnostics

Runs MOEAFramework v5.0 runtime diagnostics (hypervolume convergence, epsilon progress, seed reliability).

```bash
bash workflow/03_run_diagnostics.sh
```

### Step 4: Plot Diagnostics

Generates diagnostic figures from Step 3 output.

```bash
bash workflow/04_plot_diagnostics.sh
```

### Step 5: Re-evaluate Pareto Solutions

Re-simulates Pareto-approximate solutions with the full model and saves full HDF5 output for detailed post-hoc analysis.

```bash
bash workflow/05_reevaluate.sh [formulation] [--max-solutions N]
```

Output: `outputs/reevaluation/{formulation}/solution_XXXX.hdf5` and `outputs/reevaluation/{formulation}/objectives_summary.csv`


## Borg MOEA Setup

### Compilation

Place `borgmm.c` and `mt19937ar.c` in the `borg/` directory (from the `passNFE_ALH_PyCheckpoint` branch of MMBorgMOEA). Also place the revised `borg.py` wrapper from the BorgTraining repository.

```bash
# Linux (HPC)
mpicc -shared -fPIC -O3 -o borg/libborgmm.so borg/borgmm.c borg/mt19937ar.c -lm

# Verify
mpirun -np 4 python3 src/mmborg_cli.py --seed 1 --islands 1 --nfe 100
```

### Required Files

```
borg/
├── borg.py          # Python ctypes wrapper (from BorgTraining repo)
├── borgmm.c         # MM Borg C source (passNFE_ALH_PyCheckpoint branch)
├── mt19937ar.c      # Mersenne Twister RNG
└── libborgmm.so     # Compiled shared library (generated)
```

### HPC Environment

Both SLURM templates set thread-pinning environment variables to prevent numpy/BLAS thread contention across MPI ranks:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```


## Repository Structure

```
NYCOptimization/
├── config.py                           # Central config: paths, bounds, formulations, objectives
├── requirements.txt
│
├── workflow/                           # Numbered pipeline scripts — run in order
│   ├── 00_generate_presim.sh           # Step 0: Generate STARFIT presimulated releases (one-time)
│   ├── 01_run_baseline.sh              # Step 1: Evaluate default FFMP baseline
│   ├── 02_run_mmborg.sh                # Step 2: Launch MM Borg optimization via MPI
│   ├── 03_run_diagnostics.sh           # Step 3: MOEAFramework v5.0 runtime diagnostics
│   ├── 04_plot_diagnostics.sh          # Step 4: Generate diagnostic figures
│   └── 05_reevaluate.sh               # Step 5: Re-simulate Pareto solutions (full model)
│
├── slurm/                              # HPC submission templates and MPI smoke tests
│   ├── submit_mmborg.slurm             # Anvil (NSF ACCESS)
│   ├── submit_mmborg_hopper.slurm      # Hopper (Cornell)
│   └── test_mpi_tiny.sh               # 1-node, 2-NFE MPI sanity check
│
├── src/                                # Core library
│   ├── simulation.py                   # DVs → objectives evaluation wrapper
│   ├── objectives.py                   # ObjectiveSet + DEFAULT_OBJECTIVES
│   ├── external_policy.py              # ExternalPolicyParameter + PLMR integration
│   ├── mmborg.py / mmborg_cli.py       # MM Borg optimization driver + CLI
│   ├── diagnostics.py                  # MOEAFramework diagnostic pipeline
│   ├── policies/                       # RBF, Tree, ANN policy classes
│   ├── load/                           # HDF5 and .set/.ref file loaders
│   └── plotting/
│       ├── style.py                    # Shared rcParams, ARCH_COLORS, label dicts
│       ├── hypervolume_convergence.py
│       ├── parallel_coordinates.py
│       ├── pareto_evolution.py
│       └── seed_reliability.py
│
├── figures/                            # Manuscript figure scripts (one per figure)
│   ├── fig01_system_map.py
│   ├── fig02_architecture_schematic.py
│   ├── fig03_pareto_comparison.py      # Main result: Pareto front comparison
│   ├── fig04_resolution_curve.py       # HV vs. degrees of freedom
│   ├── fig05_robustness_degradation.py # OSST robustness analysis
│   ├── fig06_vulnerability_maps.py     # SHAP attribution
│   ├── figSI_lhs_diagnostics.py
│   ├── figSI_convergence.py
│   └── make_all_figures.py             # Regenerate all figures
│
├── scripts/                            # Operational entry points (called by workflow/)
│   ├── generate_presim.py
│   ├── run_baseline.py
│   ├── run_diagnostics.py
│   └── quick_sample.py                 # LHS sampling + diagnostic scatter plots
│
├── tests/
│   └── test_simulation_api.py
│
├── docs/                               # Documentation
│   ├── README.md                       # This index + project structure
│   └── architecture/                   # External policy (PLMR) design docs
│       ├── ARCHITECTURE_INDEX.md
│       ├── ARCHITECTURE_DESIGN.md
│       ├── DESIGN_SUMMARY.md
│       ├── DESIGN_Q_AND_A.md
│       └── IMPLEMENTATION_EXAMPLES.md
│
├── outputs/                            # Generated data (git-ignored)
│   ├── baseline/
│   ├── presim/
│   ├── optimization/<arch>/sets/
│   ├── diagnostics/
│   ├── figures/                        # Diagnostic plots
│   ├── manuscript_figures/             # Publication-quality figures
│   └── reevaluation/
│
└── borg/                               # Borg MOEA files (git-ignored, licensed)
    ├── borg.py
    └── libborgmm.so
```

## Relevant Repositories

- [Pywr-DRB nyc_opt branch](https://github.com/Pywr-DRB/Pywr-DRB/tree/nyc_opt) (local: ../Pywr-DRB) contains the Pywr-DRB model with parameterized NYC reservoir operational rules to allow for simulation-based optimization. This will be installed in the local venv/ and used to perform the simulation evaluation. Currently, this repo only supports the current FFMP rule structure, however later versions will contain the alternative parameter rule options.

- [NYCOperationExploration](https://github.com/Pywr-DRB/NYCOperationExploration) (local: ../NYCOperationExplorations) runs a sensitivity analysis of the 2017 FFMP parameterization scheme used in Pywr-DRB nyc_opt branch. This will not be used in this NYCOptimization study, however provides additional supporting analyses and code workflows.

- [StochasticExploratoryExperiment](https://github.com/Pywr-DRB/StochasticExploratoryExperiment) (local: ../) contains all code to generate, simulate, and analyze ensembles of stochastic streamflow scenarios. This will not be used directly in this NYCOptimization study, however will be helpful reference code when generating synthetic streamflow sequences for policy re-evaluation and robustness analysis.


## Multi-Master Borg MOEA

This project uses the multi-master Borg MOEA (Hadka and Reed, 2015) to perform the simulation-based optimization of NYC operations. Additionally, MOEA diagnostics are run to verify hypervolume convergence and reliability.

Relevant WaterProgramming blog posts:

- [Everything You Need to Run Borg MOEA and serial python wrapper – Part 1](https://waterprogramming.wordpress.com/2025/02/04/everything-you-need-to-run-borg-moea-and-serial-python-wrapper-part-1/)

- [Everything You Need to Run Borg MOEA and Python Wrapper – Part 2](https://waterprogramming.wordpress.com/2025/02/19/everything-you-need-to-run-borg-moea-and-python-wrapper-part-2/)

- [MM Borg Training Part 1: Setting up parallel scaling experiments with Borg MOEA](https://waterprogramming.wordpress.com/2024/07/30/mm-borg-training-part-1-setting-up-parallel-scaling-experiments-with-borg-moea/)

- [MM Borg Training Part 2: Post-processing parallel scaling experiments using MOEAFramework](https://waterprogramming.wordpress.com/2024/09/24/mm-borg-training-part-2-post-processing-parallel-scaling-experiments-using-moeaframework/)

- [MM Borg MOEA Python Wrapper – Checkpointing, Runtime and Operator Dynamics using MOEAFramework 5.0](https://waterprogramming.wordpress.com/2025/08/14/mm-borg-moea-python-wrapper-checkpointing-runtime-and-operator-dynamics-using-moea-framework-5-0/)

- [Performing runtime diagnostics using MOEAFramework](https://waterprogramming.wordpress.com/2024/04/22/performing-runtime-diagnostics-using-moeaframework/)

- [Measuring the parallel performance of the Borg MOEA](https://waterprogramming.wordpress.com/2021/07/26/measuring-the-parallel-performance-of-the-borg-moea/)


## References

Giuliani, M., Castelletti, A., Pianosi, F., Mason, E., & Reed, P. M. (2016). Curses, tradeoffs, and scalable management: Advancing evolutionary multiobjective direct policy search to improve water reservoir operations. Journal of Water Resources Planning and Management, 142(2).

Hadka, D., & Reed, P. (2013). Borg: An auto-adaptive many-objective evolutionary computing framework. Evolutionary Computation, 21(2), 231-259.

Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg multiobjective evolutionary algorithm to enhance the management of complex environmental systems. Environmental Modelling & Software, 69, 353-369.

Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment. Environmental Modelling & Software, 106185.

Herman, J. D., Reed, P. M., Zeff, H. B., & Characklis, G. W. (2015). How should robustness be defined for water systems planning under change? Journal of Water Resources Planning and Management, 141(10).

Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013). Many objective robust decision making for complex environmental systems undergoing change. Environmental Modelling & Software, 42, 55-71.

Kolesar, P., & Serio, J. (2011). Breaking the deadlock: Improving water-release policies on the Delaware River. Interfaces, 41(1), 18-34.

Quinn, J. D., Reed, P. M., Giuliani, M., & Castelletti, A. (2019). What is controlling our control rules? Opening the black box of multireservoir operating policies using time-varying sensitivity analysis. Water Resources Research, 55(7), 5962-5984.

Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). Evolutionary multiobjective optimization in water resources: The past, present, and future. Advances in Water Resources, 51, 438-456.
