

# NYCOptimization

Trevor Amestoy, PhD Candidate
Cornell University, Dept of Civil and Environmental Engineering
Reed Research Group

## Study Scope

This repository performs multi-objective optimization of NYC reservoir operations implemented in the Pywr-DRB model. The study seeks to identify sets of operational policies that outperform the current 2017 Flexible Flow Management Program (FFMP) rules across multiple competing objectives and hydrological conditions.

The study applies Many-Objective Robust Decision Making (MORDM) to: (1) discover Pareto-approximate tradeoff sets under a reference hydrological scenario, (2) compare alternative policy formulations of varying flexibility, and (3) evaluate policy robustness under stochastic streamflow ensembles.

### Policy Formulations

Three nested formulations are considered, each representing increasing operational flexibility:

- **Formulation A (Parameterized FFMP)**: Re-optimize the existing FFMP parameters (MRF baselines, drought factors, storage zone thresholds, flood release limits) within the current rule structure. ~25-35 decision variables.
- **Formulation B (FFMP + Enhanced Flexibility)**: Extend the FFMP framework with additional degrees of freedom (seasonal drought factors, asymmetric storage curves, dynamic blending weights). ~40-60 decision variables.
- **Formulation C (State-Aware Direct Policy Search)**: Replace rule-based operations with nonlinear policy functions (radial basis functions or multi-layer perceptrons) that map system state to release/diversion decisions. ~100-300 decision variables.

### Candidate Objectives

| Objective | Direction | Stakeholder Relevance |
|-----------|-----------|----------------------|
| NYC water supply reliability | Maximize | NYC |
| NYC drought severity | Minimize | NYC |
| Montague flow compliance | Maximize | All decree parties |
| Trenton flow compliance | Maximize | NJ, PA, DE (salt front) |
| Ecological flow quality | Maximize | Environmental stakeholders |
| Flood risk | Minimize | Upper Delaware communities |
| Storage resilience | Maximize | All parties |


## Workflow

### Phase 1: Problem Setup and Baseline
- Finalize objectives, decision variable bounds, and constraints for each formulation
- Implement Borg MOEA Python wrapper for Pywr-DRB coupling
- Characterize default 2017 FFMP baseline performance

### Phase 2: Optimization
- MOEA diagnostics (random seed analysis, hypervolume convergence)
- MM-Borg optimization on HPC for Formulations A, B, C
- Generate Pareto-approximate reference sets

### Phase 3: Re-evaluation Under Uncertainty
- Stochastic streamflow ensemble generation (Kirsch-Nowak)
- Re-evaluate Pareto-approximate policies across ensemble
- Compute robustness metrics (satisficing, regret, domain criterion)

### Phase 4: Scenario Discovery and Analysis
- PRIM / CART scenario discovery to identify vulnerability conditions
- Cross-formulation robustness comparison
- Stakeholder-perspective tradeoff visualization

### Phase 5: Synthesis
- Quantify "value of flexibility" across objective space
- Identify actionable policy insights for DRB stakeholders


## Repository Structure

```
NYCOptimization/
├── 00_generate_presim.sh               # Step 0: One-time setup - generate presimulated releases
├── 01_run_baseline.sh                  # Step 1: Evaluate default FFMP baseline (full model)
├── 02_run_mmborg.sh                    # Step 2: Launch MM Borg optimization (MPI, trimmed model)
├── 02_submit_mmborg.slurm              # Step 2: SLURM submission template for Anvil
├── 03_run_diagnostics.sh               # Step 3: MOEAFramework v5.0 runtime diagnostics
├── 04_plot_diagnostics.sh              # Step 4: Generate diagnostic figures
├── 05_reevaluate.sh                    # Step 5: Re-evaluate Pareto solutions (full model)
├── config.py                           # Central configuration (formulations, objectives, settings)
├── requirements.txt                    # Python dependencies
│
├── src/                                # Core modules
│   ├── simulation.py                   # Pywr-DRB simulation wrapper (DVs -> objectives)
│   ├── objectives.py                   # Objective metric computation functions
│   ├── mmborg.py                       # Multi-Master Borg optimization driver
│   ├── mmborg_cli.py                   # CLI entry point for mmborg.py
│   ├── diagnostics.py                  # MOEAFramework v5.0 diagnostic pipeline
│   ├── load/                           # Data loading utilities
│   │   ├── results.py                  # Load pywrdrb HDF5 simulation output
│   │   └── reference_set.py            # Load .ref and .set files
│   └── plotting/                       # One plot per file (manuscript-quality)
│       ├── hypervolume_convergence.py
│       ├── seed_reliability.py
│       └── parallel_coordinates.py
│
├── scripts/                            # Supporting Python scripts (called by numbered bash scripts)
│   ├── generate_presim.py              # Presimulated releases generation (called by 00_*)
│   └── run_baseline.py                 # Baseline evaluation logic (called by 01_*)
│
├── tests/                              # Test scripts
│   └── test_simulation_api.py          # Verify pywrdrb model build/run/extract pipeline
│
├── outputs/                            # Generated outputs (git-ignored)
│   ├── baseline/                       # Baseline FFMP performance
│   ├── optimization/<formulation>/     # Runtime files and solution sets per seed
│   ├── diagnostics/<formulation>/      # MOEAFramework metrics
│   ├── reference_sets/                 # Cross-seed reference sets (.ref)
│   ├── reevaluation/<formulation>/     # Robustness analysis results
│   ├── presim/                         # Pre-simulated releases for trimmed model
│   └── figures/                        # Generated plots
│
├── borg/                               # Borg MOEA files (git-ignored, licensed)
│   ├── borg.py                         # Python wrapper (from BorgTraining repo)
│   └── libborgmm.so                   # Compiled MMBorg shared library
│
├── notes/                              # Research planning and literature reviews
│   ├── STUDY_PLAN.md
│   ├── notes_drb_operations_review.md
│   ├── notes_dmuu_optimization_review.md
│   └── brainstorm_methodological_contributions.md
│
└── archive/                            # Deprecated scripts (pending deletion)
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
