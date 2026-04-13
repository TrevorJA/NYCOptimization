# Task List

## Completed

### Infrastructure
- [x] Project reorganization: `workflow/`, `slurm/`, `figures/`, `src/plotting/style.py`
- [x] Config decomposition: `src/formulations/` module (`ffmp.py`, `external.py`, `__init__.py`)
- [x] `config.py` slimmed to paths/settings/constants; re-exports from `src.formulations`
- [x] N-zone FFMP: `generate_ffmp_formulation(n_zones)` with `_interpolate_factors()`
- [x] External policy registry: `register_architecture()`, `is_external_policy()`, `get_architecture()`
- [x] Unified objective function dispatch: `make_objective_function()` routes FFMP vs. external
- [x] `FIXED_TARGET_OBJECTIVES` objective set (Montague 1,131 MGD, Trenton 1,939 MGD)
- [x] `ExternalPolicyParameter` + PLMR integration validated (`src/external_policy.py`)
- [x] Policy classes: `RBFPolicy`, `TreePolicy`, `ANNPolicy` in `src/policies/`
- [x] `src/mmborg.py` updated to use `make_objective_function()` dispatch
- [x] SLURM templates for all 4 architectures (rbf, tree, ann, ffmp_vr)
- [x] Manuscript figure stubs: `fig01`–`fig06`, `figSI_*`, `make_all_figures.py`
- [x] LHS sampling figures generated for all 5 architectures
- [x] `docs/CLAUDE.md` updated to reflect current project structure
- [x] `docs/README.md` updated with `src/formulations/` and full project tree

---

## Remaining

### Pre-HPC: State Vector Reduction (Physics-Based)
- [x] Physics-based state vector design using pywrdrb routing topology (`notes/state_vector_design.md`)
- [x] Verified routing travel times: Can/Pep→Montague 2d, Nev→Montague 1d, Can/Pep→Trenton 4d, Nev→Trenton 3d
- [x] Defined Minimal (6-dim) and Extended (9-dim) state specs (approved)
- [x] Implement configurable `STATE_SPECS` in `build_state_config()` (minimal/extended/full) — commit 82e9545
- [x] Update policy DV counts in `src/formulations/external.py` for 6-dim and 9-dim
- [ ] Run LHS comparison: 6-dim vs 9-dim vs 15-dim for all three architectures
- [ ] Decide final state vector for HPC runs based on LHS results

### Pre-HPC: Publication-Scale Readiness (2026-04-13)
- [x] Production SLURM scripts for ffmp, rbf, tree, ann (seed array jobs)
- [x] FFMP_VR N-sweep SLURM script + `FFMP_VR_N_SWEEP` in config.py
- [x] Env-var overrides for STATE_SPEC and ACTIVE_OBJECTIVE_SET
- [x] Reproducibility logging (git rev + config snapshot) in SLURM scripts
- [x] Parallelized re-evaluation (`--njobs` + per-seed output dirs)
- [x] Parallelized diagnostics (background jobs per architecture)
- [ ] **DECISION NEEDED**: final `ACTIVE_OBJECTIVE_SET` for publication runs.
      `fixed_target` is NOT currently registered in `OBJECTIVE_SETS`
      despite being referenced in docs — register it before HPC submission.
- [ ] **DECISION NEEDED**: final `STATE_SPEC` for publication runs (awaits LHS comparison)
- [ ] Validate `test_mpi_tiny.sh` on target HPC cluster
- [ ] Benchmark trimmed-model eval cost (s/NFE) to set realistic wall times

### HPC Optimization Runs
- [ ] Obtain HPC allocation (cluster access)
- [ ] Test MPI launch: `slurm/test_mpi_tiny.sh`
- [ ] Run MM Borg for all 4 architectures × 10 seeds × 1M NFE/island
  - [ ] rbf (10 seeds)
  - [ ] tree (10 seeds)
  - [ ] ann (10 seeds)
  - [ ] ffmp_vr (10 seeds, variable resolution)
- [ ] Monitor runtime files (`outputs/optimization/{arch}/runtime/`)
- [ ] Verify `.set` files written for all seeds

### MOEA Diagnostics
- [ ] Run MOEAFramework v5.0 diagnostics (`workflow/03_run_diagnostics.sh`)
- [ ] Generate diagnostic figures (`workflow/04_plot_diagnostics.sh`)
- [ ] Assess convergence: hypervolume, epsilon-indicator
- [ ] Assess seed reliability across 10 seeds

### OSST Pipeline (Optimal Stable Sets of Trees)
- [ ] Library generation: sample operating policies from Pareto fronts
- [ ] Characterize drought scenarios (duration, severity, spatial pattern)
- [ ] LHS sampling over scenario space
- [ ] Evaluate policies × scenarios (re-simulation with full model)
- [ ] XGBoost/SHAP analysis: which state variables drive policy differences?
- [ ] Identify interpretable policy clusters

### Re-evaluation
- [ ] Re-simulate Pareto solutions with full Pywr-DRB model (`workflow/05_reevaluate.sh`)
- [ ] Compare objectives under full vs. trimmed model (verify trimmed model fidelity)
- [ ] Generate robustness metrics: satisficing, regret

### Manuscript Figures
- [ ] `fig01_system_map.py` — DRB system map with NYC reservoirs
- [ ] `fig02_architecture_schematic.py` — FFMP → Variable-Res → RBF/Tree/ANN schematic
- [ ] `fig03_pareto_comparison.py` — Pareto front comparison across architectures
- [ ] `fig04_resolution_curve.py` — Complexity vs. performance frontier
- [ ] `fig05_robustness_degradation.py` — Robustness vs. climate scenario
- [ ] `fig06_vulnerability_maps.py` — Spatial vulnerability under each architecture
- [ ] `figSI_lhs_diagnostics.py` — LHS space-filling diagnostics
- [ ] `figSI_convergence.py` — Hypervolume convergence across seeds

### Manuscript Writing
- [ ] Methods: policy architecture descriptions (RBF, Tree, ANN, Variable-Res FFMP)
- [ ] Methods: objective function formulation (7 objectives, fixed vs. dynamic targets)
- [ ] Results: complexity frontier analysis
- [ ] Results: MORDM robustness analysis
- [ ] Discussion: equity implications (NJ supply objectives)
- [ ] Supplemental: LHS diagnostics, convergence, OSST details
