# Session Log

Developer sessions for the NYCOptimization project.

---

## Session 14 — 2026-04-10

### Topics

**Project Reorganization (merged from `claude/zen-dubinsky`)**

Restructured the project root to separate concerns:
- `workflow/` — numbered pipeline bash scripts (`00_` through `05_`)
- `slurm/` — HPC submission templates
- `figures/` — manuscript figure generation scripts (`fig01`–`fig06`, `figSI_*`, `make_all_figures.py`)
- `src/plotting/style.py` — shared matplotlib rcParams, color palettes, label dictionaries

Stub figure scripts created for all planned manuscript figures. `src/plotting/` reorganized with one module per visualization type.

**Config Decomposition (merged from `claude/goofy-margulis`)**

Split `config.py` (~400 lines) into `config.py` (paths/settings/constants) + `src/formulations/` module:

- `src/formulations/ffmp.py` — `FFMP_FORMULATION` dict (24 DVs), `_interpolate_factors()`, `generate_ffmp_formulation(n_zones)`
- `src/formulations/external.py` — `ARCHITECTURES` registry, `register_architecture()`, `is_external_policy()`, `get_architecture()`
- `src/formulations/__init__.py` — unified registry API: `get_bounds()`, `get_var_names()`, `get_n_vars()`, `get_baseline_values()`, `make_objective_function()`, plus `ffmp_N` dynamic dispatch

Key design decisions:
- Circular import avoided via lazy (call-time) imports of `src.objectives` inside `make_objective_function()` and `get_objective_set()`
- `ffmp_N` pattern: `get_formulation("ffmp_10")` parses N, calls `generate_ffmp_formulation(10)` at runtime
- External policy dispatch: `make_objective_function("rbf")` routes to `evaluate_with_policy()` (PLMR path); `make_objective_function("ffmp")` routes to `evaluate()` (direct path)
- `config.py` re-exports everything from `src.formulations` for backward compatibility

**Fixed-Target Objectives Architecture Design**

Documented the fair-comparison objective architecture: `FIXED_TARGET_OBJECTIVES` uses static MRF targets (Montague 1,131 MGD, Trenton 1,939 MGD) instead of FFMP drought-level-dependent targets. This ensures all policy architectures (FFMP, RBF, Tree, ANN) are evaluated against the same performance standards. Toggle via `ACTIVE_OBJECTIVE_SET` in `config.py`.

**SLURM Templates**

SLURM submission scripts in `slurm/` for all 4 non-FFMP architectures (rbf, tree, ann, ffmp_vr). Templates parameterized for 10 seeds × 1M NFE/island. Thread pinning (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) included to prevent numpy/BLAS contention across MPI ranks.

MPI rank formula: `ntasks = 1 + N_ISLANDS × (workers_per_island + 1)`

**Manuscript Outline**

Planned figure set documented in `docs/README.md`:
- Fig 1: System map
- Fig 2: Architecture schematic (FFMP → Variable-Res → RBF/Tree/ANN)
- Fig 3: Pareto front comparison across architectures
- Fig 4: Resolution curve (complexity vs. performance)
- Fig 5: Robustness degradation
- Fig 6: Vulnerability maps
- Fig SI: LHS diagnostics, convergence plots

---

## Session 13 — (prior session)

ExternalPolicyParameter + PLMR validated. LHS sampling figures generated for all 5 architectures (FFMP, Variable-Res FFMP, RBF, Tree, ANN). `FIXED_TARGET_OBJECTIVES` objective set added. HPC SLURM templates drafted.
