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

**Physics-Based State Vector Design**

Replaced statistical feature selection approach (Morris/Sobol/HIS) with physics-based state vector design driven by pywrdrb routing topology. Verified travel times from NYC reservoirs to MRF compliance points using pywrdrb node topology and the FFMP's `TotalReleaseNeededForDownstreamMRF` implementation.

Two approved state vector configurations:
- **Minimal (6-dim):** combined storage, Montague lag2, Trenton lag4, NJ demand lag4, sin/cos. DV counts: RBF 48, Tree 57, ANN 89.
- **Extended (9-dim):** adds Neversink storage, Montague lag1, Trenton lag3. DV counts: RBF 66, Tree 78, ANN 121.

Key design insight: the current 15-dim vector contains 6 redundant lag predictions that don't match routing travel times, plus redundant individual storage variables. Aggregate mode means the policy controls total release (volume balancer distributes), so combined storage is the primary state variable. Neversink separated in extended set due to different routing (1-day to Montague vs 2-day for Can/Pep).

Note: in `perfect_foresight` mode, lag predictions are exact and sufficient to meet flow targets. State vector design is based on routing physics, valid in both forecast modes.

See `notes/state_vector_design.md` for full design rationale and implementation plan.

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
