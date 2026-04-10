# docs/

Project documentation for NYCOptimization.

## Contents

| Path | Description |
|------|-------------|
| [`architecture/`](architecture/) | External policy architecture design docs (PLMR implementation) |
| [`architecture/ARCHITECTURE_INDEX.md`](architecture/ARCHITECTURE_INDEX.md) | Navigation index for the architecture doc set |
| [`architecture/ARCHITECTURE_DESIGN.md`](architecture/ARCHITECTURE_DESIGN.md) | Full specification (~100 pages) |
| [`architecture/DESIGN_SUMMARY.md`](architecture/DESIGN_SUMMARY.md) | 10-minute overview |
| [`architecture/DESIGN_Q_AND_A.md`](architecture/DESIGN_Q_AND_A.md) | Design rationale Q&A |
| [`architecture/IMPLEMENTATION_EXAMPLES.md`](architecture/IMPLEMENTATION_EXAMPLES.md) | Worked code examples |

## Project Structure

```
NYCOptimization/
├── config.py                  # Central config: paths, bounds, formulations, objectives
├── requirements.txt
│
├── workflow/                  # Numbered pipeline scripts (run in order)
│   ├── 00_generate_presim.sh  # One-time: generate STARFIT presimulated releases
│   ├── 01_run_baseline.sh     # Evaluate default FFMP baseline
│   ├── 02_run_mmborg.sh       # Launch MM Borg optimization via MPI
│   ├── 03_run_diagnostics.sh  # MOEAFramework runtime diagnostics
│   ├── 04_plot_diagnostics.sh # Generate diagnostic figures
│   └── 05_reevaluate.sh       # Re-simulate Pareto solutions with full model
│
├── slurm/                     # HPC submission scripts and MPI tests
│   └── test_mpi_tiny.sh
│
├── src/                       # Core library
│   ├── simulation.py          # DVs → objectives evaluation wrapper
│   ├── objectives.py          # ObjectiveSet + DEFAULT_OBJECTIVES
│   ├── external_policy.py     # ExternalPolicyParameter + PLMR integration
│   ├── mmborg.py / mmborg_cli.py
│   ├── diagnostics.py
│   ├── policies/              # RBF, Tree, ANN policy classes
│   ├── load/                  # HDF5 and .set/.ref file loaders
│   └── plotting/
│       ├── style.py           # Shared rcParams, ARCH_COLORS, label dicts
│       ├── hypervolume_convergence.py
│       ├── parallel_coordinates.py
│       ├── pareto_evolution.py
│       └── seed_reliability.py
│
├── figures/                   # Manuscript figure scripts (one per figure)
│   ├── fig01_system_map.py
│   ├── fig02_architecture_schematic.py
│   ├── fig03_pareto_comparison.py
│   ├── fig04_resolution_curve.py
│   ├── fig05_robustness_degradation.py
│   ├── fig06_vulnerability_maps.py
│   ├── figSI_lhs_diagnostics.py
│   ├── figSI_convergence.py
│   └── make_all_figures.py    # Regenerate all figures
│
├── scripts/                   # Operational entry points (called by workflow/)
│   ├── generate_presim.py
│   ├── run_baseline.py
│   ├── run_diagnostics.py
│   └── quick_sample.py        # LHS sampling + diagnostic plots
│
├── tests/
├── docs/                      # This directory
└── outputs/                   # Generated data (gitignored)
    ├── baseline/
    ├── presim/
    ├── optimization/{arch}/sets/
    ├── diagnostics/
    ├── figures/               # Diagnostic plots
    ├── manuscript_figures/    # Publication-quality figures
    └── reevaluation/
```

---

# External Policy Architecture Design

**Complete specification for integrating external policy control with pywrdrb, without modifying existing source code.**

---

## Quick Start

**For a 10-minute overview:** Read [DESIGN_SUMMARY.md](architecture/DESIGN_SUMMARY.md)

**For implementation:** Read [IMPLEMENTATION_EXAMPLES.md](architecture/IMPLEMENTATION_EXAMPLES.md) + [ARCHITECTURE_DESIGN.md](architecture/ARCHITECTURE_DESIGN.md) § 3 (Details)

**For design review:** Read [DESIGN_SUMMARY.md](architecture/DESIGN_SUMMARY.md) + [DESIGN_Q_AND_A.md](architecture/DESIGN_Q_AND_A.md)

**Full navigation:** See [ARCHITECTURE_INDEX.md](architecture/ARCHITECTURE_INDEX.md)

---

## The Problem

Enable three operating modes without modifying existing pywrdrb files:

- **Mode A:** Standard FFMP (baseline)
- **Mode B:** Variable-resolution FFMP (N zones: 3, 6, 10, 15...)
- **Modes C/D/E:** External policy (RBF, Tree, ANN) replaces FFMP logic

---

## The Solution (Summary)

### Four New Components

1. **ExternalPolicyParameter** (pywrdrb/parameters/external_policy.py)
   - Custom Pywr Parameter that calls a policy function at each timestep
   - Reads state from model nodes, returns action (release amount)
   - Caches results for efficient multi-output policies

2. **Model Replacement Utilities** (src/model_builder_utils.py)
   - Post-load replacement: swap FFMP parameters with ExternalPolicyParameter
   - Also replace dynamic MRF targets with fixed constants (fair comparison)

3. **PolicyBase + Subclasses** (optimization/policies/)
   - Abstract base class for RBF, Tree, ANN policies
   - Common interface for Borg MOEA optimization

4. **New Evaluation Path** (src/simulation_policy.py)
   - Borg-compatible evaluation for policy-based architectures
   - Wraps model loading, parameter replacement, simulation, objectives

### Minimal Changes

- **New files:** ~2,000 lines (10 files)
- **Modified files:** ~20 lines (src/simulation.py, config.py, objectives.py)
- **No changes to existing pywrdrb source code**

---

## Core Design Decisions

### 1. Interception Point: Post-Load Model Replacement

**Why:** Policy functions (RBF, Tree, ANN) cannot be JSON-serialized. Load the standard FFMP model, then replace parameter instances in memory.

### 2. Fair Comparison: Fixed MRF Targets

**Why:** FFMP declares drought levels and relaxes flow targets. All policies measured against same fixed targets (Montague=1,131 MGD, Trenton=1,939 MGD) for fair comparison.

### 3. State & LP Solver Interaction

**Policy reads state via ExternalPolicyParameter:**
- Extracts from Pywr nodes at each timestep (storage, inflows, time of year)
- Normalized state vector (0-1 fractions + temporal features)
- Result cached to avoid redundant calls for multi-output policies

**LP Solver respects policy output as hard constraint:**
- Policy output is max_flow on release links
- LP solver allocates water respecting policy + physical constraints
- If insufficient to meet fixed MRF targets, solver either adjusts or produces violations

---

## Execution Flows

### Mode A: Standard FFMP
```
"ffmp" → dvs_to_config() [existing] → run_simulation_inmemory() [existing]
       → compute_objectives(..., fixed_targets=True) [CHANGED]
```

### Mode B: Variable-Resolution FFMP
```
"ffmp_vr_10" → generate_ffmp_formulation(n_zones=10) [NEW]
            → dvs_to_config() [existing, extended]
            → run_simulation_inmemory() [existing]
            → compute_objectives(..., fixed_targets=True)
```

### Mode C/D/E: External Policy
```
"rbf"|"tree"|"ann" → policy.set_params(dv_vector) [NEW]
                  → evaluate_with_policy(policy) [NEW]
                     ├─ Load base model
                     ├─ Model.load(json)
                     ├─ replace_release_parameters(model, policy) [NEW]
                     ├─ set_fixed_mrf_targets(model) [NEW]
                     ├─ model.run()
                     └─ compute_objectives(..., fixed_targets=True)
```

---

## State → Action

### State Vector (5 dimensions)
```
[
  Cannonsville storage / 95,706 MG,     # 0-1
  Pepacton storage / 140,190 MG,        # 0-1
  Neversink storage / 34,941 MG,        # 0-1
  sin(2π DOY / 366),                    # -1 to 1
  cos(2π DOY / 366)                     # -1 to 1
]
```

### Action Vector (3 dimensions)
```
[
  Cannonsville release (MGD),
  Pepacton release (MGD),
  Neversink release (MGD)
]
```

---

## Performance

| Scenario | Runtime | Overhead |
|----------|---------|----------|
| FFMP baseline (78 years) | ~150 sec | baseline |
| Policy RBF | ~160-180 sec | +10-20% |
| Policy Tree | ~160-180 sec | +10-20% |
| Policy ANN | ~170-200 sec | +15-30% |

For 1M Borg evaluations: ~4-5 CPU-years (parallelizes linearly with MPI).

---

## Files

### New (No Existing Code Modified)

```
pywrdrb/
  parameters/
    external_policy.py              ExternalPolicyParameter, CacheClearParameter
    fixed_target.py                 FixedMRFParameter

NYCOptimization/
  src/
    model_builder_utils.py          Post-load replacement utilities
    simulation_policy.py            evaluate_with_policy()
  
  optimization/
    formulations.py                 generate_ffmp_formulation(), registry
    policies/
      __init__.py, base.py          PolicyBase ABC
      rbf_policy.py, tree_policy.py, ann_policy.py
```

### Modified (Minimal)

```
NYCOptimization/
  src/
    simulation.py                   +5 lines (dispatch in evaluate())
  config.py                         +10 lines (formulation registry)
  docs/objectives.py                +5 lines (use fixed targets)
```

---

## Next Steps

1. Read [DESIGN_SUMMARY.md](architecture/DESIGN_SUMMARY.md) (5 min)
2. Read [IMPLEMENTATION_EXAMPLES.md](architecture/IMPLEMENTATION_EXAMPLES.md) (30 min)
3. Reference [ARCHITECTURE_DESIGN.md](architecture/ARCHITECTURE_DESIGN.md) for details during implementation
4. See deployment checklist in [ARCHITECTURE_DESIGN.md](architecture/ARCHITECTURE_DESIGN.md) § I (Part IX)

---

## Questions?

- **Where should I intercept?** → See [DESIGN_Q_AND_A.md](architecture/DESIGN_Q_AND_A.md) § Q1
- **How do I handle MRF targets?** → See [DESIGN_Q_AND_A.md](architecture/DESIGN_Q_AND_A.md) § Q2
- **How does state access work?** → See [DESIGN_Q_AND_A.md](architecture/DESIGN_Q_AND_A.md) § Q3
- **Show me code** → See [IMPLEMENTATION_EXAMPLES.md](architecture/IMPLEMENTATION_EXAMPLES.md)
- **I need everything** → See [ARCHITECTURE_DESIGN.md](architecture/ARCHITECTURE_DESIGN.md)

---

**Document Set:** 4 markdown files, ~100 pages total, production-ready design

**Status:** Complete, awaiting implementation

**Date:** 2026-04-09
