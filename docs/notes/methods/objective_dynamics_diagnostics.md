# Objective-Dynamics Anatomy Figures

Two local diagnostic suites draw the operational dynamics behind each objective
(NYC delivery, Montague flow, Trenton flow, NYC storage, downstream flooding)
for the default FFMP baseline and one interpretable storage-conservative
contrast policy. The figures compute nothing the objective functions do not.
Every threshold, resample basis, window cut and tail rule is imported from
`src.objectives` / `src.objectives_ensemble`, and each annotated score is
asserted equal (within the objective epsilon) to the value the registered
objective set computes.

| Suite | Driver (`scripts/supplemental/`) | Plot module | Scored with |
|---|---|---|---|
| Historic single trace | `objective_dynamics_figures.py` | `objective_dynamics.py` | whole-trace (§1) performance metrics, plus the annual-unit (§2) strip the optimizer targets |
| Local KN ensemble (`kn_50yr_n5`, 5 x 50 yr, trimmed model) | `ensemble_objective_dynamics_figures.py` | `ensemble_objective_dynamics.py` | pooled annual-unit (§2) objectives via `build_ensemble_objective_set` |

Each suite renders figures A to E (one per operational quantity), F (the
contrast policy's FFMP rules over the baseline) and G (parallel axes of the
scores). Figure-level detail lives in the `plot_*` docstrings.

The contrast policy is identical in both suites. It sets `mrf_target_scale_*`
to 0.80 and `zone_vshift_*` to +0.03 of capacity, so the system holds more NYC
storage at the cost of downstream Decree flow. It is built by
`build_contrast_dv()` and its feasibility is checked with
`compute_constraint_violations`.

Run from the repo root with the project venv active:

    python scripts/supplemental/objective_dynamics_figures.py
    python scripts/supplemental/ensemble_objective_dynamics_figures.py

Outputs, including the cached simulations, go to
`supplemental_config.OBJDYN_OUTPUT_ROOT` and `ENSOBJDYN_OUTPUT_ROOT`
(`outputs/supplemental/objective_dynamics/` and
`outputs/supplemental/ensemble_objective_dynamics/`). Re-runs reuse the caches
and only re-plot. The historic suite reuses the artifact-of-record baseline HDF5
from step 05 when its objective columns match the active set. Env toggles:
`NYCOPT_OBJDYN_NOSTRICT=1` and `NYCOPT_ENSOBJDYN_NOSTRICT=1` disable the
figure-vs-score assertions, and `NYCOPT_ENSOBJDYN_REFRESH=1` ignores the
ensemble simulation cache.
