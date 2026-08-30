## Manuscript figure sequence

The planned main-text figure sequence. Figures 3–9 are rendered by `scripts/main/figures.py` from `src/figures/registry.py` (tier `manuscript`); the stem `fig{NN}_{name}` is decided there, and figures 1–2 are made outside the registry. The sequence is subject to change.

| Figure | Content | Current file |
|---|---|---|
| 1 | Basin map (made outside this repo); a condensed FFMP operating-rules panel may join it or stand alone in Section 2 | not yet produced |
| 2 | Graphical experimental-design diagram, panels (a) workflow, (b) hazard-filling selection, (c) re-evaluation and robustness comparison | not yet produced; working sketch `methods_diagram/sketch_v5.png` |
| 3 | Deeply uncertain forcing space: CMIP6 change-factor profiles, harmonic fit, sampled box, change in the flow duration curve | `fig03_forcing_space.png` |
| 4 | Realized hazard-space composition of the MC and HF search ensembles vs the candidate pool and the historical record | `fig04_ensemble_composition.png` |
| 5 | Each design's Pareto-approximate set on parallel axes (search-time objectives, shown per design, never compared across designs) | `fig05_pareto_set_parallel_axes.png` |
| 6 | Robustness rankings: each design's policies sorted by E_test satisficing robustness under each criterion set (RQ2) | `fig06_criteria_robustness.png` |
| 7 | All-Parties robustness vs no-harm frequency against the FFMP incumbent (RQ1 headline) | `fig07_regret_vs_incumbent.png` |
| 8 | Boosted-tree surfaces over the DU forcing space for each design's max-robustness/min-regret policy: success/failure probability and high/low regret vs the incumbent | `fig08_robustness_regret_surfaces.png` |
| 9 | Regret over the DU space (RQ1). Undecided among the three `number=9` registry entries `regret_surfaces`, `regret_surfaces_worst` and `regret_exposure`; their stems carry the focal-criterion key | none |

Every cross-design robustness panel (6–9) shows E_test re-evaluated quantities only.
