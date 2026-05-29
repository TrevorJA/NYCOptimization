"""
subsample_lhs_ensemble.py - Subsample the large stochastic ensemble using
Latin Hypercube Sampling over hydrologic-metric space.

Placeholder. Produces the three space-filling presets (lhs_small/medium/large)
and the three matched probabilistic presets (prob_small/medium/large) from
config.ENSEMBLE_PRESETS.

The choice of hydrologic metrics defining the LHS space — and the LHS
construction (Maximin, Sobol, etc.) — is deferred until the ensemble-design
discussion.

Expected inputs:
    Master stochastic ensemble produced by generate_stochastic_ensemble.py.

Expected outputs:
    One realization-index file per preset name in config.ENSEMBLE_PRESETS,
    plus a hydrologic-metric matrix snapshot for downstream diagnostics.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Step 2: LHS subsample over hydrologic-metric space. "
        "Implementation deferred — see config.ENSEMBLE_PRESETS for sizes."
    )


if __name__ == "__main__":
    main()
