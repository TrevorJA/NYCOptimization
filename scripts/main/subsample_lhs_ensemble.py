"""
subsample_lhs_ensemble.py - Subsample the large stochastic ensemble using
Latin Hypercube Sampling over hydrologic-metric space.

Placeholder. Produces the search-ensemble realization indices for the
subsampling-based scenario designs in src/scenario_designs.py (input_stratified,
hazard_filling, and the probabilistic designs).

The choice of hydrologic/hazard metrics defining the space — and the
construction (Maximin, Sobol, conditioned LHS, etc.) — is deferred until the
ensemble-design discussion.

Expected inputs:
    Master stochastic ensemble produced by generate_stochastic_ensemble.py.

Expected outputs:
    One realization-index file per scenario design, plus a hazard-metric matrix
    snapshot for downstream diagnostics.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Step 2: subsample over hazard-metric space. Implementation deferred — "
        "see src/scenario_designs.py (SCENARIO_DESIGNS) for the target designs."
    )


if __name__ == "__main__":
    main()
