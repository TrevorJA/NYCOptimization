"""
generate_stochastic_ensemble.py - Generate the large stochastic streamflow
ensemble that the probabilistic and LHS space-filling sub-samples will
draw from.

Placeholder. Inputs, generator method (e.g. Kirsch-Nowak, copula-based,
PUB synthetics), realization count, and output layout are deferred until
the ensemble-design discussion finalizes them.

Expected outputs:
    outputs/synthetic_ensembles/_master/{inflow_type}/  (path TBD)
        per-realization streamflow traces large enough to support both
        probabilistic and LHS sub-samples at the three planned sizes.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Step 1: generate large stochastic streamflow ensemble. "
        "Implementation deferred — see config.ENSEMBLE_PRESETS for the "
        "downstream sub-sample design."
    )


if __name__ == "__main__":
    main()
