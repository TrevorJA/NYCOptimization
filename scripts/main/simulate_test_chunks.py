"""simulate_test_chunks.py - Simulate + score a chunked test ensemble (metrics-only).

Re-evaluates a policy set against every chunk of the chunked forcing master that
``NYCOPT_REEVAL_ENSEMBLE_PRESET`` resolves to, writing objectives + robustness from in-memory reduced
metrics (no simulation-output timeseries persisted). MPI chunk-and-aggregate; degrades to serial when
MPI is unavailable. All configuration is env/registry-driven (no CLI value flags); ``--formulation``
and ``--seed`` are identifiers.

Policies (env ``NYCOPT_CHUNK_POLICIES``):
    ``baseline`` (default) - the default FFMP policy only (1 solution).
    a ``.ref`` path          - a reference set of Pareto policies (var1..varN [obj...] per line).

Launch (env-driven; via workflow/09_simulate_test_chunks.sh):

    NYCOPT_REEVAL_ENSEMBLE_PRESET=master_5yr_n128000 \\
    mpirun -np 64 python3 -m scripts.main.simulate_test_chunks --formulation ffmp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_policies(formulation: str) -> np.ndarray:
    """Resolve the policy DV matrix from ``NYCOPT_CHUNK_POLICIES`` (baseline or a .ref file)."""
    from src.formulations import get_baseline_values, get_bounds

    spec = os.environ.get("NYCOPT_CHUNK_POLICIES", "baseline")
    if spec == "baseline":
        return np.atleast_2d(get_baseline_values(formulation))
    ref = Path(spec)
    if not ref.exists():
        raise FileNotFoundError(f"NYCOPT_CHUNK_POLICIES='{spec}' is neither 'baseline' nor a file.")
    from src.load.reference_set import load_reference_set
    n_vars = len(get_bounds(formulation)[0])
    dvs, _objs = load_reference_set(ref, n_vars)
    return np.atleast_2d(np.asarray(dvs, dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", default="ffmp")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from src.chunk_reeval import simulate_test_chunks

    dvs = _load_policies(args.formulation)
    out = simulate_test_chunks(args.formulation, dvs, seed=args.seed)
    if out is not None:
        print(f"[chunk-reeval] done -> {out}")


if __name__ == "__main__":
    main()
