"""merge_test_chunks.py - Standalone merge for the chunked test-ensemble re-eval.

Merges the per-(solution, chunk) unit files a ``NYCOPT_CHUNK_MERGE=off``
simulate run (workflow step 09) left under ``partial/units/`` into the re-eval
cube + robustness scorecards, via :func:`src.chunk_reeval.merge_test_chunks`.
Single process; resumable (stateless over the unit files, overwrites its own
outputs). Refuses on missing units unless ``NYCOPT_CHUNK_MERGE_ALLOW_PARTIAL=1``.

The campaign identity (env file, ``NYCOPT_REEVAL_ENSEMBLE_PRESET``,
``NYCOPT_CHUNK_POLICIES``) must match the simulate jobs exactly — the policy
set defines the solution ids the units are keyed by. ``--formulation`` and
``--seed`` are identifiers, everything else is env (no CLI value flags).

Launch via workflow/09b_merge_test_chunks.sh.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", default="ffmp")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    from scripts.main.simulate_test_chunks import _load_policies
    from src.chunk_reeval import merge_test_chunks

    dvs = _load_policies(args.formulation)
    out = merge_test_chunks(args.formulation, list(range(dvs.shape[0])),
                            seed=args.seed)
    print(f"[chunk-reeval] merged -> {out}")


if __name__ == "__main__":
    main()
