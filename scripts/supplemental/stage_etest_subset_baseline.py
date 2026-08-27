"""
stage_etest_subset_baseline.py - Symlink the incumbent's full-E_test cube under a subset re-eval tag.

The incumbent-relative regret family needs a baseline per-SOW matrix at
``reeval/{tag}[/seed_NN]/baseline/`` (see ``src.chunk_reeval._persist_and_score``). For a
chunk-prefix E_test subset no re-simulation is needed: the incumbent's EXISTING full-E_test
cube is a strict superset of the subset's SOWs, and ``src.robustness._aligned_baseline`` joins
baseline to policy cube BY SOW LABEL (never by position), so the full cube aligns exactly to
the prefix labels. This script just plants a symlink.

The incumbent cube is design-independent (one fixed policy on the one common measuring stick),
so the same source directory serves every scenario design's subset re-eval. The
scenario-matched-baseline rule applies to search-ensemble objective baselines (step 05
``--search-ensemble``), not here.

Run it with the SAME env the step-09 submission will use, so the target directory resolves
identically (scenario, slug, re-eval tag, optional seed):

    set -a; source workflow/envs/ffmp_obj8_historic_production.env; set +a
    export NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch   # the campaign subset
    python3 -m scripts.supplemental.stage_etest_subset_baseline

Run it once per scenario design env (the target dir is design-specific) before that
design's step-09 re-evaluation on the subset.

Pass ``--seed N`` iff the step-09 run will pass ``--seed N`` (SEED env in the launcher).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: The incumbent-on-E_test per-SOW cube, regenerated on the unified substrate 2026-08-08
#: (jobs 19733672 + 19738752). Lives under the historic mm_full tree but is design-independent.
DEFAULT_BASELINE_SRC = (
    "outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symlink the incumbent full-E_test cube under a subset re-eval tag.")
    parser.add_argument("--baseline-src", default=DEFAULT_BASELINE_SRC,
                        help=f"Incumbent cube directory (default: {DEFAULT_BASELINE_SRC}).")
    parser.add_argument("--formulation", default="ffmp")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set iff the step-09 run will pass --seed.")
    args = parser.parse_args()

    src = Path(args.baseline_src).resolve()
    if not any((src / f).exists() for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")):
        raise FileNotFoundError(
            f"{src} carries no reeval_raw.parquet/.csv.gz — not a baseline cube.")

    from config import REEVAL_ENSEMBLE_SPEC, active_scenario_name, derive_slug
    from src.reeval_core import reeval_output_dir, reeval_tag

    if REEVAL_ENSEMBLE_SPEC is None or not REEVAL_ENSEMBLE_SPEC.is_ensemble:
        raise ValueError("NYCOPT_REEVAL_ENSEMBLE_PRESET must resolve to the subset ensemble.")
    reeval_dir = reeval_output_dir(active_scenario_name(), derive_slug(args.formulation),
                                   REEVAL_ENSEMBLE_SPEC, args.seed)
    link = reeval_dir / "baseline"

    if link.is_symlink():
        if link.resolve() == src:
            print(f"[subset-baseline] already staged: {link} -> {src}")
            return
        raise FileExistsError(f"{link} is a symlink to {link.resolve()}, not {src}; "
                              f"remove it first if that is intended.")
    if link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink; refusing to touch it.")
    link.symlink_to(src)
    print(f"[subset-baseline] {active_scenario_name()} / tag "
          f"{reeval_tag(REEVAL_ENSEMBLE_SPEC)}: {link} -> {src}")


if __name__ == "__main__":
    sys.exit(main())
