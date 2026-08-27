"""
extract_runtime_archive.py - Equal-NFE archives from the runtime files.

The campaign's ``production`` MOEA config runs seed 1 to 187,500 NFE per
island and every later seed to 125,000 (``max_evaluations_by_seed``), and
reports all seeds at 125,000. The reporting NFE is the config's default
``max_evaluations``; it is read from config, never passed as a flag.

Two modes, run from the repo root with the run's env file sourced (the same
identity contract as workflow step 06; ``DRAW`` / ``NYCOPT_ENSEMBLE_DRAW``
selects the draw):

    # seed 1's archive at the reporting NFE, from its island runtime files
    python scripts/main/extract_runtime_archive.py --seed 1
        -> outputs/{scenario}/{slug}/sets/seed_01_{slug}_nfe125000.set

    # the per-design equal-NFE reference set across all seeds
    python scripts/main/extract_runtime_archive.py --merge [--install]
        -> outputs/{scenario}/{slug}/sets/{slug}_merged_nfe125000.set

``--merge`` takes, for every seed 1..n_seeds, the snapshot file above when the
seed's budget exceeds the reporting NFE and the final ``seed_SS_{slug}.set``
when it equals it, unions them, and epsilon-box filters the union under the
campaign vector. ``--install`` additionally copies the result to
``{slug}_merged.set`` (backing up an existing one as
``{slug}_merged_pre_install.set``) so the step-08/09 re-evaluation, which
resolves ``{slug}_merged.set`` first, consumes the equal-NFE front without
any change to the pipeline. Step 07's own ``{slug}_merged.set`` is built from
EVERY runtime snapshot, including seed 1's tail beyond the reporting NFE, so
it is not the campaign reference set.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (  # noqa: E402
    ACTIVE_MOEA_CONFIG,
    active_scenario_name,
    derive_slug,
    get_epsilons,
    run_output_dir,
)
from src.formulations import get_obj_names, get_var_names  # noqa: E402
from src.load.reference_set import load_set_file  # noqa: E402
from src.runtime_archive import epsilon_merge, snapshot_at, write_set_file  # noqa: E402


def _snapshot_set_path(sets_dir: Path, slug: str, seed: int, nfe: int) -> Path:
    return sets_dir / f"seed_{seed:02d}_{slug}_nfe{nfe}.set"


def extract_seed(seed: int, formulation: str, nfe: int) -> Path:
    """Write seed ``seed``'s epsilon archive at ``nfe`` from its runtime files."""
    scenario = active_scenario_name()
    slug = derive_slug(formulation)
    runtime_dir = run_output_dir(scenario, slug, "runtime")
    sets_dir = run_output_dir(scenario, slug, "sets")
    files = sorted(runtime_dir.glob(f"seed_{seed:02d}_{slug}_*.runtime"))
    if not files:
        raise FileNotFoundError(f"no runtime files for seed {seed} in {runtime_dir}")
    var_names = get_var_names(formulation)
    rows = snapshot_at(files, nfe)
    n_raw = len(rows)
    rows = epsilon_merge(rows, len(var_names), get_epsilons())
    out = _snapshot_set_path(sets_dir, slug, seed, nfe)
    write_set_file(out, rows, var_names, get_obj_names(), header_lines=[
        f"Formulation: {formulation}, Seed: {seed}",
        f"Runtime snapshot at NFE/island={nfe} from {len(files)} island file(s); "
        f"{n_raw} rows unioned, {len(rows)} retained under the campaign epsilons",
    ])
    print(f"[extract] seed {seed}: {len(files)} islands @ NFE {nfe}: "
          f"{n_raw} -> {len(rows)} rows -> {out}")
    return out


def merge_equal_nfe(formulation: str, nfe: int, install: bool) -> Path:
    """Union every seed's equal-NFE archive and epsilon-box filter it."""
    scenario = active_scenario_name()
    slug = derive_slug(formulation)
    sets_dir = run_output_dir(scenario, slug, "sets")
    mc = ACTIVE_MOEA_CONFIG
    var_names = get_var_names(formulation)
    parts, provenance = [], []
    for seed in range(1, (mc.n_seeds or 1) + 1):
        budget = mc.max_evaluations_for_seed(seed)
        if budget is None or budget < nfe:
            raise ValueError(f"seed {seed} budget {budget} is below the reporting NFE {nfe}")
        src = (_snapshot_set_path(sets_dir, slug, seed, nfe) if budget > nfe
               else sets_dir / f"seed_{seed:02d}_{slug}.set")
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing (seed {seed} budget {budget}); run --seed {seed} first"
                if budget > nfe else f"{src} missing (seed {seed} has not completed)")
        rows = load_set_file(src)
        parts.append(rows)
        provenance.append(f"{src.name} ({len(rows)} rows)")
    union = np.vstack([p for p in parts if p.size])
    merged = epsilon_merge(union, len(var_names), get_epsilons())
    out = sets_dir / f"{slug}_merged_nfe{nfe}.set"
    write_set_file(out, merged, var_names, get_obj_names(), header_lines=[
        f"Formulation: {formulation}, equal-NFE merge at NFE/island={nfe}",
        "Sources: " + "; ".join(provenance),
        f"{len(union)} rows unioned, {len(merged)} retained under the campaign epsilons",
    ])
    print(f"[merge] {len(provenance)} seeds @ NFE {nfe}: {len(union)} -> {len(merged)} rows -> {out}")
    if install:
        target = sets_dir / f"{slug}_merged.set"
        if target.exists():
            backup = sets_dir / f"{slug}_merged_pre_install.set"
            shutil.copy2(target, backup)
            print(f"[merge] existing {target.name} backed up as {backup.name}")
        shutil.copy2(out, target)
        print(f"[merge] installed as {target}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--formulation", default="ffmp")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int, help="extract this seed's snapshot")
    mode.add_argument("--merge", action="store_true",
                      help="build the equal-NFE merged set across seeds")
    parser.add_argument("--install", action="store_true",
                        help="with --merge: also write {slug}_merged.set")
    args = parser.parse_args()

    nfe = ACTIVE_MOEA_CONFIG.max_evaluations
    if nfe is None:
        raise SystemExit("active MOEA config has no max_evaluations (schema-only)")
    if args.merge:
        merge_equal_nfe(args.formulation, nfe, args.install)
    else:
        extract_seed(args.seed, args.formulation, nfe)


if __name__ == "__main__":
    main()
