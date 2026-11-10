"""rescore_thresholds.py - Re-snapshot persisted re-eval cubes onto the current
satisficing thresholds and rescore them. Zero simulation.

Every re-eval cube carries the satisficing thresholds in force when step 10
wrote it (``reeval_raw_meta.json``), and the scorer reads only that snapshot
(``src.robustness``). The per-SOW objective values do not depend on the
thresholds, so a change of the adopted vector (``src.objectives_ensemble.
_DEFAULT_THRESHOLDS``) is applied to existing cubes by rewriting the snapshot
and rescoring, not by re-simulating.

For each cube on the E_test tag -- the incumbent cube
``outputs/baseline/reeval/{tag}/`` first, then every design leaf
``outputs/{design}/{slug}/reeval/{tag}[/seed_NN]/`` that holds a cube -- the
script:

1. builds the objective set for the cube's own ``obj_names`` and reads each
   objective's current threshold and kind from the live registry;
2. refuses a kind change (a direction change is a different objective, not a
   threshold change);
3. when the thresholds differ, appends the replaced vector to the meta's
   ``threshold_history`` with a timestamp and writes the new vector;
4. reruns :func:`src.robustness.run`, with the incumbent cube as the regret
   baseline for design leaves.

Identity comes from the environment (repo rule, no CLI value flags):
``NYCOPT_REEVAL_TAG`` (default: the campaign E_test tag) and
``NYCOPT_RESULTS_SLUG`` (default ``ffmp_obj8``). Source the production env file
first so the regret tolerance ladder resolves (``NYCOPT_REGRET_TAU``).

Run through ``workflow/supplemental/results_diagnostics/rescore_thresholds.sh``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src import robustness as rob  # noqa: E402
from src.figures.registry import default_reeval_tag  # noqa: E402
from src.objectives_ensemble import build_ensemble_objective_set  # noqa: E402
from src.reeval_core import (_CUBE_FILES, find_incumbent_cube,  # noqa: E402
                             incumbent_cube_dir)
from src.scenario_designs import campaign_designs  # noqa: E402


def _has_cube(leaf: Path) -> bool:
    return any((leaf / f).exists() for f in _CUBE_FILES)


def cube_leaves(tag: str, slug: str) -> list[Path]:
    """Incumbent cube first, then every design leaf (and seed subdir) with a cube."""
    leaves = []
    inc = incumbent_cube_dir(config.OUTPUTS_DIR / "baseline" / "reeval" / tag)
    if _has_cube(inc):
        leaves.append(inc)
    for design in campaign_designs():
        leaf = config.OUTPUTS_DIR / design / slug / "reeval" / tag
        candidates = [leaf] + sorted(leaf.glob("seed_*"))
        leaves += [c for c in candidates if c.is_dir() and _has_cube(c)]
    return leaves


def resnapshot(leaf: Path) -> dict:
    """Rewrite one cube's threshold snapshot from the live registry.

    Returns:
        ``{objective: (old, new)}`` for every objective whose threshold changed.

    Raises:
        SystemExit: On a satisficing-kind change or an objective the registry
            cannot resolve.
    """
    meta_path = leaf / "reeval_raw_meta.json"
    meta = json.loads(meta_path.read_text())
    names = list(meta["obj_names"])
    objs = list(build_ensemble_objective_set(names))
    if [o.name for o in objs] != names:
        sys.exit(f"[rescore] {leaf}: objective order mismatch "
                 f"{[o.name for o in objs]} != {names}")
    new_thr = {o.name: o.sat_threshold for o in objs}
    new_kind = {o.name: o.sat_kind for o in objs}
    old_thr = dict(meta.get("thresholds", {}))
    bad_kind = [n for n in names if meta.get("kinds", {}).get(n) != new_kind[n]]
    if bad_kind:
        sys.exit(f"[rescore] {leaf}: satisficing kind differs for {bad_kind}; "
                 f"that is an objective change, not a threshold change")
    changed = {n: (old_thr.get(n), new_thr[n]) for n in names
               if old_thr.get(n) != new_thr[n]}
    if changed:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = list(meta.get("threshold_history", []))
        history.append({"replaced": stamp, "thresholds": old_thr})
        meta["threshold_history"] = history
        meta["thresholds"] = new_thr
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
    return changed


def main() -> int:
    tag = default_reeval_tag()
    slug = os.environ.get("NYCOPT_RESULTS_SLUG", config.CAMPAIGN_RESULTS_SLUG)
    leaves = cube_leaves(tag, slug)
    if not leaves:
        print(f"[rescore] no re-eval cube on tag '{tag}' (slug '{slug}')")
        return 1
    print(f"[rescore] tag={tag} slug={slug}: {len(leaves)} cube(s)")
    for leaf in leaves:
        changed = resnapshot(leaf)
        if changed:
            for n, (a, b) in changed.items():
                print(f"[rescore]   {leaf.relative_to(config.OUTPUTS_DIR)}: "
                      f"{n} {a} -> {b}")
        else:
            print(f"[rescore]   {leaf.relative_to(config.OUTPUTS_DIR)}: "
                  f"snapshot already current")
        is_incumbent = leaf.parts[-3:-1] == ("baseline", "reeval")
        baseline = None if is_incumbent else find_incumbent_cube(leaf)
        rob.run(leaf, baseline)
    print("[rescore] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
