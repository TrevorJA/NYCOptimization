"""Confirmatory checks on Borg search archives (.set files).

Verifies the archive shape and the NYC weekly reliability floor constraint for
every member of every seed archive in a run's ``sets/`` directory:

  * every row carries exactly ``n_dv + n_obj`` columns (feasible-only archives
    carry no constraint columns);
  * the NYC weekly delivery reliability of every member sits at or above
    ``config.NYC_RELIABILITY_FLOOR`` on the natural 0-1 scale (archives store
    maximize objectives negated for Borg's minimization);
  * archive sizes, reported per seed for comparison against the expected range.

Usage:
    python scripts/supplemental/check_search_archives.py \
        outputs/historic/ffmp_obj8_mm_full/sets [--expect-min 800 --expect-max 1600]

Exit status is 1 if any member violates the floor or any row is mis-shaped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
import src.objectives_ensemble as objectives_ensemble  # noqa: E402
from src.formulations import (  # noqa: E402
    get_formulation,
    reliability_floor_objective_index,
)


def _load_set(path: Path) -> tuple[np.ndarray, list[int]]:
    """Return the numeric rows of a ``.set`` file and any off-width row widths.

    MOEAFramework ``.set`` files carry ``#``-prefixed comment lines and a
    trailing ``#`` terminator; everything else is a whitespace-separated
    solution row.
    """
    rows: list[list[float]] = []
    widths: list[int] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            widths.append(len(parts))
            rows.append([float(p) for p in parts])
    if not rows:
        return np.empty((0, 0)), widths
    width = widths[0]
    if any(w != width for w in widths):
        return np.empty((0, 0)), widths
    return np.asarray(rows, dtype=float), widths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sets_dir", type=Path, help="run's sets/ directory")
    ap.add_argument("--expect-min", type=int, default=None,
                    help="lower end of the expected members-per-seed range")
    ap.add_argument("--expect-max", type=int, default=None,
                    help="upper end of the expected members-per-seed range")
    ap.add_argument("--formulation", default="ffmp",
                    help="formulation whose DV count sets the expected width")
    args = ap.parse_args()

    n_dv = len(get_formulation(args.formulation)["decision_variables"])
    obj_names = config.get_obj_names()
    n_obj = len(obj_names)
    expected_width = n_dv + n_obj

    rel_idx = reliability_floor_objective_index(obj_names)
    floor = config.NYC_RELIABILITY_FLOOR
    rel_col = n_dv + rel_idx
    rel_spec = objectives_ensemble._build_registry()[obj_names[rel_idx]]
    # Borg minimizes: maximize objectives are stored negated in the archive.
    sign = -1.0 if rel_spec.direction == "maximize" else 1.0

    print(f"expected width      : {expected_width} ({n_dv} DV + {n_obj} obj)")
    print(f"floor objective     : {obj_names[rel_idx]} (obj index {rel_idx}, "
          f"column {rel_col}, direction {rel_spec.direction})")
    print(f"reliability floor   : {floor}")
    print()

    set_files = sorted(p for p in args.sets_dir.glob("*.set")
                       if "_merged" not in p.name)
    if not set_files:
        print(f"FAIL: no per-seed .set files under {args.sets_dir}")
        return 1

    failed = False
    for path in set_files:
        rows, widths = _load_set(path)
        if rows.size == 0:
            print(f"FAIL {path.name}: no uniform numeric rows "
                  f"(widths seen: {sorted(set(widths))})")
            failed = True
            continue
        n_members, width = rows.shape
        ok_width = width == expected_width
        rel = sign * rows[:, rel_col]
        rel_min, rel_max = float(rel.min()), float(rel.max())
        n_below = int((rel < floor - 1e-9).sum())
        n_at = int((np.abs(rel - floor) <= 1e-9).sum())

        print(f"{path.name}")
        print(f"  members           : {n_members}")
        print(f"  columns           : {width} "
              f"({'OK' if ok_width else 'MISMATCH'}; all rows equal width: "
              f"{len(set(widths)) == 1})")
        print(f"  NYC weekly rel    : min {rel_min:.4f}  max {rel_max:.4f}")
        print(f"  below floor       : {n_below}   exactly at floor: {n_at}")
        if args.expect_min is not None and args.expect_max is not None:
            in_range = args.expect_min <= n_members <= args.expect_max
            print(f"  size vs expected  : [{args.expect_min}, {args.expect_max}] "
                  f"-> {'in range' if in_range else 'OUT OF RANGE'}")
        print()

        if not ok_width or n_below:
            failed = True

    print("VERDICT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
