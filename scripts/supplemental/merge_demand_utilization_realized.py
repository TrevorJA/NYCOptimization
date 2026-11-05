"""merge_demand_utilization_realized.py - Merge the per-chunk realized-utilization tables.

Every E_test chunk staged with per-SOW demand scaling
(``src.ensemble_prep.stage_nyc_demand``) writes ``demand_utilization_realized.csv``
(global_id, sow, u_nominal, k, u_realized) into its chunk dir. This concatenates
them into the variant's parent dir as ``demand_utilization_realized.csv`` and
reduces them per SOW to ``demand_utilization_realized_by_sow.csv``
(sow, u_nominal, k, clipped, u_realized_mean, u_realized_min, u_realized_max,
n_realizations), then prints nominal versus realized ``u``. Realized exceeds
nominal in dry SOWs by design (negative regression slopes); the table is a
diagnostic, never a correction.

Usage (identifier flag only):
    python scripts/supplemental/merge_demand_utilization_realized.py --variant kn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

#: Per-realization table written by the chunk staging.
REALIZED_CSV = "demand_utilization_realized.csv"
#: Per-SOW reduction written beside it in the parent dir.
REALIZED_BY_SOW_CSV = "demand_utilization_realized_by_sow.csv"
#: Column order of the per-realization table.
REALIZED_COLUMNS = ("global_id", "sow", "u_nominal", "k", "u_realized")


def merge_realized(slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the chunk tables of ``slug`` and reduce them per SOW.

    Args:
        slug: Staged parent E_test slug (chunked).

    Returns:
        ``(per_realization, per_sow)`` frames, also written to the parent dir.

    Raises:
        FileNotFoundError: Naming every chunk whose table is missing.
    """
    from src.ensembles import staged_ensemble_dir

    staged = staged_ensemble_dir(slug)
    idx_path = staged / "chunk_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} is missing; '{slug}' is not a chunked E_test.")
    nominal_path = staged / "demand_utilization_nominal.csv"
    if not nominal_path.exists():
        raise FileNotFoundError(
            f"{nominal_path} is missing; run append_demand_utilization.py first.")
    entries = json.loads(idx_path.read_text()).get("chunks") or []
    frames, missing = [], []
    for entry in entries:
        path = staged_ensemble_dir(entry["slug"]) / REALIZED_CSV
        if not path.exists():
            missing.append(entry["slug"])
            continue
        frames.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chunk(s) carry no {REALIZED_CSV} (e.g. {missing[:5]}); stage "
            f"their demand first (workflow/supplemental/staging/etest_stage_nyc_demand.sh).")
    per_real = (pd.concat(frames, ignore_index=True)[list(REALIZED_COLUMNS)]
                .sort_values("global_id").reset_index(drop=True))
    if per_real["global_id"].duplicated().any():
        raise ValueError("duplicate global ids across chunk tables.")

    nominal = pd.read_csv(nominal_path)
    grouped = per_real.groupby("sow")["u_realized"]
    per_sow = (nominal.merge(
        pd.DataFrame({
            "u_realized_mean": grouped.mean(),
            "u_realized_min": grouped.min(),
            "u_realized_max": grouped.max(),
            "n_realizations": grouped.size(),
        }).reset_index(), on="sow", how="left")
        .rename(columns={"u": "u_nominal"}))
    per_sow = per_sow[["sow", "u_nominal", "k", "clipped", "u_realized_mean",
                       "u_realized_min", "u_realized_max", "n_realizations"]]

    per_real.to_csv(staged / REALIZED_CSV, index=False)
    per_sow.to_csv(staged / REALIZED_BY_SOW_CSV, index=False)

    scored = per_sow.dropna(subset=["u_realized_mean"])
    diff = (scored["u_realized_mean"] - scored["u_nominal"]).to_numpy(dtype=float)
    print(f"[merge-u] '{slug}': {len(per_real)} realizations in {len(frames)} chunks, "
          f"{len(scored)}/{len(per_sow)} SOWs realized; realized - nominal u: mean "
          f"{diff.mean():+.4f}, min {diff.min():+.4f}, max {diff.max():+.4f}, "
          f"{int((diff > 0).sum())} SOWs above nominal.")
    print(f"[merge-u] wrote {staged / REALIZED_CSV} and {staged / REALIZED_BY_SOW_CSV}")
    return per_real, per_sow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the per-chunk realized-utilization tables of an E_test.")
    parser.add_argument("--variant", default=None,
                        help="E_test variant (default: src.etest.E_TEST_VARIANT).")
    parser.add_argument("--slug", default=None,
                        help="Chunked E_test slug to merge (default: the variant's slug).")
    args = parser.parse_args()

    from src.etest import E_TEST_VARIANT, get_etest_variant

    merge_realized(args.slug or get_etest_variant(args.variant or E_TEST_VARIANT).slug)


if __name__ == "__main__":
    main()
