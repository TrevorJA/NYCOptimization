"""
random_sample_salinity_test.py - Smoke-test the salinity LSTM objective by
sampling N random FFMP decision vectors uniformly within bounds, simulating
each, and reporting the distribution of `salt_front_max_rm_excursion`.

This is a development-time tool, not a research artifact: it confirms that
the salinity LSTM is responsive to NYC operational decisions (i.e., not
returning a constant for every DV vector) and gives a quick read on the
worst-case downstream salt-front intrusion the policy class can produce.

Required env (set by `slurm/envs/ffmp_obj8_sal.env` or equivalent):
    NYCOPT_SALINITY_ON=1
    NYCOPT_OBJECTIVES=...,salt_front_max_rm_excursion

Optional env:
    PYWRDRB_SIM_START_DATE=2018-01-01    # shorter sim window for speed
    PYWRDRB_SIM_END_DATE=2022-12-31

Usage:
    python scripts/random_sample_salinity_test.py [--n 10] [--seed 42]
        [--formulation ffmp]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, derive_slug, ACTIVE_OBJECTIVES, INCLUDE_SALINITY_MODEL
from src.formulations import (
    get_baseline_values,
    get_formulation,
    get_var_names,
    get_objective_set,
)
from src.simulation import dvs_to_config, run_simulation_inmemory


def _sample_random_dvs(formulation: str, rng: np.random.Generator,
                       n_samples: int) -> np.ndarray:
    """Uniform random samples within DV bounds. Shape (n_samples, n_vars)."""
    f = get_formulation(formulation)
    var_names = get_var_names(formulation)
    lows = np.array([f["decision_variables"][n]["bounds"][0] for n in var_names])
    highs = np.array([f["decision_variables"][n]["bounds"][1] for n in var_names])
    return rng.uniform(low=lows, high=highs, size=(n_samples, len(lows)))


def _run_one(dv: np.ndarray, formulation: str, objective_set) -> dict:
    """Simulate one DV vector and return summary dict.

    Includes:
      - All active objective values (NaN on failure)
      - Salt-front time-series statistics (min/median/max RM)
      - Wall time
    """
    out = {"elapsed_s": float("nan")}
    t0 = time.perf_counter()
    try:
        cfg = dvs_to_config(dv, formulation)
        data = run_simulation_inmemory(cfg, use_trimmed=True)
        objs = objective_set.compute(data)
        out["elapsed_s"] = time.perf_counter() - t0
        for name, val in zip(objective_set.names, objs):
            out[name] = float(val)
        if "salinity" in data:
            sf = data["salinity"]["salt_front_location_mu"].dropna()
            if len(sf):
                out["sf_min_RM"] = float(sf.min())
                out["sf_median_RM"] = float(sf.median())
                out["sf_max_RM"] = float(sf.max())
            else:
                out["sf_min_RM"] = out["sf_median_RM"] = out["sf_max_RM"] = float("nan")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["elapsed_s"] = time.perf_counter() - t0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10,
                        help="Number of random DV samples (default 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default 42)")
    parser.add_argument("--formulation", default="ffmp",
                        help="Formulation name (default ffmp)")
    args = parser.parse_args()

    if not INCLUDE_SALINITY_MODEL:
        sys.exit("ERROR: NYCOPT_SALINITY_ON must be 1. "
                 "Source slurm/envs/ffmp_obj8_sal.env first.")
    if "salt_front_max_rm_excursion" not in ACTIVE_OBJECTIVES:
        sys.exit("ERROR: NYCOPT_OBJECTIVES must include 'salt_front_max_rm_excursion'. "
                 "Source slurm/envs/ffmp_obj8_sal.env first.")

    objective_set = get_objective_set()
    rng = np.random.default_rng(args.seed)

    # Run baseline first as a reference row.
    baseline_dv = np.asarray(get_baseline_values(args.formulation), dtype=float)
    print(f"=== Random-sample salinity test ===")
    print(f"  formulation: {args.formulation}")
    print(f"  n samples:   {args.n} (+ baseline)")
    print(f"  seed:        {args.seed}")
    print(f"  active obj:  {len(ACTIVE_OBJECTIVES)}")
    print()

    rows = []
    print("[baseline]", end="", flush=True)
    base_row = _run_one(baseline_dv, args.formulation, objective_set)
    base_row["sample_id"] = -1
    rows.append(base_row)
    print(f"  excursion={base_row.get('salt_front_max_rm_excursion'):.2f} RM"
          f"  ({base_row.get('elapsed_s', 0):.1f}s)")

    samples = _sample_random_dvs(args.formulation, rng, args.n)
    for i in range(args.n):
        print(f"[sample {i+1:02d}]", end="", flush=True)
        r = _run_one(samples[i], args.formulation, objective_set)
        r["sample_id"] = i
        rows.append(r)
        if "error" in r:
            print(f"  FAIL: {r['error']}")
        else:
            print(f"  excursion={r.get('salt_front_max_rm_excursion'):.2f} RM"
                  f"  sf_min={r.get('sf_min_RM', float('nan')):.2f}"
                  f"  sf_max={r.get('sf_max_RM', float('nan')):.2f}"
                  f"  ({r.get('elapsed_s', 0):.1f}s)")

    df = pd.DataFrame(rows).set_index("sample_id")
    out_dir = OUTPUTS_DIR / "random_sample_tests" / derive_slug(args.formulation)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"random_samples_seed{args.seed}_n{args.n}.csv"
    df.to_csv(out_csv)
    print(f"\n=== Saved {out_csv} ===")

    print("\n=== Summary (random samples only, baseline excluded) ===")
    rs = df.drop(index=-1, errors="ignore")
    if "salt_front_max_rm_excursion" in rs.columns:
        ex = rs["salt_front_max_rm_excursion"].dropna()
        if len(ex):
            print(f"salt_front_max_rm_excursion (RM, lower = better):")
            print(f"  mean   = {ex.mean():.2f}")
            print(f"  median = {ex.median():.2f}")
            print(f"  min    = {ex.min():.2f}   (best policy in sample)")
            print(f"  max    = {ex.max():.2f}   (WORST CASE in sample)")
            worst_idx = int(ex.idxmax())
            print(f"  worst sample_id: {worst_idx}")
    if "sf_min_RM" in rs.columns:
        sf_min = rs["sf_min_RM"].dropna()
        if len(sf_min):
            print(f"\nsf_min_RM (instantaneous worst RM observed during sim):")
            print(f"  median across samples = {sf_min.median():.2f}")
            print(f"  most-upstream excursion across all samples = {sf_min.min():.2f}")


if __name__ == "__main__":
    main()
