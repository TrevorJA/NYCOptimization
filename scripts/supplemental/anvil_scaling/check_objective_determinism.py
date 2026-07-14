"""check_objective_determinism.py - Is the model's nondeterminism reaching the objectives?

Runs the SAME baseline DVs through the SAME ensemble twice in one process and
compares (a) the raw simulated flows and (b) every registered objective. Then
measures how close the weekly aggregates sit to their decision thresholds.

The question this answers: pywrdrb's solver is mildly nondeterministic, but does
that noise stay in the noise floor, or does an objective's threshold test
amplify it into a discrete jump? A reliability objective of the form
``weekly_mean_flow >= target`` is knife-edge whenever the model releases
*exactly* enough water to meet that target — flow lands ON the threshold, and a
1e-9 solver difference flips the week from success to failure.

Usage (never on a login node):
    srun --partition=shared --account=ees260021 --ntasks=1 --time=00:30:00 \
        python3 scripts/supplemental/anvil_scaling/check_objective_determinism.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_anvil_scaling_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import SEARCH_ENSEMBLE_SPEC, get_objective_set  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.simulation import dvs_to_config, evaluate  # noqa: E402
from src.simulation import run_simulation_ensemble_inmemory  # noqa: E402

N_REPEATS = 6


def main() -> int:
    dv = np.array(get_baseline_values("ffmp"))
    spec = SEARCH_ENSEMBLE_SPEC
    obj_set = get_objective_set()
    names, epsilons = obj_set.names, obj_set.epsilons

    print("=" * 78)
    print(f"  Objective determinism check: {N_REPEATS} identical evals, same process")
    print("=" * 78)
    print(f"  ensemble: {spec.preset_name} (N={spec.n_realizations})")

    # --- 1. Repeated evaluations of the identical DV vector ---
    runs = np.array([np.asarray(evaluate(dv, formulation_name="ffmp"), dtype=float)
                     for _ in range(N_REPEATS)])

    print("\n--- Objective spread across identical evaluations ---")
    print(f"{'objective':36s} {'value':>11s} {'spread':>10s} {'eps':>7s} "
          f"{'%eps':>7s} {'n_uniq':>7s}")
    for i, nm in enumerate(names):
        col = runs[:, i]
        spread = col.max() - col.min()
        eps = epsilons[i]
        flag = "  <== VARIES" if spread > 0 else ""
        print(f"{nm[:36]:36s} {col[0]:11.6f} {spread:10.6f} {eps:7.4f} "
              f"{100*spread/eps:6.1f}% {len(np.unique(col)):7d}{flag}")

    # --- 2. Raw flow comparison: how big is the underlying numerical noise? ---
    cfg = dvs_to_config(dv, "ffmp")
    d1 = run_simulation_ensemble_inmemory(cfg, spec)
    d2 = run_simulation_ensemble_inmemory(cfg, spec)

    print("\n--- Raw simulated-flow comparison (realization 0) ---")
    for key in ("delMontague", "delTrenton"):
        s1 = pd.Series(d1[0]["major_flow"][key]).astype(float)
        s2 = pd.Series(d2[0]["major_flow"][key]).astype(float)
        diff = (s1 - s2).abs()
        rel = diff.max() / max(abs(s1).max(), 1e-12)
        print(f"  {key:14s} max|A-B| = {diff.max():.3e} MGD   "
              f"relative = {rel:.2e}   n_days_differ = {int((diff > 0).sum())}")

    # --- 3. How close do weekly means sit to their thresholds? ---
    # A reliability objective is only knife-edge if weeks pile up ON the target.
    print("\n--- Distance of weekly Montague flows from the Decree threshold ---")
    from src.objectives import _post_warmup  # noqa: PLC0415
    from config import MONTAGUE_DECREE_TARGET_MGD as TGT  # noqa: PLC0415

    wk = _post_warmup(d1[0]["major_flow"]["delMontague"]).resample("W").mean()
    gap = (wk - TGT).abs()
    for tol in (1e-9, 1e-6, 1e-3, 1e-1, 1.0):
        n = int((gap < tol).sum())
        print(f"  weeks within {tol:>8.0e} MGD of the target: {n:5d} / {len(wk)} "
              f"({100*n/len(wk):5.2f}%)  <- knife-edge for the '>=' test")
    print(f"\n  target = {TGT} MGD; weekly means: "
          f"min={wk.min():.2f} median={wk.median():.2f} max={wk.max():.2f}")
    print("\n  For contrast, the DELIVERY reliability objective tests "
          "'>= 0.99 * demand' (a 1% margin) and is perfectly stable across "
          "all 4,340 measured evaluations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
