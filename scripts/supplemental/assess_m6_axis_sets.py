"""assess_m6_axis_sets.py - Score candidate m=6 axis sets against the adequacy gate.

One-off decision support for the (m, N, P) call after the nested-P verdict
(``nested_P_saturation.md``): the full m = 8 set cannot pass the tail-share gate
at any affordable P, so a reduced set must be chosen. This scores candidate
six-axis sets on prefix rungs of the staged P = 1e6 pool image with exactly the
battery conventions of ``diagnose_hazard_selectors.py`` (campaign ``lhs_nn`` at
N = 100, 10 selector seeds + 50-seed random null, robust p1/p99 bounds and pool
P90s recomputed per prefix and per axis subset; gate statistic = within-seed
minimum per-axis tail share, seed-averaged; criterion >= 0.30).

Candidates (fixed here, with the reasoning in the nested-P results discussion):

    proposed   drop the two structurally hard axes — drought_duration
               (quasi-discrete; |rho_S| = 0.87 with deficit volume, the most
               redundant pair retained) and flood_rise_rate (entangled with its
               flood-group partners; generator-limited daily-extreme statistic).
               Keeps every hazard concept group with its most enrichable members.
    swap_dur   as proposed but keep drought_duration instead of
               drought_peak_depth (sensitivity: values the duration concept over
               instantaneous depth at the cost of higher redundancy with deficit).
    diag_m6    the nested-P diagnostic m6 nesting (reference/validation row; its
               numbers must reproduce the ladder's).

All configuration is via environment variables (no CLI value flags):

    NYCOPT_SELDIAG_POOL_SLUG   staged pool slug (default statpool_10yr_n1000000_d0)
    NYCOPT_M6ASSESS_RUNGS      prefix sizes (default "100000 1000000")

Output -> ``outputs/supplemental/hazard_selector_diagnostics/m6_axis_set_assessment.json``
plus a table on stdout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from scengen import selector_diagnostics as sd  # noqa: E402
from scengen import subsample as ss  # noqa: E402
from scengen.diagnostics import load_hazard_image  # noqa: E402
from scengen.hazard_filling import screen_hazard_axes  # noqa: E402

POOL_SLUG = os.environ.get("NYCOPT_SELDIAG_POOL_SLUG", "statpool_10yr_n1000000_d0")
RUNGS = [int(p) for p in os.environ.get("NYCOPT_M6ASSESS_RUNGS", "100000 1000000").split()]
N_SELECT, N_SEEDS, N_NULL_SEEDS = 100, 10, 50
TAIL_CRITERION = 0.30

CANDIDATES: dict[str, list[str]] = {
    "proposed": [
        "drought_deficit_volume", "drought_peak_depth", "drought_onset_rate",
        "drought_recovery_rate", "flood_peak_magnitude", "flood_pulse_duration",
    ],
    "swap_dur": [
        "drought_deficit_volume", "drought_duration", "drought_onset_rate",
        "drought_recovery_rate", "flood_peak_magnitude", "flood_pulse_duration",
    ],
    "diag_m6": [
        "drought_deficit_volume", "drought_onset_rate", "flood_peak_magnitude",
        "flood_pulse_duration", "drought_duration", "flood_rise_rate",
    ],
}


def _score(H: np.ndarray, axes: list[str]) -> dict:
    """Gate statistics + snap concentration for one axis set on one prefix."""
    records = []
    for selector, seeds in (("lhs_nn", range(N_SEEDS)), ("random", range(N_NULL_SEEDS))):
        for seed in seeds:
            rows = (ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
                    if selector == "lhs_nn" else
                    ss.random_subsample(H, N_SELECT, seed=seed))
            for axis, m in sd.per_axis_selection_metrics(H, rows, axes).items():
                records.append({"selector": selector, "seed": seed, "axis": axis,
                                "tail": m["tail_share_p90"]})
    t = pd.DataFrame.from_records(records)
    sel = t.loc[t.selector == "lhs_nn"]
    by_seed = sel.groupby("seed")["tail"]
    axis_means = sel.groupby("axis")["tail"].mean()

    X = ss.minmax_normalize(H)
    concs = []
    for seed in range(N_SEEDS):
        res = sd.select_lhs_nn(X, N_SELECT, seed=seed)
        concs.append(sd.distance_concentration(
            X, res.info["snap_distances"], seed=seed)["concentration_ratio"])

    return {
        "tail_share_min": float(by_seed.min().mean()),
        "tail_share_mean": float(by_seed.mean().mean()),
        "worst_axis": str(axis_means.idxmin()),
        "per_axis_seed_mean": {str(a): float(v) for a, v in axis_means.items()},
        "null_mean": float(t.loc[t.selector == "random", "tail"].mean()),
        "concentration_ratio": float(np.mean(concs)),
        "gate_pass": bool(by_seed.min().mean() >= TAIL_CRITERION),
    }


def main() -> None:
    """Score every candidate set at every rung; print the decision table."""
    path = config.STAGED_ENSEMBLE_DIR / POOL_SLUG / "hazard_image.npz"
    img = load_hazard_image(path)
    H_full, candidate_axes = img["H"], list(img["hazard_axes"])

    results: dict = {"pool_slug": POOL_SLUG, "n_select": N_SELECT, "seeds": N_SEEDS,
                     "null_seeds": N_NULL_SEEDS, "tail_criterion": TAIL_CRITERION,
                     "candidates": {k: list(v) for k, v in CANDIDATES.items()},
                     "scores": {}}
    for p in RUNGS:
        H_prefix = H_full[:p]
        screen = screen_hazard_axes(H_prefix, candidate_axes)
        assert all(a in screen["retained"] for c in CANDIDATES.values() for a in c)
        for name, axes in CANDIDATES.items():
            cols = [candidate_axes.index(a) for a in axes]
            s = _score(H_prefix[:, cols], axes)
            results["scores"].setdefault(str(p), {})[name] = s
            print(f"[m6assess] P={p:>9,} {name:>9}: min={s['tail_share_min']:.3f} "
                  f"mean={s['tail_share_mean']:.3f} conc={s['concentration_ratio']:.3f} "
                  f"worst={s['worst_axis']} ({s['per_axis_seed_mean'][s['worst_axis']]:.3f}) "
                  f"{'PASS' if s['gate_pass'] else 'fail'}")

    out = config.OUTPUTS_DIR / "supplemental" / "hazard_selector_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m6_axis_set_assessment.json").write_text(json.dumps(results, indent=2))
    print(f"[m6assess] wrote {out / 'm6_axis_set_assessment.json'}")


if __name__ == "__main__":
    main()
