"""verify_kirsch_baseline_persistence.py - Independent check of the no-persistence baseline.

Re-verifies the baseline claim behind the persistence-axis decision
(``docs/notes/methods/persistence_axis_diagnostics.md``): the unperturbed Kirsch
generator produces annual lag-1 autocorrelation ~ 0 in aggregate NYC inflow, versus
~0.22 (calendar-year) / ~0.26 (water-year) on the fitted record.

Independence from the prototype: this driver uses only the public
``KirschGenerator.generate()`` ensemble path (internal ``rng.choice`` bootstrap; no
injected index matrix and no ``scengen.persistence`` import), so it exercises the
exact code path production staging uses. Alongside rho1 it records multi-year
drought proxies (longest below-median run, min 3-yr rolling total) and the annual
standard-deviation ratio (synthetic vs fitted record), the companion symptom of
missing interannual dependence (annual-total under-dispersion).

Configuration (no CLI value flags):

    NYCOPT_PERSVERIFY_R        realizations             (default 64)
    NYCOPT_PERSVERIFY_YEARS    years per realization    (default 50)
    NYCOPT_PERSVERIFY_SEED     master seed              (default 20260729)

Outputs -> ``outputs/supplemental/persistence_axis/baseline_verification/``:
    realization_metrics.csv   per-realization metrics
    findings.json             ensemble summaries vs historical references
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

import numpy as np
import pandas as pd

import config

from src.ensemble_generation import _fit_kirsch, _load_masked_flows

R = int(os.environ.get("NYCOPT_PERSVERIFY_R", "64"))
YEARS = int(os.environ.get("NYCOPT_PERSVERIFY_YEARS", "50"))
SEED = int(os.environ.get("NYCOPT_PERSVERIFY_SEED", "20260729"))

NYC = ["cannonsville", "pepacton", "neversink"]

OUT_DIR = config.OUTPUTS_DIR / "supplemental" / "persistence_axis" / "baseline_verification"


def _annual_rho1(ann: np.ndarray) -> float:
    """Lag-1 autocorrelation of annual totals."""
    x = ann - ann.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else np.nan


def _annual_totals(agg: pd.Series, *, water_year: bool) -> np.ndarray:
    """Complete-year annual totals of a monthly aggregate series."""
    year = agg.index.year + ((agg.index.month >= 10).astype(int) if water_year else 0)
    tot = agg.groupby(year).agg(["sum", "size"])
    return tot[tot["size"] == 12]["sum"].to_numpy(dtype=float)


def _series_metrics(df: pd.DataFrame) -> dict:
    """Persistence metrics of one monthly realization (aggregate NYC inflow)."""
    agg = df[NYC].sum(axis=1)
    out = {}
    for wy, tag in ((False, "cy"), (True, "wy")):
        ann = _annual_totals(agg, water_year=wy)
        out[f"rho1_{tag}"] = _annual_rho1(ann)
        if wy:
            below = ann < np.median(ann)
            runs, run = [], 0
            for b in below:
                run = run + 1 if b else 0
                runs.append(run)
            roll3 = np.convolve(ann, np.ones(3), mode="valid")
            out["max_dry_run"] = int(np.max(runs))
            out["min_roll3_frac"] = float(roll3.min() / (3.0 * ann.mean()))
            out["annual_std"] = float(np.std(ann, ddof=1))
            out["annual_mean"] = float(np.mean(ann))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Q_gage, _, _ = _load_masked_flows("pub_nhmv10_BC_withObsScaled")
    Q_gen = Q_gage.loc["1945-10-01":"2022-09-30"]
    kirsch = _fit_kirsch(Q_gen)

    agg_h = Q_gen[NYC].sum(axis=1).resample("MS").sum()
    hist = {}
    for wy, tag in ((False, "cy"), (True, "wy")):
        ann_h = _annual_totals(agg_h, water_year=wy)
        hist[f"rho1_{tag}"] = _annual_rho1(ann_h)
        if wy:
            hist["annual_std"] = float(np.std(ann_h, ddof=1))
            hist["annual_mean"] = float(np.mean(ann_h))
            hist["n_years"] = len(ann_h)

    ensemble = kirsch.generate(n_realizations=R, n_years=YEARS, seed=SEED)
    rows = [_series_metrics(df) for df in ensemble.data_by_realization.values()]
    m = pd.DataFrame(rows)
    m["annual_std_ratio"] = (m["annual_std"] / m["annual_mean"]) / (
        hist["annual_std"] / hist["annual_mean"]
    )
    m.to_csv(OUT_DIR / "realization_metrics.csv", index=False)

    findings = {
        "R": R, "years": YEARS, "seed": SEED,
        "historical": hist,
        "synthetic": {
            "rho1_cy_mean": float(m["rho1_cy"].mean()),
            "rho1_cy_se": float(m["rho1_cy"].std() / np.sqrt(R)),
            "rho1_wy_mean": float(m["rho1_wy"].mean()),
            "rho1_wy_se": float(m["rho1_wy"].std() / np.sqrt(R)),
            "max_dry_run_mean": float(m["max_dry_run"].mean()),
            "min_roll3_frac_p10": float(m["min_roll3_frac"].quantile(0.10)),
            "annual_cv_ratio_mean": float(m["annual_std_ratio"].mean()),
        },
    }
    (OUT_DIR / "findings.json").write_text(json.dumps(findings, indent=2))

    s = findings["synthetic"]
    print(f"[persverify] historical rho1: cy={hist['rho1_cy']:+.3f}, wy={hist['rho1_wy']:+.3f} "
          f"(n={hist['n_years']})")
    print(f"[persverify] synthetic rho1: cy={s['rho1_cy_mean']:+.3f} (se {s['rho1_cy_se']:.3f}), "
          f"wy={s['rho1_wy_mean']:+.3f} (se {s['rho1_wy_se']:.3f})")
    print(f"[persverify] annual CV ratio (syn/hist) = {s['annual_cv_ratio_mean']:.3f}, "
          f"max dry run = {s['max_dry_run_mean']:.1f} yr, "
          f"p10 min 3-yr total frac = {s['min_roll3_frac_p10']:.3f}")
    print(f"[persverify] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
