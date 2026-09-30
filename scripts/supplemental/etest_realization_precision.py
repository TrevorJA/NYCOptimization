"""etest_realization_precision.py - What realizations per SOW buy on E_test.

Supplemental table (campaign_design.md §5; SI Text S8.5) behind the choice of
R, the number of stochastic realizations crossed with each state of the world
(SOW) of the re-evaluation ensemble. Table-only, no simulation.

Three tables:

``etr_per_sow_noise.csv``
    Per objective and R in ``ETR_R_LEVELS``: pooled unit-years per SOW, the
    per-SOW estimator noise sigma_i(R) scaled from the persisted pass-A noise
    floor measured at R = 25 on the current FFMP policy's E_test cube
    (``rtol_noise_floor.csv``; realizations within a SOW are independent, so
    sigma scales as sqrt(25 / R)), the false-harm tolerance floor
    z sqrt(2) sigma_i(R), its ratio to the epsilon precision, and the expected
    share of SOWs misclassified against the current satisficing criterion
    (the criterion's density of per-SOW values, from the incumbent's threshold
    sweep, convolved with the noise). The pass-A floor is the unpaired upper
    bound; the paired floors of the regret-tolerance note scale identically.

``etr_cross_sow_se.csv``
    The worst-case binomial standard error of a satisficing fraction for
    N_theta in ``ETR_N_THETA_LEVELS``. Independent of R.

``etr_options.csv``
    SU pricing of the E_test options against the remaining balance, from the
    measured re-evaluation rate and the staging costs named in
    ``supplemental_config.py`` (``ETR_*``).

Settings in ``supplemental_config.py``; no CLI value flags. Run::

    python scripts/supplemental/etest_realization_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_dryenv_env()

import config  # noqa: E402
from src.etest import E_TEST_R, E_TEST_REEVAL_N_THETA, E_TEST_YEARS  # noqa: E402

TABLES = scfg.ETR_TABLES_DIR


def _incumbent_cdf(sweep: pd.DataFrame, objective: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-SOW value bins and masses of the current FFMP policy from its threshold sweep."""
    s = sweep[sweep["objective"] == objective].sort_values("threshold")
    if s.empty:
        return None
    t = s["threshold"].to_numpy(dtype=float)
    kind = str(s["kind"].iloc[0])
    F = 1.0 - s["frac_sow"].to_numpy(dtype=float) if kind == "ge" else s["frac_sow"].to_numpy(dtype=float)
    F = np.clip(np.maximum.accumulate(F), 0.0, 1.0)
    mass = np.diff(np.concatenate([[0.0], F, [1.0]]))
    centres = np.concatenate([[t[0]], (t[:-1] + t[1:]) / 2, [t[-1]]])
    return centres, mass


def misclassified_share(centres: np.ndarray, mass: np.ndarray, criterion: float,
                        sigma: float) -> float:
    """Expected share of SOWs whose noisy per-SOW value falls on the wrong side of ``criterion``."""
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    return float(np.sum(mass * norm.cdf(-np.abs(centres - criterion) / sigma)))


def per_sow_noise_table() -> pd.DataFrame:
    floor = pd.read_csv(scfg.ETR_NOISE_FLOOR_CSV)
    sweep = pd.read_csv(scfg.ETR_THRESHOLD_SWEEP_CSV)
    rec = pd.read_csv(scfg.ETR_THRESHOLD_RECOMMENDATION_CSV).set_index("objective")
    names = [o.name for o in config.get_objective_set()]
    eps = dict(zip(names, config.get_epsilons()))
    r_ref = scfg.ETR_NOISE_FLOOR_R
    units = scfg.ETR_UNITS_PER_REALIZATION
    rows = []
    for obj in names:
        f = floor[floor["objective"] == obj]
        measured = not f.empty
        sigma_ref = float(f["sigma_local"].iloc[0]) if measured else np.nan
        cdf = _incumbent_cdf(sweep, obj) if measured else None
        crit = float(rec.loc[obj, "current_threshold"]) if obj in rec.index else np.nan
        for R in scfg.ETR_R_LEVELS:
            sigma = sigma_ref * np.sqrt(r_ref / R) if measured else np.nan
            tau_floor = scfg.ETR_FALSE_HARM_Z * np.sqrt(2.0) * sigma
            z2 = scfg.ETR_FALSE_HARM_Z * np.sqrt(2.0)
            tau_paired = scfg.ETR_PAIRED_TAU_FLOORS_R25[obj] * np.sqrt(r_ref / R)
            sigma_paired = tau_paired / z2
            rows.append({
                "objective": obj, "R": R, "units_per_sow": R * units,
                "measured_at_R25": measured,
                "sigma_per_sow_unpaired": sigma,
                "tau_floor_unpaired": tau_floor,
                "sigma_per_sow_paired": sigma_paired,
                "tau_floor_paired": tau_paired,
                "paired_floor_measured": obj not in scfg.ETR_PAIRED_FLOOR_UNMEASURED,
                "epsilon": eps[obj],
                "tau_floor_unpaired_over_epsilon": tau_floor / eps[obj] if measured else np.nan,
                "tau_floor_paired_over_epsilon": tau_paired / eps[obj],
                "current_criterion": crit,
                "misclassified_share_incumbent_unpaired": (
                    misclassified_share(cdf[0], cdf[1], crit, sigma)
                    if (cdf is not None and np.isfinite(crit)) else np.nan),
                "misclassified_share_incumbent_paired": (
                    misclassified_share(cdf[0], cdf[1], crit, sigma_paired)
                    if (cdf is not None and np.isfinite(crit)) else np.nan),
            })
    return pd.DataFrame(rows)


def cross_sow_table() -> pd.DataFrame:
    return pd.DataFrame({
        "n_theta": list(scfg.ETR_N_THETA_LEVELS),
        "worst_case_se_satisficing_fraction": [0.5 / np.sqrt(n) for n in scfg.ETR_N_THETA_LEVELS],
        "note": "binomial, p = 0.5; independent of realizations per SOW",
    })


def options_table() -> pd.DataFrame:
    """SU pricing of the E_test options against the remaining balance."""
    per_real_year = scfg.ETR_SU_PER_POLICY_CHUNK / (scfg.ETR_CHUNK_REALIZATIONS * E_TEST_YEARS)
    n_theta = E_TEST_REEVAL_N_THETA
    cap = scfg.ETR_POLICY_CAP

    def reeval_su(R: int) -> float:
        return cap * n_theta * R * E_TEST_YEARS * per_real_year

    def regen_su(R: int, n_theta_gen: int) -> dict:
        real_years = n_theta_gen * R * E_TEST_YEARS
        scale = real_years / (scfg.ETR_ETEST_REALIZATION_YEARS_REF)
        return {
            "generation_su_est": scfg.ETR_GENERATION_SU_REF_EST * scale,
            "presim_su": scfg.ETR_PRESIM_SU_REF * scale,
            "hazard_image_su_bound": scfg.ETR_HAZARD_IMAGE_SU_BOUND * scale,
            "baselines_su": scfg.ETR_N_DESIGNS * scfg.ETR_BASELINE_SU_REF * (R / E_TEST_R),
        }

    opts = [
        ("A_keep_R25_current_box", E_TEST_R, False, "re-score the stale hazard images only"),
        ("B_keep_R25_drier_bound", E_TEST_R, True,
         "regenerate the 1,000 x 25 design over a wider box"),
        ("C_R50_current_box", 50, True, "regenerate as 1,000 x 50 (or 500 x 50)"),
        ("D_R50_drier_bound", 50, True, "regenerate as 1,000 x 50 over a wider box"),
    ]
    rows = []
    for name, R, regen, note in opts:
        re_su = reeval_su(R)
        rg = regen_su(R, scfg.ETR_N_THETA_GENERATED) if regen else {
            "generation_su_est": 0.0, "presim_su": 0.0,
            "hazard_image_su_bound": scfg.ETR_HAZARD_IMAGE_SU_BOUND, "baselines_su": 0.0}
        staging = sum(rg.values())
        for basis, total in scfg.ETR_CAMPAIGN_TOTALS.items():
            campaign_wo_reeval = total - reeval_su(E_TEST_R)
            new_total = campaign_wo_reeval + re_su + staging
            rows.append({
                "option": name, "R": R, "n_theta_reevaluated": n_theta,
                "regenerate_etest": regen, "note": note, "basis": basis,
                "reeval_su_at_cap": round(re_su), **{k: round(v) for k, v in rg.items()},
                "staging_total_su": round(staging),
                "campaign_total_su": round(new_total),
                "reserve_su": round(scfg.ETR_BALANCE - new_total),
                "reserve_frac": round((scfg.ETR_BALANCE - new_total) / scfg.ETR_BALANCE, 3),
            })
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    noise = per_sow_noise_table()
    noise.to_csv(TABLES / "etr_per_sow_noise.csv", index=False)
    cross = cross_sow_table()
    cross.to_csv(TABLES / "etr_cross_sow_se.csv", index=False)
    opts = options_table()
    opts.to_csv(TABLES / "etr_options.csv", index=False)
    show = ["objective", "R", "units_per_sow", "tau_floor_unpaired", "tau_floor_paired",
            "tau_floor_paired_over_epsilon", "misclassified_share_incumbent_unpaired",
            "misclassified_share_incumbent_paired"]
    print(noise[noise["R"].isin([25, 50])][show].to_string(index=False))
    print(cross.to_string(index=False))
    print(opts[["option", "basis", "reeval_su_at_cap", "staging_total_su", "campaign_total_su",
                "reserve_su", "reserve_frac"]].to_string(index=False))
    print(f"[etr] tables in {TABLES}")


if __name__ == "__main__":
    main()
