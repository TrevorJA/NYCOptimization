"""flood_objective_run.py - Simulation pass for the flood-objective diagnostics.

Decides the flood objective definition: the incumbent any-gauge minor-stage day
count (``downstream_flood_days_minor``) vs magnitude-weighted exceedance
candidates. This script produces every measured input the figures script
(``flood_objective_figures.py``) reduces:

  1. Content-based staleness audit + targeted re-stage of every flood-augmented
     inflow file consumed here (the KN fixture's file is re-staged with
     ``force=True`` when stale; a stale historic CSV in the sibling Pywr-DRB
     data tree fails loudly, since regenerating it belongs to that repo).
  2. Policy evaluation: the FFMP baseline + FLOODOBJ_N_POLICIES feasible-uniform
     policies + a FLOODOBJ_SWEEP_POINTS-point flood-release-scale ladder, each
     simulated (trimmed model) on the historic trace AND the KN stationary
     fixture. Persists a cube of per-policy x realization x FFMP-year-unit
     values for six candidate metrics, plus per-gauge flood-day records and
     per-realization flow/stage maxima (rating-curve exposure).
  3. Sim-vs-obs scoring (zero simulation): computes all six candidates on the
     completed flood-gauge diagnostic experiment's post-fix 2000-2023 output
     and on observed flows, via that experiment's own helpers.

Candidates (all minimized; window values use the ``_flood_days_anygauge``
normalization, i.e. metric-window total / (n_days / 365.25)):

  C1  days/yr any gauge >= NWS minor flood stage        [days/yr]   (incumbent)
  C2  days/yr any gauge >= FFMP cautionary stage        [days/yr]   (reference)
  C3  sum over gauges+days of (stage - minor)+          [gauge-ft-days/yr]
  C4  sum over days of max-gauge (stage - minor)+       [ft-days/yr]
  C5  C3 with each gauge scaled by 1/(major - minor)    [norm-gauge-days/yr]
  C6  sum over gauges+days of (Q - Q_minor)+            [MG/yr]

Outputs -> outputs/supplemental/flood_objective/{cube,tables}
Configuration lives in supplemental_config.py (FLOODOBJ_* section) — no CLI
value flags.

Usage:
    python scripts/supplemental/flood_objective_run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_floodobj_env()

from src.ensembles import (  # noqa: E402
    get_ensemble_spec,
    register_ensemble_path,
    staged_ensemble_dir,
)
from src.formulations import (  # noqa: E402
    get_baseline_values,
    get_bounds,
    get_var_names,
)
from src.objectives import (  # noqa: E402
    _DOWNSTREAM_GAUGES,
    _metric_window,
    build_objective_set,
)
from src.objectives_ensemble import ffmp_year_unit_slices  # noqa: E402
from src.sensitivity_common import sample_feasible_dvs  # noqa: E402
from src.simulation import (  # noqa: E402
    compute_constraint_violations,
    dvs_to_config,
    run_simulation_ensemble_inmemory,
    run_simulation_inmemory,
)

from pywrdrb.flood_thresholds import flood_stage_thresholds  # noqa: E402
from pywrdrb.utils.constants import cfs_to_mgd  # noqa: E402
from pywrdrb.utils.rating_curves import (  # noqa: E402
    load_all_flood_monitoring_curves,
)

CANDIDATES = ["C1_days_minor", "C2_days_action", "C3_sum_ft",
              "C4_max_ft", "C5_norm_ft", "C6_flow_mg"]
GAUGES = list(_DOWNSTREAM_GAUGES)

#: Post-fix content signature: Hale Eddy and Fishs Eddy local inflows are fixed
#: fractions of the SAME donor, so their column-mean ratio is exactly
#: fractions["01426500"] / fractions["01421000"] in any post-fix file.
_RATIO_TOL = 1e-4


def _threshold_series(level: str) -> pd.Series:
    return pd.Series({g: flood_stage_thresholds[g][level] for g in GAUGES})


MINOR = _threshold_series("minor")
ACTION = _threshold_series("action")
MAJOR = _threshold_series("major")


def _q_minor_mgd() -> pd.Series:
    """Discharge at NWS minor flood stage per gauge (MGD), by curve inversion."""
    curves = load_all_flood_monitoring_curves()
    q = {}
    for g in GAUGES:
        cfs = float(np.asarray(curves[g].stage_to_discharge(MINOR[g])).ravel()[0])
        q[g] = cfs * cfs_to_mgd
    return pd.Series(q)


Q_MINOR = _q_minor_mgd()


###############################################################################
# Candidate metrics
###############################################################################

def candidate_daily(stage: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    """Daily per-candidate contributions on an already-windowed frame pair.

    Args:
        stage: Daily stage (ft), columns = gauge ids.
        flow: Daily gauge flow (MGD), same index and columns.

    Returns:
        DataFrame indexed like ``stage`` with one column per candidate; the
        window metric is the column sum divided by window years, the annual
        unit metric the within-unit column sum.
    """
    stage = stage[GAUGES]
    flow = flow[GAUGES]
    exc = stage.sub(MINOR, axis=1).clip(lower=0)
    out = pd.DataFrame(index=stage.index)
    out["C1_days_minor"] = stage.ge(MINOR, axis=1).any(axis=1).astype(float)
    out["C2_days_action"] = stage.ge(ACTION, axis=1).any(axis=1).astype(float)
    out["C3_sum_ft"] = exc.sum(axis=1)
    out["C4_max_ft"] = exc.max(axis=1)
    out["C5_norm_ft"] = exc.div(MAJOR - MINOR, axis=1).sum(axis=1)
    out["C6_flow_mg"] = flow.sub(Q_MINOR, axis=1).clip(lower=0).sum(axis=1)
    return out


def window_values(daily: pd.DataFrame) -> np.ndarray:
    """Per-year-normalized whole-window candidate values (order CANDIDATES)."""
    n_days = len(daily)
    if n_days == 0:
        return np.zeros(len(CANDIDATES))
    return (daily[CANDIDATES].sum(axis=0) / (n_days / 365.25)).to_numpy()


def unit_values(daily: pd.DataFrame) -> np.ndarray:
    """(n_units, n_candidates) un-normalized within-FFMP-year-unit sums."""
    slices = ffmp_year_unit_slices(daily.index)
    return np.asarray(
        [daily[CANDIDATES].iloc[sl].sum(axis=0).to_numpy() for sl in slices],
        dtype=float,
    )


def flood_day_records(stage: pd.DataFrame, flow: pd.DataFrame) -> list:
    """Per-gauge records for days at/above minor stage (or minor discharge).

    Returns a list of ``(gauge_index, date, stage_ft, q_mgd)`` tuples covering
    the union of the stage-basis and flow-basis exceedance sets, so the
    figures script can audit the 1-day stage lag and rating-curve exposure.
    """
    recs = []
    for gi, g in enumerate(GAUGES):
        mask = (stage[g] >= MINOR[g]) | (flow[g] >= Q_MINOR[g])
        if not mask.any():
            continue
        sub_s, sub_q = stage.loc[mask, g], flow.loc[mask, g]
        for dt, s_ft in sub_s.items():
            recs.append((gi, dt, float(s_ft), float(sub_q.loc[dt])))
    return recs


###############################################################################
# Staleness audit
###############################################################################

def _flood_file_ratio_ok(he_mean: float, fe_mean: float) -> tuple:
    """Compare the Hale-Eddy/Fishs-Eddy mean ratio to the post-fix fraction."""
    from pywrdrb.pre.flood_node_inflows import flood_node_inflow_fractions

    frac = flood_node_inflow_fractions()
    expected = frac["01426500"] / frac["01421000"]
    measured = he_mean / fe_mean
    return abs(measured - expected) < _RATIO_TOL, measured, expected


def audit_and_restage(spec) -> dict:
    """Verify (and if needed re-stage) every flood-inflow file this run reads.

    The ensemble file is re-staged in place with ``force=True`` when stale; a
    stale historic CSV aborts the run (that file belongs to the sibling
    Pywr-DRB data tree and is regenerated by its own preprocessor).

    Returns:
        Audit manifest: per file, mtime, measured/expected content ratio, and
        the action taken.
    """
    import h5py

    audit = {}

    hist_csv = Path(
        # pywrdrb's own input tree for the historic single trace
        scfg.FLOODOBJ_GAUGE_EXPERIMENT_DIR.parents[1]
        / "src" / "pywrdrb" / "data" / "flows" / "pub_nhmv10_BC_withObsScaled"
        / "catchment_inflow_with_flood_nodes_mgd.csv"
    )
    df = pd.read_csv(hist_csv, index_col=0)
    ok, measured, expected = _flood_file_ratio_ok(
        df["01426500"].mean(), df["01421000"].mean())
    audit["historic_csv"] = {
        "path": str(hist_csv),
        "mtime": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(hist_csv.stat().st_mtime)),
        "he_fe_ratio": round(measured, 6),
        "expected": round(expected, 6),
        "action": "fresh" if ok else "STALE",
    }
    if not ok:
        sys.exit(
            f"[flood_run] STALE historic flood inflows: {hist_csv} has "
            f"HE/FE ratio {measured:.4f}, expected {expected:.4f} post-fix. "
            "Regenerate it with pywrdrb's FloodNodeInflowPreprocessor before "
            "re-running.")

    register_ensemble_path(spec.inflow_type)
    ens_h5 = staged_ensemble_dir(spec.inflow_type) / \
        "catchment_inflow_with_flood_nodes_mgd.hdf5"

    def _ens_ratio_ok() -> tuple:
        with h5py.File(ens_h5, "r") as f:
            he = np.asarray(f["01426500"]["0"]).ravel().mean()
            fe = np.asarray(f["01421000"]["0"]).ravel().mean()
        return _flood_file_ratio_ok(he, fe)

    action = "fresh"
    ok, measured, expected = _ens_ratio_ok() if ens_h5.exists() else (
        False, float("nan"), float("nan"))
    if not ok:
        from pywrdrb.pre.flood_node_inflows import (
            FloodNodeInflowEnsemblePreprocessor,
        )

        print(f"[flood_run] re-staging stale ensemble flood inflows "
              f"({ens_h5.name}, ratio {measured:.4f}) ...", flush=True)
        pp = FloodNodeInflowEnsemblePreprocessor(
            inflow_type=spec.inflow_type,
            realization_ids=list(spec.realization_indices),
            force=True,
        )
        pp.load()
        pp.process()
        pp.save()
        ok, measured, expected = _ens_ratio_ok()
        if not ok:
            sys.exit(f"[flood_run] re-stage failed content check on {ens_h5}")
        action = "re-staged (force=True)"
    audit["ensemble_hdf5"] = {
        "path": str(ens_h5),
        "mtime": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(ens_h5.stat().st_mtime)),
        "he_fe_ratio": round(measured, 6),
        "expected": round(expected, 6),
        "action": action,
    }
    return audit


###############################################################################
# Policy set
###############################################################################

def build_policies() -> tuple:
    """Baseline + feasible-uniform sample + flood-release-scale sweep ladder.

    Returns:
        (dvs (P, n_var), kind (P,) str, sweep_t (P,) float NaN off-ladder,
         violations (P, 2), sample_info)
    """
    formulation = scfg.FLOODOBJ_FORMULATION
    baseline = get_baseline_values(formulation)
    samples, info = sample_feasible_dvs(
        formulation, scfg.FLOODOBJ_SEED, scfg.FLOODOBJ_N_POLICIES)

    # Sweep ladder: one common multiplier per reservoir applied to BOTH flood
    # zones, ramped over that reservoir's L1a bounds. A common multiplier
    # preserves the effective L1b <= L1a ordering elementwise (the default
    # rows satisfy it, and min(row*m*b, cap) is monotone in row), so every
    # ladder point is feasible under the formal flood-ordering constraint —
    # ramping the two zones' DVs over their OWN bounds is not (L1b's upper
    # 2.0 overtakes L1a's 1.35 and the constraint fires).
    names = get_var_names(formulation)
    lo, hi = get_bounds(formulation)
    idx = {n: i for i, n in enumerate(names)}
    ts = np.linspace(0.0, 1.0, scfg.FLOODOBJ_SWEEP_POINTS)
    sweep = []
    for t in ts:
        dv = baseline.copy()
        for res in ("cannonsville", "pepacton", "neversink"):
            ia = idx[f"flood_release_scale_l1a_{res}"]
            m = lo[ia] + t * (hi[ia] - lo[ia])
            dv[ia] = m
            dv[idx[f"flood_release_scale_l1b_{res}"]] = m
        sweep.append(dv)

    dvs = np.vstack([baseline[None, :], samples, np.vstack(sweep)])
    kind = np.array(
        ["baseline"] + ["random"] * len(samples) + ["sweep"] * len(ts))
    sweep_t = np.concatenate(
        [[np.nan], np.full(len(samples), np.nan), ts])
    viol = np.asarray(
        [compute_constraint_violations(dv, formulation) for dv in dvs])
    return dvs, kind, sweep_t, viol, info


###############################################################################
# Evaluation
###############################################################################

def evaluate_all(dvs: np.ndarray, spec) -> dict:
    """Simulate every policy on the historic trace and the KN fixture.

    Returns the raw reduction arrays persisted into the cube.
    """
    formulation = scfg.FLOODOBJ_FORMULATION
    n_pol = len(dvs)
    n_real = spec.n_realizations

    hist_window = np.zeros((n_pol, len(CANDIDATES)))
    hist_units = None
    ens_window = np.zeros((n_pol, n_real, len(CANDIDATES)))
    ens_units = None
    hist_max = np.zeros((n_pol, len(GAUGES), 2))       # stage_ft, q_mgd
    ens_max = np.zeros((n_pol, n_real, len(GAUGES), 2))
    eval_secs = np.zeros((n_pol, 2))
    records = []  # (policy, domain 0=hist/1=ens, realization, gauge, date, stage, q)

    for p, dv in enumerate(dvs):
        cfg = dvs_to_config(dv, formulation)

        t0 = time.perf_counter()
        data = run_simulation_inmemory(cfg)
        eval_secs[p, 0] = time.perf_counter() - t0
        stage = _metric_window(data["flood_stage"][GAUGES])
        flow = _metric_window(data["major_flow"][GAUGES])
        daily = candidate_daily(stage, flow)
        hist_window[p] = window_values(daily)
        u = unit_values(daily)
        if hist_units is None:
            hist_units = np.zeros((n_pol, u.shape[0], len(CANDIDATES)))
        hist_units[p] = u
        hist_max[p, :, 0] = stage.max(axis=0).to_numpy()
        hist_max[p, :, 1] = flow.max(axis=0).to_numpy()
        for gi, dt, s_ft, q in flood_day_records(stage, flow):
            records.append((p, 0, 0, gi, dt, s_ft, q))

        t0 = time.perf_counter()
        data_per_real = run_simulation_ensemble_inmemory(cfg, spec)
        eval_secs[p, 1] = time.perf_counter() - t0
        for r, data_r in enumerate(data_per_real):
            stage = _metric_window(data_r["flood_stage"][GAUGES])
            flow = _metric_window(data_r["major_flow"][GAUGES])
            daily = candidate_daily(stage, flow)
            ens_window[p, r] = window_values(daily)
            u = unit_values(daily)
            if ens_units is None:
                ens_units = np.zeros((n_pol, n_real, u.shape[0],
                                      len(CANDIDATES)))
            ens_units[p, r] = u
            ens_max[p, r, :, 0] = stage.max(axis=0).to_numpy()
            ens_max[p, r, :, 1] = flow.max(axis=0).to_numpy()
            for gi, dt, s_ft, q in flood_day_records(stage, flow):
                records.append((p, 1, r, gi, dt, s_ft, q))

        print(f"[flood_run] policy {p + 1}/{n_pol} "
              f"(hist {eval_secs[p, 0]:.1f}s, ens {eval_secs[p, 1]:.1f}s)",
              flush=True)

    rec = pd.DataFrame(
        records,
        columns=["policy", "domain", "realization", "gauge_idx",
                 "date", "stage_ft", "q_mgd"],
    )
    return {
        "hist_window": hist_window, "hist_units": hist_units,
        "ens_window": ens_window, "ens_units": ens_units,
        "hist_max": hist_max, "ens_max": ens_max,
        "eval_secs": eval_secs, "records": rec,
    }


def crosscheck_incumbent(dvs: np.ndarray) -> dict:
    """C1 on the baseline historic trace must equal the registered objective."""
    objs = build_objective_set(["downstream_flood_days_minor"])
    cfg = dvs_to_config(dvs[0], scfg.FLOODOBJ_FORMULATION)
    data = run_simulation_inmemory(cfg)
    registered = float(objs.compute(data)[0])
    stage = _metric_window(data["flood_stage"][GAUGES])
    flow = _metric_window(data["major_flow"][GAUGES])
    ours = float(window_values(candidate_daily(stage, flow))[0])
    ok = np.isclose(registered, ours, rtol=1e-12, atol=1e-12)
    if not ok:
        sys.exit(f"[flood_run] C1 cross-check FAILED: registered "
                 f"{registered!r} vs candidate machinery {ours!r}")
    return {"registered": registered, "candidate_c1": ours, "match": bool(ok)}


###############################################################################
# Sim-vs-obs (zero simulation; reuses the flood-gauge experiment)
###############################################################################

def _water_year(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.year) + (np.asarray(index.month) >= 10).astype(int)


def sim_vs_obs_tables() -> dict:
    """Score every candidate sim-vs-obs on the gauge experiment's 2000-2023 run.

    Stage is recomputed from flow through the shared rating curves on BOTH
    sides (the experiment's convention, so curve artifacts cancel). The whole
    2000-2023 window is used without the 6-month exclusion, matching the
    experiment's ``summary.csv``; annual series are complete water years
    (WY2001-WY2023). Also reproduces the experiment's aggregate
    (per-gauge-summed) minor-stage days/yr as a wiring check.
    """
    sys.path.insert(0, str(scfg.FLOODOBJ_GAUGE_EXPERIMENT_DIR))
    import diagnostics as gauge_diag  # noqa: E402  (the experiment's module)

    bundle = gauge_diag.load_all()
    sim_stage, obs_stage = bundle["sim_stage"], bundle["obs_stage"]
    sim_flow, obs_flow = bundle["sim_flow"], bundle["obs_flow"]

    def _daily(stage, flow):
        stage, flow = stage.align(flow, join="inner", axis=0)
        valid = stage[GAUGES].notna().all(axis=1) & \
            flow[GAUGES].notna().all(axis=1)
        return candidate_daily(stage.loc[valid], flow.loc[valid])

    sim_daily, obs_daily = _daily(sim_stage, sim_flow), _daily(obs_stage,
                                                               obs_flow)

    rows = []
    for name, daily in (("sim", sim_daily), ("obs", obs_daily)):
        vals = window_values(daily)
        rows.append(pd.Series(vals, index=CANDIDATES, name=name))
    summary = pd.DataFrame(rows).T
    summary["ratio_sim_obs"] = summary["sim"] / summary["obs"]
    summary.index.name = "candidate"
    out = scfg.floodobj_table_path("simobs_summary")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)

    annual_rows = []
    for name, daily in (("sim", sim_daily), ("obs", obs_daily)):
        wy = _water_year(daily.index)
        complete = pd.Series(daily.index.date, index=daily.index).groupby(
            wy).size() >= 364
        for cand in CANDIDATES:
            grp = daily[cand].groupby(wy).sum()
            for year, val in grp[complete].items():
                annual_rows.append((int(year), cand, name, float(val)))
    annual = pd.DataFrame(annual_rows,
                          columns=["water_year", "candidate", "source",
                                   "value"])
    annual = annual.pivot_table(index=["water_year", "candidate"],
                                columns="source", values="value").reset_index()
    annual.to_csv(scfg.floodobj_table_path("simobs_annual_candidates"),
                  index=False)

    # Wiring check vs the experiment's summary.csv (aggregate = per-gauge sum).
    freq = gauge_diag.gauge_threshold_frequency(sim_stage, "minor")
    exp_summary = pd.read_csv(
        Path(scfg.FLOODOBJ_GAUGE_EXPERIMENT_DIR) / "output" / "summary.csv")
    return {
        "aggregate_minor_days_sim": freq["aggregate"],
        "experiment_summary_rows": len(exp_summary),
        "summary_table": str(out),
    }


###############################################################################
# Main
###############################################################################

def main() -> None:
    spec = get_ensemble_spec(scfg.FLOODOBJ_ENSEMBLE_SLUG)
    audit = audit_and_restage(spec)
    print("[flood_run] staleness audit:",
          json.dumps(audit, indent=2), flush=True)

    dvs, kind, sweep_t, viol, sample_info = build_policies()
    n_sweep_infeasible = int((viol[kind == "sweep"].sum(axis=1) > 0).sum())
    print(f"[flood_run] {len(dvs)} policies "
          f"(1 baseline + {int((kind == 'random').sum())} random + "
          f"{int((kind == 'sweep').sum())} sweep; "
          f"{n_sweep_infeasible} sweep points violate a formal constraint)",
          flush=True)

    check = crosscheck_incumbent(dvs)
    print(f"[flood_run] C1 cross-check vs registered objective: "
          f"{check['registered']:.6f} days/yr (match)", flush=True)

    results = evaluate_all(dvs, spec)
    simobs = sim_vs_obs_tables()

    cube_path = scfg.floodobj_cube_path("flood_cube")
    cube_path.parent.mkdir(parents=True, exist_ok=True)
    rec = results["records"]
    np.savez_compressed(
        cube_path,
        dvs=dvs, kind=kind, sweep_t=sweep_t, violations=viol,
        hist_window=results["hist_window"], hist_units=results["hist_units"],
        ens_window=results["ens_window"], ens_units=results["ens_units"],
        hist_max=results["hist_max"], ens_max=results["ens_max"],
        eval_secs=results["eval_secs"],
        rec_policy=rec["policy"].to_numpy(int),
        rec_domain=rec["domain"].to_numpy(int),
        rec_realization=rec["realization"].to_numpy(int),
        rec_gauge=rec["gauge_idx"].to_numpy(int),
        rec_date=rec["date"].to_numpy("datetime64[D]"),
        rec_stage_ft=rec["stage_ft"].to_numpy(float),
        rec_q_mgd=rec["q_mgd"].to_numpy(float),
        candidates=np.array(CANDIDATES), gauges=np.array(GAUGES),
        minor_ft=MINOR.to_numpy(), action_ft=ACTION.to_numpy(),
        major_ft=MAJOR.to_numpy(), q_minor_mgd=Q_MINOR.to_numpy(),
    )

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cube": str(cube_path),
        "ensemble": scfg.FLOODOBJ_ENSEMBLE_SLUG,
        "n_policies": int(len(dvs)),
        "seed": scfg.FLOODOBJ_SEED,
        "audit": audit,
        "feasible_sample": {
            "n_draws": int(sample_info["n_draws"]),
            "acceptance_rate": float(sample_info["acceptance_rate"]),
        },
        "n_sweep_constraint_violations": n_sweep_infeasible,
        "crosscheck_incumbent": check,
        "sim_vs_obs": simobs,
        "q_minor_mgd": {g: round(float(Q_MINOR[g]), 1) for g in GAUGES},
        "mean_eval_secs": {
            "historic": round(float(results["eval_secs"][:, 0].mean()), 2),
            "ensemble": round(float(results["eval_secs"][:, 1].mean()), 2),
        },
    }
    with open(scfg.FLOODOBJ_CUBE_DIR / "flood_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[flood_run] cube -> {cube_path}")
    print(f"[flood_run] done in "
          f"{results['eval_secs'].sum() / 60:.1f} simulation-minutes")


if __name__ == "__main__":
    main()
