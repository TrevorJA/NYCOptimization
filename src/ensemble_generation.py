"""ensemble_generation.py - Build a Kirsch-Nowak synthetic streamflow ensemble.

Single-node, serial port of
``../StochasticExploratoryExperiment/methods/generate.py::generate_ensemble_set``,
stripped of MPI plumbing, per-set/per-batch abstractions, and climate-adjustment
branches. Produces two HDF5 files in pywrdrb's ``FlowEnsemble`` format:

    {output_dir}/gage_flow_mgd.hdf5         - cumulative gage flows per node
    {output_dir}/catchment_inflow_mgd.hdf5  - marginal per-catchment inflows
    {output_dir}/_meta.json                 - provenance (seed, n_years, etc.)

The two HDF5s are auto-loadable by pywrdrb once ``inflow_type`` (== the slug)
is registered with the path navigator (see ``src.ensembles.register_ensemble_path``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from synhydro import KirschGenerator, NowakDisaggregator, Ensemble
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pywr_drb_node_data import (
    immediate_downstream_nodes_dict,
    downstream_node_lags,
)

from src.load.historical_flows import load_historical_flows


# Node lists derived once from pywrdrb's node-data dict, matching the
# reference's convention: numeric (USGS gauge) names go through the KDE
# regression branch; all others are generated directly by Kirsch-Nowak.
_PYWRDRB_NODES = list(immediate_downstream_nodes_dict.keys())
NODES_TO_GENERATE = [n for n in _PYWRDRB_NODES if n[0] != "0" and n != "delTrenton"]
NODES_TO_REGRESS = [n for n in _PYWRDRB_NODES if n[0] == "0"]


def generate_kirsch_nowak_ensemble(
    *,
    n_years: int,
    n_realizations: int,
    seed: int,
    output_dir: Path,
    flowtype: str = "pub_nhmv10_BC_withObsScaled",
    start_date: str = "1945-10-01",
) -> dict:
    """Generate a synthetic streamflow ensemble and write pywrdrb HDF5s.

    Args:
        n_years: Length of each synthetic realization, in years.
        n_realizations: Number of realizations to generate.
        seed: Master random seed for Kirsch + KDE sampling.
        output_dir: Directory to write the two HDF5 files and ``_meta.json``.
            Must already exist.
        flowtype: pywrdrb inflow-dataset key for the historical record fed to
            Kirsch. Defaults to the BC-reconstructed 1945-2023 record.
        start_date: Date assigned to day 0 of each synthetic realization. The
            simulation layer is responsible for any further window clipping.

    Returns:
        Dict of provenance written to ``_meta.json``: ``slug``, ``flowtype``,
        ``n_years``, ``n_realizations``, ``seed``, ``sites``, ``start_date``,
        ``end_date``.
    """
    output_dir = Path(output_dir)

    Q_full = load_historical_flows(gage=True, period="full", flowtype=flowtype)
    Q_inflow = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    Q_full = Q_full.loc[:, NODES_TO_GENERATE]

    # Zeros in the historical record are artifacts (USGS reporting nulls etc.)
    # and would otherwise propagate into synthetic flows through the Kirsch fit
    # and Nowak disaggregation. Mask before fitting.
    n_zeros = int((Q_full == 0.0).sum().sum())
    if n_zeros > 0:
        Q_full = Q_full.replace(0, np.nan)
        Q_inflow = Q_inflow.replace(0, np.nan)
        print(f"[gen] Masked {n_zeros} zero values as NaN before fitting.")

    print(f"[gen] Fitting KirschGenerator on full record "
          f"({Q_full.index[0].date()} -> {Q_full.index[-1].date()}, "
          f"{Q_full.shape[1]} sites)...")
    kirsch = KirschGenerator(generate_using_log_flow=True, debug=False)
    kirsch.preprocessing(Q_full)
    kirsch.fit()

    print(f"[gen] Fitting NowakDisaggregator...")
    nowak = NowakDisaggregator(debug=False)
    nowak.preprocessing(Q_full)
    nowak.fit()

    print(f"[gen] Generating monthly ensemble "
          f"(n_realizations={n_realizations}, n_years={n_years}, seed={seed})...")
    monthly_ensemble = kirsch.generate(
        n_realizations=n_realizations,
        n_years=n_years,
        seed=seed,
    )

    print(f"[gen] Disaggregating monthly -> daily...")
    daily_ensemble = nowak.disaggregate(monthly_ensemble)
    syn_by_real = daily_ensemble.data_by_realization  # dict[int, DataFrame]

    print(f"[gen] Filling non-major nodes via KDE regression...")
    kdes = _fit_downstream_kdes(Q_inflow)
    _apply_kde_downstream(syn_by_real, kdes, base_seed=seed)

    print(f"[gen] Computing marginal catchment inflows...")
    n_days = next(iter(syn_by_real.values())).shape[0]
    syn_dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    inflow_by_real: dict[int, pd.DataFrame] = {}
    for real_id, gage_df in syn_by_real.items():
        # delTrenton is treated as coincident with delDRCanal (see pywrdrb node docs)
        gage_df["delTrenton"] = 0.0
        gage_df.index = syn_dates
        inflow_by_real[real_id] = _subtract_upstream_catchment_inflows(gage_df.copy())

    # Site order is whatever _subtract_upstream_catchment_inflows produced;
    # cast to float32 to halve disk footprint without affecting precision.
    sites = list(next(iter(inflow_by_real.values())).columns)
    for real_id in syn_by_real:
        syn_by_real[real_id] = syn_by_real[real_id][sites].astype(np.float32)
        inflow_by_real[real_id] = inflow_by_real[real_id][sites].astype(np.float32)

    print(f"[gen] Writing HDF5s to {output_dir}/...")
    gage_path = output_dir / "gage_flow_mgd.hdf5"
    inflow_path = output_dir / "catchment_inflow_mgd.hdf5"
    Ensemble(syn_by_real).to_hdf5(str(gage_path))
    Ensemble(inflow_by_real).to_hdf5(str(inflow_path))

    meta = {
        "slug": output_dir.name,
        "flowtype": flowtype,
        "n_years": n_years,
        "n_realizations": n_realizations,
        "seed": seed,
        "sites": sites,
        "start_date": str(syn_dates[0].date()),
        "end_date": str(syn_dates[-1].date()),
    }
    (output_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def generate_input_spanning_master(
    *,
    n_profiles: int,
    n_years: int,
    seed: int,
    output_dir: Path,
    mean_frac_csv: str | Path,
    variance_axis: bool = False,
    mean_abs_csv: str | Path | None = None,
    std_csv: str | Path | None = None,
    bound_pct: tuple[float, float] = (5.0, 95.0),
    margin: float = 0.0,
    flowtype: str = "pub_nhmv10_BC_withObsScaled",
    start_date: str = "1945-10-01",
) -> dict:
    """Generate a master ensemble that SPANS the input (forcing-parameter) space.

    Fits the Kirsch-Nowak generator once, then draws ``n_profiles`` forcing profiles by LHS over the
    CMIP6-based interpretable harmonic-parameter hypercube (``scengen.forcing_space.sample_harmonic_forcing``)
    and generates ONE realization per profile under that profile's climate adjustment. Every master
    member therefore carries both its input coordinates ``theta`` (the 12-month change-factor profile,
    saved to ``forcing_profiles.npz``) and a streamflow realization.

    Two richness rungs are supported via ``variance_axis``:
      - ``False`` (impoverished baseline): perturb the monthly mean only, with ``c_j = 1`` (absolute
        real-space SD preserved); ``theta`` is the 12-month mean factor ``a_j``.
      - ``True`` (enriched): add an independent CMIP6-derived CV-change axis ``v_j`` so
        ``c_j = a_j * v_j``; ``theta`` is the 24-dim ``(a_j, v_j)``.

    Args:
        n_profiles: Number of forcing profiles = number of realizations.
        n_years: Length of each realization, in years.
        seed: Master seed for forcing draw, Kirsch generation, Nowak disaggregation, and KDE fill.
        output_dir: Existing directory for the two HDF5s, ``_meta.json``, ``forcing_profiles.npz``.
        mean_frac_csv: CMIP6 multiplicative mean change-factor (``_frac_``) CSV.
        variance_axis: If True, add the independent variance axis (requires ``mean_abs_csv``/``std_csv``).
        mean_abs_csv, std_csv: Absolute monthly mean/std tables (for the CV-change envelope).
        bound_pct: Percentiles defining each harmonic-parameter's CMIP6 range (default (5, 95) = the
            empirical 90% range; robust to outlier GCM runs).
        margin: Optional fractional widening of the harmonic-parameter hypercube.
        flowtype: pywrdrb inflow-dataset key for the historical record fed to Kirsch.
        start_date: Date assigned to day 0 of each realization.

    Returns:
        Provenance dict written to ``_meta.json``.
    """
    from scengen import forcing_space as fs

    output_dir = Path(output_dir)

    Q_full = load_historical_flows(gage=True, period="full", flowtype=flowtype)
    Q_inflow = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    Q_full = Q_full.loc[:, NODES_TO_GENERATE]
    n_zeros = int((Q_full == 0.0).sum().sum())
    if n_zeros > 0:
        Q_full = Q_full.replace(0, np.nan)
        Q_inflow = Q_inflow.replace(0, np.nan)
        print(f"[gen] Masked {n_zeros} zero values as NaN before fitting.")

    print(f"[gen] Fitting KirschGenerator on full record "
          f"({Q_full.index[0].date()} -> {Q_full.index[-1].date()}, {Q_full.shape[1]} sites)...")
    kirsch = KirschGenerator(generate_using_log_flow=True, debug=False)
    kirsch.preprocessing(Q_full)
    kirsch.fit()
    nowak = NowakDisaggregator(debug=False)
    nowak.preprocessing(Q_full)
    nowak.fit()

    # Baseline log-space monthly stats (calendar-ordered period index); restored per profile.
    base_mean = kirsch.mean_period.copy()
    base_std = kirsch.std_period.copy()
    base_mean_vals = np.asarray(base_mean, dtype=float)
    base_std_vals = np.asarray(base_std, dtype=float)

    # Sample the forcing space over the CMIP6-based harmonic-parameter hypercube (water-year ordered).
    mean_env = fs.load_cmip6_envelope(mean_frac_csv)
    a_wy = fs.sample_harmonic_forcing(n_profiles, mean_env, seed=seed, bound_pct=bound_pct, margin=margin)
    v_wy = None
    if variance_axis:
        if mean_abs_csv is None or std_csv is None:
            raise ValueError("variance_axis=True requires mean_abs_csv and std_csv")
        cv_env = fs.derive_variance_envelope(mean_abs_csv, std_csv)
        v_wy = fs.sample_harmonic_forcing(n_profiles, cv_env, seed=seed + 1, bound_pct=bound_pct, margin=margin)

    print(f"[gen] Generating {n_profiles} climate-adjusted realizations "
          f"(variance_axis={variance_axis}, n_years={n_years}, seed={seed})...")
    monthly_by_real: dict[int, pd.DataFrame] = {}
    monthly_metadata = None  # reuse the generator's metadata (carries 'MS' time_resolution)
    for i in range(n_profiles):
        a_cal = fs.water_year_to_calendar(a_wy[i])
        c_cal = a_cal * fs.water_year_to_calendar(v_wy[i]) if variance_axis else np.ones(12)
        mean_new, std_new = fs.apply_climate_adjustment(
            base_mean_vals, base_std_vals, a_cal, c_profile=c_cal
        )
        kirsch.mean_period = pd.DataFrame(mean_new, index=base_mean.index, columns=base_mean.columns)
        kirsch.std_period = pd.DataFrame(std_new, index=base_std.index, columns=base_std.columns)
        ens_i = kirsch.generate(n_realizations=1, n_years=n_years, realization_indices=[i], seed=seed)
        monthly_by_real[i] = ens_i.data_by_realization[i]
        if monthly_metadata is None:
            monthly_metadata = ens_i.metadata
        if (i + 1) % 100 == 0:
            print(f"[gen]   generated {i + 1}/{n_profiles} monthly realizations")

    print(f"[gen] Disaggregating monthly -> daily (batched)...")
    daily_ensemble = nowak.disaggregate(Ensemble(monthly_by_real, metadata=monthly_metadata), seed=seed)
    syn_by_real = daily_ensemble.data_by_realization

    print(f"[gen] Filling non-major nodes via KDE regression...")
    kdes = _fit_downstream_kdes(Q_inflow)
    _apply_kde_downstream(syn_by_real, kdes, base_seed=seed)

    print(f"[gen] Computing marginal catchment inflows...")
    n_days = next(iter(syn_by_real.values())).shape[0]
    syn_dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    inflow_by_real: dict[int, pd.DataFrame] = {}
    for real_id, gage_df in syn_by_real.items():
        gage_df["delTrenton"] = 0.0
        gage_df.index = syn_dates
        inflow_by_real[real_id] = _subtract_upstream_catchment_inflows(gage_df.copy())

    sites = list(next(iter(inflow_by_real.values())).columns)
    for real_id in syn_by_real:
        syn_by_real[real_id] = syn_by_real[real_id][sites].astype(np.float32)
        inflow_by_real[real_id] = inflow_by_real[real_id][sites].astype(np.float32)

    print(f"[gen] Writing HDF5s + forcing profiles to {output_dir}/...")
    Ensemble(syn_by_real).to_hdf5(str(output_dir / "gage_flow_mgd.hdf5"))
    Ensemble(inflow_by_real).to_hdf5(str(output_dir / "catchment_inflow_mgd.hdf5"))

    # Persist input coordinates theta (water-year ordered) so the diagnostic can subsample in input
    # space; realization_ids align with the HDF5 realization keys (0..n_profiles-1).
    profiles_payload = {
        "realization_ids": np.arange(n_profiles, dtype=int),
        "mean_factor_a": a_wy.astype(float),
        "water_year_months": np.array(fs.WATER_YEAR_MONTHS, dtype=int),
        "richness": "mean_var" if variance_axis else "mean_only",
    }
    if variance_axis:
        profiles_payload["cv_factor_v"] = v_wy.astype(float)
    np.savez(output_dir / "forcing_profiles.npz", **profiles_payload)

    forcing_hash = fs.forcing_hash(a_wy, envelope_csv=mean_frac_csv, margin=margin, seed=seed)
    meta = {
        "slug": output_dir.name,
        "kind": "input_spanning_master",
        "richness": "mean_var" if variance_axis else "mean_only",
        "variance_axis": variance_axis,
        "flowtype": flowtype,
        "n_years": n_years,
        "n_realizations": n_profiles,
        "seed": seed,
        "bound_pct": list(bound_pct),
        "margin": margin,
        "forcing_param": "harmonic",
        "mean_frac_csv": str(mean_frac_csv),
        "forcing_hash": forcing_hash,
        "sites": sites,
        "start_date": str(syn_dates[0].date()),
        "end_date": str(syn_dates[-1].date()),
    }
    (output_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _fit_downstream_kdes(Q_inflow: pd.DataFrame) -> dict[str, stats.gaussian_kde]:
    """Fit a per-pair KDE of downstream/upstream inflow ratio.

    For every (upstream, downstream) pair where upstream is in ``NODES_TO_GENERATE``
    and downstream is in ``NODES_TO_REGRESS``, fit a KDE over the historical
    fractional ratio. The downstream gage flow is later sampled as
    ``upstream_flow * frac_sample`` plus a routing-lag correction.
    """
    kdes: dict[str, stats.gaussian_kde] = {}
    for upstream in NODES_TO_GENERATE:
        downstream = immediate_downstream_nodes_dict[upstream]
        if downstream not in NODES_TO_REGRESS:
            continue
        frac = Q_inflow[downstream] / Q_inflow[upstream]
        frac = frac[np.isfinite(frac)]
        kdes[f"{upstream}_to_{downstream}"] = stats.gaussian_kde(frac.values)
    return kdes


def _apply_kde_downstream(
    syn_by_real: dict[int, pd.DataFrame],
    kdes: dict[str, stats.gaussian_kde],
    *,
    base_seed: int,
) -> None:
    """Inject regression-sampled downstream flows into each realization.

    Mutates ``syn_by_real`` in place — adds a column per downstream gage.
    Seeded per (base_seed, kde_name) so results are reproducible.
    """
    n_days = next(iter(syn_by_real.values())).shape[0]
    n_realizations = len(syn_by_real)
    real_ids = list(syn_by_real.keys())

    for upstream in NODES_TO_GENERATE:
        downstream = immediate_downstream_nodes_dict[upstream]
        if downstream not in NODES_TO_REGRESS:
            continue
        kde_name = f"{upstream}_to_{downstream}"
        kde = kdes[kde_name]
        kde_seed = base_seed + (hash(kde_name) % 10000)

        samples = kde.resample(n_days * n_realizations, seed=kde_seed)
        samples = samples.reshape((n_days, n_realizations))
        samples = np.clip(samples, 0.0, 1.0)

        lag = downstream_node_lags[downstream]
        for i, real_id in enumerate(real_ids):
            upstream_flow = syn_by_real[real_id][upstream].values
            downstream_inflow = upstream_flow * samples[:, i]
            if lag > 0:
                gage = downstream_inflow.copy()
                gage[lag:] += upstream_flow[:-lag]
                gage[:lag] += upstream_flow[:lag]
            else:
                gage = downstream_inflow + upstream_flow
            syn_by_real[real_id][downstream] = gage
