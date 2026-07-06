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
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

from synhydro import KirschGenerator, NowakDisaggregator, Ensemble
from synhydro.core.seeding import as_seed_sequence, spawn_realization_seed
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

    print(f"[gen] Disaggregating monthly -> daily + KDE fill + catchment inflows...")
    kdes = _fit_downstream_kdes(Q_inflow)
    syn_by_real, inflow_by_real, sites = _disaggregate_fill_inflow(
        monthly_ensemble, nowak=nowak, kdes=kdes, master_seed=seed, start_date=start_date
    )

    print(f"[gen] Writing HDF5s to {output_dir}/...")
    Ensemble(syn_by_real).to_hdf5(str(output_dir / "gage_flow_mgd.hdf5"))
    Ensemble(inflow_by_real).to_hdf5(str(output_dir / "catchment_inflow_mgd.hdf5"))

    syn_dates = next(iter(syn_by_real.values())).index
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

    print(f"[gen] Disaggregating monthly -> daily (batched) + KDE fill + catchment inflows...")
    kdes = _fit_downstream_kdes(Q_inflow)
    syn_by_real, inflow_by_real, sites = _disaggregate_fill_inflow(
        Ensemble(monthly_by_real, metadata=monthly_metadata),
        nowak=nowak, kdes=kdes, master_seed=seed, start_date=start_date,
    )
    syn_dates = next(iter(syn_by_real.values())).index

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


def _fit_kirsch(Q: pd.DataFrame) -> KirschGenerator:
    """Fit a log-space ``KirschGenerator`` on the (zero-masked) gage record ``Q``."""
    kirsch = KirschGenerator(generate_using_log_flow=True, debug=False)
    kirsch.preprocessing(Q)
    kirsch.fit()
    return kirsch


def _fit_nowak(Q: pd.DataFrame) -> NowakDisaggregator:
    """Fit a ``NowakDisaggregator`` on the (zero-masked) gage record ``Q``."""
    nowak = NowakDisaggregator(debug=False)
    nowak.preprocessing(Q)
    nowak.fit()
    return nowak


def _load_masked_flows(
    flowtype: str, *, baseline_period: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load the full-record gage + catchment-inflow flows, masking zero artifacts.

    Zeros in the historical record (USGS reporting nulls etc.) would otherwise propagate into the
    Kirsch fit and Nowak disaggregation, so they are masked to NaN before fitting.

    Args:
        flowtype: pywrdrb inflow-dataset key.
        baseline_period: When given, also return the gage record sliced to this ``(start, end)``
            window (for the two-generator climate-adjustment reference of methods §3.2). ``None``
            returns ``None`` for the baseline slice (single-generator callers).

    Returns:
        ``(Q_gage_full, Q_inflow_full, Q_gage_baseline)`` — the baseline slice is ``None`` when
        ``baseline_period`` is not requested.
    """
    Q_gage = load_historical_flows(gage=True, period="full", flowtype=flowtype).loc[:, NODES_TO_GENERATE]
    Q_inflow = load_historical_flows(gage=False, period="full", flowtype=flowtype)
    n_zeros = int((Q_gage == 0.0).sum().sum())
    if n_zeros > 0:
        Q_gage = Q_gage.replace(0, np.nan)
        Q_inflow = Q_inflow.replace(0, np.nan)
        print(f"[gen] Masked {n_zeros} zero values as NaN before fitting.")
    Q_base = None
    if baseline_period is not None:
        Q_base = Q_gage.loc[baseline_period[0]:baseline_period[1]]
    return Q_gage, Q_inflow, Q_base


def _disaggregate_fill_inflow(
    monthly_ensemble: Ensemble,
    *,
    nowak: NowakDisaggregator,
    kdes: dict[str, stats.gaussian_kde],
    master_seed: int | np.random.SeedSequence,
    start_date: str,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], list[str]]:
    """Disaggregate a monthly ensemble to daily gage flows and marginal catchment inflows.

    The shared back half of every generator: Nowak daily disaggregation (seed-deterministic, keyed by
    the ensemble's integer realization ids), per-realization KDE downstream fill, and
    ``_subtract_upstream_catchment_inflows``. The integer realization keys are preserved throughout
    (re-keying would change the daily output — the SynHydro determinism caveat), and both frames are
    cast to float32.

    Args:
        monthly_ensemble: Monthly ``Ensemble`` keyed by global realization index, carrying the
            generator metadata (``'MS'`` time resolution).
        nowak: Fitted disaggregator.
        kdes: Per-pair downstream KDEs from :func:`_fit_downstream_kdes`.
        master_seed: Master seed forwarded to the Nowak and KDE streams.
        start_date: Date assigned to day 0 of each realization.

    Returns:
        ``(gage_by_real, inflow_by_real, sites)`` — float32 frames keyed by the same global indices,
        with a shared column order ``sites``.
    """
    daily = nowak.disaggregate(monthly_ensemble, seed=master_seed)
    syn_by_real = daily.data_by_realization

    _apply_kde_downstream(syn_by_real, kdes, master_seed=master_seed)

    n_days = next(iter(syn_by_real.values())).shape[0]
    syn_dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    inflow_by_real: dict[int, pd.DataFrame] = {}
    for real_id, gage_df in syn_by_real.items():
        # delTrenton is treated as coincident with delDRCanal (see pywrdrb node docs).
        gage_df["delTrenton"] = 0.0
        gage_df.index = syn_dates
        inflow_by_real[real_id] = _subtract_upstream_catchment_inflows(gage_df.copy())

    # Site order is whatever _subtract_upstream_catchment_inflows produced; float32 halves disk.
    sites = list(next(iter(inflow_by_real.values())).columns)
    for real_id in syn_by_real:
        syn_by_real[real_id] = syn_by_real[real_id][sites].astype(np.float32)
        inflow_by_real[real_id] = inflow_by_real[real_id][sites].astype(np.float32)
    return syn_by_real, inflow_by_real, sites


def _kde_stream(
    master: np.random.SeedSequence, realization_id: int, pair_name: str
) -> np.random.Generator:
    """Return the KDE-fill RNG for one (realization, downstream-pair), keyed to the global index.

    The stream is realization ``realization_id``'s SynHydro child seed
    (``spawn_realization_seed``), namespaced by ``crc32(pair_name)``. It therefore depends only on
    ``(master, realization_id, pair_name)`` — invariant to how the realization range is partitioned
    across MPI ranks / batches — and is disjoint from SynHydro's ``"generation"``/``"disaggregation"``
    sub-streams (whose spawn indices are 0 and 1, never a large CRC32). This is the only randomness in
    the pipeline not already covered by the SynHydro sub-streams (methods §7 determinism contract).
    """
    child = spawn_realization_seed(master, int(realization_id))
    namespaced = np.random.SeedSequence(
        entropy=child.entropy,
        spawn_key=tuple(child.spawn_key) + (zlib.crc32(pair_name.encode()),),
        pool_size=child.pool_size,
    )
    return np.random.default_rng(namespaced)


def _apply_kde_downstream(
    syn_by_real: dict[int, pd.DataFrame],
    kdes: dict[str, stats.gaussian_kde],
    *,
    master_seed: int | np.random.SeedSequence,
) -> None:
    """Inject regression-sampled downstream flows into each realization.

    Mutates ``syn_by_real`` in place — adds a column per downstream gage. Each realization's fill is
    resampled independently from its own global-index-keyed stream (:func:`_kde_stream`), so the
    result is bit-for-bit reproducible and invariant to the batch/MPI partition. A single downstream
    fraction series of length ``n_days`` is drawn per (realization, pair) — do NOT batch the resample
    across realizations, which would couple them (methods §7).
    """
    master = as_seed_sequence(master_seed)
    n_days = next(iter(syn_by_real.values())).shape[0]

    for upstream in NODES_TO_GENERATE:
        downstream = immediate_downstream_nodes_dict[upstream]
        if downstream not in NODES_TO_REGRESS:
            continue
        pair_name = f"{upstream}_to_{downstream}"
        kde = kdes[pair_name]
        lag = downstream_node_lags[downstream]

        for real_id, gage_df in syn_by_real.items():
            rng = _kde_stream(master, real_id, pair_name)
            frac = np.clip(kde.resample(n_days, seed=rng)[0], 0.0, 1.0)
            upstream_flow = gage_df[upstream].values
            downstream_inflow = upstream_flow * frac
            if lag > 0:
                gage = downstream_inflow.copy()
                gage[lag:] += upstream_flow[:-lag]
                gage[:lag] += upstream_flow[:lag]
            else:
                gage = downstream_inflow + upstream_flow
            gage_df[downstream] = gage


###############################################################################
# Master ensemble (methods §3.2): N_forcing x realizations_per_profile, streaming H
###############################################################################

@dataclass
class _MasterSetup:
    """Fitted generators + forcing draw shared by full generation and single-realization regen."""

    kirsch: KirschGenerator            # full-record generator that produces realizations
    nowak: NowakDisaggregator
    kdes: dict[str, stats.gaussian_kde]
    base_mean: pd.DataFrame            # baseline-period log-space mean (eqs 10-11 reference)
    base_std: pd.DataFrame             # baseline-period log-space std
    a_wy: np.ndarray                   # (N_forcing, 12) water-year mean change factors
    v_wy: np.ndarray | None            # (N_forcing, 12) CV change factors, or None


def _sample_master_forcing(config) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw the master's forcing profiles (mean ``a`` and optional CV ``v``) from the CMIP6 hypercube."""
    from scengen import forcing_space as fs

    mean_env = fs.load_cmip6_envelope(config.mean_frac_csv)
    a_wy = fs.sample_harmonic_forcing(
        config.n_forcing_profiles, mean_env,
        seed=config.master_seed, bound_pct=config.bound_pct, margin=config.margin,
    )
    v_wy = None
    if config.variance_axis:
        if config.mean_abs_csv is None or config.std_csv is None:
            raise ValueError("variance_axis=True requires mean_abs_csv and std_csv on the config")
        cv_env = fs.derive_variance_envelope(config.mean_abs_csv, config.std_csv)
        v_wy = fs.sample_harmonic_forcing(
            config.n_forcing_profiles, cv_env,
            seed=config.master_seed + 1, bound_pct=config.bound_pct, margin=config.margin,
        )
    return a_wy, v_wy


def _prepare_master(config) -> _MasterSetup:
    """Fit the two Kirsch generators (§3.2) + Nowak + KDEs and draw the forcing profiles.

    The baseline-period generator supplies the eqs-10-11 reference moments; the full-record
    generator produces the realizations. Both are deterministic given ``config``.
    """
    Q_gage, Q_inflow, Q_base = _load_masked_flows(
        config.flowtype, baseline_period=config.baseline_period
    )
    Q_gen = Q_gage.loc[config.full_period[0]:config.full_period[1]]
    kirsch_base = _fit_kirsch(Q_base)
    a_wy, v_wy = _sample_master_forcing(config)
    return _MasterSetup(
        kirsch=_fit_kirsch(Q_gen),
        nowak=_fit_nowak(Q_gen),
        kdes=_fit_downstream_kdes(Q_inflow),
        base_mean=kirsch_base.mean_period.copy(),
        base_std=kirsch_base.std_period.copy(),
        a_wy=a_wy,
        v_wy=v_wy,
    )


def _generate_profile_monthly(
    setup: _MasterSetup, config, profile_idx: int, *, indices: list[int] | None = None,
) -> tuple[dict[int, pd.DataFrame], object]:
    """Climate-adjust ``setup.kirsch`` to profile ``profile_idx`` and generate its realizations.

    Global indices default to the profile's full block ``[p*R, …, p*R+R-1]``; pass ``indices`` to
    regenerate a subset (each index draws from its own ``realization_rng(master, k, "generation")``
    stream, so a subset is identical to the full-block draw for those indices). Returns the monthly
    frames keyed by global index and the generator metadata.
    """
    from scengen import forcing_space as fs

    R = config.realizations_per_profile
    a_cal = fs.water_year_to_calendar(setup.a_wy[profile_idx])
    if setup.v_wy is not None:
        c_cal = a_cal * fs.water_year_to_calendar(setup.v_wy[profile_idx])
    else:
        c_cal = np.ones(12)
    mean_new, std_new = fs.apply_climate_adjustment(
        np.asarray(setup.base_mean, dtype=float), np.asarray(setup.base_std, dtype=float),
        a_cal, c_profile=c_cal,
    )
    setup.kirsch.mean_period = pd.DataFrame(mean_new, index=setup.base_mean.index, columns=setup.base_mean.columns)
    setup.kirsch.std_period = pd.DataFrame(std_new, index=setup.base_std.index, columns=setup.base_std.columns)

    idxs = list(indices) if indices is not None else [profile_idx * R + j for j in range(R)]
    ens = setup.kirsch.generate(
        n_years=config.realization_years, realization_indices=idxs, seed=config.master_seed
    )
    return {k: ens.data_by_realization[k] for k in idxs}, ens.metadata


def _hazard_block(
    inflow_by_real: dict[int, pd.DataFrame], ordered_ids: list[int], nyc_nodes,
    reference_monthly: np.ndarray, reference_daily: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute the candidate hazard image rows for one block of realizations, in ``ordered_ids`` order.

    Aggregates the NYC-inflow catchments to a single series per scenario and calls
    :func:`scengen.hazard_metrics.compute_candidate_hazard_image` once for the block.
    """
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import compute_candidate_hazard_image

    daily_rows, monthly_rows = [], []
    for k in ordered_ids:
        agg = inflow_by_real[k].loc[:, list(nyc_nodes)].sum(axis=1)  # daily pd.Series
        daily_rows.append(agg.to_numpy(dtype=float))
        monthly_rows.append(daily_to_monthly(agg, agg="mean"))
    H_block, axes = compute_candidate_hazard_image(
        np.vstack(monthly_rows), np.vstack(daily_rows), reference_monthly, reference_daily
    )
    return H_block, list(axes)


def _write_chunk_hdf5(
    chunk_dir: Path,
    global_ids: list[int],
    gage_by_real: dict[int, pd.DataFrame],
    inflow_by_real: dict[int, pd.DataFrame],
) -> None:
    """Write one chunk's daily HDF5s, renumbering its realizations to local keys ``0..S-1``.

    pywrdrb resolves a staged ensemble's realizations as ``0..N-1``, so each chunk is a standalone
    ensemble whose local key ``i`` maps to global index ``global_ids[i]`` (recorded in the chunk's
    ``_meta.json``). Only this chunk's frames are resident, so peak memory is bounded by the chunk.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(global_ids)
    gage_local = {i: gage_by_real[g] for i, g in enumerate(ordered)}
    inflow_local = {i: inflow_by_real[g] for i, g in enumerate(ordered)}
    Ensemble(gage_local).to_hdf5(str(chunk_dir / "gage_flow_mgd.hdf5"))
    Ensemble(inflow_local).to_hdf5(str(chunk_dir / "catchment_inflow_mgd.hdf5"))


def _write_chunk_meta(
    chunk_dir: Path, global_ids: list[int], config, sites: list[str], forcing_hash: str,
    *, chunk_idx: int,
) -> None:
    """Write a chunk's ``_meta.json`` so it resolves as a standalone staged ensemble.

    Carries ``n_realizations`` (local ``0..S-1``, what pywrdrb needs) plus ``global_realization_ids``
    (the local->global map) and provenance so downstream analysis re-keys chunk rows to global ids.
    """
    ordered = sorted(global_ids)
    L = config.realization_years
    meta = {
        "slug": chunk_dir.name,
        "kind": "forcing_master_chunk",
        "master_slug": chunk_dir.name.split("__chunk")[0],
        "chunk_index": chunk_idx,
        "n_realizations": len(ordered),
        "realization_years": L,
        "n_years": L,
        "global_realization_ids": [int(g) for g in ordered],
        "seed": config.master_seed,
        "master_seed": config.master_seed,
        "forcing_hash": forcing_hash,
        "flowtype": config.flowtype,
        "sites": sites,
        "source_kind": "synhydro_kn",
        "start_date": config.start_date,
    }
    (chunk_dir / "_meta.json").write_text(json.dumps(meta, indent=2))


def generate_master_ensemble(config) -> "EnsembleManifest":  # noqa: F821 (scengen contract)
    """Generate the master ensemble M and its streaming hazard image H (methods §3.2).

    Generalizes :func:`generate_input_spanning_master` to ``n_forcing_profiles x
    realizations_per_profile`` realizations, keyed by the global index ``k = p*R + j``. For each
    forcing profile the Kirsch moments are climate-adjusted (baseline-period reference), the profile's
    ``R`` realizations are generated, batch-disaggregated, KDE-filled, and reduced to marginal
    catchment inflows; the candidate hazard image is accumulated per block and the daily traces are
    discarded when ``config.store_daily`` is False (the ~1e6 production mode). Persists (always)
    ``forcing_profiles.npz``, ``hazard_image.npz``, ``_meta.json`` and ``manifest.json``, and (when
    ``store_daily``) the two daily HDF5s that workflow/02 consumes unchanged.

    Args:
        config: A ``scengen.master_ensemble.MasterEnsembleConfig`` (with forcing fields set).

    Returns:
        The ``scengen.manifest.EnsembleManifest`` describing M (design ``"master"``).
    """
    from scengen import forcing_space as fs
    from scengen import diagnostics as dg
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
    from scengen.hazard_filling import daily_to_monthly
    from scengen.manifest import EnsembleManifest

    if config.mean_frac_csv is None:
        raise ValueError("MasterEnsembleConfig.mean_frac_csv is required for forcing-based generation")
    if config.output_dir is None:
        raise ValueError("MasterEnsembleConfig.output_dir is required")

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N_forcing, R = config.n_forcing_profiles, config.realizations_per_profile
    N_M, L = N_forcing * R, config.realization_years

    print(f"[master] Fitting generators (baseline {config.baseline_period}, "
          f"full {config.full_period}); drawing {N_forcing} forcing profiles...")
    setup = _prepare_master(config)

    # Historical reference for the SSI/POT hazard fit (aggregate NYC inflow; fixed once).
    ref = load_historical_flows(gage=False, period="full", flowtype=config.flowtype)
    ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    reference_monthly = daily_to_monthly(ref_daily, agg="mean")
    reference_daily = ref_daily.to_numpy(dtype=float)

    forcing_hash = fs.forcing_hash(
        setup.a_wy, envelope_csv=config.mean_frac_csv, margin=config.margin, seed=config.master_seed
    )

    # Chunk plan: split the N_forcing profiles into contiguous chunks of `chunk_profiles` profiles
    # (= chunk_size realizations). chunk_size<=0 or >=N_M keeps the single-directory layout (daily
    # HDF5 in out_dir, backward-compatible); otherwise each chunk is written to a sibling staged dir
    # `{slug}__chunk{JJJ}` and out_dir holds only the global artifacts. Only one chunk's daily traces
    # are ever resident, so peak memory is bounded by chunk_size (methods §3.2).
    chunked = config.store_daily and 0 < config.chunk_size < N_M
    if chunked and config.chunk_size % R != 0:
        raise ValueError(
            f"chunk_size ({config.chunk_size}) must be a multiple of "
            f"realizations_per_profile ({R}) so chunks align to forcing profiles."
        )
    chunk_profiles = (config.chunk_size // R) if chunked else N_forcing
    block = max(1, min(config.hazard_block_size, chunk_profiles))

    H_blocks: list[np.ndarray] = []
    hazard_axes: list[str] = []
    sites: list[str] = []
    chunk_index: list[dict] = []

    for chunk_idx, pf0 in enumerate(range(0, N_forcing, chunk_profiles)):
        pf1 = min(pf0 + chunk_profiles, N_forcing)
        chunk_gage: dict[int, pd.DataFrame] = {}
        chunk_inflow: dict[int, pd.DataFrame] = {}
        for b0 in range(pf0, pf1, block):
            monthly_by_real: dict[int, pd.DataFrame] = {}
            metadata = None
            for p in range(b0, min(b0 + block, pf1)):
                block_monthly, md = _generate_profile_monthly(setup, config, p)
                monthly_by_real.update(block_monthly)
                if metadata is None:
                    metadata = md
            gage_by_real, inflow_by_real, sites = _disaggregate_fill_inflow(
                Ensemble(monthly_by_real, metadata=metadata),
                nowak=setup.nowak, kdes=setup.kdes,
                master_seed=config.master_seed, start_date=config.start_date,
            )
            H_block, hazard_axes = _hazard_block(
                inflow_by_real, sorted(inflow_by_real), DEFAULT_NYC_INFLOW_NODES,
                reference_monthly, reference_daily,
            )
            H_blocks.append(H_block)
            if config.store_daily:
                chunk_gage.update(gage_by_real)
                chunk_inflow.update(inflow_by_real)

        global_ids = list(range(pf0 * R, pf1 * R))
        if config.store_daily:
            chunk_dir = out_dir.parent / f"{out_dir.name}__chunk{chunk_idx:03d}" if chunked else out_dir
            _write_chunk_hdf5(chunk_dir, global_ids, chunk_gage, chunk_inflow)
            if chunked:
                _write_chunk_meta(chunk_dir, global_ids, config, sites, forcing_hash,
                                  chunk_idx=chunk_idx)
            chunk_index.append({
                "chunk_index": chunk_idx,
                "slug": chunk_dir.name,
                "global_start": global_ids[0],
                "global_end": global_ids[-1] + 1,
                "n_realizations": len(global_ids),
            })
        print(f"[master]   chunk {chunk_idx}: profiles [{pf0},{pf1}) -> "
              f"realizations [{pf0 * R},{pf1 * R}); H rows {sum(len(b) for b in H_blocks)}/{N_M}")

    if not config.store_daily:
        print("[master] stream_only: daily traces discarded; persisting H + seeds + params only.")

    H = np.vstack(H_blocks)  # (N_M, m), rows in global-index order 0..N_M-1
    realization_ids = np.arange(N_M, dtype=int)

    # Per-realization theta (repeat each profile's factor R times) aligned with the HDF5 keys.
    profiles_payload = {
        "realization_ids": realization_ids,
        "mean_factor_a": np.repeat(setup.a_wy, R, axis=0).astype(float),
        "water_year_months": np.array(fs.WATER_YEAR_MONTHS, dtype=int),
        "richness": "mean_var" if config.variance_axis else "mean_only",
        "n_forcing_profiles": N_forcing,
        "realizations_per_profile": R,
    }
    if config.variance_axis:
        profiles_payload["cv_factor_v"] = np.repeat(setup.v_wy, R, axis=0).astype(float)
    np.savez(out_dir / "forcing_profiles.npz", **profiles_payload)

    dg.save_hazard_image(
        out_dir / "hazard_image.npz",
        H=H, hazard_axes=hazard_axes, realization_ids=realization_ids, selected_rows=realization_ids,
    )

    # Redundancy screen on H (informational; per-design selection screen runs in workflow/02).
    spread = dg.per_metric_spread(H, hazard_axes)
    clusters = dg.spearman_clusters(H, hazard_axes)
    screen_result = {
        "candidate_axes": list(hazard_axes),
        "clusters": clusters["clusters"],
        "representatives": clusters["representatives"],
        "degenerate": {name: bool(spread[name]["degenerate"]) for name in hazard_axes},
    }

    # Chunk index maps each staged daily chunk to its global realization range (consumed by the
    # chunk-aware materializer + chunk simulation). Empty when store_daily is False (regenerate).
    n_chunks = len(chunk_index)
    (out_dir / "chunk_index.json").write_text(json.dumps(
        {"master_slug": out_dir.name, "n_realizations": N_M, "chunk_size": config.chunk_size,
         "n_chunks": n_chunks, "chunks": chunk_index}, indent=2,
    ))

    # Informational staged-spec meta (consumed by src.ensembles._spec_from_staged_dir + workflow/02).
    meta = {
        "slug": out_dir.name,
        "kind": "forcing_master",
        "richness": "mean_var" if config.variance_axis else "mean_only",
        "variance_axis": config.variance_axis,
        "flowtype": config.flowtype,
        "n_years": L,
        "realization_years": L,
        "n_realizations": N_M,
        "n_forcing_profiles": N_forcing,
        "realizations_per_profile": R,
        "seed": config.master_seed,
        "store_daily": config.store_daily,
        "chunk_size": config.chunk_size,
        "n_chunks": n_chunks,
        "bound_pct": list(config.bound_pct),
        "margin": config.margin,
        "forcing_param": "harmonic",
        "mean_frac_csv": str(config.mean_frac_csv),
        "forcing_hash": forcing_hash,
        "sites": sites,
        "source_kind": "synhydro_kn",
        "start_date": config.start_date,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

    manifest = EnsembleManifest(
        design="master",
        draw=0,
        n_realizations=N_M,
        realization_years=L,
        master_seed=config.master_seed,
        realization_global_indices=realization_ids.tolist(),
        forcing_hash=forcing_hash,
        hazard_axes=list(hazard_axes),
        screen_result=screen_result,
        slug=out_dir.name,
        created=datetime.now(timezone.utc).isoformat(),
        source_kind="synhydro_kn",
        notes=(f"forcing master: {N_forcing} profiles x {R} realizations = {N_M}, L={L}yr, "
               f"richness={'mean_var' if config.variance_axis else 'mean_only'}, "
               f"store_daily={config.store_daily}"),
    )
    manifest.write(out_dir, filename="manifest.json")
    print(f"[master] Done '{out_dir.name}': N_M={N_M}, H={H.shape}, store_daily={config.store_daily}.")
    return manifest


def regenerate_realization(master_seed: int, global_index: int, *, config) -> pd.DataFrame:
    """Regenerate a single master realization's daily gage flows bit-for-bit from its global index.

    Re-fits the (deterministic) generators, re-derives the forcing profile ``p = global_index // R``,
    and regenerates only realization ``global_index`` through the same Kirsch -> Nowak -> KDE path,
    keyed to the same global-index RNG streams — identical regardless of N or MPI layout. The marginal
    catchment inflows are a pure function of these gage flows, so gage-array equality implies inflow
    equality. Validated by ``tests/test_master_ensemble_determinism.py``.

    Args:
        master_seed: Must equal ``config.master_seed`` (generation is fully keyed to it).
        global_index: The master realization to regenerate, ``0 <= k < n_realizations``.
        config: The same ``MasterEnsembleConfig`` used to generate the master.

    Returns:
        Daily multi-node gage flows (float32) for the requested realization.
    """
    if master_seed != config.master_seed:
        raise ValueError(
            f"master_seed ({master_seed}) must equal config.master_seed ({config.master_seed}); "
            f"generation is fully determined by config.master_seed."
        )
    if not 0 <= global_index < config.n_realizations:
        raise ValueError(
            f"global_index {global_index} out of range [0, {config.n_realizations})."
        )

    profile_idx = global_index // config.realizations_per_profile
    setup = _prepare_master(config)
    monthly, metadata = _generate_profile_monthly(
        setup, config, profile_idx, indices=[global_index]
    )
    gage_by_real, _inflow, _sites = _disaggregate_fill_inflow(
        Ensemble({global_index: monthly[global_index]}, metadata=metadata),
        nowak=setup.nowak, kdes=setup.kdes,
        master_seed=master_seed, start_date=config.start_date,
    )
    return gage_by_real[global_index]
