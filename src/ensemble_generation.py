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
from synhydro.core.ensemble import EnsembleMetadata
from synhydro.core.seeding import as_seed_sequence, spawn_realization_seed
from synhydro.methods.generation.parametric import MultiSiteHMMGenerator
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

# SynHydro's timescale-generalized NowakDisaggregator defaults
# boundary_blend_timesteps to 0 (published Nowak et al. 2010, no boundary
# correction). This project's ensembles were generated with the pre-upgrade
# default of 2-day smoothing at month boundaries; pin it so regenerated
# ensembles stay bit-identical to prior runs.
NOWAK_BOUNDARY_BLEND_TIMESTEPS = 2


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
        seed: Root random seed for Kirsch + KDE sampling.
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

    Q_full, Q_inflow, _ = _load_masked_flows(flowtype)
    kirsch = _fit_kirsch(Q_full)
    nowak = _fit_nowak(Q_full)

    monthly_ensemble = kirsch.generate(
        n_realizations=n_realizations,
        n_years=n_years,
        seed=seed,
    )

    syn_by_real, inflow_by_real, sites = _disaggregate_fill_inflow(
        monthly_ensemble, nowak=nowak, kdes=_fit_downstream_kdes(Q_inflow),
        root_seed=seed, start_date=start_date,
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
    nowak = NowakDisaggregator(
        boundary_blend_timesteps=NOWAK_BOUNDARY_BLEND_TIMESTEPS, debug=False
    )
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
    root_seed: int | np.random.SeedSequence,
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
        root_seed: Root seed forwarded to the Nowak and KDE streams.
        start_date: Date assigned to day 0 of each realization.

    Returns:
        ``(gage_by_real, inflow_by_real, sites)`` — float32 frames keyed by the same global indices,
        with a shared column order ``sites``.
    """
    daily = nowak.disaggregate(monthly_ensemble, seed=root_seed)
    syn_by_real = daily.data_by_realization

    _apply_kde_downstream(syn_by_real, kdes, root_seed=root_seed)

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
    root: np.random.SeedSequence, realization_id: int, pair_name: str
) -> np.random.Generator:
    """Return the KDE-fill RNG for one (realization, downstream-pair), keyed to the global index.

    The stream is realization ``realization_id``'s SynHydro child seed
    (``spawn_realization_seed``), namespaced by ``crc32(pair_name)``. It therefore depends only on
    ``(root, realization_id, pair_name)`` — invariant to how the realization range is partitioned
    across MPI ranks / batches — and is disjoint from SynHydro's ``"generation"``/``"disaggregation"``
    sub-streams (whose spawn indices are 0 and 1, never a large CRC32). This is the only randomness in
    the pipeline not already covered by the SynHydro sub-streams (methods §7 determinism contract).
    """
    child = spawn_realization_seed(root, int(realization_id))
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
    root_seed: int | np.random.SeedSequence,
) -> None:
    """Inject regression-sampled downstream flows into each realization.

    Mutates ``syn_by_real`` in place — adds a column per downstream gage. Each realization's fill is
    resampled independently from its own global-index-keyed stream (:func:`_kde_stream`), so the
    result is bit-for-bit reproducible and invariant to the batch/MPI partition. A single downstream
    fraction series of length ``n_days`` is drawn per (realization, pair) — do NOT batch the resample
    across realizations, which would couple them (methods §7).
    """
    root = as_seed_sequence(root_seed)
    n_days = next(iter(syn_by_real.values())).shape[0]

    for upstream in NODES_TO_GENERATE:
        downstream = immediate_downstream_nodes_dict[upstream]
        if downstream not in NODES_TO_REGRESS:
            continue
        pair_name = f"{upstream}_to_{downstream}"
        kde = kdes[pair_name]
        lag = downstream_node_lags[downstream]

        for real_id, gage_df in syn_by_real.items():
            rng = _kde_stream(root, real_id, pair_name)
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
# Forcing ensembles / candidate pools (methods §3.2):
#   N_forcing x realizations_per_profile realizations, optional streaming hazard image
###############################################################################

@dataclass
class _GeneratorSetup:
    """Fitted generators + (for ``du_forced``) the forcing draw, shared by generation and regen."""

    kirsch: KirschGenerator | None      # ``generator="kn"``: monthly generator
    nowak: NowakDisaggregator           # monthly -> daily (shared by both generators)
    kdes: dict[str, stats.gaussian_kde]
    base_mean: pd.DataFrame | None      # baseline-period log-space mean (eqs 10-11 reference)
    base_std: pd.DataFrame | None       # baseline-period log-space std
    a_wy: np.ndarray | None             # (N_forcing, 12) water-year mean change factors
    v_wy: np.ndarray | None             # (N_forcing, 12) CV change factors, or None
    theta_params: np.ndarray | None     # (N_forcing, d) intrinsic harmonic coordinates
    theta_names: list[str]              # axis labels of ``theta_params``
    hmm: MultiSiteHMMGenerator | None = None   # ``generator="hmm"``: annual generator
    ann2mon: NowakDisaggregator | None = None  # ``generator="hmm"``: annual -> monthly


def _sample_forcing(
    config,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Draw the forcing profiles (mean ``a``, optional CV ``v``) from the CMIP6 hypercube.

    A ``stationary`` population has no theta at all — there is nothing to sample and nothing to
    adjust — so this returns all-``None`` and :func:`_generate_profile_monthly` leaves the fitted
    full-record moments untouched.

    ``config.theta_sampler`` is forwarded to BOTH harmonic draws (mean and CV axes). A candidate
    pool that is later subsampled MUST be ``"iid"``: a uniform subset of an i.i.d. pool is
    distributionally identical to i.i.d. draws, which is what makes a generate-N design the exact
    control for a subsample-N-from-a-pool design; the equivalence fails for LHS (see
    ``scengen.forcing_space.sample_harmonic_forcing``).

    Returns:
        ``(a_wy, v_wy, theta_params, theta_names)`` — ``a_wy``/``v_wy`` are ``(N_forcing, 12)``
        water-year change factors (``v_wy`` is ``None`` unless ``variance_axis``), ``theta_params``
        the ``(N_forcing, d)`` intrinsic harmonic coordinates (the mean axes, plus the ``cv_``-
        prefixed CV axes when ``variance_axis``), and ``theta_names`` its axis labels.
    """
    from scengen import forcing_space as fs
    from scengen.seeds import design_seed

    if config.population == "stationary":
        return None, None, None, []

    mean_env = fs.load_cmip6_envelope(config.mean_frac_csv)
    a_wy, a_params, a_names = fs.sample_harmonic_forcing(
        config.n_forcing_profiles, mean_env,
        seed=config.root_seed, bound_pct=config.bound_pct, margin=config.margin,
        method=config.theta_sampler, return_params=True,
    )
    v_wy = None
    theta_params, theta_names = np.asarray(a_params, dtype=float), list(a_names)
    if config.variance_axis:
        if config.mean_abs_csv is None or config.std_csv is None:
            raise ValueError("variance_axis=True requires mean_abs_csv and std_csv on the config")
        cv_env = fs.derive_variance_envelope(config.mean_abs_csv, config.std_csv)
        # The CV axis is an INDEPENDENT stream: namespace it rather than using the old `root+1`
        # adjacency, which the namespaced seed scheme (scengen.seeds) exists to retire.
        v_wy, v_params, v_names = fs.sample_harmonic_forcing(
            config.n_forcing_profiles, cv_env,
            seed=design_seed(config.root_seed, "cv_axis", 0),
            bound_pct=config.bound_pct, margin=config.margin,
            method=config.theta_sampler, return_params=True,
        )
        theta_params = np.hstack([theta_params, np.asarray(v_params, dtype=float)])
        theta_names += [f"cv_{n}" for n in v_names]
    return a_wy, v_wy, theta_params, theta_names


def _complete_calendar_years(Q: pd.DataFrame) -> pd.DataFrame:
    """Trim a daily record to whole calendar years (Jan 1 .. Dec 31).

    The HMM is fitted on annual totals, so a partial leading/trailing year would enter the fit as a
    spuriously dry year. The calendar (not water) year is the right boundary because the whole
    monthly pipeline — Kirsch's ``mean_period`` and the Nowak proportion pools — is calendar-indexed.
    """
    idx = Q.index
    start = idx[idx.month == 1][0] if (idx.month == 1).any() else idx[0]
    start = pd.Timestamp(year=start.year, month=1, day=1)
    end = pd.Timestamp(year=idx[-1].year - (0 if (idx[-1].month == 12 and idx[-1].day == 31) else 1),
                       month=12, day=31)
    return Q.loc[start:end]


def _fit_hmm(Q: pd.DataFrame, *, seed: int) -> MultiSiteHMMGenerator:
    """Fit a multi-site Gaussian-mixture HMM (Gold et al. 2024) to ANNUAL totals of ``Q``.

    The structurally different generator of the held-out test ensemble: flows are drawn from
    state-conditional multivariate log-normals with a Markov state chain, so interannual wet/dry
    sequencing is a fitted transition process rather than the shuffled-historical-year structure
    Kirsch inherits. Annual totals are the HMM's native timestep; the seasonal cycle is restored by
    the annual->monthly Nowak disaggregator.

    Args:
        Q: Daily multi-site gage record (zero-masked).
        seed: EM ``random_state``, so the fit is reproducible.
    """
    annual = _complete_calendar_years(Q).resample("YS").sum()
    hmm = MultiSiteHMMGenerator(n_states=2, covariance_type="full", debug=False)
    hmm.preprocessing(annual)
    hmm.fit(random_state=seed)
    return hmm


def _fit_annual_to_monthly(Q: pd.DataFrame) -> NowakDisaggregator:
    """Fit the annual->monthly Nowak disaggregator on observed monthly totals of ``Q``.

    Restores the seasonal cycle of an HMM annual realization by KNN-resampling a historical year's
    monthly proportion vector, so the disaggregated months sum to the synthetic annual total and the
    monthly frames land in EXACTLY the convention the Kirsch path emits (calendar-indexed monthly
    totals), which is what lets both generators share :func:`_disaggregate_fill_inflow`.
    """
    monthly = _complete_calendar_years(Q).resample("MS").sum()
    disagg = NowakDisaggregator(input_timestep="annual", output_timestep="monthly", debug=False)
    disagg.preprocessing(monthly)
    disagg.fit()
    return disagg


def _profile_stream_seed(root_seed: int, profile_idx: int, stream: str) -> int:
    """Deterministic integer seed for one (forcing profile, RNG stream) of the HMM path.

    The HMM generator has no ``realization_indices`` API (unlike Kirsch), so its draws are keyed to
    the forcing PROFILE rather than the global realization index. Chunks are profile-aligned by
    construction, so this is still invariant to the chunk/MPI partition; the daily disaggregation and
    KDE fill downstream remain keyed to the global index.
    """
    child = spawn_realization_seed(as_seed_sequence(root_seed), int(profile_idx))
    namespaced = np.random.SeedSequence(
        entropy=child.entropy,
        spawn_key=tuple(child.spawn_key) + (zlib.crc32(stream.encode()),),
        pool_size=child.pool_size,
    )
    return int(namespaced.generate_state(1, dtype=np.uint32)[0])


def _prepare_generators(config) -> _GeneratorSetup:
    """Fit the flow generator (§3.2) + Nowak + KDEs and draw the forcing profiles.

    ``config.generator`` selects the family. ``"kn"``: a full-record Kirsch generator produces the
    realizations, and for ``du_forced`` a second baseline-period Kirsch supplies the eqs-10-11
    reference moments. ``"hmm"``: an annual multi-site HMM plus an annual->monthly Nowak; it needs no
    log-moment reference because the forcing is applied as a monthly delta-change (see
    :func:`_generate_profile_monthly`). Deterministic given ``config``.
    """
    generator = getattr(config, "generator", "kn")
    needs_reference = config.population != "stationary" and generator == "kn"
    baseline = config.baseline_period if needs_reference else None
    Q_gage, Q_inflow, Q_base = _load_masked_flows(config.flowtype, baseline_period=baseline)
    Q_gen = Q_gage.loc[config.full_period[0]:config.full_period[1]]
    base_mean = base_std = None
    if Q_base is not None:
        kirsch_base = _fit_kirsch(Q_base)
        base_mean = kirsch_base.mean_period.copy()
        base_std = kirsch_base.std_period.copy()
    a_wy, v_wy, theta_params, theta_names = _sample_forcing(config)
    return _GeneratorSetup(
        kirsch=_fit_kirsch(Q_gen) if generator == "kn" else None,
        nowak=_fit_nowak(Q_gen),
        kdes=_fit_downstream_kdes(Q_inflow),
        base_mean=base_mean,
        base_std=base_std,
        a_wy=a_wy,
        v_wy=v_wy,
        theta_params=theta_params,
        theta_names=theta_names,
        hmm=_fit_hmm(Q_gen, seed=config.root_seed) if generator == "hmm" else None,
        ann2mon=_fit_annual_to_monthly(Q_gen) if generator == "hmm" else None,
    )


def _generate_profile_monthly(
    setup: _GeneratorSetup, config, profile_idx: int, *, indices: list[int] | None = None,
) -> tuple[dict[int, pd.DataFrame], object]:
    """Generate profile ``profile_idx``'s realizations under the configured generator.

    Dispatches to :func:`_generate_profile_monthly_hmm` when ``config.generator == "hmm"``; the
    Kirsch path is below.

    When ``setup.a_wy is None`` (a ``stationary`` population) the climate adjustment is SKIPPED
    ENTIRELY: ``setup.kirsch`` keeps the ``mean_period``/``std_period`` it was fitted with on the
    full record. Passing ``a = 1`` instead would be algebraically the identity but would *overwrite*
    those moments with the baseline-period ones, silently changing the generator.

    Global indices default to the profile's full block ``[p*R, …, p*R+R-1]``; pass ``indices`` to
    regenerate a subset (each index draws from its own ``realization_rng(root, k, "generation")``
    stream, so a subset is identical to the full-block draw for those indices). Returns the monthly
    frames keyed by global index and the generator metadata.
    """
    from scengen import forcing_space as fs

    if getattr(config, "generator", "kn") == "hmm":
        return _generate_profile_monthly_hmm(setup, config, profile_idx, indices=indices)

    R = config.realizations_per_profile
    if setup.a_wy is not None:
        a_cal = fs.water_year_to_calendar(setup.a_wy[profile_idx])
        if setup.v_wy is not None:
            c_cal = a_cal * fs.water_year_to_calendar(setup.v_wy[profile_idx])
        else:
            c_cal = np.ones(12)
        mean_new, std_new = fs.apply_climate_adjustment(
            np.asarray(setup.base_mean, dtype=float), np.asarray(setup.base_std, dtype=float),
            a_cal, c_profile=c_cal,
        )
        setup.kirsch.mean_period = pd.DataFrame(
            mean_new, index=setup.base_mean.index, columns=setup.base_mean.columns
        )
        setup.kirsch.std_period = pd.DataFrame(
            std_new, index=setup.base_std.index, columns=setup.base_std.columns
        )

    idxs = list(indices) if indices is not None else [profile_idx * R + j for j in range(R)]
    ens = setup.kirsch.generate(
        n_years=config.realization_years, realization_indices=idxs, seed=config.root_seed
    )
    return {k: ens.data_by_realization[k] for k in idxs}, ens.metadata


def _generate_profile_monthly_hmm(
    setup: _GeneratorSetup, config, profile_idx: int, *, indices: list[int] | None = None,
) -> tuple[dict[int, pd.DataFrame], object]:
    """Generate profile ``profile_idx``'s realizations with the annual HMM + annual->monthly Nowak.

    The forcing enters as a **monthly delta-change**: the disaggregated monthly totals are multiplied
    by the profile's calendar-ordered change factors ``a_j``, so theta shifts both the annual level
    and the seasonal shape exactly as it prescribes. This is a DIFFERENT (and equally published)
    mechanism from the Kirsch path's log-moment adjustment (Kirsch et al. 2013 eqs. 10-11), which
    perturbs fitted log-space moments and has no analogue for state-conditional HMM emissions. That
    is acceptable — indeed useful — HERE and ONLY here: the test ensemble is a measuring stick, never
    a control, so its two constructions are meant to differ structurally. The search-side designs are
    all ``generator="kn"`` and are therefore still exactly controlled against each other.

    The whole profile block is generated even when ``indices`` requests a subset: the HMM draws its R
    realizations from one RNG in sequence, so realization ``j`` is only reproducible as part of its
    block. Regeneration is therefore block-exact rather than realization-exact, at R x the cost of a
    single realization.

    Args:
        setup: Fitted setup carrying ``hmm`` and ``ann2mon``.
        config: The ``ForcingEnsembleConfig``.
        profile_idx: Forcing-profile index p.
        indices: Optional subset of the block's global indices to return.

    Returns:
        ``(monthly_frames_by_global_index, ensemble_metadata)``.

    Raises:
        NotImplementedError: If a CV/variance forcing axis is requested (the delta-change mechanism
            perturbs monthly means only; a CV axis would need a variance model the HMM does not
            expose).
    """
    from scengen import forcing_space as fs

    if setup.v_wy is not None:
        raise NotImplementedError(
            "generator='hmm' does not support the CV/variance forcing axis: the delta-change "
            "mechanism perturbs monthly means only. Generate the HMM test ensemble with "
            "variance_axis=False, or extend the HMM emission covariances explicitly."
        )

    R = config.realizations_per_profile
    annual_raw = setup.hmm.generate(
        n_realizations=R,
        n_years=config.realization_years,
        seed=_profile_stream_seed(config.root_seed, profile_idx, "hmm_generation"),
    )
    annual = Ensemble(
        annual_raw.data_by_realization, metadata=EnsembleMetadata(time_resolution="YS"),
    )
    monthly = setup.ann2mon.disaggregate(
        annual, seed=_profile_stream_seed(config.root_seed, profile_idx, "hmm_annual_to_monthly"),
    )
    frames = monthly.data_by_realization  # keyed 0..R-1 by the HMM's local realization ids

    if setup.a_wy is not None:
        a_cal = fs.water_year_to_calendar(setup.a_wy[profile_idx])
        for j, df in frames.items():
            factors = pd.Series(a_cal[df.index.month - 1], index=df.index)
            frames[j] = df.mul(factors, axis=0)

    block = {profile_idx * R + j: frames[j] for j in range(R)}
    if indices is not None:
        block = {k: block[k] for k in indices}
    return block, monthly.metadata


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
    R = config.realizations_per_profile
    meta = {
        "slug": chunk_dir.name,
        "kind": "forcing_pool_chunk",
        "pool_slug": chunk_dir.name.split("__chunk")[0],
        "chunk_index": chunk_idx,
        "n_realizations": len(ordered),
        "realization_years": L,
        "n_years": L,
        "global_realization_ids": [int(g) for g in ordered],
        "population": config.population,
        "generator": getattr(config, "generator", "kn"),
        "seed_domain": getattr(config, "seed_domain", None),
        "theta_sampler": config.theta_sampler if config.population != "stationary" else None,
        # A chunk is a standalone staged ensemble, so it must carry the SOW grouping itself: the
        # re-eval cube is keyed to global ids, but a chunk-local scorer (src.chunk_reeval) resolves
        # its spec from this file. p = global_id // R.
        "realizations_per_profile": R,
        "n_forcing_profiles": len(ordered) // R,
        "seed": config.root_seed,
        "root_seed": config.root_seed,
        "forcing_hash": forcing_hash,
        "flowtype": config.flowtype,
        "sites": sites,
        "source_kind": f"synhydro_{getattr(config, 'generator', 'kn')}",
        "start_date": config.start_date,
    }
    (chunk_dir / "_meta.json").write_text(json.dumps(meta, indent=2))


def _validate_config(config) -> None:
    """Fail fast on an inconsistent :class:`scengen.forcing_ensemble.ForcingEnsembleConfig`."""
    if config.output_dir is None:
        raise ValueError("ForcingEnsembleConfig.output_dir is required")
    if config.population not in ("stationary", "du_forced"):
        raise ValueError(
            f"unknown population {config.population!r}; expected 'stationary' or 'du_forced'"
        )
    if config.population == "stationary":
        # There is no theta, so a profile IS a realization: the global-index keying (k = p*R + j)
        # and the chunking machinery then work unchanged with n_forcing_profiles = N.
        if config.realizations_per_profile != 1:
            raise ValueError(
                "population='stationary' has no forcing profiles: set realizations_per_profile=1 "
                f"and n_forcing_profiles=N (got realizations_per_profile="
                f"{config.realizations_per_profile})."
            )
    elif config.mean_frac_csv is None:
        raise ValueError("population='du_forced' requires ForcingEnsembleConfig.mean_frac_csv")
    if config.theta_sampler not in ("iid", "lhs"):
        raise ValueError(
            f"unknown theta_sampler {config.theta_sampler!r}; expected 'iid' or 'lhs'"
        )
    if getattr(config, "generator", "kn") not in ("kn", "hmm"):
        raise ValueError(
            f"unknown generator {config.generator!r}; expected 'kn' or 'hmm'"
        )


def generate_forcing_ensemble(config) -> "EnsembleManifest":  # noqa: F821 (scengen contract)
    """Generate one design's realizations (or its candidate pool) and persist them (methods §3.2).

    Generates ``n_forcing_profiles x realizations_per_profile`` realizations keyed by the global index
    ``k = p*R + j``. For a ``du_forced`` population each forcing profile's Kirsch moments are
    climate-adjusted against the baseline-period reference; for a ``stationary`` population the
    fitted full-record moments are used unchanged (R is 1, so a "profile" is just a realization).
    Each block is generated, batch-disaggregated, KDE-filled, and reduced to marginal catchment
    inflows; the daily traces are discarded when ``config.store_daily`` is False.

    The candidate hazard image is streamed only when ``config.compute_hazard_image`` is True — only
    the hazard-filling designs subsample, and the SSI-6 + POT pass is pure waste for the rest.

    Persists ``_meta.json``, ``chunk_index.json`` and ``manifest.json`` always; ``forcing_profiles.npz``
    for a ``du_forced`` population; ``hazard_image.npz`` when the hazard image is computed; and the two
    daily HDF5s (per chunk) when ``store_daily``.

    Args:
        config: A ``scengen.forcing_ensemble.ForcingEnsembleConfig``.

    Returns:
        The ``scengen.manifest.EnsembleManifest`` describing the generated ensemble.
    """
    from scengen import forcing_space as fs
    from scengen import diagnostics as dg
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES
    from scengen.hazard_filling import daily_to_monthly
    from scengen.manifest import EnsembleManifest

    _validate_config(config)
    forced = config.population == "du_forced"

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N_forcing, R = config.n_forcing_profiles, config.realizations_per_profile
    N, L = N_forcing * R, config.realization_years

    print(f"[gen] Fitting generators (generator={getattr(config, 'generator', 'kn')}, "
          f"population={config.population}, full {config.full_period})"
          + (f"; drawing {N_forcing} {config.theta_sampler} forcing profiles" if forced else "")
          + "...")
    setup = _prepare_generators(config)

    # Historical reference for the SSI/POT hazard fit (aggregate NYC inflow; fixed once).
    reference_monthly = reference_daily = None
    if config.compute_hazard_image:
        ref = load_historical_flows(gage=False, period="full", flowtype=config.flowtype)
        ref_daily = ref.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
        reference_monthly = daily_to_monthly(ref_daily, agg="mean")
        reference_daily = ref_daily.to_numpy(dtype=float)

    forcing_hash = fs.forcing_hash(
        setup.a_wy, envelope_csv=config.mean_frac_csv, margin=config.margin, seed=config.root_seed
    ) if forced else ""

    # Chunk plan: split the N_forcing profiles into contiguous chunks of `chunk_profiles` profiles
    # (= chunk_size realizations). chunk_size<=0 or >=N keeps the single-directory layout (daily
    # HDF5 in out_dir); otherwise each chunk is written to a sibling staged dir `{slug}__chunk{JJJ}`
    # and out_dir holds only the global artifacts. Only one chunk's daily traces are ever resident,
    # so peak memory is bounded by chunk_size (methods §3.2).
    chunked = config.store_daily and 0 < config.chunk_size < N
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
                root_seed=config.root_seed, start_date=config.start_date,
            )
            if config.compute_hazard_image:
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
        print(f"[gen]   chunk {chunk_idx}: profiles [{pf0},{pf1}) -> "
              f"realizations [{pf0 * R},{pf1 * R}) of {N}")

    realization_ids = np.arange(N, dtype=int)

    # Per-realization theta (repeat each profile's coordinates R times) aligned with the HDF5 keys.
    if forced:
        profiles_payload = {
            "realization_ids": realization_ids,
            "mean_factor_a": np.repeat(setup.a_wy, R, axis=0).astype(float),
            "theta_params": np.repeat(setup.theta_params, R, axis=0).astype(float),
            "theta_param_names": np.array(setup.theta_names),
            "water_year_months": np.array(fs.WATER_YEAR_MONTHS, dtype=int),
            "theta_sampler": config.theta_sampler,
            "richness": "mean_var" if config.variance_axis else "mean_only",
            "n_forcing_profiles": N_forcing,
            "realizations_per_profile": R,
        }
        if config.variance_axis:
            profiles_payload["cv_factor_v"] = np.repeat(setup.v_wy, R, axis=0).astype(float)
        np.savez(out_dir / "forcing_profiles.npz", **profiles_payload)

    # Redundancy screen on H (informational; per-design selection screen runs in workflow/03).
    screen_result = None
    if config.compute_hazard_image:
        H = np.vstack(H_blocks)  # (N, m), rows in global-index order 0..N-1
        dg.save_hazard_image(
            out_dir / "hazard_image.npz",
            H=H, hazard_axes=hazard_axes, realization_ids=realization_ids,
            selected_rows=realization_ids,
        )
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
        {"pool_slug": out_dir.name, "n_realizations": N, "chunk_size": config.chunk_size,
         "n_chunks": n_chunks, "chunks": chunk_index}, indent=2,
    ))

    # Informational staged-spec meta (consumed by src.ensembles._spec_from_staged_dir + workflow/03).
    generator = getattr(config, "generator", "kn")
    meta = {
        "slug": out_dir.name,
        "kind": "forcing_pool",
        "population": config.population,
        "generator": generator,
        # Namespace of root_seed. config.py compares the SEARCH and TEST ensembles on this key and
        # HARD-ERRORS when they match, so a test ensemble can never be drawn from a search seed
        # stream (selection bias, Bonham et al. 2024). It is recorded per-ensemble because two
        # ensembles can share a generator stream under different slugs.
        "seed_domain": getattr(config, "seed_domain", None),
        "theta_sampler": config.theta_sampler if forced else None,
        "richness": ("mean_var" if config.variance_axis else "mean_only") if forced else None,
        "variance_axis": config.variance_axis if forced else False,
        "flowtype": config.flowtype,
        "n_years": L,
        "realization_years": L,
        "n_realizations": N,
        # (n_forcing_profiles, realizations_per_profile) IS the SOW grouping: realization k belongs
        # to forcing profile (SOW) k // R. src.reeval_core reads it back so the SOW-unit robustness
        # metric is computable offline from the persisted re-eval cube.
        "n_forcing_profiles": N_forcing,
        "realizations_per_profile": R,
        "seed": config.root_seed,
        "root_seed": config.root_seed,
        "store_daily": config.store_daily,
        "compute_hazard_image": config.compute_hazard_image,
        "chunk_size": config.chunk_size,
        "n_chunks": n_chunks,
        "bound_pct": list(config.bound_pct) if forced else None,
        "margin": config.margin if forced else None,
        "forcing_param": "harmonic" if forced else None,
        "mean_frac_csv": str(config.mean_frac_csv) if forced else None,
        "forcing_hash": forcing_hash,
        "sites": sites,
        "source_kind": f"synhydro_{generator}",
        "start_date": config.start_date,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

    manifest = EnsembleManifest(
        design=f"{config.population}_pool",
        draw=0,
        n_realizations=N,
        realization_years=L,
        master_seed=config.root_seed,
        realization_global_indices=realization_ids.tolist(),
        forcing_hash=forcing_hash,
        hazard_axes=list(hazard_axes),
        screen_result=screen_result,
        slug=out_dir.name,
        created=datetime.now(timezone.utc).isoformat(),
        source_kind=f"synhydro_{generator}",
        notes=(f"{config.population} pool: {N_forcing} profiles x {R} realizations = {N}, L={L}yr, "
               f"generator={generator}, "
               f"theta_sampler={config.theta_sampler if forced else 'n/a'}, "
               f"richness={'mean_var' if config.variance_axis else 'mean_only'}, "
               f"store_daily={config.store_daily}, hazard_image={config.compute_hazard_image}"),
    )
    manifest.write(out_dir, filename="manifest.json")
    print(f"[gen] Done '{out_dir.name}': N={N}, store_daily={config.store_daily}, "
          f"hazard_image={config.compute_hazard_image}.")
    return manifest


def regenerate_realization(root_seed: int, global_index: int, *, config) -> pd.DataFrame:
    """Regenerate a single realization's daily gage flows bit-for-bit from its global index.

    Re-fits the (deterministic) generators, re-derives the forcing profile ``p = global_index // R``,
    and regenerates only realization ``global_index`` through the same Kirsch -> Nowak -> KDE path,
    keyed to the same global-index RNG streams — identical regardless of N or MPI layout. The marginal
    catchment inflows are a pure function of these gage flows, so gage-array equality implies inflow
    equality. Validated by ``tests/test_master_ensemble_determinism.py``.

    Args:
        root_seed: Must equal ``config.root_seed`` (generation is fully keyed to it).
        global_index: The realization to regenerate, ``0 <= k < n_realizations``.
        config: The same ``ForcingEnsembleConfig`` used to generate the ensemble.

    Returns:
        Daily multi-node gage flows (float32) for the requested realization.
    """
    if root_seed != config.root_seed:
        raise ValueError(
            f"root_seed ({root_seed}) must equal config.root_seed ({config.root_seed}); "
            f"generation is fully determined by config.root_seed."
        )
    if not 0 <= global_index < config.n_realizations:
        raise ValueError(
            f"global_index {global_index} out of range [0, {config.n_realizations})."
        )

    profile_idx = global_index // config.realizations_per_profile
    setup = _prepare_generators(config)
    monthly, metadata = _generate_profile_monthly(
        setup, config, profile_idx, indices=[global_index]
    )
    gage_by_real, _inflow, _sites = _disaggregate_fill_inflow(
        Ensemble({global_index: monthly[global_index]}, metadata=metadata),
        nowak=setup.nowak, kdes=setup.kdes,
        root_seed=root_seed, start_date=config.start_date,
    )
    return gage_by_real[global_index]
