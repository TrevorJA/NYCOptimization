"""diagnose_hazard_selectors.py - Selector + axis-set + sizing diagnostics for hazard filling.

Supplemental (SI) experiment characterizing the hazard-filling selection
machinery on a staged candidate pool at the selection level (no simulation);
design and findings in ``docs/notes/methods/hazard_selector_diagnostics.md``.

The axis set under test is the campaign selection set
(``config.HAZARD_SELECTION_AXES``), screened on the pool image; the candidate
metrics are the 21 metrics of the hazard image. Analysis blocks:

  A. Selection-axis screen report, and the redundancy of the 21 candidate
     metrics: Spearman matrix, |rho| >= 0.7 clusters, and PCA on normal
     scores (eigenvalues, participation ratio, components for 90% of variance,
     highest-loading metric per leading component). SSI fit check: the
     historical record's SSI under the reference fit, by calendar month
     (standard deviation, count at or below the qualifying level and the
     count a standard normal gives).
  1. Selector comparison at the campaign bounds vs a many-seed random null.
  2. Normalization-bounds sweep.
  3. Sub-pool draw stability (disjoint random halves of the pool).
  B. Per-axis marginal coverage + tail enrichment vs the null.
  C. Selection-set profile: per seed, one lhs_nn selection gives the snap
     statistics (mean target displacement, the share of targets far from every
     pool member, lhs_nn vs lhs_assign order-dependence), the hazard-direction
     tail share on every candidate metric, the minimum own-axis tail share and
     attainment (share over the exact-snap limit), and the n_eff / N of the
     nearest-member weights.
  D. N-sweep: tail enrichment and coverage vs N against the matched random null.
  E. Selection invariance / implicit weighting (leave-one-axis-out,
     add-one-metric-back from the other candidate metrics, per-axis
     snap-distance contributions).
  Truncation summary: the pool, selected-member and drought-metric top-decile
  fractions of controlling events truncated at the window's onset or end.

Descriptor settings (tail direction, cluster level, variance share, far-target
distance) are the ``SELDIAG_*`` entries of ``supplemental_config.py``.

Configuration via environment variables (no CLI value flags). Defaults are
the dev scale; the campaign values (P=1e6 pools, N=300) come from the env.

    NYCOPT_SELDIAG_POOL_SLUG          staged pool slug (default statpool_10yr_n4000_d0)
    NYCOPT_SELDIAG_N                  ensemble size N (default 100)
    NYCOPT_SELDIAG_SEEDS              selector seeds (default 10)
    NYCOPT_SELDIAG_NULL_SEEDS         random-null seeds (default 50)
    NYCOPT_SELDIAG_PREFIX_P           truncate the image to its first P' rows
                                      (0 = full); a prefix is an exact i.i.d.
                                      pool of its size (global-index child
                                      streams); outputs go to {pool_slug}_prefix{P'}
    NYCOPT_SELDIAG_SATURATION         1 = lean mode: axis screen, per-axis
                                      coverage and snap block only (lhs_nn);
                                      no figures
    NYCOPT_SELDIAG_N_SWEEP            N ladder for block D (default "100 150 200 300")
    NYCOPT_SELDIAG_SATURATION_NSWEEP  1 = also run block D inside saturation mode

Run after staging the pool hazard image (workflow step 02 with
``NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary``; locally use
``NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1`` so only ``hazard_image.npz`` is kept)::

    python scripts/supplemental/diagnose_hazard_selectors.py

Outputs -> ``outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import config  # noqa: E402
import supplemental_config as scfg  # noqa: E402
from scengen import selector_diagnostics as sd  # noqa: E402
from scengen import subsample as ss  # noqa: E402
from scengen.diagnostics import load_hazard_image  # noqa: E402
from scengen.hazard_filling import screen_hazard_axes  # noqa: E402
from src.plotting.style import apply_style, save_figure  # noqa: E402

POOL_SLUG = os.environ.get("NYCOPT_SELDIAG_POOL_SLUG", "statpool_10yr_n4000_d0")
N_SELECT = int(os.environ.get("NYCOPT_SELDIAG_N", "100"))
N_SEEDS = int(os.environ.get("NYCOPT_SELDIAG_SEEDS", "10"))
N_NULL_SEEDS = int(os.environ.get("NYCOPT_SELDIAG_NULL_SEEDS", "50"))

#: Truncate the staged image to its first P' rows (0 = full image). A prefix is
#: an honest i.i.d. pool of its size (global-index child streams), so the
#: nested-P ladder scores every rung exactly as a standalone pool would be.
PREFIX_P = int(os.environ.get("NYCOPT_SELDIAG_PREFIX_P", "0"))

#: Lean saturation mode: axis screen + per-axis coverage + snap/concentration
#: only (no comparator selectors, no bounds/N sweeps, no figures).
SATURATION = os.environ.get("NYCOPT_SELDIAG_SATURATION", "").strip().lower() in (
    "1", "true", "yes", "on",
)

#: Saturation mode additionally runs block D (the N sweep) when set, so the
#: nested-prefix ladder doubles as the joint (N, P) ladder.
SATURATION_NSWEEP = os.environ.get(
    "NYCOPT_SELDIAG_SATURATION_NSWEEP", "").strip().lower() in ("1", "true", "yes", "on")

#: Output slug: prefix rungs get their own directory so rungs don't clobber.
OUT_SLUG = f"{POOL_SLUG}_prefix{PREFIX_P}" if PREFIX_P else POOL_SLUG

#: Designed (non-null) selectors, in presentation order.
DESIGNED = ("lhs_nn", "lhs_assign", "maximin", "eps_cell")

#: (lo_pct, hi_pct) pairs for the normalization sweep. (1, 99) is the campaign
#: default; (0, 100) is the full-range sensitivity.
BOUNDS_SWEEP: tuple[tuple[float, float], ...] = ((0.0, 100.0), (0.5, 99.5), (1.0, 99.0), (2.0, 98.0))

#: Sub-pool halves for the draw-stability block.
N_SUBPOOLS = 2

#: Ensemble sizes for the sizing decision surface (block D). Env-configurable
#: so the ensemble-size diagnostic can run its wider ladder without a second
#: driver.
N_SWEEP: tuple[int, ...] = tuple(
    int(n) for n in os.environ.get("NYCOPT_SELDIAG_N_SWEEP", "100 150 200 300").split()
)

#: Upper pool quantile reported alongside the P90 tail share in block D (the
#: severe-corner supply that grows scarce first as N rises at fixed P).
TAIL_UPPER_PCT: float = 99.0

#: Share of an i.i.d. selection above the pool P90: the reference against which
#: the minimum per-axis tail share of the designed selectors is reported.
TAIL_NULL_SHARE: float = 0.10

#: The one axis set of blocks C, D and E: the campaign selection axes.
_SET_NAME = "campaign"

_COLORS = {
    "random": "0.55", "lhs_nn": "#1f6fb4", "lhs_assign": "#7db3d9",
    "maximin": "#2c8c5a", "eps_cell": "#c1272d",
}
_SET_COLOR = "#7d3f9b"


def _out_dir() -> Path:
    out = config.OUTPUTS_DIR / "supplemental" / "hazard_selector_diagnostics" / OUT_SLUG
    # Saturation mode writes tables only, so don't leave an empty figures/ dir.
    (out / "figures" if not SATURATION else out).mkdir(parents=True, exist_ok=True)
    return out


def _load_pool() -> tuple[dict, dict]:
    """Load the staged pool hazard image (optionally a prefix) and screen the selection axes.

    With ``NYCOPT_SELDIAG_PREFIX_P`` set, only the first P' rows are analyzed;
    the screen and every downstream bound (robust p1/p99, pool P90s) are then
    computed on the prefix alone, so the rung is scored exactly as a standalone
    pool of size P' would be.

    Returns:
        ``(descriptors, screen)``: the 21-metric matrix of the rows
        (:func:`descriptor_image`) and the live screen of
        ``config.HAZARD_SELECTION_AXES`` on it.
    """
    path = config.STAGED_ENSEMBLE_DIR / POOL_SLUG / "hazard_image.npz"
    if not path.exists():
        raise SystemExit(
            f"[seldiag] pool hazard image not staged: {path}. Run workflow step 02 "
            f"(NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary, "
            f"NYCOPT_CANDIDATE_POOL_N=<P>, NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1) first."
        )
    img = load_hazard_image(path)
    H, S = img["H"], img["supplement"]
    if PREFIX_P:
        if PREFIX_P > len(H):
            raise SystemExit(
                f"[seldiag] NYCOPT_SELDIAG_PREFIX_P={PREFIX_P} exceeds the staged "
                f"image size P={len(H)} ({path})."
            )
        H, S = H[:PREFIX_P], S[:PREFIX_P]
    descriptors = descriptor_image(H, list(img["hazard_axes"]), S, img["supplement_names"])
    axes = list(config.HAZARD_SELECTION_AXES)
    missing = [a for a in axes if a not in descriptors["names"]]
    if missing:
        raise SystemExit(f"[seldiag] selection axes absent from the image's metrics: {missing}")
    screen = screen_hazard_axes(_sub(descriptors["D"], descriptors["names"], axes), axes)
    return descriptors, screen


def _reference_ssi() -> pd.Series:
    """SSI of the historical record under the reference fit the pool was scored with.

    The record is transformed by the fitted calculator every scenario goes
    through (``scengen.hazard_metrics.get_reference_fits``), on the flow
    dataset the pool's ``_meta.json`` names.
    """
    from scengen.hazard_metrics import _REFERENCE_START, flows_to_series, get_reference_fits
    from scripts.main.compute_etest_hazard_image import _reference_series

    meta = json.loads((config.STAGED_ENSEMBLE_DIR / POOL_SLUG / "_meta.json").read_text())
    reference_monthly, reference_daily = _reference_series(meta["flowtype"])
    dry_calc, _, _ = get_reference_fits(reference_monthly, reference_daily)
    return dry_calc.transform(flows_to_series(reference_monthly, start_date=_REFERENCE_START))


def _sub(D: np.ndarray, names: list[str], axes: list[str]) -> np.ndarray:
    return D[:, [names.index(a) for a in axes]]


def _seeds(k: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(k)]


###############################################################################
# Pure image helpers live in src/hazard_metric_image.py (re-exported here so the
# descriptor blocks and their tests keep their names)
###############################################################################

from src.hazard_metric_image import (  # noqa: E402,F401
    descriptor_image, descriptor_redundancy, exact_snap_limit, hazard_tail_mask,
    nearest_member_ess_ratio, normal_scores, ssi_fit_check,
)




def _main_comparison(H: np.ndarray, axes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Block 1: all selectors at campaign bounds; wide random null."""
    table_d, details = sd.run_selector_comparison(
        H, axes, N_SELECT, seeds=_seeds(N_SEEDS), selectors=DESIGNED,
    )
    table_r, details_r = sd.run_selector_comparison(
        H, axes, N_SELECT, seeds=_seeds(N_NULL_SEEDS), selectors=("random",),
    )
    details.update(details_r)
    return pd.concat([table_r, table_d], ignore_index=True), details


def _bounds_sweep(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block 2: designed selectors under each normalization-bounds pair."""
    frames = []
    for lo, hi in BOUNDS_SWEEP:
        t, _ = sd.run_selector_comparison(
            H, axes, N_SELECT, seeds=_seeds(N_SEEDS), selectors=DESIGNED,
            lo_pct=lo, hi_pct=hi,
        )
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _subpool_stability(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block 3: block 1 re-run on disjoint random halves of the pool.

    A random partition of an i.i.d. pool yields independent i.i.d. pools, so
    between-half spread is an honest (cheap) stand-in for pool-re-roll variance.
    """
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(H))
    frames = []
    for h, part in enumerate(np.array_split(perm, N_SUBPOOLS)):
        Hh = H[np.sort(part)]
        t, _ = sd.run_selector_comparison(
            Hh, axes, N_SELECT, seeds=_seeds(N_SEEDS),
            selectors=("random",) + DESIGNED, pool_label=f"half{h}",
        )
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _per_axis_coverage(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block B: per-axis marginal coverage + tail enrichment, lhs_nn vs null."""
    records = []
    for selector, seeds in (("lhs_nn", _seeds(N_SEEDS)), ("random", _seeds(N_NULL_SEEDS))):
        for seed in seeds:
            if selector == "lhs_nn":
                rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
            else:
                rows = ss.random_subsample(H, N_SELECT, seed=seed)
            for axis, m in sd.per_axis_selection_metrics(H, rows, axes).items():
                records.append({"selector": selector, "seed": seed, "axis": axis, **m})
    return pd.DataFrame.from_records(records)


def _dimension_sweep(
    D: np.ndarray, names: list[str], axis_sets: dict[str, list[str]],
    *, include_assign: bool = True, descriptors: dict | None = None,
) -> pd.DataFrame:
    """Block C: the selection-set profile (snap statistics, order-dependence, tails).

    One ``lhs_nn`` selection per (set, seed) (``subsample.lhs_nn_assignment``,
    so the targets are kept). ``include_assign=False`` (saturation mode) skips
    the Hungarian comparator, since its anchor-by-pool cost matrix is
    memory-heavy at large P and contributes nothing to the tail-share record.

    With ``descriptors`` (:func:`descriptor_image`), each record also carries
    the hazard-direction tail share of the selection on every candidate
    metric (``tail__{name}``), the minimum own-axis tail share and
    attainment (share over :func:`exact_snap_limit`), the share of targets
    farther than ``SELDIAG_FAR_TARGET_DISTANCE`` from every pool member, the
    ``n_eff / N`` of the nearest-member weights, and the selected members'
    truncation-flag fractions. ``snap_mean`` is the mean target displacement.
    """
    tail_mask = limits = None
    if descriptors is not None:
        tail_mask = hazard_tail_mask(descriptors["D"], descriptors["sign"],
                                     pct=scfg.SELDIAG_TAIL_PCT)
        limits = dict(zip(names,
                          exact_snap_limit(D, tail_pct=scfg.SELDIAG_TAIL_PCT)))
    records = []
    for mset, axes in axis_sets.items():
        H = _sub(D, names, axes)
        X = ss.minmax_normalize(H)
        tree = cKDTree(X) if descriptors is not None else None
        for seed in _seeds(N_SEEDS):
            a = ss.lhs_nn_assignment(X, N_SELECT, seed=seed)
            rows = np.sort(a.rows)
            conc = sd.distance_concentration(X, a.displacement, seed=seed)
            lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
            cov = ss.coverage_metrics(X[rows], lb, ub)
            jac = np.nan
            if include_assign:
                res_as = sd.select_lhs_assign(X, N_SELECT, seed=seed)
                jac = sd.jaccard(rows, res_as.rows)
            rec = {
                "m_set": mset, "m": len(axes), "seed": seed,
                "snap_mean": float(np.mean(a.displacement)),
                "snap_p95": float(np.percentile(a.displacement, 95)),
                "nn_min_abs": float(cov.get("nn_min", 0.0)),
                "L2_star_abs": float(cov["L2_star_discrepancy"]),
                **conc,
                "jaccard_nn_vs_assign": jac,
            }
            if descriptors is not None:
                tails = tail_mask[rows].mean(axis=0)
                own = np.array([tails[descriptors["names"].index(x)] for x in axes])
                lim = np.array([limits[x] for x in axes])
                att = np.divide(own, lim, out=np.full_like(own, np.nan), where=lim > 0)
                d_near, _ = tree.query(a.targets, k=1)
                flags = descriptors["flags"][rows].mean(axis=0)
                rec.update({
                    "tail_share_min": float(own.min()),
                    "attainment_min": float(np.nanmin(att)),
                    "frac_targets_far": float(np.mean(d_near > scfg.SELDIAG_FAR_TARGET_DISTANCE)),
                    "ess_over_n": nearest_member_ess_ratio(X, rows),
                    **{f"selected__{f}": float(v)
                       for f, v in zip(descriptors["flag_names"], flags)},
                    **{f"tail__{n}": float(v) for n, v in zip(descriptors["names"], tails)},
                })
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _selection_set_profile(
    dim: pd.DataFrame, axis_sets: dict[str, list[str]], D: np.ndarray,
    names: list[str], descriptor_names: list[str],
) -> pd.DataFrame:
    """The selection-set profile as one tidy table (seed means and SDs of block C).

    Rows: per (set, descriptor) the hazard-direction ``tail_share``; per (set,
    own axis) the ``attainment`` and its ``exact_snap_limit``; per set the
    within-seed minima ``tail_share_min_own`` and ``attainment_min_own``,
    ``displacement_mean`` (mean target displacement), ``frac_targets_far`` and
    ``ess_over_n``.
    """
    limits = dict(zip(names, exact_snap_limit(D, tail_pct=scfg.SELDIAG_TAIL_PCT)))
    rows = []
    for mset, axes in axis_sets.items():
        g = dim.loc[dim.m_set == mset]
        base = {"m_set": mset, "m": len(axes)}

        def add(statistic: str, values: pd.Series, descriptor: str = "",
                in_set: bool | None = None) -> None:
            rows.append({**base, "statistic": statistic, "descriptor": descriptor,
                         "in_set": in_set, "mean": float(values.mean()),
                         "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan})

        for name in descriptor_names:
            share = g[f"tail__{name}"]
            add("tail_share", share, name, name in axes)
            if name in axes:
                limit = limits[name]
                add("attainment", share / limit if limit > 0 else share * np.nan, name, True)
                add("exact_snap_limit", pd.Series([limit]), name, True)
        for statistic, column in (("tail_share_min_own", "tail_share_min"),
                                  ("attainment_min_own", "attainment_min"),
                                  ("displacement_mean", "snap_mean"),
                                  ("frac_targets_far", "frac_targets_far"),
                                  ("ess_over_n", "ess_over_n")):
            add(statistic, g[column])
    return pd.DataFrame.from_records(rows)


def _truncation_summary(
    D: np.ndarray, names: list[str], descriptors: dict, dim: pd.DataFrame,
) -> pd.DataFrame:
    """Fractions of controlling events truncated at the window onset or end.

    Rows: the pool; the selected members of each axis set (seed mean, block C);
    the pool members above the p90 of each drought metric (its top decile).
    """
    flags, flag_names = descriptors["flags"], descriptors["flag_names"]
    rows = [{"population": "pool", "m_set": "", "axis": "", "n": int(len(flags)),
             **dict(zip(flag_names, flags.mean(axis=0).tolist()))}]
    for mset in dict.fromkeys(dim["m_set"]):
        g = dim.loc[dim.m_set == mset]
        rows.append({"population": "selected", "m_set": mset, "axis": "", "n": N_SELECT,
                     **{f: float(g[f"selected__{f}"].mean()) for f in flag_names}})
    for k, axis in enumerate(names):
        if not axis.startswith("drought_"):
            continue
        top = D[:, k] > np.percentile(D[:, k], scfg.SELDIAG_TAIL_PCT)
        rows.append({"population": "top_decile", "m_set": "", "axis": axis,
                     "n": int(top.sum()),
                     **dict(zip(flag_names, (flags[top].mean(axis=0) if top.any()
                                             else np.full(len(flag_names), np.nan)).tolist()))})
    return pd.DataFrame.from_records(rows)


def _redundancy_table(red: dict) -> pd.DataFrame:
    """Block-A descriptor redundancy as one long table.

    Statistics: ``spearman_rho`` (every descriptor pair), ``cluster`` (member
    rows, value = cluster size), ``pca_eigenvalue`` / ``pca_explained`` per
    component, ``pca_top_loading`` per leading component, and the scalars
    ``participation_ratio`` and ``n_components_variance_share``.
    """
    names, rho = red["names"], red["rho"]
    rows = [{"statistic": "spearman_rho", "descriptor": a, "other": b, "component": np.nan,
             "value": float(rho[i, j])}
            for i, a in enumerate(names) for j, b in enumerate(names)]
    rows += [{"statistic": "cluster", "descriptor": a, "other": "", "component": c + 1,
              "value": len(members)}
             for c, members in enumerate(red["clusters"]) for a in members]
    for c, (ev, ex) in enumerate(zip(red["eigenvalues"], red["explained"])):
        rows.append({"statistic": "pca_eigenvalue", "descriptor": "", "other": "",
                     "component": c + 1, "value": float(ev)})
        rows.append({"statistic": "pca_explained", "descriptor": "", "other": "",
                     "component": c + 1, "value": float(ex)})
    rows += [{"statistic": "pca_top_loading", "descriptor": d, "other": "",
              "component": c, "value": v} for c, d, v in red["top_loadings"]]
    rows += [
        {"statistic": "participation_ratio", "descriptor": "", "other": "",
         "component": np.nan, "value": red["participation_ratio"]},
        {"statistic": "n_components_variance_share", "descriptor": "", "other": "",
         "component": np.nan, "value": red["n_components"]},
    ]
    return pd.DataFrame.from_records(rows)


def n_sweep_record(
    H: np.ndarray, X: np.ndarray, rows: np.ndarray, axes: list[str], *,
    upper_pct: float = TAIL_UPPER_PCT,
) -> dict:
    """One block-D row: per-axis tail shares, stratification, joint coverage.

    Shared with the ensemble-size diagnostic
    (``scripts/supplemental/ensemble_size_hazard.py``) so both drivers score a
    selection identically.

    Args:
        H: ``(M, d)`` pool sub-image on the selection axes (raw metric values).
        X: The same sub-image normalized once to the campaign unit box
           (``ss.minmax_normalize(H)``), passed in so repeated calls do not
           re-normalize a 1e6-row image.
        rows: Selected row indices.
        axes: Axis names (columns of ``H``).
        upper_pct: The upper pool quantile whose tail share is reported beside
            the P90 share.

    Returns:
        Flat dict: ``tail_share_min/mean`` (P90), ``tail_share_pXX_min/mean``
        (upper quantile), ``ks_mean``, ``L2_star_abs``, ``nn_min_abs``,
        ``mst_edge_mean``, ``mst_edge_min``.
    """
    per_axis = sd.per_axis_selection_metrics(H, rows, axes)
    tails = [m["tail_share_p90"] for m in per_axis.values()]
    kss = [m["ks_to_uniform"] for m in per_axis.values()]
    p_up = np.percentile(H, upper_pct, axis=0)
    tails_up = [float(np.mean(H[rows, k] > p_up[k])) for k in range(H.shape[1])]
    lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
    cov = ss.coverage_metrics(X[rows], lb, ub)
    mst = sd._mst_edge_stats(X[rows])
    tag = f"p{int(upper_pct)}"
    return {
        "tail_share_min": float(np.min(tails)),
        "tail_share_mean": float(np.mean(tails)),
        f"tail_share_{tag}_min": float(np.min(tails_up)),
        f"tail_share_{tag}_mean": float(np.mean(tails_up)),
        "ks_mean": float(np.mean(kss)),
        "L2_star_abs": float(cov["L2_star_discrepancy"]),
        "nn_min_abs": float(cov.get("nn_min", 0.0)),
        "mst_edge_mean": float(mst["mst_edge_mean"]),
        "mst_edge_min": float(mst["mst_edge_min"]),
    }


def _n_sweep(
    D: np.ndarray, names: list[str], axis_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """Block D: the (axis set × N) sizing decision surface, vs matched random nulls."""
    records = []
    for mset, axes in axis_sets.items():
        H = _sub(D, names, axes)
        X = ss.minmax_normalize(H)
        for n in N_SWEEP:
            for selector, seeds in (
                ("lhs_nn", _seeds(N_SEEDS)), ("random", _seeds(N_NULL_SEEDS)),
            ):
                for seed in seeds:
                    if selector == "lhs_nn":
                        rows = ss.absolute_filling_subsample(H, n, seed=seed)
                    else:
                        rows = ss.random_subsample(H, n, seed=seed)
                    records.append({
                        "m_set": mset, "m": len(axes), "n": n,
                        "selector": selector, "seed": seed,
                        **n_sweep_record(H, X, rows, axes),
                    })
    return pd.DataFrame.from_records(records)


def _invariance(
    D: np.ndarray, names: list[str], axis_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict]:
    """Block E: LOO / add-one-metric-back overlap + per-axis snap contributions."""
    retained = axis_sets[_SET_NAME]
    H_ret = _sub(D, names, retained)
    full_rows = {s: ss.absolute_filling_subsample(H_ret, N_SELECT, seed=s)
                 for s in _seeds(N_SEEDS)}

    records = []
    for axis in retained:  # leave-one-axis-out
        axes = [a for a in retained if a != axis]
        H = _sub(D, names, axes)
        for seed in _seeds(N_SEEDS):
            rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
            records.append({
                "variant": "loo", "axis": axis, "m": len(axes), "seed": seed,
                "jaccard_vs_set": sd.jaccard(rows, full_rows[seed]),
            })
    for axis in [m for m in names if m not in retained]:  # add-one-metric-back
        axes = retained + [axis]
        H = _sub(D, names, axes)
        for seed in _seeds(N_SEEDS):
            rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
            records.append({
                "variant": "add_one", "axis": axis, "m": len(axes), "seed": seed,
                "jaccard_vs_set": sd.jaccard(rows, full_rows[seed]),
            })

    # Per-axis contribution to the snap distance (and dry/wet group shares).
    X = ss.minmax_normalize(H_ret)
    shares_per_seed = [
        sd.snap_axis_contributions(X, N_SELECT, retained, seed=s)
        for s in _seeds(N_SEEDS)
    ]
    mean_shares = {a: float(np.mean([sh[a] for sh in shares_per_seed])) for a in retained}
    dry = sum(v for a, v in mean_shares.items() if a.startswith("drought"))
    contributions = {
        "per_axis": mean_shares,
        "dry_group": float(dry),
        "wet_group": float(1.0 - dry),
    }
    return pd.DataFrame.from_records(records), contributions


###############################################################################
# Figures
###############################################################################

def _fig_selection_scatter(H, axes, details, out) -> None:
    """F1: where each rule lands on the (dry, wet) magnitude plane (seed 0)."""
    dry = next((i for i, a in enumerate(axes) if a.startswith("drought")), 0)
    wet = next((i for i, a in enumerate(axes) if a.startswith("flood")), min(1, len(axes) - 1))
    names = ("random",) + DESIGNED
    fig, ax = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.3), sharex=True, sharey=True)
    for c, name in enumerate(names):
        a = ax[c]
        rows = details[name]["rows"][0]
        a.scatter(H[:, dry], H[:, wet], s=6, c="0.85", edgecolors="none", rasterized=True)
        a.scatter(H[rows, dry], H[rows, wet], s=22, c=_COLORS[name],
                  edgecolors="white", linewidths=0.3)
        a.set_title(name)
        a.set_xlabel(axes[dry])
        if c == 0:
            a.set_ylabel(axes[wet])
    fig.suptitle(f"Selected members by rule (N={N_SELECT}, pool P={len(H)}, seed 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out / "figures" / "F1_selection_scatter")
    plt.close(fig)


def _fig_coverage_vs_null(table, out) -> None:
    """F2: L2-star per rule (both geometries) against the random-null band."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for p, metric in enumerate(("L2_star_abs", "L2_star_cdf")):
        a = ax[p]
        null = table.loc[table.selector == "random", metric]
        a.axhspan(null.mean() - 2 * null.std(), null.mean() + 2 * null.std(),
                  color="0.9", zorder=0, label="random null (±2σ)")
        a.axhline(null.mean(), color="0.6", lw=1, zorder=1)
        for i, name in enumerate(DESIGNED):
            vals = table.loc[table.selector == name, metric]
            a.scatter(np.full(len(vals), i), vals, s=18, c=_COLORS[name], zorder=3)
            a.scatter([i], [vals.mean()], marker="_", s=500, c="black", zorder=4)
        a.set_xticks(range(len(DESIGNED)))
        a.set_xticklabels(DESIGNED, rotation=20)
        a.set_ylabel("L2-star discrepancy")
        a.set_title({"L2_star_abs": "absolute (campaign) geometry",
                     "L2_star_cdf": "rank geometry"}[metric])
        if p == 0:
            a.legend(loc="upper right")
    fig.suptitle("Coverage vs the random null (lower = more uniform)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out / "figures" / "F2_coverage_vs_null")
    plt.close(fig)


def _fig_tail_and_atom(table, out) -> None:
    """F3: tail enrichment + zero-event atom per rule (seed spread)."""
    metrics = [
        ("tail_share_p90", "mean share above pool P90\n(unbiased ≈ 0.10)", 0.10),
        ("corner_share_p90", "share in any-axis P90 corner", None),
        ("zero_event_share_selected", "zero-drought-event share\n(pool line = atom mass)", None),
    ]
    pool_atom = table["zero_event_share_pool"].iloc[0] if "zero_event_share_pool" in table else None
    names = ("random",) + DESIGNED
    fig, ax = plt.subplots(1, len(metrics), figsize=(11.4, 3.6))
    for p, (metric, label, ref) in enumerate(metrics):
        a = ax[p]
        for i, name in enumerate(names):
            vals = table.loc[table.selector == name, metric]
            a.scatter(np.full(len(vals), i), vals, s=16, c=_COLORS[name])
            a.scatter([i], [vals.mean()], marker="_", s=450, c="black")
        if metric == "zero_event_share_selected" and pool_atom is not None:
            ref = pool_atom
        if ref is not None:
            a.axhline(ref, color="0.5", lw=1, ls="--")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=25)
        a.set_ylabel(label)
    fig.suptitle("Tail enrichment and the dry zero-event atom, by selection rule")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F3_tail_and_atom")
    plt.close(fig)


def _fig_snap_and_separation(table, details, out) -> None:
    """F4: anchor snap distances (LHS rules) + realized minimum separation."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a = ax[0]
    for name in ("lhs_nn", "lhs_assign"):
        snaps = np.concatenate([info["snap_distances"] for info in details[name]["info"]])
        a.hist(snaps, bins=40, histtype="step", lw=1.8, density=True,
               color=_COLORS[name], label=f"{name} (mean {snaps.mean():.3f})")
    a.set_xlabel("anchor→selected distance (unit box)")
    a.set_ylabel("density")
    a.set_title("Snap distances: off-manifold anchor cost")
    a.legend()
    a2 = ax[1]
    names = ("random",) + DESIGNED
    for i, name in enumerate(names):
        vals = table.loc[table.selector == name, "nn_min_abs"]
        a2.scatter(np.full(len(vals), i), vals, s=16, c=_COLORS[name])
        a2.scatter([i], [vals.mean()], marker="_", s=450, c="black")
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, rotation=25)
    a2.set_ylabel("min pairwise separation (abs geometry)")
    a2.set_title("Near-duplicate guard: minimum separation")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F4_snap_and_separation")
    plt.close(fig)


def _fig_bounds_sweep(sweep, out) -> None:
    """F5: how the normalization bounds move tail enrichment and coverage."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = [f"({lo:g},{hi:g})" for lo, hi in BOUNDS_SWEEP]
    for p, (metric, label) in enumerate((
        ("tail_share_p90", "mean share above pool P90"),
        ("L2_star_abs", "L2-star, absolute geometry"),
    )):
        a = ax[p]
        for name in DESIGNED:
            means = [
                sweep.loc[(sweep.selector == name) & (sweep.hi_pct == hi), metric].mean()
                for _, hi in BOUNDS_SWEEP
            ]
            a.plot(x, means, "o-", color=_COLORS[name], label=name)
        a.set_xlabel("(lo_pct, hi_pct) normalization bounds")
        a.set_ylabel(label)
        if p == 0:
            a.axhline(0.10, color="0.5", lw=1, ls="--")
            a.legend()
    fig.suptitle("Normalization-bounds sweep (campaign default = (1, 99))")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F5_bounds_sweep")
    plt.close(fig)


def _fig_axis_screen(red: dict, screen: dict, selection_axes: list[str], out: Path) -> None:
    """F6: metric redundancy — |rho| heatmap, cluster tree, normal-score PCA spectrum.

    Covers the 21 candidate metrics (selection axes in black, the others in
    gray), ordered by the cluster tree; the tree carries the axis screen's
    near-duplicate cut and the cluster cut.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    names, rho = red["names"], np.asarray(red["rho"])
    d = 1.0 - np.abs(rho)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(np.clip((d + d.T) / 2.0, 0.0, None), checks=False),
                method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    fig, (a, a2, a3) = plt.subplots(1, 3, figsize=(16.5, 5.6),
                                    gridspec_kw={"width_ratios": [1.2, 1.1, 0.75]})
    im = a.imshow(np.abs(rho)[np.ix_(order, order)], vmin=0, vmax=1, cmap="magma_r")
    labels = [names[i] for i in order]
    a.set_xticks(range(len(labels)))
    a.set_yticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=90, fontsize=6)
    a.set_yticklabels(labels, fontsize=6)
    for tick in a.get_xticklabels() + a.get_yticklabels():
        tick.set_color("black" if tick.get_text() in selection_axes else "0.45")
    a.set_title("|Spearman ρ| (selection axes black, other metrics gray)", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)

    dendrogram(Z, labels=names, ax=a2, color_threshold=0.0,
               above_threshold_color="0.3", leaf_rotation=90, leaf_font_size=6)
    for tick in a2.get_xticklabels():
        tick.set_color("black" if tick.get_text() in selection_axes else "0.45")
    dup = 1.0 - screen["dedupe_threshold"]
    cut = 1.0 - scfg.SELDIAG_CLUSTER_RHO
    a2.axhline(dup, color="#c1272d", lw=1.2, ls="--",
               label=f"near-duplicate cut (1−|ρ| = {dup:g})")
    a2.axhline(cut, color="0.2", lw=1.0, ls=":",
               label=f"cluster cut (1−|ρ| = {cut:g}; {len(red['clusters'])} clusters)")
    a2.set_ylabel("1 − |ρ_S| (average linkage)")
    a2.set_title("Cluster tree (diagnostic only)", fontsize=9)
    a2.legend(loc="upper left", fontsize=7)

    k = np.arange(1, len(red["explained"]) + 1)
    a3.bar(k, red["explained"], color="#1f6fb4", width=0.7, label="component share")
    a3.plot(k, np.cumsum(red["explained"]), "o-", color="0.2", ms=3, lw=1.2,
            label="cumulative share")
    a3.axhline(scfg.SELDIAG_PCA_VARIANCE_SHARE, color="0.5", lw=1, ls="--")
    a3.axvline(red["n_components"], color="0.5", lw=1, ls=":")
    a3.set_xlabel("principal component (normal scores)")
    a3.set_ylabel("share of variance")
    a3.set_ylim(0, 1.02)
    a3.set_title(f"PCA: {red['n_components']} components reach "
                 f"{scfg.SELDIAG_PCA_VARIANCE_SHARE:.0%}; participation ratio "
                 f"{red['participation_ratio']:.1f}", fontsize=9)
    a3.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F6_axis_screen")
    plt.close(fig)


def _fig_selection_set_profile(table: pd.DataFrame, dim: pd.DataFrame,
                               axis_sets: dict[str, list[str]], out: Path) -> None:
    """F11: the selection-set profile — tail share on every metric and the set scalars.

    Left: seed-mean hazard-direction tail share of the selection on every
    candidate metric (own axes in the set colour; the i.i.d. share is 0.10).
    Right: one small panel per set scalar, seeds as dots and the mean as a bar.
    """
    axes = axis_sets[_SET_NAME]
    ts = table.loc[(table.statistic == "tail_share") & (table.m_set == _SET_NAME)]
    names = list(dict.fromkeys(ts["descriptor"]))
    scalars = (("tail_share_min", "min own-axis tail share"),
               ("attainment_min", "min own-axis attainment"),
               ("snap_mean", "mean target displacement"),
               ("frac_targets_far", f"targets > {scfg.SELDIAG_FAR_TARGET_DISTANCE:g} "
                                    "from the pool"),
               ("ess_over_n", "n_eff / N (nearest-member weights)"))
    fig = plt.figure(figsize=(15.0, 4.6))
    gs = fig.add_gridspec(1, 2 + len(scalars), width_ratios=[6.0, 0.4] + [1.0] * len(scalars),
                          wspace=0.6, left=0.05, right=0.98, top=0.86, bottom=0.3)
    a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    colors = [_SET_COLOR if n in axes else "0.6" for n in names]
    a.bar(x, ts["mean"].to_numpy(), yerr=2 * ts["sd"].to_numpy(), color=colors, width=0.7,
          capsize=2)
    a.axhline(TAIL_NULL_SHARE, color="0.4", lw=1, ls="--", label="i.i.d. share")
    a.set_xticks(x)
    a.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    a.set_ylabel("share in the hazard-direction pool tail")
    a.set_title(f"Selection set (m={len(axes)}): tail share on every candidate metric "
                f"(N={N_SELECT}, {N_SEEDS} seeds; own axes coloured; whisker = 2 seed SD)",
                fontsize=9)
    a.legend(fontsize=7)
    g = dim.loc[dim.m_set == _SET_NAME]
    for p, (column, label) in enumerate(scalars):
        b = fig.add_subplot(gs[0, 2 + p])
        vals = g[column]
        b.scatter(np.zeros(len(vals)), vals, s=12, c=_SET_COLOR)
        b.scatter([0], [vals.mean()], marker="_", s=300, c="black")
        if column == "tail_share_min":
            b.axhline(TAIL_NULL_SHARE, color="0.5", lw=1, ls="--")
        b.set_xticks([])
        b.set_title(label, fontsize=7.5)
    save_figure(fig, out / "figures" / "F11_selection_set_profile")
    plt.close(fig)


def _fig_per_axis_coverage(per_axis, axes, out) -> None:
    """F7: per-axis KS-to-uniform + tail share, lhs_nn seeds vs the null band."""
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
    xs = np.arange(len(axes))
    for p, (metric, label, ref) in enumerate((
        ("ks_to_uniform", "KS distance to uniform (scaled coords)", None),
        ("tail_share_p90", "share above pool P90", TAIL_NULL_SHARE),
    )):
        a = ax[p]
        for i, axis in enumerate(axes):
            null = per_axis.loc[(per_axis.selector == "random") & (per_axis.axis == axis), metric]
            a.errorbar([i - 0.12], [null.mean()], yerr=[2 * null.std()], fmt="o",
                       color=_COLORS["random"], ms=4, capsize=3,
                       label="random null (±2σ)" if i == 0 else None)
            vals = per_axis.loc[(per_axis.selector == "lhs_nn") & (per_axis.axis == axis), metric]
            a.scatter(np.full(len(vals), i + 0.12), vals, s=14, c=_COLORS["lhs_nn"],
                      label="lhs_nn (seeds)" if i == 0 else None)
            a.scatter([i + 0.12], [vals.mean()], marker="_", s=300, c="black")
        if ref is not None:
            a.axhline(ref, color="0.5", lw=1, ls="--",
                      label=f"i.i.d. reference ({ref:g})")
        a.set_xticks(xs)
        a.set_xticklabels(axes, rotation=40, ha="right", fontsize=7)
        a.set_ylabel(label)
        a.legend(fontsize=7)
    fig.suptitle(f"Per-axis marginal coverage + tail enrichment on the selection axes "
                 f"(N={N_SELECT})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out / "figures" / "F7_per_axis_coverage")
    plt.close(fig)


def _fig_n_sweep(nsw, out) -> None:
    """F9: the sizing decision surface — worst-axis tail enrichment + coverage vs N."""
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.6))
    msets = list(dict.fromkeys(nsw["m_set"]))
    for p, (metric, label) in enumerate((
        ("tail_share_min", "min per-axis tail share"),
        ("ks_mean", "mean per-axis KS to uniform"),
        ("L2_star_abs", "joint L2-star (abs geometry)"),
    )):
        a = ax[p]
        for mset in msets:
            sel = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "lhs_nn")]
            means = sel.groupby("n")[metric].mean()
            a.plot(means.index, means.values, "o-", color=_SET_COLOR,
                   label=f"lhs_nn {mset}")
            null = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "random")]
            nmeans = null.groupby("n")[metric].mean()
            a.plot(nmeans.index, nmeans.values, "--", lw=1, color=_SET_COLOR,
                   alpha=0.55, label=f"null {mset}")
        if metric == "tail_share_min":
            a.axhline(TAIL_NULL_SHARE, color="0.5", lw=1, ls="--",
                      label=f"i.i.d. reference ({TAIL_NULL_SHARE:g})")
        a.set_xlabel("ensemble size N")
        a.set_ylabel(label)
        if p == 0:
            a.legend(fontsize=6.5)
    fig.suptitle("Tail enrichment and coverage vs N on the selection axes "
                 "(dashed = matched random null)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F9_n_sweep")
    plt.close(fig)


def _fig_invariance(inv, contributions, out) -> None:
    """F10: selection invariance (LOO / add-one-back) + snap-distance weighting."""
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
    a = ax[0]
    loo = inv.loc[inv.variant == "loo"].groupby("axis")["jaccard_vs_set"]
    order = list(loo.mean().sort_values().index)
    a.barh(range(len(order)), [loo.mean()[x] for x in order],
           xerr=[2 * loo.std()[x] for x in order], color="#1f6fb4", height=0.55,
           label="leave-one-axis-out")
    add = inv.loc[inv.variant == "add_one"].groupby("axis")["jaccard_vs_set"].mean()
    for i, axis in enumerate(order):
        if axis in add.index:
            a.plot([add[axis]], [i], "d", color="#c1272d", ms=6,
                   label="add-one-metric-back" if i == min(
                       j for j, x in enumerate(order) if x in add.index) else None)
    a.set_yticks(range(len(order)))
    a.set_yticklabels(order, fontsize=7)
    a.set_xlabel("Jaccard overlap with the selection-set selection")
    a.set_xlim(0, 1)
    a.legend(fontsize=7)
    a.set_title("Selection invariance to single axes")

    a2 = ax[1]
    per = contributions["per_axis"]
    names = list(per)
    colors = ["#8c5a2c" if n.startswith("drought") else "#1f6fb4" for n in names]
    a2.bar(range(len(names)), [per[n] for n in names], color=colors)
    a2.axhline(1.0 / len(names), color="0.5", lw=1, ls="--", label="equal weighting")
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    a2.set_ylabel("mean share of squared snap distance")
    a2.set_title(f"Implicit axis weighting (dry {contributions['dry_group']:.2f} / "
                 f"wet {contributions['wet_group']:.2f})")
    a2.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F10_invariance")
    plt.close(fig)


###############################################################################
# Driver
###############################################################################

def _run_saturation(
    out: Path, D: np.ndarray, names: list[str], screen: dict,
    axis_sets: dict[str, list[str]],
) -> None:
    """Lean saturation mode: only the metrics the nested-P tail-share record needs.

    On the selection set: per-axis marginal coverage + tail enrichment
    (lhs_nn seeds vs the random null) and the snap/concentration block (lhs_nn
    only). The tail statistic follows the block-D convention: within-seed
    minimum per-axis tail share, averaged over selector seeds, reported against
    the 0.10 share of an i.i.d. selection with no threshold applied.
    """
    frames = []
    for mset, axes in axis_sets.items():
        t = _per_axis_coverage(_sub(D, names, axes), axes)
        t.insert(0, "m_set", mset)
        frames.append(t)
    per_axis = pd.concat(frames, ignore_index=True)
    dim = _dimension_sweep(D, names, axis_sets, include_assign=False)

    per_axis.to_csv(out / "per_axis_coverage.csv", index=False)
    dim.to_csv(out / "dimension_sweep.csv", index=False)
    if SATURATION_NSWEEP:
        _n_sweep(D, names, axis_sets).to_csv(
            out / "n_sweep.csv", index=False)

    adequacy = {}
    for mset, axes in axis_sets.items():
        sel = per_axis.loc[(per_axis.m_set == mset) & (per_axis.selector == "lhs_nn")]
        by_seed = sel.groupby("seed")["tail_share_p90"]
        axis_means = sel.groupby("axis")["tail_share_p90"].mean()
        conc = dim.loc[dim.m_set == mset, "concentration_ratio"]
        adequacy[mset] = {
            "m": len(axes),
            "tail_share_min": float(by_seed.min().mean()),
            "tail_share_mean": float(by_seed.mean().mean()),
            "worst_axis": str(axis_means.idxmin()),
            "worst_axis_seed_mean": float(axis_means.min()),
            "per_axis_seed_mean": {str(a): float(v) for a, v in axis_means.items()},
            "concentration_ratio": float(conc.mean()),
            "snap_mean": float(dim.loc[dim.m_set == mset, "snap_mean"].mean()),
        }

    summary = {
        "pool_slug": POOL_SLUG, "prefix_p": PREFIX_P or None, "out_slug": OUT_SLUG,
        "P": int(len(D)), "n_select": N_SELECT,
        "seeds": N_SEEDS, "null_seeds": N_NULL_SEEDS,
        "saturation_mode": True,
        "saturation_nsweep": SATURATION_NSWEEP,
        "n_sweep": list(N_SWEEP) if SATURATION_NSWEEP else None,
        "axis_screen": {k: v for k, v in screen.items() if k != "spread"},
        "selection_axes": list(axis_sets[_SET_NAME]),
        "tail_null_share": TAIL_NULL_SHARE,
        "adequacy": adequacy,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[seldiag] saturation mode: wrote {3 if SATURATION_NSWEEP else 2} tables "
          f"+ summary.json -> {out}")


def main() -> None:
    """Run all analysis blocks and write tables, summary, and SI figures."""
    apply_style()
    out = _out_dir()
    descriptors, screen = _load_pool()
    D, names = descriptors["D"], descriptors["names"]
    retained = screen["retained"]
    H_ret = _sub(D, names, retained)
    axis_sets = {_SET_NAME: list(retained)}
    print(f"[seldiag] pool '{POOL_SLUG}': P={len(D)}"
          + (f" (prefix of first {PREFIX_P} rows)" if PREFIX_P else "")
          + f", selection axes (m={len(retained)})={retained}, "
          f"dropped={list(screen['dropped'])}, "
          f"N={N_SELECT}, seeds={N_SEEDS} (+{N_NULL_SEEDS} null)"
          + (", saturation mode" if SATURATION else ""))

    if SATURATION:
        _run_saturation(out, D, names, screen, axis_sets)
        return

    redundancy = descriptor_redundancy(
        descriptors["D"], descriptors["names"], threshold=scfg.SELDIAG_CLUSTER_RHO,
        variance_share=scfg.SELDIAG_PCA_VARIANCE_SHARE,
    )
    fit_check = ssi_fit_check(_reference_ssi(), level=scfg.SELDIAG_SSI_QUALIFYING_LEVEL)
    table, details = _main_comparison(H_ret, retained)
    sweep = _bounds_sweep(H_ret, retained)
    halves = _subpool_stability(H_ret, retained)
    per_axis = _per_axis_coverage(H_ret, retained)
    dim = _dimension_sweep(D, names, axis_sets, descriptors=descriptors)
    comparison = _selection_set_profile(dim, axis_sets, D, names,
                                        descriptors["names"])
    truncation = _truncation_summary(D, names, descriptors, dim)
    nsw = _n_sweep(D, names, axis_sets)
    inv, contributions = _invariance(D, names, axis_sets)

    table.to_csv(out / "selector_comparison.csv", index=False)
    sweep.to_csv(out / "normalization_sweep.csv", index=False)
    halves.to_csv(out / "subpool_stability.csv", index=False)
    per_axis.to_csv(out / "per_axis_coverage.csv", index=False)
    dim.to_csv(out / "dimension_sweep.csv", index=False)
    nsw.to_csv(out / "n_sweep.csv", index=False)
    inv.to_csv(out / "selection_invariance.csv", index=False)
    _redundancy_table(redundancy).to_csv(out / "descriptor_redundancy.csv", index=False)
    fit_check.to_csv(out / "ssi_fit_check.csv", index=False)
    comparison.to_csv(out / "selection_set_profile.csv", index=False)
    truncation.to_csv(out / "truncation_summary.csv", index=False)

    # Tail-share record: min per-axis tail share (seed mean) at each N per axis set.
    adequacy = {}
    for mset in axis_sets:
        sel = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "lhs_nn")]
        means = sel.groupby("n")["tail_share_min"].mean()
        adequacy[mset] = {
            "min_tail_share_by_n": {int(n): float(means[n]) for n in means.index},
        }

    summary = {
        "pool_slug": POOL_SLUG, "prefix_p": PREFIX_P or None, "out_slug": OUT_SLUG,
        "P": int(len(D)), "n_select": N_SELECT,
        "seeds": N_SEEDS, "null_seeds": N_NULL_SEEDS,
        "axis_screen": {k: v for k, v in screen.items() if k != "spread"},
        "selection_axes": list(retained),
        "tail_null_share": TAIL_NULL_SHARE,
        "adequacy": adequacy,
        "snap_contributions": contributions,
        "zero_event_share_pool": float(table["zero_event_share_pool"].iloc[0])
        if "zero_event_share_pool" in table else None,
        "jaccard_across_seeds": {
            name: details[name]["jaccard_across_seeds"] for name in details
        },
        "eps_cell": details["eps_cell"]["info"][0] if "eps_cell" in details else None,
        "selector_means_at_campaign_bounds": {
            name: table.loc[table.selector == name]
            .drop(columns=["pool", "selector"]).mean(numeric_only=True).to_dict()
            for name in ("random",) + DESIGNED
        },
        "descriptor_redundancy": {
            "descriptors": redundancy["names"],
            "constant": redundancy["constant"],
            "clusters": redundancy["clusters"],
            "eigenvalues": [float(v) for v in redundancy["eigenvalues"]],
            "participation_ratio": redundancy["participation_ratio"],
            "n_components_variance_share": redundancy["n_components"],
            "top_loadings": redundancy["top_loadings"],
        },
        "ssi_fit_check": {
            "level": scfg.SELDIAG_SSI_QUALIFYING_LEVEL,
            **{f"{column}_range": [float(fit_check[column].min()),
                                   float(fit_check[column].max())]
               for column in ("sd", "n_at_or_below", "n_expected")},
        },
        "selection_set_profile": {
            mset: {
                row.statistic: row.mean for row in comparison.loc[
                    (comparison.m_set == mset) & (comparison.descriptor == "")
                ].itertuples()
            }
            for mset in axis_sets
        },
        "truncation_pool": {f: float(v) for f, v in zip(
            descriptors["flag_names"], descriptors["flags"].mean(axis=0))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _fig_selection_scatter(H_ret, retained, details, out)
    _fig_coverage_vs_null(table, out)
    _fig_tail_and_atom(table, out)
    _fig_snap_and_separation(table, details, out)
    _fig_bounds_sweep(sweep, out)
    _fig_axis_screen(redundancy, screen, retained, out)
    _fig_per_axis_coverage(per_axis, retained, out)
    _fig_n_sweep(nsw, out)
    _fig_invariance(inv, contributions, out)
    _fig_selection_set_profile(comparison, dim, axis_sets, out)

    print(f"[seldiag] wrote 11 tables + summary.json + 10 figures -> {out}")


if __name__ == "__main__":
    main()
