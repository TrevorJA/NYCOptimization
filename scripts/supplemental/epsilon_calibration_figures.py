"""epsilon_calibration_figures.py - Tables + figures for the epsilon-calibration
experiment.

Pure post-processing of the per-unit annual-metric cubes written by
``epsilon_calibration_run.py`` (one per scenario design); it never re-runs
simulations, so all analysis can be regenerated freely. For every design cube
found at the current sample settings it derives, per annual-unit objective:

  1. **Signal scale** — IQR of the natural-unit search scalar across the
     feasible random policies; ``eps_signal = IQR / 10`` (Reed et al. 2013:
     archive resolution matches the signal scale).
  2. **Noise floor** — bootstrap SD of the unit-operator estimator under
     resampling of the realization axis (unit-years for the single-trace
     historic design, an i.i.d. approximation disclosed in the meta); an
     epsilon below this resolves sampling noise, not policy differences
     (Kasprzyk et al. 2013).
  3. **Granularity floor** — 1 / (N x units) for the failure-frequency
     objectives (their scalar is a fraction with that exact step).
  4. **Recommendation** — ``eps_rec = ceil_to_clean_step(max(1, 2, 3))`` in
     native units, plus a plain-language interpretation.
  5. **Archive-size sweep** — ε-box nondominated archive size of the evaluated
     policies for the ADOPTED CAMPAIGN vector scaled by ``EPS_SCALE_GRID`` and
     for the previous (pre-calibration, provisional) registry vector recorded
     in the cube. Random policies under-fill a converged front, so the SWEEP
     TREND is the signal, not the absolute sizes — disclosed in the table.

The final campaign vector is the clean-rounded per-objective maximum of the
raw requirement across the CAMPAIGN designs (``EPS_CAMPAIGN_DESIGNS``; the
Borg problem/JARs carry one epsilon set for every design). The historic
single-trace reference arm is analyzed and reported but excluded from the max
— its small-NL noise floor would otherwise coarsen the
shared vector ~3-4x beyond what the ensemble search measures need.

Outputs (all under ``outputs/supplemental/epsilon_calibration/``):
  tables/  : epsilon_diagnostics_{design}, archive_sweep_{design},
             epsilon_recommendation (combined)  [CSV]
  figures/ : eps_calibration_ladder (F1, combined),
             archive_size_vs_scale (F2, combined),
             scalar_distributions_{design} (F3, per design),
             parallel_axes_{design} (F4, per design: the evaluated policies
             on the active objectives with the adopted-epsilon archive
             highlighted)  [PNG]

Figure conventions (manuscript SI): Okabe-Ito colors keyed to the DESIGN
(never to plot order) — fixed_probabilistic #0072B2, hazard_filling_stationary
#D55E00, historic reference #B0B0B0 — with the adopted campaign epsilon in
#009E73 and the previous provisional epsilon in black; the palette was
validated for normal-vision and protan/deutan separation (OKLab), and every
color distinction is doubled by marker shape.

Configuration and paths come from ``supplemental_config.py`` — no CLI flags.

Usage:
    python scripts/supplemental/epsilon_calibration_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402  (env-then-config contract)

scfg.configure_epsilon_env()  # set experiment env before config is imported

import config  # noqa: E402
from src.objectives_ensemble import (  # noqa: E402
    ENSEMBLE_OBJECTIVES,
    FailureFrequencyOp,
    build_ensemble_objective_set,
)
from src.plotting.parallel_coordinates import (  # noqa: E402
    custom_parallel_coordinates,
)
from src.plotting.style import (  # noqa: E402
    apply_style,
    label_for,
    save_figure,
)
from src.sensitivity_common import (  # noqa: E402
    apply_operator_rows,
    ceil_to_clean_step,
    epsilon_nondominated,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# Labels + colors (Okabe-Ito, keyed to the DESIGN — entity-stable, never
# assigned by plot order; validated for CVD + normal-vision separation)
# ---------------------------------------------------------------------------

#: Native-unit phrase per objective-name pattern, for axis labels and the
#: interpretation column.
_UNIT_PHRASES: list[tuple[str, str]] = [
    ("reliability_annual", "fraction of unit-years"),
    ("_p99_pct", "% of target (P99 unit-year)"),
    ("flood_days_annual_p99", "days/yr (P99 unit-year)"),
    ("flood_days_annual", "days/yr (mean unit-year)"),
    ("storage_min_p01_pct", "% of NYC capacity (P1 annual minimum)"),
]

#: Per-design plotting identity (color + SI display name + reference flag).
_DESIGN_STYLE: dict[str, dict] = {
    "fixed_probabilistic": {
        "color": "#0072B2", "label": "Fixed probabilistic (i.i.d. control)",
        "reference": False},
    "hazard_filling_stationary": {
        "color": "#D55E00", "label": "Hazard-filling (stationary)",
        "reference": False},
    "historic": {
        "color": "#B0B0B0", "label": "Historic trace (reference)",
        "reference": True},
}
#: Fallback colors for designs outside the campaign trio (assigned by sorted
#: name so the mapping is deterministic across runs, not by plot order).
_FALLBACK_COLORS = ["#56B4E9", "#CC79A7", "#E69F00"]

_CAMPAIGN_COLOR = "#009E73"   # adopted campaign epsilon vector
_PREVIOUS_COLOR = "#000000"   # previous provisional registry vector

#: Suffix for ladder row labels of objectives outside the default active set.
_NON_DEFAULT_NOTE = {
    "downstream_flood_days_annual": " (diagnostic)",
    "downstream_flood_days_annual_p99": " (diagnostic)",
}


def _design_style(design: str, fallback_rank: int = 0) -> dict:
    """Entity-stable style for ``design`` (deterministic fallback if unknown)."""
    if design in _DESIGN_STYLE:
        return _DESIGN_STYLE[design]
    return {"color": _FALLBACK_COLORS[fallback_rank % len(_FALLBACK_COLORS)],
            "label": design, "reference": False}


def _unit_phrase(name: str) -> str:
    """Native-unit phrase for an annual-unit objective name."""
    for pattern, phrase in _UNIT_PHRASES:
        if pattern in name:
            return phrase
    return "native units"


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the plain (ε -> 0) Pareto-nondominated rows, minimized."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        if dominates_i.any():
            keep[i] = False
    return keep


# ---------------------------------------------------------------------------
# Per-design analysis (tables; figures are drawn later, once the adopted
# campaign vector is known)
# ---------------------------------------------------------------------------

def _natural_scalars(units: np.ndarray, objs: list) -> np.ndarray:
    """Natural-unit search scalar per (policy, objective) from the unit cube.

    Reproduces ``AnnualUnitObjective.compute`` exactly: pool every
    realization's unit-years, apply the unit operator (operator sentinel
    semantics handle NaN units).

    Args:
        units: Cube of shape ``(n_dv, n_real, n_obj, n_units)``.
        objs: Ordered :class:`AnnualUnitObjective` list matching axis 2.

    Returns:
        Array of shape ``(n_dv, n_obj)``.
    """
    n_dv, n_real, n_obj, n_units = units.shape
    out = np.full((n_dv, n_obj), np.nan)
    for k, obj in enumerate(objs):
        pooled = units[:, :, k, :].reshape(n_dv, n_real * n_units)
        out[:, k] = apply_operator_rows(obj.unit_operator, pooled)
    return out


def _bootstrap_sd(units: np.ndarray, objs: list, policy_mask: np.ndarray,
                  is_ensemble: bool) -> np.ndarray:
    """Bootstrap SD of each policy x objective search scalar.

    Resamples the realization axis with replacement (``EPS_BOOTSTRAP_B``
    draws); the single-trace historic cube (n_real = 1) falls back to
    resampling the unit-year axis — an i.i.d. approximation that understates
    noise under serial dependence (disclosed, per the project convention).
    One shared index draw serves every policy and objective so their noise
    estimates are comparable.

    Args:
        units: Cube of shape ``(n_dv, n_real, n_obj, n_units)``.
        objs: Ordered :class:`AnnualUnitObjective` list.
        policy_mask: Policies to bootstrap (others stay NaN).
        is_ensemble: Whether the realization axis is the sampling axis.

    Returns:
        Array of shape ``(n_dv, n_obj)`` of bootstrap SDs (ddof=1).
    """
    n_dv, n_real, n_obj, n_units = units.shape
    rng = np.random.default_rng(scfg.EPS_BOOTSTRAP_SEED)
    B = scfg.EPS_BOOTSTRAP_B
    axis_n = n_real if is_ensemble else n_units
    idx = rng.integers(0, axis_n, size=(B, axis_n))

    sd = np.full((n_dv, n_obj), np.nan)
    for i in np.flatnonzero(policy_mask):
        for k, obj in enumerate(objs):
            if is_ensemble:
                u = units[i, :, k, :]                       # (n_real, n_units)
                pools = u[idx].reshape(B, axis_n * n_units)
            else:
                u = units[i, 0, k, :]                       # (n_units,)
                pools = u[idx]                              # (B, n_units)
            stats = apply_operator_rows(obj.unit_operator, pools)
            sd[i, k] = float(np.std(stats, ddof=1))
    return sd


def analyze_design(cube_path: Path) -> dict:
    """Per-design epsilon diagnostics: floors, per-design recommendation, QC.

    Writes the per-design diagnostics CSV; figure drawing is deferred to
    :func:`main` because the combined ladder and the archive sweep need the
    ADOPTED campaign vector, which is only known after every design's floors
    are on the table.

    Args:
        cube_path: The design's unit-cube HDF5 from the run script.

    Returns:
        Dict with the design name, diagnostics DataFrame, natural-unit scalar
        matrix + policy masks (consumed by the figure/sweep stages), and QC.
    """
    with h5py.File(cube_path, "r") as f:
        units = f["units"][:]
        sample_ids = f["sample_ids"][:]
        obj_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["objective_names"][:]]
        eps_previous = f["epsilons_current"][:]
        design = str(f.attrs["design"])
        is_ensemble = bool(f.attrs["is_ensemble"])
        acceptance = float(f.attrs["acceptance_rate"])

    objs = [ENSEMBLE_OBJECTIVES[n] for n in obj_names]
    n_dv, n_real, n_obj, n_units = units.shape
    nl = n_real * n_units

    failed = np.isnan(units).all(axis=(1, 2, 3))
    random_ok = (sample_ids >= 0) & ~failed
    valid_ok = ~failed  # baseline included where it succeeded

    natural = _natural_scalars(units, objs)
    natural[failed] = np.nan  # a failed policy never reaches the archive
    sd = _bootstrap_sd(units, objs, valid_ok, is_ensemble)

    rows = []
    for k, (name, obj) in enumerate(zip(obj_names, objs)):
        col = natural[random_ok, k]
        col = col[np.isfinite(col)]
        iqr = (float(np.percentile(col, 75) - np.percentile(col, 25))
               if col.size else float("nan"))
        eps_signal = iqr / 10.0
        noise_med = float(np.nanmedian(sd[random_ok, k]))
        noise_p90 = float(np.nanpercentile(sd[random_ok, k], 90))
        granularity = (1.0 / nl
                       if isinstance(obj.unit_operator, FailureFrequencyOp)
                       else 0.0)
        floor_raw = np.nanmax([eps_signal, noise_med, granularity])
        eps_rec = ceil_to_clean_step(floor_raw)
        binding = (["signal (IQR/10)", "noise", "granularity"]
                   [int(np.nanargmax([eps_signal, noise_med, granularity]))]
                   if np.isfinite(floor_raw) else "degenerate")
        interp = f"{eps_rec:g} {_unit_phrase(name)}"
        if granularity > 0.0 and np.isfinite(eps_rec):
            interp += f" = {eps_rec * nl:.1f} of {nl} unit-years"
        rows.append({
            "objective": name,
            "direction": obj.direction,
            "n_valid": int(col.size),
            "median": float(np.median(col)) if col.size else float("nan"),
            "iqr": iqr,
            "eps_signal_iqr10": eps_signal,
            "eps_noise_median": noise_med,
            "eps_noise_p90": noise_p90,
            "eps_granularity": granularity,
            "eps_floor_raw": float(floor_raw),
            "eps_recommended": eps_rec,
            "eps_previous": float(eps_previous[k]),
            "previous_over_recommended": (float(eps_previous[k] / eps_rec)
                                          if np.isfinite(eps_rec) and eps_rec > 0
                                          else float("nan")),
            "binding_floor": binding,
            "interpretation": interp,
        })
    diag = pd.DataFrame(rows).set_index("objective")

    scfg.EPS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    diag.to_csv(scfg.epsilon_table_path("epsilon_diagnostics", design))

    print(f"[{design}] n_policies={n_dv} (failed={int(failed.sum())}), "
          f"NL={nl}, acceptance={acceptance:.2%}", flush=True)
    return {"design": design, "diag": diag, "nl": nl, "natural": natural,
            "random_ok": random_ok, "valid_ok": valid_ok,
            "obj_names": obj_names, "eps_previous": np.asarray(eps_previous)}


# ---------------------------------------------------------------------------
# Archive-size sweep (adopted campaign vector; runs after the combine stage)
# ---------------------------------------------------------------------------

def _archive_sweep(res: dict, active: list,
                   eps_campaign_active: np.ndarray) -> pd.DataFrame:
    """ε-archive size of one design's policies under candidate epsilon vectors.

    Sweeps the ADOPTED CAMPAIGN vector (x ``EPS_SCALE_GRID``) and evaluates the
    previous provisional registry vector once, over the ACTIVE campaign
    objective subset in Borg sign convention. NOTE: random feasible policies
    under-fill a converged Pareto front, so absolute sizes are a lower-fidelity
    proxy — the epsilon SCALING TREND is the decision signal.
    """
    obj_names = res["obj_names"]
    ka = [obj_names.index(n) for n in active]
    signs = np.array([-1.0 if ENSEMBLE_OBJECTIVES[n].direction == "maximize"
                      else 1.0 for n in active])
    F = res["natural"][:, ka] * signs
    rows_ok = res["valid_ok"] & np.isfinite(F).all(axis=1)
    F = F[rows_ok]

    n_pareto = int(_nondominated_mask(F).sum())

    entries = []
    if np.isfinite(eps_campaign_active).all() and (eps_campaign_active > 0).all():
        for scale in scfg.EPS_SCALE_GRID:
            size = len(epsilon_nondominated(F, eps_campaign_active * scale))
            entries.append({"design": res["design"], "vector": "campaign",
                            "scale": float(scale), "archive_size": size})
    prev = np.asarray(res["eps_previous"], dtype=float)[ka]
    entries.append({"design": res["design"], "vector": "previous", "scale": 1.0,
                    "archive_size": len(epsilon_nondominated(F, prev))})
    sweep = pd.DataFrame(entries)
    sweep["n_policies"] = int(rows_ok.sum())
    sweep["n_pareto_eps0"] = n_pareto
    return sweep


# ---------------------------------------------------------------------------
# Figures (manuscript SI)
# ---------------------------------------------------------------------------

def _row_order(obj_names: list) -> list:
    """Ladder row order: active campaign objectives first, non-default last."""
    active = list(build_ensemble_objective_set(config.ACTIVE_OBJECTIVES).names)
    return ([n for n in active if n in obj_names]
            + [n for n in obj_names if n not in active])


def _fig_eps_ladder(results: list, combined: pd.DataFrame) -> None:
    """F1 (combined): per-objective epsilon ladder across scenario designs.

    One row per objective on a shared log axis (native units span decades but
    read comparably as orders of magnitude). Per design: the bootstrap noise
    floor (filled circle), the signal scale IQR/10 (open square), and the
    frequency-granularity step (vertical tick). The adopted campaign epsilon
    (filled diamond) is by construction at or right of every ensemble-design
    floor; the previous provisional epsilon (x) shows what was replaced. The
    historic reference design is drawn in light gray — it is excluded from
    the campaign max. Numerically-zero floors (saturated estimators, e.g. a
    median bootstrap SD of exactly 0 when most policies sit at a constant
    scalar) cannot render on a log axis and are omitted — the per-design
    diagnostics CSV records them.
    """
    order = _row_order(list(combined.index))
    n = len(order)
    n_active = sum(1 for nm in order if nm not in _NON_DEFAULT_NOTE)
    ys = {nm: n - 1 - i for i, nm in enumerate(order)}
    dodge = {"historic": 0.22, "fixed_probabilistic": 0.0,
             "hazard_filling_stationary": -0.22}
    tiny = 1e-6  # below any attainable floor; masks exact-zero float noise
    plotted: list = []

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * n + 2.3))
    for rank, res in enumerate(sorted(results, key=lambda r: r["design"])):
        d = res["design"]
        st = _design_style(d, rank)
        dg = res["diag"]
        dy = dodge.get(d, 0.0)
        yy = np.array([ys[nm] + dy for nm in order])
        noise = dg.loc[order, "eps_noise_median"].to_numpy(dtype=float)
        signal = dg.loc[order, "eps_signal_iqr10"].to_numpy(dtype=float)
        gran = dg.loc[order, "eps_granularity"].to_numpy(dtype=float)
        m = np.isfinite(noise) & (noise > tiny)
        ax.scatter(noise[m], yy[m], s=30, marker="o", color=st["color"],
                   zorder=3, linewidths=0)
        plotted.extend(noise[m])
        m = np.isfinite(signal) & (signal > tiny)
        ax.scatter(signal[m], yy[m], s=58, marker="s", facecolors="none",
                   edgecolors=st["color"], linewidths=1.5, zorder=3)
        plotted.extend(signal[m])
        m = np.isfinite(gran) & (gran > tiny)
        ax.scatter(gran[m], yy[m], s=90, marker="|", color=st["color"],
                   linewidths=1.5, zorder=2)
        plotted.extend(gran[m])

    y0 = np.array([ys[nm] for nm in order])
    prev = combined.loc[order, "eps_previous"].to_numpy(dtype=float)
    camp = combined.loc[order, "eps_campaign"].to_numpy(dtype=float)
    ax.scatter(prev, y0, s=60, marker="x", color=_PREVIOUS_COLOR,
               linewidths=1.6, zorder=4)
    ax.scatter(camp, y0, s=80, marker="D", color=_CAMPAIGN_COLOR,
               edgecolors="white", linewidths=0.7, zorder=5)
    plotted.extend(prev[np.isfinite(prev) & (prev > tiny)])
    plotted.extend(camp[np.isfinite(camp) & (camp > tiny)])
    ax.set_xlim(min(plotted) * 0.4, max(plotted) * 2.5)

    if n_active < n:  # separator between the campaign set and non-default rows
        ax.axhline(n - n_active - 0.5, color="0.4", lw=0.8, ls=(0, (4, 3)))

    ax.set_xscale("log")
    ax.set_yticks(n - 1 - np.arange(n))
    ax.set_yticklabels([label_for(nm) + _NON_DEFAULT_NOTE.get(nm, "")
                        for nm in order], fontsize=8.5)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlabel("epsilon value")
    ax.set_title("Search-epsilon calibration: per-design floors and the "
                 "adopted campaign vector")
    ax.grid(axis="x", alpha=0.3)

    design_handles = [
        Line2D([], [], marker="s", ls="", markersize=8,
               markerfacecolor=_design_style(d, i)["color"],
               markeredgecolor="none", label=_design_style(d, i)["label"])
        for i, d in enumerate(sorted({r["design"] for r in results}))
    ]
    quantity_handles = [
        Line2D([], [], marker="o", ls="", color="0.25", markersize=6,
               label="noise floor (median bootstrap SD; Kasprzyk et al. 2013)"),
        Line2D([], [], marker="s", ls="", markersize=7, markerfacecolor="none",
               markeredgecolor="0.25",
               label="signal scale (IQR/10; Reed et al. 2013)"),
        Line2D([], [], marker="|", ls="", color="0.25", markersize=9,
               label="frequency granularity (1 / pooled unit-years)"),
        Line2D([], [], marker="D", ls="", color=_CAMPAIGN_COLOR, markersize=7,
               label="adopted campaign epsilon (max over ensemble designs)"),
        Line2D([], [], marker="x", ls="", color=_PREVIOUS_COLOR, markersize=7,
               label="previous provisional epsilon"),
    ]
    ax.legend(handles=design_handles + quantity_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7.6, frameon=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, scfg.epsilon_figure_path("eps_calibration_ladder")
                .with_suffix(""))
    plt.close(fig)


def _fig_scalar_distributions(res: dict, combined: pd.DataFrame) -> None:
    """F3 (per design): search-scalar distribution with epsilon widths.

    A histogram per objective across the design's feasible random policies;
    the horizontal bars above each distribution show the ADOPTED campaign
    epsilon width (green) and the previous provisional width (black) at the
    same scale — the visual check that one epsilon box holds neither the
    whole signal (too coarse) nor pure noise (too fine).
    """
    st = _design_style(res["design"])
    obj_names = _row_order(res["obj_names"])
    n = len(obj_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 2.7 * nrows))
    axes = np.atleast_2d(axes)
    for k, name in enumerate(obj_names):
        ax = axes[k // ncols][k % ncols]
        col = res["natural"][res["random_ok"], res["obj_names"].index(name)]
        col = col[np.isfinite(col)]
        if col.size:
            ax.hist(col, bins=30, color=st["color"], alpha=0.8)
        x0 = float(np.min(col)) if col.size else 0.0
        y0, y1 = ax.get_ylim()
        for offset, value, color, lw in (
                (0.06, float(combined.loc[name, "eps_campaign"]),
                 _CAMPAIGN_COLOR, 3.0),
                (0.14, float(combined.loc[name, "eps_previous"]),
                 _PREVIOUS_COLOR, 1.8)):
            if np.isfinite(value) and value > 0:
                ax.plot([x0, x0 + value], [y1 * (1 - offset)] * 2,
                        color=color, lw=lw, solid_capstyle="butt", zorder=4)
        ax.set_title(label_for(name) + _NON_DEFAULT_NOTE.get(name, ""),
                     fontsize=8)
        ax.set_xlabel(_unit_phrase(name), fontsize=7)
        ax.tick_params(labelsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()

    handles = [
        Line2D([], [], color=_CAMPAIGN_COLOR, lw=3,
               label="adopted campaign epsilon width"),
        Line2D([], [], color=_PREVIOUS_COLOR, lw=1.8,
               label="previous provisional epsilon width"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Search-scalar spread across "
                 f"{int(res['random_ok'].sum())} constraint-feasible random "
                 f"policies — {st['label']}", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    save_figure(fig, scfg.epsilon_figure_path(
        "scalar_distributions", res["design"]).with_suffix(""))
    plt.close(fig)


def _fig_parallel_axes(res: dict, active: list,
                       eps_campaign_active: np.ndarray) -> None:
    """F4 (per design): parallel-axes view of the adopted epsilon's effect.

    Every constraint-feasible evaluated policy is a polyline over the ACTIVE
    campaign objectives (axes oriented so up = preferred, native ranges
    annotated); the subset retained by ε-box nondominance under the ADOPTED
    campaign vector is highlighted, the merged remainder is faint grey, and
    the FFMP baseline is bold. The visual check that the adopted resolution
    thins the set without collapsing the span of any tradeoff axis. Rendered
    by the shared :func:`custom_parallel_coordinates` (same normalization and
    annotation conventions as the Pareto-set viewer).
    """
    st = _design_style(res["design"])
    obj_names = res["obj_names"]
    ka = [obj_names.index(n) for n in active]
    signs = np.array([-1.0 if ENSEMBLE_OBJECTIVES[n].direction == "maximize"
                      else 1.0 for n in active])
    finite = np.isfinite(res["natural"][:, ka]).all(axis=1)
    rows_ok = res["valid_ok"] & finite

    raw = res["natural"][rows_ok][:, ka]
    keep = np.zeros(raw.shape[0], dtype=bool)
    keep[epsilon_nondominated(raw * signs, eps_campaign_active)] = True

    baseline_rows = np.flatnonzero(rows_ok & ~res["random_ok"])
    baseline_raw = (res["natural"][baseline_rows[0], ka]
                    if baseline_rows.size else None)

    custom_parallel_coordinates(
        pd.DataFrame(raw, columns=list(active)),
        axis_labels=[label_for(n) for n in active],
        minmaxs=["max" if ENSEMBLE_OBJECTIVES[n].direction == "maximize"
                 else "min" for n in active],
        title=("Effect of the adopted campaign epsilons — "
               f"{st['label']}"),
        baseline=baseline_raw,
        highlight_mask=keep,
        baseline_label="FFMP baseline (status quo)",
        highlight_label="retained in the epsilon-archive",
        exclude_label="merged by epsilon-dominance",
        legend_loc="center left",
        legend_bbox=(1.005, 0.5),  # outside right: the cloud fills the axes
        legend_ncol=1,
        save_fig_filename=scfg.epsilon_figure_path("parallel_axes", res["design"]),
    )


def _fig_archive_sweep(results: list, sweeps: dict) -> None:
    """F2 (combined): archive cardinality vs scaling of the adopted vector.

    One line per design: the ε-box nondominated archive size of the evaluated
    policies as the adopted campaign vector is scaled by ``EPS_SCALE_GRID``
    (log2 axis; the adopted vector is the x = 1 vertical line). The x marker
    (drawn at x = 1; its abscissa is nominal) is the archive size under the
    previous provisional vector, and the dotted line is the unconstrained
    (ε -> 0) Pareto count.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for rank, res in enumerate(sorted(results, key=lambda r: r["design"])):
        d = res["design"]
        st = _design_style(d, rank)
        sweep = sweeps[d]
        rec = sweep[sweep["vector"] == "campaign"]
        if len(rec):
            ax.plot(rec["scale"], rec["archive_size"], marker="o", lw=2,
                    markersize=5, color=st["color"], label=st["label"],
                    zorder=3)
        prv = sweep[sweep["vector"] == "previous"]
        if len(prv):
            ax.scatter([1.0], prv["archive_size"], marker="x", s=70,
                       color=st["color"], linewidths=1.8, zorder=4)
        ax.axhline(sweep["n_pareto_eps0"].iloc[0], color=st["color"], ls=":",
                   lw=1.1, alpha=0.7, zorder=2)
    ax.axvline(1.0, color=_CAMPAIGN_COLOR, lw=1.2, ls="--", alpha=0.9,
               zorder=1)
    ax.text(1.0, ax.get_ylim()[1], " adopted vector (x1)", fontsize=8,
            color=_CAMPAIGN_COLOR, ha="left", va="top")

    ax.set_xscale("log", base=2)
    scales = sorted(float(s) for s in scfg.EPS_SCALE_GRID)
    ax.set_xticks(scales)
    ax.set_xticklabels([f"1/{round(1 / s):d}" if s < 1 and (1 / s).is_integer()
                        else f"{s:g}" for s in scales])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("scale factor applied to the adopted campaign epsilon vector")
    ax.set_ylabel("epsilon-nondominated archive size "
                  f"(of {int(list(sweeps.values())[0]['n_policies'].iloc[0])} "
                  "evaluated policies)")
    ax.set_title("Archive cardinality vs epsilon resolution")
    ax.grid(alpha=0.3)
    extra = [
        Line2D([], [], color="0.35", ls=":", lw=1.1,
               label="unconstrained Pareto count (epsilon -> 0)"),
        Line2D([], [], marker="x", ls="", color="0.35", markersize=7,
               label="previous provisional vector (abscissa nominal)"),
    ]
    ax.legend(handles=(ax.get_legend_handles_labels()[0] + extra), fontsize=8,
              frameon=True)
    save_figure(fig, scfg.epsilon_figure_path("archive_size_vs_scale")
                .with_suffix(""))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined recommendation
# ---------------------------------------------------------------------------

def combine_recommendations(results: list) -> pd.DataFrame:
    """Cross-design synthesis: one campaign epsilon per objective.

    The Borg problem (and the MOEAFramework JARs) carry ONE epsilon set used
    by every design, so the campaign value is the clean-rounded per-objective
    maximum of the raw requirement across the CAMPAIGN designs
    (``scfg.EPS_CAMPAIGN_DESIGNS``) — no campaign design's archive may resolve
    below its own noise floor. Reference designs (the historic single trace)
    keep their per-design columns in the table for context but do NOT enter
    the max: the historic 76-unit estimator's noise floor would otherwise
    coarsen the shared vector ~3-4x beyond what the ensemble measures need,
    so the historic arm's archive is allowed to resolve below its own noise
    floor (disclosed).
    """
    designs = [r["design"] for r in results]
    campaign = [d for d in designs if d in scfg.EPS_CAMPAIGN_DESIGNS] or designs
    obj_names = list(results[0]["diag"].index)
    table = pd.DataFrame(index=pd.Index(obj_names, name="objective"))
    for r in results:
        table[f"eps_raw__{r['design']}"] = r["diag"]["eps_floor_raw"]
        table[f"eps_rec__{r['design']}"] = r["diag"]["eps_recommended"]
    raw_cols = [f"eps_raw__{d}" for d in campaign]
    table["eps_campaign"] = [
        ceil_to_clean_step(v) for v in table[raw_cols].max(axis=1)]
    table["binding_design"] = table[raw_cols].idxmax(axis=1).str.replace(
        "eps_raw__", "", regex=False)
    table["eps_previous"] = results[0]["diag"]["eps_previous"]
    table["previous_over_campaign"] = (table["eps_previous"]
                                       / table["eps_campaign"])
    table["interpretation"] = [
        f"{table.loc[n, 'eps_campaign']:g} {_unit_phrase(n)}"
        for n in obj_names]
    return table


def main() -> None:
    apply_style()
    scfg.EPS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cube_paths = sorted(scfg.EPS_CUBE_DIR.glob(scfg.epsilon_cube_glob()))
    if not cube_paths:
        sys.exit(f"[epsilon_figures] no cubes matching "
                 f"'{scfg.epsilon_cube_glob()}' under {scfg.EPS_CUBE_DIR} — "
                 "run epsilon_calibration_run.py first.")

    results = [analyze_design(p) for p in cube_paths]

    combined = combine_recommendations(results)
    out = scfg.epsilon_table_path("epsilon_recommendation")
    combined.to_csv(out)

    # Archive sweeps + figures need the adopted campaign vector, so they run
    # after the combine stage.
    active = [n for n in
              build_ensemble_objective_set(config.ACTIVE_OBJECTIVES).names
              if n in combined.index]
    eps_campaign_active = combined.loc[active, "eps_campaign"].to_numpy(float)
    sweeps: dict[str, pd.DataFrame] = {}
    for res in results:
        sweep = _archive_sweep(res, active, eps_campaign_active)
        sweeps[res["design"]] = sweep
        sweep.to_csv(scfg.epsilon_table_path("archive_sweep", res["design"]),
                     index=False)

    _fig_eps_ladder(results, combined)
    for res in results:
        _fig_scalar_distributions(res, combined)
        _fig_parallel_axes(res, active, eps_campaign_active)
    _fig_archive_sweep(results, sweeps)

    designs = [r["design"] for r in results]
    campaign = [d for d in designs if d in scfg.EPS_CAMPAIGN_DESIGNS] or designs
    reference = [d for d in designs if d not in campaign]
    print(f"\n=== Campaign epsilon recommendation "
          f"(max over {campaign}; reference-only: {reference}) ===", flush=True)
    cols = ["eps_campaign", "eps_previous", "previous_over_campaign",
            "binding_design"]
    print(combined[cols].to_string(), flush=True)
    spread = combined[[f"eps_raw__{d}" for d in campaign]]
    hetero = (spread.max(axis=1) / spread.min(axis=1)).replace(np.inf, np.nan)
    worst = hetero.max()
    if np.isfinite(worst) and worst > 4.0:
        print(f"\nWARN: raw epsilon requirement differs {worst:.1f}x across "
              "campaign designs for at least one objective — review the "
              "per-design diagnostics before adopting the combined vector.",
              flush=True)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
