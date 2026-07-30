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
     policies for the recommended vector scaled by ``EPS_SCALE_GRID`` and for
     the current registry vector (how strongly epsilon controls Pareto-set
     cardinality). Random policies under-fill a converged front, so the SWEEP
     TREND is the signal, not the absolute sizes — disclosed in the table.

The final campaign vector is the clean-rounded per-objective maximum of the
raw requirement across all analyzed designs (the Borg problem/JARs carry one
epsilon set for every design).

Outputs (all under ``outputs/supplemental/epsilon_calibration/``):
  tables/  : epsilon_diagnostics_{design}, archive_sweep_{design},
             epsilon_recommendation (combined)  [CSV]
  figures/ : eps_ladder_{design} (F1), scalar_distributions_{design} (F3),
             archive_size_vs_scale (F2, combined)  [PNG]

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

# ---------------------------------------------------------------------------
# Labels (fixed, colorblind-safe Okabe-Ito assignments)
# ---------------------------------------------------------------------------

#: Native-unit phrase per objective-name pattern, for the interpretation column.
_UNIT_PHRASES: list[tuple[str, str]] = [
    ("reliability_annual", "fraction of unit-years"),
    ("_p99_pct", "% of target (P99 unit-year)"),
    ("flood_days_annual_p99", "days/yr (P99 unit-year)"),
    ("flood_days_annual", "days/yr (mean unit-year)"),
    ("storage_min_p01_pct", "% of NYC capacity (P1 annual minimum)"),
]

#: Quantity colors for the epsilon-ladder figure (Okabe-Ito).
_LADDER_COLORS = {
    "granularity": "#999999",
    "noise": "#E69F00",
    "signal": "#0072B2",
    "current": "#000000",
    "recommended": "#009E73",
}

#: Design colors for the combined archive-size figure (Okabe-Ito).
_DESIGN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


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
# Per-design analysis
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
    """Full per-design epsilon analysis: tables + per-design figures.

    Args:
        cube_path: The design's unit-cube HDF5 from the run script.

    Returns:
        Dict with the design name, diagnostics DataFrame, archive-sweep
        DataFrame, and QC counts (consumed by the combined stage).
    """
    with h5py.File(cube_path, "r") as f:
        units = f["units"][:]
        sample_ids = f["sample_ids"][:]
        obj_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["objective_names"][:]]
        eps_current = f["epsilons_current"][:]
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
            "eps_current": float(eps_current[k]),
            "current_over_recommended": (float(eps_current[k] / eps_rec)
                                         if np.isfinite(eps_rec) and eps_rec > 0
                                         else float("nan")),
            "binding_floor": binding,
            "interpretation": interp,
        })
    diag = pd.DataFrame(rows).set_index("objective")

    sweep = _archive_sweep(natural, valid_ok, obj_names, eps_current, diag,
                           design)

    scfg.EPS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    diag.to_csv(scfg.epsilon_table_path("epsilon_diagnostics", design))
    sweep.to_csv(scfg.epsilon_table_path("archive_sweep", design), index=False)

    _fig_eps_ladder(diag, design)
    _fig_scalar_distributions(natural, random_ok, obj_names, diag, design)

    print(f"[{design}] n_policies={n_dv} (failed={int(failed.sum())}), "
          f"NL={nl}, acceptance={acceptance:.2%}", flush=True)
    return {"design": design, "diag": diag, "sweep": sweep, "nl": nl}


def _archive_sweep(natural: np.ndarray, valid_ok: np.ndarray,
                   obj_names: list, eps_current: np.ndarray,
                   diag: pd.DataFrame, design: str) -> pd.DataFrame:
    """ε-archive size of the evaluated policies for candidate epsilon vectors.

    Uses the ACTIVE campaign objective subset (the set Borg optimizes) in Borg
    sign convention. NOTE: random feasible policies under-fill a converged
    Pareto front, so absolute sizes are a lower-fidelity proxy — the epsilon
    SCALING TREND is the decision signal.
    """
    active = list(build_ensemble_objective_set(config.ACTIVE_OBJECTIVES).names)
    ka = [obj_names.index(n) for n in active]
    signs = np.array([-1.0 if ENSEMBLE_OBJECTIVES[n].direction == "maximize"
                      else 1.0 for n in active])
    F = natural[:, ka] * signs
    rows_ok = valid_ok & np.isfinite(F).all(axis=1)
    F = F[rows_ok]

    eps_rec = diag.loc[active, "eps_recommended"].to_numpy(dtype=float)
    n_pareto = int(_nondominated_mask(F).sum())

    entries = []
    if np.isfinite(eps_rec).all() and (eps_rec > 0).all():
        for scale in scfg.EPS_SCALE_GRID:
            size = len(epsilon_nondominated(F, eps_rec * scale))
            entries.append({"design": design, "vector": "recommended",
                            "scale": float(scale), "archive_size": size})
    cur = np.asarray(eps_current, dtype=float)[ka]
    entries.append({"design": design, "vector": "current", "scale": 1.0,
                    "archive_size": len(epsilon_nondominated(F, cur))})
    sweep = pd.DataFrame(entries)
    sweep["n_policies"] = int(rows_ok.sum())
    sweep["n_pareto_eps0"] = n_pareto
    return sweep


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_eps_ladder(diag: pd.DataFrame, design: str) -> None:
    """F1: per-objective epsilon ladder — floors, current, and recommendation.

    One row per objective on a shared log axis (native units span decades but
    read comparably as orders of magnitude); the recommendation is by
    construction at or right of every floor.
    """
    names = list(diag.index)
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * n + 1.8))
    specs = [
        ("eps_granularity", "granularity floor", "|", "granularity", 90),
        ("eps_noise_median", "noise floor (median bootstrap SD)", "o",
         "noise", 45),
        ("eps_signal_iqr10", "signal scale (IQR/10)", "s", "signal", 45),
        ("eps_current", "current epsilon", "x", "current", 60),
        ("eps_recommended", "recommended epsilon", "D", "recommended", 55),
    ]
    for col, lab, marker, ckey, size in specs:
        vals = diag[col].to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals > 0)
        kwargs = {"s": size, "marker": marker, "label": lab, "zorder": 3,
                  "linewidths": 1.5}
        if marker in ("o", "s"):  # hollow so overlapping points stay legible
            kwargs.update(facecolors="none", edgecolors=_LADDER_COLORS[ckey])
        else:
            kwargs.update(color=_LADDER_COLORS[ckey])
        ax.scatter(vals[mask], (n - 1 - np.arange(n))[mask], **kwargs)
    ax.set_xscale("log")
    ax.set_yticks(n - 1 - np.arange(n))
    ax.set_yticklabels([label_for(nm) for nm in names])
    ax.set_xlabel("epsilon-scale quantities (native units, log)")
    ax.set_title(f"Epsilon calibration ladder — {design}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    save_figure(fig, scfg.epsilon_figure_path("eps_ladder", design).with_suffix(""))
    plt.close(fig)


def _fig_scalar_distributions(natural: np.ndarray, random_ok: np.ndarray,
                              obj_names: list, diag: pd.DataFrame,
                              design: str) -> None:
    """F3: search-scalar distribution per objective, with epsilon widths.

    A histogram per objective across the feasible random policies; the bars
    under each axis show the current and recommended epsilon widths at the
    same scale — the visual check that one epsilon box holds neither the whole
    signal (too coarse) nor pure noise (too fine).
    """
    n = len(obj_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    for k, name in enumerate(obj_names):
        ax = axes[k // ncols][k % ncols]
        col = natural[random_ok, k]
        col = col[np.isfinite(col)]
        if col.size:
            ax.hist(col, bins=30, color="#0072B2", alpha=0.75)
        x0 = float(np.min(col)) if col.size else 0.0
        y0, y1 = ax.get_ylim()
        for offset, key, ckey in ((0.06, "eps_current", "current"),
                                  (0.13, "eps_recommended", "recommended")):
            width = float(diag.loc[name, key])
            if np.isfinite(width) and width > 0:
                ax.plot([x0, x0 + width],
                        [y1 * (1 - offset)] * 2,
                        color=_LADDER_COLORS[ckey], lw=3, solid_capstyle="butt")
        ax.set_title(label_for(name), fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.suptitle(f"Search-scalar spread across feasible random policies — "
                 f"{design}\n(bars: current [black] and recommended [green] "
                 "epsilon widths)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, scfg.epsilon_figure_path(
        "scalar_distributions", design).with_suffix(""))
    plt.close(fig)


def _fig_archive_sweep(results: list) -> None:
    """F2 (combined): archive size vs epsilon scale, one line per design."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, res in enumerate(results):
        sweep = res["sweep"]
        rec = sweep[sweep["vector"] == "recommended"]
        color = _DESIGN_COLORS[i % len(_DESIGN_COLORS)]
        if len(rec):
            ax.plot(rec["scale"], rec["archive_size"], marker="o", lw=2,
                    color=color, label=res["design"])
        cur = sweep[sweep["vector"] == "current"]
        if len(cur):
            ax.scatter([1.0], cur["archive_size"], marker="x", s=70,
                       color=color, zorder=4)
        ax.axhline(sweep["n_pareto_eps0"].iloc[0], color=color, ls=":",
                   lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("epsilon scale factor (x recommended vector)")
    ax.set_ylabel("epsilon-nondominated archive size")
    ax.set_title("Archive resolution vs epsilon scale\n"
                 "(x = current registry vector; dotted = plain Pareto count)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
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
    maximum of the raw requirement across the analyzed designs — no design's
    archive may resolve below its own noise floor.
    """
    designs = [r["design"] for r in results]
    obj_names = list(results[0]["diag"].index)
    table = pd.DataFrame(index=pd.Index(obj_names, name="objective"))
    for r in results:
        table[f"eps_raw__{r['design']}"] = r["diag"]["eps_floor_raw"]
        table[f"eps_rec__{r['design']}"] = r["diag"]["eps_recommended"]
    raw_cols = [f"eps_raw__{d}" for d in designs]
    table["eps_campaign"] = [
        ceil_to_clean_step(v) for v in table[raw_cols].max(axis=1)]
    table["binding_design"] = table[raw_cols].idxmax(axis=1).str.replace(
        "eps_raw__", "", regex=False)
    table["eps_current"] = results[0]["diag"]["eps_current"]
    table["current_over_campaign"] = table["eps_current"] / table["eps_campaign"]
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

    _fig_archive_sweep(results)
    combined = combine_recommendations(results)
    out = scfg.epsilon_table_path("epsilon_recommendation")
    combined.to_csv(out)

    print(f"\n=== Campaign epsilon recommendation ({len(results)} designs) ===",
          flush=True)
    cols = ["eps_campaign", "eps_current", "current_over_campaign",
            "binding_design"]
    print(combined[cols].to_string(), flush=True)
    spread = combined[[c for c in combined.columns
                       if c.startswith("eps_raw__")]]
    hetero = (spread.max(axis=1) / spread.min(axis=1)).replace(np.inf, np.nan)
    worst = hetero.max()
    if np.isfinite(worst) and worst > 4.0:
        print(f"\nWARN: raw epsilon requirement differs {worst:.1f}x across "
              "designs for at least one objective — review the per-design "
              "diagnostics before adopting the combined vector.", flush=True)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
