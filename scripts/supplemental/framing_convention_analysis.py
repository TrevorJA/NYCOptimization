"""framing_convention_analysis.py - Cube-based framing-convention diagnostics.

Post-hoc reductions of the epsilon-calibration per-unit annual-metric cubes
(``supplemental_config.epsilon_cube_glob``): 512 constraint-feasible random
policies + the FFMP baseline, evaluated on each campaign design's own search
ensemble, with every objective's stage-(i) annual metric stored per unit-year
(failing-week counts for the frequency objectives, annual flood-day counts,
annual deficits/storage minima). Zero simulation; every diagnostic below is a
reduction of the stored cubes.

Implements four of the pre-campaign framing closures
(``docs/notes/methods/framing_convention_diagnostics.md`` diagnostics 1 and 4,
the flood unit-operator comparison, and the annual-unit redundancy screen for
the 8th objective):

  1. **Failure-week count k sweep** - for each frequency objective and
     k in ``FRAMING_K_GRID``, the policy population's reliability values,
     saturation fractions (<= band / >= 1 - band), and Kendall tau_b of the
     induced ranking against the shipped ``_DEFAULT_FAILURE_K``. Gates the
     final k per objective.
  2. **Flood unit operator (mean vs P99)** - pooled-mean vs pooled-P99 annual
     flood days per policy: ranking agreement (tau_b) and a
     bootstrap-over-realizations noise / ranking-stability comparison at the
     campaign unit count. Gates the flood unit-operator choice.
  3. **Flood-days controllability** - empirical exogenous floor F_min(u) over
     the policy population per pooled unit-year, the policy-invariant floor
     share of the baseline's flood days, and its complement (a LOWER bound on
     the controllable fraction; the floor is a sample minimum, not an oracle).
  4. **Annual-unit redundancy screen** - Spearman matrix over the policy
     population of all nine registry objective values (shipped k), flagging
     |rho| >= ``FRAMING_RHO_FLAG_THRESHOLD`` (Olden & Poff 2003). Gates the
     8th objective (``nj_delivery_reliability_annual``) activate-or-drop.

Outputs -> ``outputs/supplemental/framing_convention/{tables,figures}``.
Configuration lives in ``supplemental_config.py`` (FRAMING_* section) — no CLI
value flags.

Usage:
    python scripts/supplemental/framing_convention_analysis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_epsilon_env()  # LSTMs off before config import (registry build)
os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")

from src.objectives_ensemble import (  # noqa: E402
    ENSEMBLE_OBJECTIVES,
    FailureFrequencyOp,
    PooledMeanOp,
    PooledPercentileOp,
    _DEFAULT_FAILURE_K,
)
from src.sensitivity_common import kendall_tau_b, spearman_and_flagged  # noqa: E402
from src.plotting.style import (  # noqa: E402
    annotated_corr_heatmap,
    apply_style,
    label_for,
    save_figure,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Campaign designs analyzed, in display order (labels/colors match the
#: epsilon-calibration SI figures: Okabe-Ito keyed to the DESIGN).
_DESIGNS: tuple = ("fixed_probabilistic", "hazard_filling_stationary", "historic")
_DESIGN_STYLE: dict = {
    "fixed_probabilistic": {
        "color": "#0072B2", "label": "Fixed probabilistic (i.i.d. control)"},
    "hazard_filling_stationary": {
        "color": "#D55E00", "label": "Hazard-filling (stationary)"},
    "historic": {
        "color": "#B0B0B0", "label": "Historic trace (reference)"},
}

#: Frequency objectives screened by the k sweep (registry display order).
_FREQ_OBJECTIVES: tuple = (
    "nyc_delivery_reliability_annual",
    "montague_flow_reliability_annual",
    "trenton_flow_reliability_annual",
    "nj_delivery_reliability_annual",
)

# The ACTIVE flood objective (flood_objective_diagnostics.md); cubes
# written before the exceedance adoption carry the day count under this role
# instead — re-run on a current cube.
_FLOOD_OBJECTIVE = "downstream_flood_exceedance_annual"


def _load_cube(design: str) -> dict:
    """Load one design's cube into pooled per-policy unit arrays.

    Returns:
        Dict with ``pooled`` (n_dv, n_pooled_units) per objective name,
        ``units3`` (n_dv, n_real, n_units) per objective name, ``sample_ids``,
        ``pop_mask`` (random-policy rows), ``baseline_row`` index, and sizes.
    """
    path = scfg.epsilon_cube_path(design)
    if not path.exists():
        sys.exit(f"[framing] missing cube for '{design}': {path}")
    with h5py.File(path, "r") as f:
        units = f["units"][:]                      # (n_dv, n_real, n_obj, n_units)
        sample_ids = f["sample_ids"][:]
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["objective_names"][:]]
    n_dv, n_real, _, n_units = units.shape
    baseline_rows = np.where(sample_ids == -1)[0]
    if baseline_rows.size != 1:
        sys.exit(f"[framing] '{design}': expected exactly one baseline row "
                 f"(sample_id -1), found {baseline_rows.size}")
    pooled = {nm: units[:, :, j, :].reshape(n_dv, n_real * n_units)
              for j, nm in enumerate(names)}
    units3 = {nm: units[:, :, j, :] for j, nm in enumerate(names)}
    return {
        "design": design, "pooled": pooled, "units3": units3,
        "sample_ids": sample_ids, "pop_mask": sample_ids >= 0,
        "baseline_row": int(baseline_rows[0]),
        "n_dv": n_dv, "n_real": n_real, "n_units": n_units,
        "objective_names": names,
    }


def _objective_values(cube: dict, k_override: "dict | None" = None) -> pd.DataFrame:
    """Final objective values per policy (vectorized registry unit operators).

    Args:
        cube: Loaded cube dict.
        k_override: Optional ``{frequency objective: k}`` replacing the shipped
            failure-week counts.

    Returns:
        DataFrame (rows = policies, columns = registry objectives present in
        the cube), on each objective's native registry scale.
    """
    out = {}
    for nm in cube["objective_names"]:
        arr = cube["pooled"][nm]
        op = ENSEMBLE_OBJECTIVES[nm].unit_operator
        if isinstance(op, FailureFrequencyOp):
            k = (k_override or {}).get(nm, op.k)
            out[nm] = (np.where(np.isfinite(arr), arr, np.inf) < k).mean(axis=1)
        elif isinstance(op, PooledPercentileOp):
            vals = np.where(np.isfinite(arr), arr, op.worst_value)
            out[nm] = np.percentile(vals, op.q, axis=1)
        elif isinstance(op, PooledMeanOp):
            vals = np.where(np.isfinite(arr), arr, op.worst_value)
            out[nm] = vals.mean(axis=1)
        else:  # pragma: no cover - registry only holds the three ops above
            out[nm] = np.array([op(row) for row in arr])
    return pd.DataFrame(out)


###############################################################################
# 1. Failure-week count k sweep
###############################################################################

def k_sweep(cubes: dict) -> pd.DataFrame:
    """Saturation + ranking-sensitivity table over ``FRAMING_K_GRID``."""
    band = scfg.FRAMING_SATURATION_BAND
    rows = []
    for design, cube in cubes.items():
        pop = cube["pop_mask"]
        for nm in _FREQ_OBJECTIVES:
            counts = cube["pooled"][nm]
            shipped = _DEFAULT_FAILURE_K[nm]
            vals_by_k = {k: (counts < k).mean(axis=1)
                         for k in scfg.FRAMING_K_GRID}
            for k, vals in vals_by_k.items():
                pv = vals[pop]
                rows.append({
                    "design": design, "objective": nm, "k": k,
                    "shipped_k": shipped,
                    "frac_low": float((pv <= band).mean()),
                    "frac_high": float((pv >= 1.0 - band).mean()),
                    "saturation_frac": float(((pv <= band)
                                              | (pv >= 1.0 - band)).mean()),
                    "pop_median": float(np.median(pv)),
                    "pop_iqr": float(np.percentile(pv, 75)
                                     - np.percentile(pv, 25)),
                    "baseline_value": float(vals[cube["baseline_row"]]),
                    "tau_vs_shipped": kendall_tau_b(pv,
                                                    vals_by_k[shipped][pop]),
                })
    df = pd.DataFrame(rows)
    out = scfg.framing_table_path("k_sweep")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def k_sweep_figure(df: pd.DataFrame) -> None:
    """Panel figure: saturation fraction vs k and tau_b vs k, per objective."""
    n = len(_FREQ_OBJECTIVES)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.9 * n),
                             sharex=True, constrained_layout=True)
    for i, nm in enumerate(_FREQ_OBJECTIVES):
        ax_sat, ax_tau = axes[i]
        for design in _DESIGNS:
            sub = df[(df["objective"] == nm) & (df["design"] == design)]
            st = _DESIGN_STYLE[design]
            ax_sat.plot(sub["k"], sub["saturation_frac"], marker="o", ms=3.5,
                        color=st["color"], label=st["label"])
            ax_tau.plot(sub["k"], sub["tau_vs_shipped"], marker="o", ms=3.5,
                        color=st["color"], label=st["label"])
        shipped = _DEFAULT_FAILURE_K[nm]
        for ax in (ax_sat, ax_tau):
            ax.axvline(shipped, color="0.35", lw=0.9, ls=(0, (4, 3)))
            ax.set_xticks(list(scfg.FRAMING_K_GRID))
        ax_sat.set_ylim(-0.03, 1.03)
        ax_tau.set_ylim(-0.03, 1.03)
        ax_sat.set_ylabel(label_for(nm), fontsize=7)
        if i == 0:
            ax_sat.set_title("Saturated policy fraction", fontsize=8)
            ax_tau.set_title(r"$\tau_b$ of ranking vs shipped $k$", fontsize=8)
    axes[-1, 0].set_xlabel("failure-week count k")
    axes[-1, 1].set_xlabel("failure-week count k")
    axes[0, 0].legend(fontsize=6, frameon=False)
    save_figure(fig, scfg.framing_figure_path("k_sweep_saturation_tau"))
    plt.close(fig)


def k_distribution_figure(cubes: dict) -> None:
    """Supporting panel: population reliability distributions per k."""
    band = scfg.FRAMING_SATURATION_BAND
    n = len(_FREQ_OBJECTIVES)
    designs = list(cubes)
    fig, axes = plt.subplots(n, len(designs), figsize=(2.4 * len(designs), 1.75 * n),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for j, design in enumerate(designs):
        cube = cubes[design]
        pop = cube["pop_mask"]
        for i, nm in enumerate(_FREQ_OBJECTIVES):
            ax = axes[i, j]
            counts = cube["pooled"][nm]
            data = [(counts < k).mean(axis=1)[pop] for k in scfg.FRAMING_K_GRID]
            ax.axhspan(0.0, band, color="0.85", zorder=0)
            ax.axhspan(1.0 - band, 1.0, color="0.85", zorder=0)
            bp = ax.boxplot(data, positions=list(scfg.FRAMING_K_GRID),
                            widths=0.55, showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(_DESIGN_STYLE[design]["color"])
                patch.set_alpha(0.55)
                patch.set_edgecolor("0.25")
            for med in bp["medians"]:
                med.set_color("0.15")
            base = [(counts < k).mean(axis=1)[cube["baseline_row"]]
                    for k in scfg.FRAMING_K_GRID]
            ax.plot(list(scfg.FRAMING_K_GRID), base, ls="none", marker="*",
                    ms=8, color="0.1", zorder=4, label="FFMP baseline")
            ax.axvline(_DEFAULT_FAILURE_K[nm], color="0.35", lw=0.9,
                       ls=(0, (4, 3)))
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(_DESIGN_STYLE[design]["label"], fontsize=7)
            if j == 0:
                ax.set_ylabel(label_for(nm), fontsize=6.5)
            if i == n - 1:
                ax.set_xlabel("failure-week count k", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower left")
    save_figure(fig, scfg.framing_figure_path("k_sweep_distributions"))
    plt.close(fig)


###############################################################################
# 2. Flood unit operator: mean vs P99
###############################################################################

def flood_operator(cubes: dict) -> pd.DataFrame:
    """Mean-vs-P99 ranking agreement + bootstrap noise / ranking stability."""
    rng = np.random.default_rng(scfg.FRAMING_BOOTSTRAP_SEED)
    rows = []
    for design, cube in cubes.items():
        pop = cube["pop_mask"]
        fl3 = cube["units3"][_FLOOD_OBJECTIVE]      # (n_dv, n_real, n_units)
        n_dv, n_real, n_units = fl3.shape
        pooled = fl3.reshape(n_dv, n_real * n_units)
        mean_full = pooled.mean(axis=1)
        p99_full = np.percentile(pooled, 99.0, axis=1)

        # Bootstrap the sampling axis: realizations for the ensembles, pooled
        # unit-years for the single-trace historic design.
        boot_axis = n_real if n_real > 1 else n_units
        stats = {"mean": {"vals": [], "tau": []}, "p99": {"vals": [], "tau": []}}
        for _ in range(scfg.FRAMING_BOOTSTRAP_B):
            idx = rng.integers(0, boot_axis, boot_axis)
            sub = (fl3[:, idx, :] if n_real > 1 else fl3[:, :, idx])
            sub = sub.reshape(n_dv, -1)
            bm = sub.mean(axis=1)
            bp = np.percentile(sub, 99.0, axis=1)
            stats["mean"]["vals"].append(bm)
            stats["p99"]["vals"].append(bp)
            stats["mean"]["tau"].append(kendall_tau_b(bm[pop], mean_full[pop]))
            stats["p99"]["tau"].append(kendall_tau_b(bp[pop], p99_full[pop]))
        for op_name, full in (("mean", mean_full), ("p99", p99_full)):
            vals = np.asarray(stats[op_name]["vals"])       # (B, n_dv)
            rows.append({
                "design": design, "operator": op_name,
                "full_baseline": float(full[cube["baseline_row"]]),
                "pop_median": float(np.median(full[pop])),
                "pop_iqr": float(np.percentile(full[pop], 75)
                                 - np.percentile(full[pop], 25)),
                "boot_std_median_policy": float(
                    np.median(vals.std(axis=0)[pop])),
                "boot_tau_vs_full_mean": float(
                    np.nanmean(stats[op_name]["tau"])),
                "boot_tau_vs_full_p05": float(
                    np.nanpercentile(stats[op_name]["tau"], 5)),
                "boot_tau_nan_frac": float(
                    np.mean(~np.isfinite(stats[op_name]["tau"]))),
                "tau_mean_vs_p99": kendall_tau_b(mean_full[pop],
                                                 p99_full[pop]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.framing_table_path("flood_operator"), index=False)
    return df


def flood_operator_figure(cubes: dict, df: pd.DataFrame) -> None:
    """Scatter of pooled-mean vs pooled-P99 annual flood days per policy."""
    designs = list(cubes)
    fig, axes = plt.subplots(1, len(designs), figsize=(2.7 * len(designs), 2.7),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, design in zip(axes, designs):
        cube = cubes[design]
        pop = cube["pop_mask"]
        pooled = cube["pooled"][_FLOOD_OBJECTIVE]
        mean_v = pooled.mean(axis=1)
        p99_v = np.percentile(pooled, 99.0, axis=1)
        st = _DESIGN_STYLE[design]
        ax.scatter(mean_v[pop], p99_v[pop], s=7, alpha=0.45,
                   color=st["color"], linewidths=0)
        ax.scatter(mean_v[cube["baseline_row"]], p99_v[cube["baseline_row"]],
                   marker="*", s=110, color="0.1", zorder=4,
                   label="FFMP baseline")
        tau = df[(df["design"] == design)
                 & (df["operator"] == "mean")]["tau_mean_vs_p99"].iloc[0]
        ax.set_title(f"{st['label']}\n" + rf"$\tau_b$(mean, P99) = {tau:.2f}",
                     fontsize=7.5)
        ax.set_xlabel("pooled mean (days/yr)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("pooled P99 (days/yr)", fontsize=7)
            ax.legend(fontsize=6, frameon=False)
    save_figure(fig, scfg.framing_figure_path("flood_operator_mean_vs_p99"))
    plt.close(fig)


###############################################################################
# 3. Flood-days controllability
###############################################################################

def flood_controllability(cubes: dict) -> pd.DataFrame:
    """Empirical exogenous floor + controllable fraction, per design."""
    rows = []
    for design, cube in cubes.items():
        pooled = cube["pooled"][_FLOOD_OBJECTIVE]   # (n_dv, n_pooled)
        pop = cube["pop_mask"]
        floor_u = pooled.min(axis=0)                # over population + baseline
        base_u = pooled[cube["baseline_row"]]
        base_total = base_u.sum()
        floor_share = (float(floor_u.sum() / base_total)
                       if base_total > 0 else np.nan)
        mean_vals = pooled.mean(axis=1)
        rows.append({
            "design": design,
            "baseline_mean_days": float(base_u.mean()),
            "floor_mean_days": float(floor_u.mean()),
            "pop_min_mean_days": float(mean_vals[pop].min()),
            "pop_max_mean_days": float(mean_vals[pop].max()),
            "floor_share_of_baseline": floor_share,
            "controllable_frac_lower_bound": (1.0 - floor_share
                                              if np.isfinite(floor_share)
                                              else np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.framing_table_path("flood_controllability"), index=False)
    return df


def flood_controllability_figure(cubes: dict, df: pd.DataFrame) -> None:
    """Per design: population distribution of mean flood days + floor lines."""
    designs = list(cubes)
    fig, axes = plt.subplots(1, len(designs), figsize=(2.7 * len(designs), 2.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, design in zip(axes, designs):
        cube = cubes[design]
        pop = cube["pop_mask"]
        mean_vals = cube["pooled"][_FLOOD_OBJECTIVE].mean(axis=1)
        row = df[df["design"] == design].iloc[0]
        st = _DESIGN_STYLE[design]
        ax.hist(mean_vals[pop], bins=40, color=st["color"], alpha=0.65)
        ax.axvline(row["baseline_mean_days"], color="0.1", lw=1.2,
                   label="FFMP baseline")
        ax.axvline(row["floor_mean_days"], color="0.1", lw=1.2,
                   ls=(0, (4, 3)), label="empirical floor")
        ax.set_title(
            f"{st['label']}\nfloor share {row['floor_share_of_baseline']:.2f}"
            f" | controllable ≥ "
            f"{row['controllable_frac_lower_bound']:.2f}", fontsize=7.5)
        ax.set_xlabel("mean annual flood days (days/yr)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("policies", fontsize=7)
            ax.legend(fontsize=6, frameon=False)
    save_figure(fig, scfg.framing_figure_path("flood_controllability"))
    plt.close(fig)


###############################################################################
# 4. Annual-unit redundancy screen
###############################################################################

def redundancy(cubes: dict) -> pd.DataFrame:
    """Spearman screen of the nine registry objectives; NJ-focused summary."""
    nj_rows = []
    for design, cube in cubes.items():
        vals = _objective_values(cube)
        pop_vals = vals[cube["pop_mask"]].reset_index(drop=True)
        names = cube["objective_names"]
        spearman, flagged, excluded = spearman_and_flagged(
            pop_vals, names, scfg.FRAMING_RHO_FLAG_THRESHOLD)
        spearman.to_csv(scfg.framing_table_path("redundancy_spearman", design))
        flagged.to_csv(scfg.framing_table_path("redundancy_flagged", design),
                       index=False)
        if excluded:
            print(f"[framing] {design}: excluded from Spearman "
                  f"(constant/sparse): {excluded}")
        nj = "nj_delivery_reliability_annual"
        if nj in spearman.index:
            for other in spearman.columns:
                if other != nj:
                    nj_rows.append({"design": design, "objective": other,
                                    "rho_vs_nj": float(spearman.loc[nj, other])})
    df = pd.DataFrame(nj_rows)
    df.to_csv(scfg.framing_table_path("nj_redundancy"), index=False)
    return df


def redundancy_figure(cubes: dict) -> None:
    """Annotated Spearman heatmap per design (one row of panels)."""
    designs = list(cubes)
    fig, axes = plt.subplots(1, len(designs),
                             figsize=(3.4 * len(designs), 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, design in zip(axes, designs):
        cube = cubes[design]
        vals = _objective_values(cube)
        pop_vals = vals[cube["pop_mask"]].reset_index(drop=True)
        names = cube["objective_names"]
        spearman, _, _ = spearman_and_flagged(
            pop_vals, names, scfg.FRAMING_RHO_FLAG_THRESHOLD)
        annotated_corr_heatmap(
            ax, spearman.values, list(spearman.columns),
            box_threshold=scfg.FRAMING_RHO_FLAG_THRESHOLD, fontsize=5)
        ax.set_title(_DESIGN_STYLE[design]["label"], fontsize=8)
    save_figure(fig, scfg.framing_figure_path("redundancy_spearman"))
    plt.close(fig)


###############################################################################
# Driver
###############################################################################

def main() -> None:
    apply_style()
    cubes = {d: _load_cube(d) for d in _DESIGNS
             if scfg.epsilon_cube_path(d).exists()}
    if not cubes:
        sys.exit("[framing] no epsilon cubes found under "
                 f"{scfg.EPS_CUBE_DIR}")
    for d, c in cubes.items():
        print(f"[framing] {d}: {c['n_dv']} policies x {c['n_real']} real "
              f"x {c['n_units']} units")
    scfg.FRAMING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ks = k_sweep(cubes)
    k_sweep_figure(ks)
    k_distribution_figure(cubes)

    fo = flood_operator(cubes)
    flood_operator_figure(cubes, fo)

    fc = flood_controllability(cubes)
    flood_controllability_figure(cubes, fc)

    nj = redundancy(cubes)
    redundancy_figure(cubes)

    print(f"[framing] tables -> {scfg.FRAMING_TABLES_DIR}")
    print(f"[framing] figures -> {scfg.FRAMING_FIGURES_DIR}")
    print("\n=== headline numbers ===")
    shipped = ks[ks["k"] == ks["shipped_k"]]
    print(shipped[["design", "objective", "saturation_frac", "frac_low",
                   "frac_high", "tau_vs_shipped"]].to_string(index=False))
    print(fo[["design", "operator", "tau_mean_vs_p99",
              "boot_std_median_policy", "boot_tau_vs_full_mean"]]
          .to_string(index=False))
    print(fc.to_string(index=False))
    if not nj.empty:
        print(nj.pivot(index="objective", columns="design",
                       values="rho_vs_nj").round(2).to_string())


if __name__ == "__main__":
    main()
