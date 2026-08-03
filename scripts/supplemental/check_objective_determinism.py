"""check_objective_determinism.py - Are the objectives deterministic under LP jitter?

The Pywr-DRB model is NOT bitwise deterministic: the LP solver introduces
minor nondeterminism in state trajectories (first measured in-process on Anvil
by ``scripts/supplemental/anvil_scaling/check_objective_determinism.py``). The
claim under test here is that the OBJECTIVES are deterministic — repeated
evaluations of the same decision-variable vector must yield identical (or
numerically negligible-difference) objective values, i.e. the LP jitter must
not propagate through the 6-month metric window and the annual-unit
aggregation into the objective vector.

Design (all knobs in ``supplemental_config.py``, ``NYCOPT_DETERMINISM_*`` env
overridable; no CLI flags):

  * Policies: the default FFMP baseline plus ``DETERMINISM_N_PERTURBED``
    feasible perturbations of it (accepted only at exactly-zero formal
    constraint violations).
  * Paths: historic single-trace and the staged local ensemble fixture, each
    on the trimmed (search) and full (baseline/re-eval) model. Each path is
    tested for determinism against ITSELF only.
  * Repeats: ``DETERMINISM_N_REPEATS`` per (policy, path), each repeat in a
    FRESH python subprocess (fresh interpreter, model build, solver instance),
    launched by this same script in worker mode (dispatch via the
    ``NYCOPT_DETERMINISM_TASK`` env var, one worker per (path, repeat)).
  * Metrics: per objective, across repeats: max absolute and max relative
    deviation. Each worker also records the daily aggregate NYC storage
    series of the baseline policy, documenting the state-level jitter the
    objectives are expected to absorb.

VERDICT RULE (stated before running): an objective counts as deterministic on
a path iff its across-repeat deviation is exactly zero or at floating-point
noise scale (max relative deviation <= ``DETERMINISM_REL_TOL`` = 1e-9);
anything larger is reported as propagation, per objective, with the worst
offender identified.

Usage (from the repo root; rerunnable — completed (path, repeat) runs are
skipped, delete ``outputs/supplemental/objective_determinism/runs/`` for a
full re-measurement)::

    ./venv/Scripts/python.exe scripts/supplemental/check_objective_determinism.py

Outputs (gitignored) under ``outputs/supplemental/objective_determinism/``:
``policies.json``, ``runs/`` (per-worker results + logs), ``summary.csv``,
``summary.json``, and ``figures/`` (F1 per-objective deviations, F2 the
state-jitter vs objective-agreement contrast).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_determinism_env()

import numpy as np  # noqa: E402

#: Plotting floor for exactly-zero deviations on log axes.
_ZERO_FLOOR = 1e-18

#: Relative-deviation denominator floor (guards objectives whose value is ~0).
_REL_DENOM_FLOOR = 1e-12

#: Fixed per-path colors (colorblind-safe seaborn-deep subset).
_PATH_COLORS = {
    "historic_trimmed": "#4C72B0",
    "historic_full": "#DD8452",
    "ensemble_trimmed": "#55A868",
    "ensemble_full": "#C44E52",
}

#: Compact two-line axis labels for the annual-unit objective set (fallback:
#: wrapped ``label_for``).
_SHORT_OBJ_LABELS = {
    "nyc_delivery_reliability_annual": "NYC Delivery\nRel.",
    "nyc_delivery_deficit_p99_pct": "NYC Deficit\nCVaR90",
    "montague_flow_reliability_annual": "Montague\nRel.",
    "montague_flow_deficit_p99_pct": "Montague Deficit\nCVaR90",
    "trenton_flow_reliability_annual": "Trenton\nRel.",
    "downstream_flood_severity_annual": "Flood Severity\n(minor, ft·d)",
    "downstream_flood_days_annual": "Flood Days\n(minor)",
    "nyc_storage_min_p01_pct": "NYC Storage\nmin P01",
}


def _short_label(name: str) -> str:
    """Compact multi-line label for one objective name."""
    import textwrap
    from src.plotting.style import label_for
    return _SHORT_OBJ_LABELS.get(name, textwrap.fill(label_for(name), 14))


###############################################################################
# Policy set
###############################################################################

def build_policies() -> dict:
    """Build (or reuse) the policy set: FFMP baseline + feasible perturbations.

    Perturbed policies are drawn as uniform perturbations of the baseline
    (± ``DETERMINISM_PERTURB_FRAC`` of each DV's bound range, clipped to
    bounds) and accepted only when both formal DV-arithmetic constraints are
    exactly zero; the fraction halves every 25 rejected draws. The draw is
    seeded, so the set is reproducible; an existing ``policies.json`` is
    reused verbatim so a resumed run measures the same vectors.

    Returns:
        The policies dict (also persisted to ``policies.json``).
    """
    from src.formulations import get_baseline_values, get_bounds
    from src.simulation import compute_constraint_violations

    path = scfg.determinism_policies_path()
    if path.exists():
        return json.loads(path.read_text())

    form = scfg.DETERMINISM_FORMULATION
    base = np.asarray(get_baseline_values(form), dtype=float)
    lo, hi = get_bounds(form)
    rng = np.random.default_rng(scfg.DETERMINISM_SEED)

    policies = [{
        "policy_id": 0,
        "label": "baseline",
        "dv_values": base.tolist(),
        "constraint_violations": compute_constraint_violations(base, form),
    }]
    for k in range(1, scfg.DETERMINISM_N_PERTURBED + 1):
        frac = scfg.DETERMINISM_PERTURB_FRAC
        for attempt in range(500):
            dv = np.clip(base + rng.uniform(-1.0, 1.0, base.size) * frac * (hi - lo),
                         lo, hi)
            cons = compute_constraint_violations(dv, form)
            if cons == [0.0, 0.0]:
                break
            if (attempt + 1) % 25 == 0:
                frac *= 0.5
        else:
            raise RuntimeError(
                f"could not draw a feasible perturbed policy {k} in 500 attempts")
        policies.append({
            "policy_id": k,
            "label": f"perturbed_{k}",
            "dv_values": dv.tolist(),
            "perturb_frac_final": frac,
            "constraint_violations": cons,
        })

    out = {
        "formulation": form,
        "seed": scfg.DETERMINISM_SEED,
        "perturb_frac_initial": scfg.DETERMINISM_PERTURB_FRAC,
        "policies": policies,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


###############################################################################
# Worker (one fresh process per (path, repeat))
###############################################################################

def worker_main() -> int:
    """Evaluate every policy once on one path; write the result JSON.

    Runs in a fresh subprocess whose environment fixes the path's
    trimmed/full switch (``NYCOPT_USE_TRIMMED_MODEL``, exported by the
    driver before ``config`` is imported). Objectives are the annual-unit
    (§2) set in NATURAL units — the same metric code search dispatches to,
    minus Borg's maximize negation (a sign flip cannot change a deviation).

    Returns:
        Process exit code (0 on success).
    """
    task = json.loads(Path(os.environ["NYCOPT_DETERMINISM_TASK"]).read_text())
    path_name: str = task["path"]
    repeat: int = task["repeat"]

    from config import USE_TRIMMED_MODEL
    from src.formulations import get_objective_set
    from src.objectives import _nyc_storage_pct_daily
    from src.simulation import (
        dvs_to_config,
        run_simulation_ensemble_inmemory,
        run_simulation_inmemory,
    )

    use_trimmed = path_name.endswith("_trimmed")
    is_ensemble = path_name.startswith("ensemble")
    if USE_TRIMMED_MODEL != use_trimmed:
        raise RuntimeError(
            f"config.USE_TRIMMED_MODEL={USE_TRIMMED_MODEL} does not match "
            f"path '{path_name}' — the driver must export NYCOPT_USE_TRIMMED_MODEL")

    spec = None
    if is_ensemble:
        from src.ensembles import get_ensemble_spec, staged_ensemble_missing
        slug = scfg.DETERMINISM_ENSEMBLE_SLUG
        missing = staged_ensemble_missing(slug)
        if missing:
            raise RuntimeError(
                f"ensemble fixture '{slug}' is not staged (missing {missing}); "
                "stage it with src/local_test_ensemble.py — this experiment "
                "never generates ensembles")
        spec = get_ensemble_spec(slug)

    obj_set = get_objective_set()
    objectives = list(obj_set)
    policies = json.loads(scfg.determinism_policies_path().read_text())

    result = {
        "path": path_name,
        "repeat": repeat,
        "objective_names": list(obj_set.names),
        "ensemble_slug": scfg.DETERMINISM_ENSEMBLE_SLUG if is_ensemble else None,
        "policies": {},
    }
    for pol in policies["policies"]:
        cfg = dvs_to_config(np.asarray(pol["dv_values"], dtype=float),
                            scfg.DETERMINISM_FORMULATION)
        t0 = time.perf_counter()
        if is_ensemble:
            data_per_real = run_simulation_ensemble_inmemory(cfg, spec)
        else:
            data_per_real = [run_simulation_inmemory(cfg, use_trimmed=use_trimmed)]
        wall_s = time.perf_counter() - t0
        entry = {
            "objectives": [float(o.compute(data_per_real)) for o in objectives],
            "wall_s": wall_s,
        }
        if pol["policy_id"] == 0:
            # State-level series (realization 0): the jitter the objectives
            # must absorb. Full float precision survives the JSON round-trip.
            storage = _nyc_storage_pct_daily(data_per_real[0])
            entry["nyc_storage_pct"] = {
                "start_date": str(storage.index[0].date()),
                "values": [float(v) for v in storage.to_numpy()],
            }
        result["policies"][str(pol["policy_id"])] = entry

    out_file = Path(task["out_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result))
    return 0


###############################################################################
# Driver: task fan-out
###############################################################################

def _run_workers() -> float:
    """Launch one worker subprocess per (path, repeat); return wall seconds.

    Completed runs (an existing, parseable result JSON) are skipped, making
    the driver resumable. Worker stdout/stderr go to a per-task ``.log``
    beside the result JSON.
    """
    scfg.DETERMINISM_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for path_name in scfg.DETERMINISM_PATHS:
        for repeat in range(scfg.DETERMINISM_N_REPEATS):
            out_file = scfg.determinism_run_path(path_name, repeat)
            if out_file.exists():
                try:
                    json.loads(out_file.read_text())
                    print(f"[determinism] {path_name} rep{repeat}: done, skipping")
                    continue
                except json.JSONDecodeError:
                    out_file.unlink()
            task_file = out_file.with_suffix(".task.json")
            task_file.write_text(json.dumps({
                "path": path_name, "repeat": repeat, "out_file": str(out_file),
            }))
            env = dict(os.environ)
            env["NYCOPT_DETERMINISM_TASK"] = str(task_file)
            env["NYCOPT_USE_TRIMMED_MODEL"] = (
                "1" if path_name.endswith("_trimmed") else "0")
            print(f"[determinism] {path_name} rep{repeat}: running ...")
            log_file = out_file.with_suffix(".log")
            with open(log_file, "w") as log:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve())],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_DIR),
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"worker failed for {path_name} rep{repeat} "
                    f"(exit {proc.returncode}); see {log_file}")
            task_file.unlink()
    return time.perf_counter() - t0


###############################################################################
# Aggregation
###############################################################################

def _load_runs() -> dict:
    """Load all per-(path, repeat) result JSONs, keyed by path name."""
    runs: dict = {}
    for path_name in scfg.DETERMINISM_PATHS:
        runs[path_name] = [
            json.loads(scfg.determinism_run_path(path_name, r).read_text())
            for r in range(scfg.DETERMINISM_N_REPEATS)
        ]
    return runs


def _storage_jitter(reps: list) -> dict:
    """Across-repeat jitter of the baseline policy's daily NYC storage series.

    Args:
        reps: The per-repeat result dicts of one path.

    Returns:
        Dict with the daily max-over-repeat-pairs |difference| series and its
        summary stats (max, count and fraction of days with any difference).
    """
    series = np.asarray(
        [r["policies"]["0"]["nyc_storage_pct"]["values"] for r in reps],
        dtype=float,
    )
    daily_span = series.max(axis=0) - series.min(axis=0)
    return {
        "start_date": reps[0]["policies"]["0"]["nyc_storage_pct"]["start_date"],
        "daily_max_absdiff": daily_span,
        "max_absdiff_pct_pts": float(daily_span.max()),
        "n_days_differ": int((daily_span > 0).sum()),
        "frac_days_differ": float((daily_span > 0).mean()),
    }


def aggregate(runs: dict, policies: dict, workers_wall_s: float) -> "tuple":
    """Reduce the raw runs to the summary table, verdict JSON, and jitter data.

    Args:
        runs: Output of :func:`_load_runs`.
        policies: The policy-set dict.
        workers_wall_s: Wall seconds spent in (or skipped by) the worker loop.

    Returns:
        ``(summary_df, summary, jitter)`` — the per (path, policy, objective)
        DataFrame, the verdict dict, and per-path storage-jitter dicts.
    """
    import pandas as pd

    labels = {str(p["policy_id"]): p["label"] for p in policies["policies"]}
    rows = []
    jitter = {}
    for path_name, reps in runs.items():
        names = reps[0]["objective_names"]
        for pid, label in labels.items():
            values = np.asarray(
                [r["policies"][pid]["objectives"] for r in reps], dtype=float)
            for j, obj_name in enumerate(names):
                col = values[:, j]
                abs_dev = float(col.max() - col.min())
                rel_dev = abs_dev / max(abs(float(np.median(col))), _REL_DENOM_FLOOR)
                rows.append({
                    "path": path_name,
                    "policy_id": int(pid),
                    "policy_label": label,
                    "objective": obj_name,
                    "value_median": float(np.median(col)),
                    "value_min": float(col.min()),
                    "value_max": float(col.max()),
                    "max_abs_dev": abs_dev,
                    "max_rel_dev": rel_dev,
                    "n_unique": int(np.unique(col).size),
                    "deterministic": rel_dev <= scfg.DETERMINISM_REL_TOL,
                })
        jitter[path_name] = _storage_jitter(reps)

    summary_df = pd.DataFrame(rows)

    worst = summary_df.loc[summary_df["max_rel_dev"].idxmax()]
    per_path = {}
    for path_name in runs:
        sub = summary_df[summary_df["path"] == path_name]
        j = jitter[path_name]
        per_path[path_name] = {
            "objectives_deterministic": bool(sub["deterministic"].all()),
            "max_abs_dev": float(sub["max_abs_dev"].max()),
            "max_rel_dev": float(sub["max_rel_dev"].max()),
            "n_nondeterministic_cells": int((~sub["deterministic"]).sum()),
            "state_jitter": {
                "series": "daily aggregate NYC storage (% capacity), baseline "
                          "policy, realization 0",
                "max_absdiff_pct_pts": j["max_absdiff_pct_pts"],
                "n_days_differ": j["n_days_differ"],
                "frac_days_differ": j["frac_days_differ"],
            },
            "mean_eval_wall_s": float(np.mean(
                [r["policies"][pid]["wall_s"] for r in runs[path_name]
                 for pid in labels])),
        }

    summary = {
        "verdict_rule": (
            "deterministic iff per-objective across-repeat deviation is exactly "
            f"zero or max relative deviation <= {scfg.DETERMINISM_REL_TOL:g}"),
        "objectives_deterministic_all_paths": bool(summary_df["deterministic"].all()),
        "worst_offender": {
            "path": worst["path"],
            "policy": worst["policy_label"],
            "objective": worst["objective"],
            "max_abs_dev": float(worst["max_abs_dev"]),
            "max_rel_dev": float(worst["max_rel_dev"]),
        },
        "per_path": per_path,
        "config": {
            "n_repeats": scfg.DETERMINISM_N_REPEATS,
            "n_policies": len(labels),
            "paths": list(scfg.DETERMINISM_PATHS),
            "formulation": scfg.DETERMINISM_FORMULATION,
            "ensemble_slug": scfg.DETERMINISM_ENSEMBLE_SLUG,
            "seed": scfg.DETERMINISM_SEED,
            "rel_tol": scfg.DETERMINISM_REL_TOL,
        },
        "workers_wall_s": workers_wall_s,
    }
    return summary_df, summary, jitter


###############################################################################
# Figures (max 2)
###############################################################################

def _floor(values: np.ndarray) -> np.ndarray:
    """Clip zeros up to the log-axis plotting floor."""
    return np.maximum(np.asarray(values, dtype=float), _ZERO_FLOOR)


def make_figures(summary_df, jitter: dict) -> None:
    """Write F1 (per-objective deviations) and F2 (state-vs-objective contrast).

    Args:
        summary_df: The per (path, policy, objective) summary table.
        jitter: Per-path storage-jitter dicts from :func:`aggregate`.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from src.plotting.style import apply_style, save_figure

    apply_style()
    scfg.DETERMINISM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [p for p in scfg.DETERMINISM_PATHS if p in set(summary_df["path"])]
    objectives = list(dict.fromkeys(summary_df["objective"]))
    short = {o: _short_label(o) for o in objectives}

    # --- F1: per-objective max relative deviation, all policies x paths ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(objectives), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(paths))
    for off, path_name in zip(offsets, paths):
        sub = summary_df[summary_df["path"] == path_name]
        vals = [
            _floor(sub[sub["objective"] == o]["max_rel_dev"].to_numpy())
            for o in objectives
        ]
        for xi, v in zip(x + off, vals):
            ax.scatter(np.full(v.size, xi), v, s=22,
                       color=_PATH_COLORS.get(path_name, "gray"),
                       alpha=0.85, linewidths=0,
                       label=path_name if xi == x[0] + off else None)
    ax.axhline(scfg.DETERMINISM_REL_TOL, color="0.35", ls="--", lw=1)
    ax.text(0.02, scfg.DETERMINISM_REL_TOL * 1.5,
            f"verdict threshold ({scfg.DETERMINISM_REL_TOL:g})",
            transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color="0.35")
    eps64 = np.finfo(float).eps
    ax.axhline(eps64, color="0.35", ls=":", lw=1)
    ax.text(0.02, eps64 * 1.5, "float64 eps",
            transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color="0.35")
    if (summary_df["max_rel_dev"] <= 0).any():
        ax.axhline(_ZERO_FLOOR, color="0.8", lw=0.8)
        ax.text(0.98, _ZERO_FLOOR * 1.6, "exact zero (plotted at floor)",
                transform=ax.get_yaxis_transform(),
                fontsize=8, color="0.5", va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(_ZERO_FLOOR / 5, max(1e-6, summary_df["max_rel_dev"].max() * 20))
    ax.set_xticks(x)
    ax.set_xticklabels([short[o] for o in objectives], fontsize=8)
    ax.set_ylabel("Max relative deviation across repeats")
    ax.set_title(
        f"Objective determinism: {scfg.DETERMINISM_N_REPEATS} fresh-process "
        "repeats per policy and path")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    save_figure(fig, scfg.determinism_figure_path("f1_objective_deviation"))
    plt.close(fig)

    # --- F2: state-trajectory jitter vs objective-level agreement ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6))

    for path_name in paths:
        j = jitter[path_name]
        span = _floor(j["daily_max_absdiff"])
        idx = pd.date_range(j["start_date"], periods=span.size, freq="D")
        ax_a.plot(idx, span, lw=0.7, color=_PATH_COLORS.get(path_name, "gray"),
                  label=f"{path_name} (max {j['max_absdiff_pct_pts']:.2e})")
    ax_a.set_yscale("log")
    ax_a.set_ylabel("Across-repeat range of daily\nNYC storage (% capacity points)")
    ax_a.set_title("(a) State-level jitter (baseline policy)")
    ax_a.legend(frameon=False, fontsize=7.5)
    if all(j["max_absdiff_pct_pts"] == 0 for j in jitter.values()):
        ax_a.text(0.5, 0.5, "no state-level jitter measured\n(all repeats "
                  "bitwise identical)", transform=ax_a.transAxes,
                  ha="center", va="center", fontsize=9, color="0.4")

    x = np.arange(len(objectives), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(paths))
    for off, path_name in zip(offsets, paths):
        sub = summary_df[summary_df["path"] == path_name]
        worst_abs = [
            _floor([sub[sub["objective"] == o]["max_abs_dev"].max()])[0]
            for o in objectives
        ]
        ax_b.scatter(x + off, worst_abs, s=26,
                     color=_PATH_COLORS.get(path_name, "gray"), alpha=0.9,
                     linewidths=0, label=path_name)
        jmax = jitter[path_name]["max_absdiff_pct_pts"]
        if jmax > 0:
            ax_b.axhline(jmax, color=_PATH_COLORS.get(path_name, "gray"),
                         ls="--", lw=1, alpha=0.6)
    ax_b.set_yscale("log")
    ax_b.set_ylim(_ZERO_FLOOR / 5, None)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([short[o] for o in objectives], fontsize=7)
    ax_b.set_ylabel("Max absolute deviation across repeats\n(natural units)")
    ax_b.set_title("(b) Objective-level agreement (worst policy;\n"
                   "dashed = that path's state-jitter max)")
    ax_b.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    save_figure(fig, scfg.determinism_figure_path("f2_state_vs_objective"))
    plt.close(fig)


###############################################################################
# Entry
###############################################################################

def driver_main() -> int:
    """Run the full experiment: policies -> workers -> summary -> figures."""
    t0 = time.perf_counter()
    scfg.DETERMINISM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    policies = build_policies()
    print(f"[determinism] {len(policies['policies'])} policies, "
          f"{scfg.DETERMINISM_N_REPEATS} repeats, paths: "
          f"{', '.join(scfg.DETERMINISM_PATHS)}")

    workers_wall_s = _run_workers()
    runs = _load_runs()
    summary_df, summary, jitter = aggregate(runs, policies, workers_wall_s)
    summary["total_wall_s"] = time.perf_counter() - t0

    summary_df.to_csv(scfg.determinism_summary_path("csv"), index=False)
    scfg.determinism_summary_path("json").write_text(json.dumps(summary, indent=2))
    make_figures(summary_df, jitter)

    verdict = ("DETERMINISTIC" if summary["objectives_deterministic_all_paths"]
               else "PROPAGATION DETECTED")
    worst = summary["worst_offender"]
    print(f"[determinism] verdict: {verdict}")
    print(f"[determinism] worst offender: {worst['objective']} on {worst['path']} "
          f"({worst['policy']}): abs {worst['max_abs_dev']:.3e}, "
          f"rel {worst['max_rel_dev']:.3e}")
    print(f"[determinism] summary -> {scfg.determinism_summary_path('csv')}")
    return 0


if __name__ == "__main__":
    if os.environ.get("NYCOPT_DETERMINISM_TASK"):
        raise SystemExit(worker_main())
    raise SystemExit(driver_main())
