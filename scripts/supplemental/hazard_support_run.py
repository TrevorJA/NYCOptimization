"""hazard_support_run.py - Hazard-support decomposition of the E_test design contrast.

Supplemental diagnostic (docs/notes/methods/hazard_support_decomposition.md; SI
Text S10): decomposes the ``hazard_filling`` - ``fixed_probabilistic`` difference
on E_test by where each state of the world (SOW) sits relative to the stationary
candidate pool's hazard support. Zero simulation - every quantity reduces
persisted artifacts. Two stages, one entry point:

Stage A (policy-free; runs as soon as the regenerated pool and E_test hazard
images exist):

  1. Score every E_test sub-window (125,000 rows of ``hazard_image_subwindows
     .npz``) against each P = 1e6 candidate-pool image on the six campaign
     selection axes, in the selector's own robust p1/p99 range-scaled
     coordinates. The pre-registered primary support statistic is the
     nearest-pool-member distance compared against the q = 0.99 quantile of the
     pool's OWN self-nearest-neighbour distances; the sensitivity is the
     p1/p99 selection-box membership.
  2. Aggregate to per-SOW support scores (``out_frac``), assign the
     pre-declared strata (in_support <= 0.05 < boundary < 0.50 <=
     out_of_support), attribute excursions to axes, and join the forcing
     coordinates theta = (m, r1, r2).
  3. Persist the SOW-level pool coverage deficit in the step-11 rank space
     (the complement of the search-ensemble deficit of
     ``scripts/main/scenario_discovery.py`` - same screened axes, same
     ``cdf_transform`` anchoring, same ``coverage_deficit`` code).

Stage B (consumes the stage-A labels unchanged): re-scores each (design, draw,
seed) run's Starr satisficing fraction and no-harm frequency per support
stratum and per forcing tercile of ``m``, with a SOW-level paired bootstrap CI
on the HF - PS difference. This is a PRE-CAMPAIGN decision instrument: it runs
on whatever re-evaluated cubes exist before the campaign (the go/no-go sets on
the interim ``first10ch`` E_test subset, ``HSD_REEVAL_TAG``) so the
stationary-vs-climate-augmented pool question is settled before search SUs are
spent, never after. Skipped with a message when no matched-design cube exists
on the tag.

All settings live in ``supplemental_config.py`` (``HSD_*``); no CLI value
flags. Smoke mode (``NYCOPT_HSD_SMOKE=1``) uses the P = 2,000 smoke pools and
the first ``HSD_SMOKE_N_SOW`` SOWs, and prefixes every artifact with
``smoke_``.

Run (wrapper: ``workflow/supplemental/hazard_support_decomposition.sh``)::

    python scripts/supplemental/hazard_support_run.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_hsd_env()

import config  # noqa: E402
from src import robustness as rob  # noqa: E402
from src.ensembles import staged_ensemble_dir  # noqa: E402
from src.factor_mapping import (  # noqa: E402
    cdf_transform, screen_hazard_axes, select_compromise,
)
from src.satisficing_criteria import (  # noqa: E402
    active_variant, criterion_by_key, focal_criterion,
)

#: The matched contrast, in (proposed - control) order.
HF_DESIGN = "hazard_filling_stationary"
PS_DESIGN = "fixed_probabilistic"


###############################################################################
# Stage A - pure helpers (tested in tests/test_hazard_support_decomposition.py)
###############################################################################

def robust_bounds(H: np.ndarray, lo_pct: float, hi_pct: float) -> tuple:
    """Per-axis robust range bounds of a hazard image block.

    Args:
        H: ``(n, m)`` raw hazard coordinates.
        lo_pct: Lower percentile (the selector's ROBUST_LO_PCT).
        hi_pct: Upper percentile.

    Returns:
        ``(lo, hi)`` length-``m`` arrays.
    """
    lo = np.percentile(H, lo_pct, axis=0)
    hi = np.percentile(H, hi_pct, axis=0)
    if np.any(hi <= lo):
        bad = np.where(hi <= lo)[0]
        raise ValueError(f"degenerate robust range on axis index {bad.tolist()}")
    return lo, hi


def scale_to_bounds(H: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Linear robust-range scaling, deliberately UNCLIPPED.

    Clipping is a selector construction detail (members beyond the box snap to
    its faces); support scoring must see the excursion beyond the box, so
    values map linearly and out-of-box points land outside [0, 1].
    """
    return (np.asarray(H, dtype=float) - lo[None, :]) / (hi - lo)[None, :]


def self_nn_distances(Z: np.ndarray, workers: int = -1) -> np.ndarray:
    """Each row's Euclidean distance to its nearest OTHER row."""
    d, _ = cKDTree(Z).query(Z, k=2, workers=workers)
    return np.asarray(d, dtype=float)[:, 1]


def nn_distances(Z_query: np.ndarray, Z_ref: np.ndarray,
                 workers: int = -1) -> np.ndarray:
    """Each query row's Euclidean distance to the nearest reference row."""
    d, _ = cKDTree(Z_ref).query(Z_query, k=1, workers=workers)
    return np.asarray(d, dtype=float).ravel()


def sow_aggregate(values: np.ndarray, theta_index: np.ndarray,
                  n_sow: int) -> np.ndarray:
    """Mean of a per-sub-window statistic within each SOW (label join).

    Args:
        values: Length-``n_sub`` per-sub-window values (bools average to a
            fraction).
        theta_index: Length-``n_sub`` SOW ids in ``[0, n_sow)``.
        n_sow: Number of SOWs.

    Returns:
        Length-``n_sow`` per-SOW means; NaN for a SOW with no sub-windows.
    """
    theta_index = np.asarray(theta_index, dtype=int)
    values = np.asarray(values, dtype=float)
    sums = np.bincount(theta_index, weights=values, minlength=n_sow)
    counts = np.bincount(theta_index, minlength=n_sow)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def assign_strata(out_frac: np.ndarray, cuts: tuple,
                  names: tuple) -> np.ndarray:
    """Pre-registered strata from the per-SOW support score.

    ``out_frac <= cuts[0]`` -> ``names[0]``; ``>= cuts[1]`` -> ``names[2]``;
    strictly between -> ``names[1]``. NaN scores raise (a SOW without
    sub-windows is a broken input, not a stratum).
    """
    out_frac = np.asarray(out_frac, dtype=float)
    if np.isnan(out_frac).any():
        raise ValueError("out_frac contains NaN - sub-window coverage is ragged")
    lo, hi = float(cuts[0]), float(cuts[1])
    strata = np.full(out_frac.shape, names[1], dtype=object)
    strata[out_frac <= lo] = names[0]
    strata[out_frac >= hi] = names[2]
    return strata


def tercile_labels(x: np.ndarray, bins: int) -> tuple:
    """Quantile-bin index of each value plus the bin edges (compare_designs
    convention: ``searchsorted`` on interior quantile edges)."""
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    labels = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, bins - 1)
    return labels, edges


def axis_excursions(Z: np.ndarray) -> np.ndarray:
    """Per-axis scaled excursion beyond the [0, 1] robust box, ``(n, m)``.

    Positive values measure how far outside the p1/p99 interval a coordinate
    sits, in range-scaled units; in-box coordinates score 0.
    """
    return np.maximum(0.0, np.maximum(Z - 1.0, -Z))


def paired_design_delta(vec_by_run: dict, idx: np.ndarray) -> float:
    """HF - PS difference of design means on one SOW index set.

    A design's value is the mean over its draws of the mean over each draw's
    seeds of the run's fixed-policy pass fraction on ``idx``. The campaign
    searches one draw per design, so this is the seed mean: the seed is the
    unit of analysis and the contrast is conditional on one draw per design.

    Args:
        vec_by_run: ``{(design, draw, seed): (G,) boolean per-SOW vector}``.
        idx: SOW row indices to score (a stratum, or a bootstrap resample).

    Returns:
        The HF - PS difference, NaN if either design is absent.
    """
    design_vals = {}
    for design in (HF_DESIGN, PS_DESIGN):
        per_draw: dict = {}
        for (d, draw, _seed), vec in vec_by_run.items():
            if d == design:
                per_draw.setdefault(draw, []).append(float(vec[idx].mean()))
        if not per_draw:
            return float("nan")
        design_vals[design] = float(np.mean(
            [np.mean(v) for v in per_draw.values()]))
    return design_vals[HF_DESIGN] - design_vals[PS_DESIGN]


###############################################################################
# Stage A - orchestration
###############################################################################

def _load_subwindow_image() -> dict:
    """The E_test sub-window hazard image, provenance-checked, smoke-sliced."""
    from scengen.hazard_metrics import _SCENARIO_STAMP_START

    path = staged_ensemble_dir(scfg.HSD_ETEST_SLUG) / "hazard_image_subwindows.npz"
    if not path.exists():
        sys.exit(f"[hsd] E_test sub-window hazard image missing: {path}\n"
                 f"[hsd] Run scripts/main/compute_etest_hazard_image.py first.")
    z = np.load(path, allow_pickle=True)
    stamp = str(z["scenario_stamp_start"]) if "scenario_stamp_start" in z else None
    if "reference_start" not in z or stamp != _SCENARIO_STAMP_START:
        sys.exit(f"[hsd] {path} carries a retired date convention "
                 f"(scenario_stamp_start={stamp!r}); regenerate it.")
    img = {
        "H": np.asarray(z["H"], dtype=float),
        "axes": [str(a) for a in z["hazard_axes"]],
        "theta_index": np.asarray(z["theta_index"], dtype=int),
        "window_years": int(z["window_years"]),
    }
    if scfg.HSD_SMOKE:
        keep = img["theta_index"] < scfg.HSD_SMOKE_N_SOW
        img["H"], img["theta_index"] = img["H"][keep], img["theta_index"][keep]
    return img


def _load_pool_image(slug: str) -> dict:
    """A candidate pool's hazard image (provenance validated by the loader)."""
    from scengen.diagnostics import load_hazard_image

    path = staged_ensemble_dir(slug) / "hazard_image.npz"
    if not path.exists():
        sys.exit(f"[hsd] pool hazard image missing: {path}")
    img = load_hazard_image(path)
    return {"H": np.asarray(img["H"], dtype=float),
            "axes": [str(a) for a in img["hazard_axes"]]}


def _crosscheck_selector_bounds(pool_slug: str, sel_axes: list,
                                lo: np.ndarray, hi: np.ndarray) -> float:
    """Max relative difference vs the realized selector's persisted bounds.

    The hazard-filling ensemble built FROM this pool persisted the exact
    normalization the selector used (`_meta.json` ``normalization``). A gross
    mismatch means the wrong pool or a drifted convention - hard error at 1e-3.
    Returns NaN when no matching realized ensemble is staged (smoke pools).
    """
    for hf_slug in scfg.HSD_HAZFILL_SLUGS:
        meta_path = staged_ensemble_dir(hf_slug) / "_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("source_pool") != pool_slug:
            continue
        norm = meta.get("normalization", {})
        if (float(norm.get("lo_pct", -1)), float(norm.get("hi_pct", -1))) != \
                tuple(map(float, scfg.HSD_BOUND_PCT)):
            raise ValueError(
                f"[hsd] {hf_slug} normalization percentiles "
                f"{norm.get('lo_pct')}/{norm.get('hi_pct')} != configured "
                f"{scfg.HSD_BOUND_PCT}")
        rel = 0.0
        for j, axis in enumerate(sel_axes):
            entry = norm.get("axes", {}).get(axis)
            if entry is None:
                raise ValueError(f"[hsd] {hf_slug} normalization lacks {axis}")
            span = float(entry["hi"]) - float(entry["lo"])
            rel = max(rel,
                      abs(lo[j] - float(entry["lo"])) / span,
                      abs(hi[j] - float(entry["hi"])) / span)
        if rel > 1e-3:
            raise ValueError(
                f"[hsd] recomputed p1/p99 bounds for {pool_slug} differ from "
                f"the selector's persisted bounds by {rel:.2e} (rel) - wrong "
                f"pool image or drifted convention.")
        return rel
    return float("nan")


def _theta_table(n_sow: int) -> pd.DataFrame:
    """Per-SOW forcing coordinates (m, em, r1, r2) + tercile of ``m``."""
    from src.plotting.forcing_space import load_etest_sample

    sample = load_etest_sample(staged_ensemble_dir(scfg.HSD_ETEST_SLUG))
    theta, names = sample["theta"], list(sample["theta_names"])
    if sample["n_sow"] < n_sow:
        sys.exit(f"[hsd] staged forcing has {sample['n_sow']} SOWs < {n_sow}")
    theta = theta[:n_sow]
    m = theta[:, names.index(scfg.HSD_TERCILE_AXIS)]
    labels, edges = tercile_labels(m, scfg.HSD_TERCILE_BINS)
    return pd.DataFrame({
        "sow_id": np.arange(n_sow),
        "m": theta[:, names.index("m")],
        "em": np.exp(theta[:, names.index("m")]),
        "r1": theta[:, names.index("r1")],
        "r2": theta[:, names.index("r2")],
        "m_tercile": labels,
    }), edges


def _pool_deficit(etest_full: dict, pool: dict, n_sow: int) -> tuple:
    """SOW-level distance to the nearest POOL member in the step-11 rank space.

    Mirrors ``scenario_discovery`` exactly: axes screened on the
    realization-level E_test image, SOW coordinates = within-SOW mean of the
    realizations' descriptors, both point sets mapped by ``cdf_transform``
    anchored on the SOW-level E_test cloud, ``coverage_deficit`` geometry.

    Args:
        etest_full: Realization-level E_test hazard image dict (``H``, ``axes``,
            ``realization_ids``) restricted to the first ``n_sow`` SOWs.
        pool: Pool hazard image dict.
        n_sow: SOW count.

    Returns:
        ``(deficit, retained_axes)``.
    """
    axes = etest_full["axes"]
    screen = screen_hazard_axes(etest_full["H"], axes)
    idx = screen["retained_idx"]
    rps = etest_full["realizations_per_sow"]
    sow_ids = np.asarray(etest_full["realization_ids"], dtype=int) // rps
    H_sow = np.vstack([
        etest_full["H"][sow_ids == g][:, idx].mean(axis=0) for g in range(n_sow)
    ])
    X_test = cdf_transform(H_sow, H_sow)
    pool_cols = [pool["axes"].index(a) for a in screen["retained"]]
    X_pool = cdf_transform(pool["H"][:, pool_cols], H_sow)
    return nn_distances(X_test, X_pool), screen["retained"]


def _load_etest_realization_image(n_sow: int) -> dict:
    """The realization-level E_test hazard image, restricted to n_sow SOWs."""
    from scengen.diagnostics import load_hazard_image

    path = staged_ensemble_dir(scfg.HSD_ETEST_SLUG) / "hazard_image.npz"
    if not path.exists():
        sys.exit(f"[hsd] E_test hazard image missing: {path}")
    img = load_hazard_image(path)
    forcing = np.load(staged_ensemble_dir(scfg.HSD_ETEST_SLUG)
                      / "forcing_profiles.npz", allow_pickle=True)
    rps = int(forcing["realizations_per_profile"])
    rid = np.asarray(img["realization_ids"], dtype=int)
    keep = (rid // rps) < n_sow
    return {"H": np.asarray(img["H"], dtype=float)[keep],
            "axes": [str(a) for a in img["hazard_axes"]],
            "realization_ids": rid[keep], "realizations_per_sow": rps}


def run_stage_a() -> pd.DataFrame:
    """Compute and persist the support-membership tables; return the SOW table."""
    sel_axes = list(config.HAZARD_SELECTION_AXES)
    sub = _load_subwindow_image()
    sub_cols = [sub["axes"].index(a) for a in sel_axes]
    theta_index = sub["theta_index"]
    n_sow = int(theta_index.max()) + 1
    n_win = int(np.bincount(theta_index).max())
    print(f"[hsd] stage A: {len(theta_index)} sub-windows, {n_sow} SOWs, "
          f"axes={sel_axes}")

    theta_df, tercile_edges = _theta_table(n_sow)
    q_all = (scfg.HSD_SELF_NN_QUANTILE,) + tuple(scfg.HSD_SELF_NN_QUANTILES_SENS)

    sow = theta_df.copy()
    bounds_rows, thr_rows, axis_rows, reach_rows = [], [], [], []
    primary_dominant = None

    etest_real = _load_etest_realization_image(n_sow)

    for k, pool_slug in enumerate(scfg.HSD_POOL_SLUGS):
        pool = _load_pool_image(pool_slug)
        pool_cols = [pool["axes"].index(a) for a in sel_axes]
        Hp = pool["H"][:, pool_cols]
        lo, hi = robust_bounds(Hp, *scfg.HSD_BOUND_PCT)
        rel = _crosscheck_selector_bounds(pool_slug, sel_axes, lo, hi)
        for j, axis in enumerate(sel_axes):
            bounds_rows.append({"pool": pool_slug, "draw": k, "axis": axis,
                                "lo": lo[j], "hi": hi[j],
                                "selector_crosscheck_reldiff": rel})

        Zp = scale_to_bounds(Hp, lo, hi)
        Zs = scale_to_bounds(sub["H"][:, sub_cols], lo, hi)
        d_self = self_nn_distances(Zp)
        d_sub = nn_distances(Zs, Zp)

        for q in q_all:
            thr = float(np.quantile(d_self, q))
            beyond = d_sub > thr
            col = f"out_frac_q{q:g}_d{k}"
            sow[col] = sow_aggregate(beyond, theta_index, n_sow)
            thr_rows.append({"pool": pool_slug, "draw": k, "kind": "self_nn",
                             "q": q, "self_nn_threshold": thr,
                             "n_pool": len(Zp),
                             "subwindows_beyond": int(beyond.sum()),
                             "subwindow_beyond_frac": float(beyond.mean()),
                             "pool_beyond_frac": float("nan")})
            if q == scfg.HSD_SELF_NN_QUANTILE:
                # Excursion attribution among above-threshold sub-windows. A
                # sub-window can exceed the NN threshold while sitting inside
                # the box (an interior void); it gets no dominant axis.
                exc = axis_excursions(Zs)
                dom = np.where(beyond & (exc.max(axis=1) > 0),
                               exc.argmax(axis=1), -1)
                if k == 0:
                    dom_share = np.full((n_sow,), -1, dtype=int)
                    for g in range(n_sow):
                        d_g = dom[(theta_index == g) & (dom >= 0)]
                        dom_share[g] = (np.bincount(d_g).argmax()
                                        if d_g.size else -1)
                    primary_dominant = dom_share
                terc_of_sub = theta_df["m_tercile"].to_numpy()[theta_index]
                for t in range(scfg.HSD_TERCILE_BINS):
                    sel_t = beyond & (terc_of_sub == t)
                    n_t = int(sel_t.sum())
                    for j, axis in enumerate(sel_axes):
                        axis_rows.append({
                            "pool": pool_slug, "draw": k, "m_tercile": t,
                            "axis": axis,
                            "n_beyond": n_t,
                            "dominant_share": (float((dom[sel_t] == j).mean())
                                               if n_t else float("nan")),
                            "mean_excursion": (float(exc[sel_t, j].mean())
                                               if n_t else float("nan")),
                        })

        # Box (D2) sensitivity + the pool's own null calibration.
        box_out_sub = ((Zs < 0.0) | (Zs > 1.0)).any(axis=1)
        sow[f"box_frac_d{k}"] = sow_aggregate(box_out_sub, theta_index, n_sow)
        pool_box_frac = float(((Zp < 0.0) | (Zp > 1.0)).any(axis=1).mean())
        thr_rows.append({"pool": pool_slug, "draw": k, "kind": "box",
                         "q": float("nan"),
                         "self_nn_threshold": float("nan"), "n_pool": len(Zp),
                         "subwindows_beyond": int(box_out_sub.sum()),
                         "subwindow_beyond_frac": float(box_out_sub.mean()),
                         "pool_beyond_frac": pool_box_frac})

        # Reach table (F4): sub-window quantiles by tercile vs the pool band.
        if k == 0:
            terc_of_sub = theta_df["m_tercile"].to_numpy()[theta_index]
            for j, axis in enumerate(sel_axes):
                for t in range(scfg.HSD_TERCILE_BINS):
                    v = sub["H"][terc_of_sub == t, sub_cols[j]]
                    reach_rows.append({
                        "axis": axis, "m_tercile": t, "n_subwindows": len(v),
                        "q50": float(np.quantile(v, 0.50)),
                        "q90": float(np.quantile(v, 0.90)),
                        "q99": float(np.quantile(v, 0.99)),
                        "pool_lo": lo[j], "pool_hi": hi[j],
                    })

        sow[f"pool_deficit_d{k}"], retained = _pool_deficit(
            etest_real, pool, n_sow)
        del pool, Hp, Zp, Zs

    primary_col = f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d0"
    sow["stratum"] = assign_strata(sow[primary_col].to_numpy(),
                                   scfg.HSD_STRATA_CUTS, scfg.HSD_STRATA_NAMES)
    for k in range(1, len(scfg.HSD_POOL_SLUGS)):
        sow[f"stratum_d{k}"] = assign_strata(
            sow[f"out_frac_q{scfg.HSD_SELF_NN_QUANTILE:g}_d{k}"].to_numpy(),
            scfg.HSD_STRATA_CUTS, scfg.HSD_STRATA_NAMES)
    sel_axes_arr = np.asarray(sel_axes, dtype=object)
    sow["dominant_axis"] = np.where(
        primary_dominant >= 0, sel_axes_arr[np.maximum(primary_dominant, 0)],
        "none")

    counts = (sow.groupby("stratum")["sow_id"].count()
              .reindex(list(scfg.HSD_STRATA_NAMES), fill_value=0))
    agree_rows = [{"draw": 0, "pool": scfg.HSD_POOL_SLUGS[0],
                   "agreement_with_d0": 1.0,
                   **{f"n_{s}": int(counts[s]) for s in scfg.HSD_STRATA_NAMES}}]
    for k in range(1, len(scfg.HSD_POOL_SLUGS)):
        ck = (sow[f"stratum_d{k}"].groupby(sow[f"stratum_d{k}"]).count()
              .reindex(list(scfg.HSD_STRATA_NAMES), fill_value=0))
        agree_rows.append({
            "draw": k, "pool": scfg.HSD_POOL_SLUGS[k],
            "agreement_with_d0": float((sow["stratum"]
                                        == sow[f"stratum_d{k}"]).mean()),
            **{f"n_{s}": int(ck[s]) for s in scfg.HSD_STRATA_NAMES}})

    scfg.HSD_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sow.to_csv(scfg.hsd_table_path("hsd_sow_support"), index=False)
    pd.DataFrame(bounds_rows).to_csv(scfg.hsd_table_path("hsd_pool_bounds"),
                                     index=False)
    pd.DataFrame(thr_rows).to_csv(scfg.hsd_table_path("hsd_pool_thresholds"),
                                  index=False)
    pd.DataFrame(axis_rows).to_csv(scfg.hsd_table_path("hsd_axis_excursion"),
                                   index=False)
    pd.DataFrame(reach_rows).to_csv(scfg.hsd_table_path("hsd_reach_by_tercile"),
                                    index=False)
    pd.DataFrame(agree_rows).to_csv(scfg.hsd_table_path("hsd_stratum_counts"),
                                    index=False)

    manifest = {
        "smoke": scfg.HSD_SMOKE, "etest_slug": scfg.HSD_ETEST_SLUG,
        "pool_slugs": list(scfg.HSD_POOL_SLUGS),
        "selection_axes": sel_axes, "bound_pct": list(scfg.HSD_BOUND_PCT),
        "self_nn_quantile": scfg.HSD_SELF_NN_QUANTILE,
        "strata_cuts": list(scfg.HSD_STRATA_CUTS),
        "n_sow": n_sow, "n_subwindows_per_sow": n_win,
        "tercile_axis": scfg.HSD_TERCILE_AXIS,
        "tercile_edges": [float(e) for e in tercile_edges],
        "pool_deficit_retained_axes": list(retained),
        "stratum_counts": {s: int(counts[s]) for s in scfg.HSD_STRATA_NAMES},
    }
    manifest_path = scfg.HSD_TABLES_DIR / (
        ("smoke_" if scfg.HSD_SMOKE else "") + "hsd_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[hsd] stage A done: strata "
          f"{ {s: int(counts[s]) for s in scfg.HSD_STRATA_NAMES} } "
          f"-> {scfg.hsd_table_path('hsd_sow_support')}")
    return sow


###############################################################################
# Stage B - design contrast by stratum (pre-campaign, on the available cubes)
###############################################################################

def _partitions(sow: pd.DataFrame, sow_labels: list) -> dict:
    """``{partition: {group: (G,) boolean mask}}`` aligned to cube SOW labels.

    The join is on the SOW label, never positional. Applies the pre-declared
    fallback merge when a support stratum is under-populated.
    """
    table = sow.set_index("sow_id")
    missing = [g for g in sow_labels if int(g) not in table.index]
    if missing:
        raise KeyError(
            f"{len(missing)} cube SOW(s) have no stage-A support labels "
            f"(e.g. {missing[:5]}); re-run stage A over at least this tag's "
            f"SOW range.")
    aligned = table.loc[[int(g) for g in sow_labels]]
    strata = aligned["stratum"].to_numpy()
    out: dict = {"support_stratum": {}}
    counts = {s: int((strata == s).sum()) for s in scfg.HSD_STRATA_NAMES}
    if min(counts.values()) < scfg.HSD_MIN_STRATUM_SOW:
        merged = np.isin(strata, scfg.HSD_STRATA_NAMES[1:])
        warnings.warn(
            f"[hsd] stratum counts {counts} fall below the pre-declared floor "
            f"({scfg.HSD_MIN_STRATUM_SOW}); applying the two-way fallback "
            f"(in_support vs beyond_support) for the headline.")
        out["support_stratum"][scfg.HSD_STRATA_NAMES[0]] = ~merged
        out["support_stratum"]["beyond_support"] = merged
    else:
        for s in scfg.HSD_STRATA_NAMES:
            out["support_stratum"][s] = strata == s
    terc = aligned["m_tercile"].to_numpy()
    out["m_tercile"] = {f"tercile_{t}": terc == t
                        for t in range(scfg.HSD_TERCILE_BINS)}
    return out


def run_stage_b(sow: pd.DataFrame) -> None:
    """Re-score the design contrast per support stratum and forcing tercile."""
    from scripts.main.compare_designs import discover_runs

    runs = [r for r in discover_runs(scfg.HSD_FORMULATION, scfg.HSD_REEVAL_TAG)]
    have = {d for r in runs for d in [r.design]}
    if not ({HF_DESIGN, PS_DESIGN} <= have):
        print(f"[hsd] stage B skipped: matched-design cubes not found on tag "
              f"'{scfg.HSD_REEVAL_TAG}' (found: {sorted(have) or 'none'}). "
              f"Point NYCOPT_HSD_REEVAL_TAG at a tag both designs were "
              f"re-evaluated on.")
        return
    if not os.environ.get("NYCOPT_REGRET_TAU", "").strip():
        sys.exit("[hsd] stage B needs the ADOPTED regret tolerance: source a "
                 "production env file (workflow/envs/*.env sets "
                 "NYCOPT_REGRET_TAU) - refusing the eps-only fallback ladder.")

    csets = [focal_criterion(), criterion_by_key("reference_all8")]
    print(f"[hsd] stage B: {len(runs)} runs on '{scfg.HSD_REEVAL_TAG}', "
          f"criteria variant '{active_variant()}', sets "
          f"{[c.key for c in csets]}")

    contrast_rows, noharm_rows = [], []
    # Per-SOW boolean vectors of each run's fixed (full-set best) policy,
    # keyed for the bootstrap: {(cset, endpoint): {(design, draw, seed): vec}}.
    fixed_vecs: dict = {}
    partitions = None
    sow_labels_ref = None

    for r in runs:
        raw = rob.load_raw(r.path)
        if partitions is None:
            partitions = _partitions(sow, raw.sow_labels)
            sow_labels_ref = list(raw.sow_labels)
        elif list(raw.sow_labels) != sow_labels_ref:
            raise ValueError(f"[hsd] {r.design} draw={r.draw} seed={r.seed} "
                             f"scores a different SOW set than the first run; "
                             f"strata masks cannot be shared.")
        base = None
        bdir = r.path / "baseline"
        if (bdir / "reeval_raw_meta.json").exists():
            base = rob.load_raw(bdir)

        noharm_sg = None
        if base is not None:
            D = rob.incumbent_advantage(raw, base)
            tau = rob.tau_ladder(raw.obj_names)
            tau_vec = np.array([tau[n] for n in raw.obj_names], dtype=float)
            finite = np.isfinite(D)
            noharm_sg = (finite & (D >= -tau_vec[None, None, :])).all(axis=2)

        for cset in csets:
            thr = cset.thresholds(raw.thresholds, raw.kinds)
            sat_sg = rob._satisfaction_cube(raw, thr).all(axis=2)  # (S, G)
            comp = select_compromise(raw, thresholds=thr)
            s_idx = comp["index"]
            fixed_vecs.setdefault((cset.key, "sat_fixed"), {})[r.key] = \
                sat_sg[s_idx]
            for pname, groups in partitions.items():
                for gname, mask in groups.items():
                    contrast_rows.append({
                        "design": r.design, "draw": r.draw, "seed": r.seed,
                        "criterion_key": cset.key, "partition": pname,
                        "group": gname, "n_sow": int(mask.sum()),
                        "fixed_solution_id": comp["solution_id"],
                        "fixed_frac": float(sat_sg[s_idx, mask].mean()),
                        "best_frac": float(sat_sg[:, mask].mean(axis=1).max()),
                    })
            if cset.key == focal_criterion().key and noharm_sg is not None:
                fixed_vecs.setdefault(("all_axes", "noharm_fixed"), {})[r.key] \
                    = noharm_sg[s_idx]
                for pname, groups in partitions.items():
                    for gname, mask in groups.items():
                        noharm_rows.append({
                            "design": r.design, "draw": r.draw, "seed": r.seed,
                            "partition": pname, "group": gname,
                            "n_sow": int(mask.sum()),
                            "fixed_solution_id": comp["solution_id"],
                            "noharm_fixed": float(noharm_sg[s_idx, mask].mean()),
                            "noharm_best": float(
                                noharm_sg[:, mask].mean(axis=1).max()),
                        })

    pd.DataFrame(contrast_rows).to_csv(
        scfg.hsd_table_path("hsd_stratum_contrast", tagged=True), index=False)
    if noharm_rows:
        pd.DataFrame(noharm_rows).to_csv(
            scfg.hsd_table_path("hsd_stratum_noharm", tagged=True), index=False)

    # SOW-level paired bootstrap on the HF - PS difference, per group.
    rng = np.random.default_rng(scfg.HSD_BOOTSTRAP_SEED)
    boot_rows = []
    for (cset_key, endpoint), vec_by_run in fixed_vecs.items():
        for pname, groups in partitions.items():
            for gname, mask in groups.items():
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue
                point = paired_design_delta(vec_by_run, idx)
                reps = np.array([
                    paired_design_delta(
                        vec_by_run, rng.choice(idx, size=idx.size,
                                               replace=True))
                    for _ in range(scfg.HSD_BOOTSTRAP_N)])
                # Per-draw sign agreement: each HF draw's seed-mean against
                # the PS design mean (draws are not paired across designs).
                hf: dict = {}
                ps: dict = {}
                for (d, draw, _s), vec in vec_by_run.items():
                    tgt = hf if d == HF_DESIGN else ps if d == PS_DESIGN else None
                    if tgt is not None:
                        tgt.setdefault(draw, []).append(float(vec[idx].mean()))
                ps_mean = float(np.mean([np.mean(v) for v in ps.values()])) \
                    if ps else float("nan")
                draw_deltas = [float(np.mean(hf[d]) - ps_mean)
                               for d in sorted(hf)]
                boot_rows.append({
                    "criterion_key": cset_key, "endpoint": endpoint,
                    "partition": pname, "group": gname, "n_sow": int(idx.size),
                    "delta_hf_minus_ps": point,
                    "ci_lo": float(np.nanquantile(reps, 0.025)),
                    "ci_hi": float(np.nanquantile(reps, 0.975)),
                    "n_boot": scfg.HSD_BOOTSTRAP_N,
                    "n_hf_draws_positive": int(sum(d > 0 for d in draw_deltas)),
                    "n_hf_draws": len(draw_deltas),
                })
    pd.DataFrame(boot_rows).to_csv(
        scfg.hsd_table_path("hsd_contrast_bootstrap", tagged=True), index=False)

    # Partition agreement: stratum x tercile cross-tab on the shared SOW set.
    aligned = sow.set_index("sow_id").loc[[int(g) for g in sow_labels_ref]]
    xtab = (pd.crosstab(aligned["stratum"], aligned["m_tercile"])
            .reindex(list(scfg.HSD_STRATA_NAMES), fill_value=0))
    xtab.to_csv(scfg.hsd_table_path("hsd_partition_agreement", tagged=True))

    # Pool-deficit failure association (descriptive; the step-11 complement).
    deficit_rows = []
    pool_deficit = aligned["pool_deficit_d0"].to_numpy(dtype=float)
    focal = focal_criterion()
    for r in runs:
        raw = rob.load_raw(r.path)
        thr = focal.thresholds(raw.thresholds, raw.kinds)
        sat_sg = rob._satisfaction_cube(raw, thr).all(axis=2)
        comp = select_compromise(raw, thresholds=thr)
        y = ~sat_sg[comp["index"]]
        try:
            codes, edges = pd.qcut(pool_deficit, scfg.HSD_DEFICIT_BINS,
                                   labels=False, retbins=True,
                                   duplicates="drop")
        except ValueError:
            continue
        frame = pd.DataFrame({"bin": codes, "deficit": pool_deficit,
                              "fail": y.astype(int)})
        binned = (frame.groupby("bin")
                  .agg(n=("fail", "size"), failure_rate=("fail", "mean"),
                       deficit_mid=("deficit", "median")).reset_index())
        for _, row in binned.iterrows():
            deficit_rows.append({
                "design": r.design, "draw": r.draw, "seed": r.seed,
                "criterion_key": focal.key,
                "solution_id": comp["solution_id"], **row.to_dict()})
    if deficit_rows:
        pd.DataFrame(deficit_rows).to_csv(
            scfg.hsd_table_path("hsd_pool_deficit_failure", tagged=True), index=False)
    print(f"[hsd] stage B done -> "
          f"{scfg.hsd_table_path('hsd_contrast_bootstrap', tagged=True)}")


def main() -> None:
    """Stage A always (reusing persisted tables when present), stage B gated."""
    sow_path = scfg.hsd_table_path("hsd_sow_support")
    if sow_path.exists() and os.environ.get("NYCOPT_HSD_REFRESH", "0") != "1":
        print(f"[hsd] stage A tables present ({sow_path}); reusing "
              f"(NYCOPT_HSD_REFRESH=1 forces recompute).")
        sow = pd.read_csv(sow_path)
    else:
        sow = run_stage_a()
    run_stage_b(sow)


if __name__ == "__main__":
    main()
