"""robustness_threshold_figures.py - Satisficing-threshold placement diagnostics.

Reduces the incumbent FFMP policy's persisted E_test re-eval cube (1,000
theta-SOWs; step 05 ``--reeval``; per-SOW annual-unit objective values) into
the SI threshold-placement evidence for the satisficing thresholds
(``objectives_ensemble._DEFAULT_THRESHOLDS``):

Tables (outputs/supplemental/robustness_threshold_diagnostics/tables/):
  rtd_sow_summary.csv                per-objective quantiles of the per-SOW
                                     value distribution
  rtd_default_stringency.csv         where each current threshold sits (SOW
                                     fraction satisficing + Wilson CI +
                                     stringency coordinate; guardrail margins;
                                     sole/co-failure attribution within the
                                     conjunction)
  rtd_threshold_sweep.csv            tidy dense sweep: satisficing fraction vs
                                     threshold per objective, defaults/candidates
                                     marked
  rtd_candidate_placements.csv       candidate threshold menu with fractions + CIs
  rtd_theta_spearman.csv             (8+3)x(8+3) Spearman: per-SOW objectives +
                                     DU factors (m, r1, r2)
  rtd_historic_anchor_comparison.csv historic-trace annual-unit anchor vs cube
                                     distribution (support flags; near-historic
                                     consistency; persisted baseline CSV as a
                                     same-space cross-check)
  rtd_joint_satisficing.csv          Starr conjunction: observed joint fraction
                                     vs independence / comonotone benchmarks
  rtd_failure_combinations.csv       most frequent failing-criteria combinations
  rtd_failing_count_distribution.csv per-SOW count of simultaneously failing
                                     criteria
  rtd_critical_m.csv                 m at which each criterion's local pass rate
                                     crosses 0.5 (hydrologic reading of the
                                     placement)
  rtd_threshold_recommendation.csv   recommended vector + basis + headline flag
                                     (+ the joint Starr row)

Figures (figures/):
  S_rtd_baseline_sow_cdfs        per-objective ECDFs of the per-SOW values with
                                 thresholds/anchors overlaid
  S_rtd_threshold_sensitivity    satisficing fraction vs threshold (decision
                                 instrument; Wilson band; degeneracy zones)
  S_rtd_theta_spearman           DU-factor attribution (objectives x theta block)
  S_rtd_factor_maps              theta-plane failure maps, all criteria, m x r1
  S_rtd_conjunction              failure-combination frequencies + sole/co-failure
                                 attribution of the Starr conjunction

The SOW is the only scoring unit (each state's unit-years pooled through the
§2 unit operator). The satisfaction rule mirrors ``src.robustness._satisfy``
exactly (inclusive comparison, non-finite = fail); thresholds/kinds come from
the cube's own ``reeval_raw_meta.json`` snapshot, never the live registry. The
historic anchor comes from the JSON cache written by
``robustness_threshold_anchor.py`` — this script never imports pywrdrb.

Figure conventions: no in-panel text annotations; exact values live in the
companion tables, and reference lines are distinguished by style + legend.

Configuration lives in supplemental_config.py (RTD_* section) — no CLI value
flags.

Usage (never on a login node):
    sbatch workflow/supplemental/robustness_threshold_diagnostics.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_rtd_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

import src.robustness as rob  # noqa: E402
from src.pareto_filter import DEFAULT_STAKEHOLDER_FLOORS  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_style, label_for, save_figure,
)

# Okabe-Ito CVD-validated colors, matching the flood-figure conventions.
SOW_COLOR = "#0072B2"        # per-SOW value distribution / pass
FAIL_COLOR = "#D55E00"       # failing SOWs in the factor maps
RESCUED_COLOR = "#009E73"    # fails current threshold, passes recommended
ANCHOR_COLOR = "#CC79A7"     # historic-trace anchor (NOT the fail color: the
                             # anchor is a reference, not a verdict)
QUANTILE_COLOR = "#E69F00"   # SOW-quantile candidates (report-only, rule 4)

#: DU-factor display labels (forcing_parameterization.md: log change-factor
#: harmonic with fixed CMIP6 phases).
THETA_LABELS = {
    "m": "θ m (annual mean, log)",
    "r1": "θ r$_1$ (seasonal amplitude)",
    "r2": "θ r$_2$ (semiannual shape)",
}

#: Compact per-objective tags for the failure-combination axis; everything
#: longer than these turns the conjunction figure into a wall of text.
SHORT_LABELS = {
    "nyc_delivery_reliability_annual": "NYC rel",
    "nyc_delivery_deficit_p99_pct": "NYC def",
    "montague_flow_reliability_annual": "Mon rel",
    "montague_flow_deficit_p99_pct": "Mon def",
    "trenton_flow_reliability_annual": "Tre rel",
    "downstream_flood_exceedance_annual": "Flood",
    "nyc_storage_min_p01_pct": "Storage",
    "nj_delivery_reliability_annual": "NJ rel",
}

FLOOD_OBJ = "downstream_flood_exceedance_annual"

#: ``pareto_filter`` records stakeholder floors by the §1 base/selector name;
#: translate to the annual reporting names so the annual-named cube still
#: picks them up. Local map rather than objectives_ensemble._BASE_TO_ENSEMBLE
#: because importing that module would pull pywrdrb into this script (the
#: anchor cache exists precisely to keep it out).
_FLOOR_NAME_TO_ANNUAL = {
    "nyc_delivery_reliability_weekly": "nyc_delivery_reliability_annual",
}
STAKEHOLDER_FLOORS = {
    _FLOOR_NAME_TO_ANNUAL.get(name, name): val
    for name, val in DEFAULT_STAKEHOLDER_FLOORS.items()
}


###############################################################################
# Pure computation helpers (unit-tested; no I/O)
###############################################################################

def sweep_fractions(values, grid, kind) -> np.ndarray:
    """Satisficing fraction at each grid threshold, mirroring ``_satisfy``.

    Inclusive comparison (``ge``: v >= t; ``le``: v <= t); non-finite values
    count as unsatisfied in the denominator.
    """
    v = np.asarray(values, dtype=float).ravel()
    t = np.asarray(grid, dtype=float).ravel()
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        if kind == "ge":
            sat = finite[None, :] & (v[None, :] >= t[:, None])
        elif kind == "le":
            sat = finite[None, :] & (v[None, :] <= t[:, None])
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return sat.mean(axis=1)


def sweep_fractions_mask(values, thr, kind) -> np.ndarray:
    """Boolean per-value satisfaction at one threshold (same rule as _satisfy)."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        return finite & (v >= thr) if kind == "ge" else finite & (v <= thr)


def sweep_grid(values, extra_points, n_points) -> np.ndarray:
    """Dense threshold grid over the finite support, extended so every value in
    ``extra_points`` (defaults, candidates) lies exactly on a sample."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to build a sweep grid from")
    extras = np.asarray([e for e in extra_points if np.isfinite(e)], dtype=float)
    lo = min(float(v.min()), *(extras.tolist() or [float(v.min())]))
    hi = max(float(v.max()), *(extras.tolist() or [float(v.max())]))
    grid = np.linspace(lo, hi, int(n_points))
    return np.union1d(grid, extras)


def stringency_coordinate(values, thr, kind) -> float:
    """Quantile position of a threshold: the fraction of finite values that fail
    the criterion marginally (``ge``: mean(v < t); ``le``: mean(v > t)) — the
    same convention as ``compare_designs.default_stringency``."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if kind == "ge":
        return float(np.mean(v < thr))
    if kind == "le":
        return float(np.mean(v > thr))
    raise ValueError(f"unknown kind {kind!r}")


def wilson_ci(frac, n, conf=None) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial fraction, vectorized over ``frac``.

    Applied to SOW-unit fractions (n independent LHS forcing draws) — the only
    counting unit under the per-SOW annual-unit substrate.
    """
    conf = scfg.RTD_CI_CONFIDENCE if conf is None else float(conf)
    z = float(norm.ppf(0.5 + conf / 2.0))
    p = np.asarray(frac, dtype=float)
    n = float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - half, centre + half


def satisfaction_matrix(values_by_name, obj_names, thresholds, kinds) -> np.ndarray:
    """``(G, M)`` boolean satisfaction across objectives on the SOW axis."""
    return np.column_stack([
        sweep_fractions_mask(np.asarray(values_by_name[n], dtype=float),
                             float(thresholds[n]), kinds[n])
        for n in obj_names
    ])


def joint_satisficing_stats(sat, obj_names) -> dict:
    """Starr conjunction vs its marginal benchmarks on a ``(G, M)`` matrix.

    The independence benchmark (product of marginals) and the comonotone
    benchmark (min marginal) bracket the joint fraction: observed ≈ min says
    failures nest inside the binding criterion's failure set; observed ≈
    product says they accumulate independently — the two readings of a
    conjunction (cf. regret_tolerance_diagnostics.joint_vs_independent).
    """
    sat = np.asarray(sat, dtype=bool)
    marg = sat.mean(axis=0)
    observed = float(sat.all(axis=1).mean())
    k = int(np.argmin(marg))
    return {
        "joint_frac": observed,
        "independence_benchmark": float(np.prod(marg)),
        "comonotone_benchmark": float(marg.min()),
        "co_occurrence_gap": observed - float(np.prod(marg)),
        "binding_criterion": obj_names[k],
        "binding_marginal_frac": float(marg[k]),
    }


def failure_combinations(sat, obj_names, top_k) -> pd.DataFrame:
    """Frequency of every observed failing-criteria combination.

    Returns the ``top_k`` most frequent failing combinations individually plus
    a pooled ``(other)`` remainder and the all-pass ``(none)`` row. The
    ``criteria`` column keeps the member names as a tuple for plotting; the
    joined string is what the CSV carries.
    """
    sat = np.asarray(sat, dtype=bool)
    n = sat.shape[0]
    counts: dict[tuple, int] = {}
    for row in ~sat:
        key = tuple(int(i) for i in np.nonzero(row)[0])
        counts[key] = counts.get(key, 0) + 1

    none_count = counts.pop((), 0)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rows = []
    for rank, (key, cnt) in enumerate(ordered[:int(top_k)], start=1):
        names = tuple(obj_names[i] for i in key)
        rows.append({"rank": rank, "criteria": names,
                     "failing_criteria": " + ".join(names),
                     "n_failing": len(key), "count": cnt, "frac": cnt / n})
    other = sum(cnt for _, cnt in ordered[int(top_k):])
    if other:
        rows.append({"rank": len(rows) + 1, "criteria": (),
                     "failing_criteria": "(other combinations)",
                     "n_failing": -1, "count": other, "frac": other / n})
    rows.append({"rank": 0, "criteria": (), "failing_criteria": "(none)",
                 "n_failing": 0, "count": none_count, "frac": none_count / n})
    return pd.DataFrame(rows)


def sole_cofailure(sat) -> tuple[np.ndarray, np.ndarray]:
    """Per-criterion fractions of SOWs where it is the SOLE failing criterion
    vs failing alongside at least one other (``(M,)`` arrays; sums to the
    marginal failure fraction)."""
    fail = ~np.asarray(sat, dtype=bool)
    nfail = fail.sum(axis=1)
    sole = (fail & (nfail == 1)[:, None]).mean(axis=0)
    co = (fail & (nfail > 1)[:, None]).mean(axis=0)
    return sole, co


def failing_count_distribution(sat) -> np.ndarray:
    """Fraction of SOWs failing exactly k criteria, k = 0..M (length M+1)."""
    fail = ~np.asarray(sat, dtype=bool)
    return np.bincount(fail.sum(axis=1),
                       minlength=fail.shape[1] + 1) / fail.shape[0]


def critical_m(theta_m, pass_mask, window) -> float:
    """m at which the centered rolling local pass rate crosses 0.5.

    Sorts the SOWs by m, smooths the pass indicator with an odd ``window``,
    and linearly interpolates each 0.5-crossing; the reported boundary is the
    median crossing (robust to noise wiggles). NaN when the criterion is
    degenerate (the smoothed rate never crosses 0.5).
    """
    m = np.asarray(theta_m, dtype=float).ravel()
    p = np.asarray(pass_mask, dtype=float).ravel()
    order = np.argsort(m)
    m, p = m[order], p[order]
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > m.size:
        raise ValueError(f"window {w} exceeds the {m.size} SOWs")
    rate = np.convolve(p, np.ones(w) / w, mode="valid")
    centers = m[(w - 1) // 2: m.size - (w - 1) // 2]
    d = rate - 0.5
    crossings = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    if crossings.size == 0:
        return float("nan")
    xs = []
    for i in crossings:
        x0, x1, y0, y1 = centers[i], centers[i + 1], d[i], d[i + 1]
        xs.append(float(x0) if y1 == y0
                  else float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return float(np.median(xs))


def nearest_to_zero_theta(theta, k) -> np.ndarray:
    """Indices of the ``k`` SOWs closest to theta = 0 (no forcing change),
    per-factor standardized L2 distance."""
    t = np.asarray(theta, dtype=float)
    sd = t.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    d = np.sqrt(((t / sd) ** 2).sum(axis=1))
    return np.argsort(d)[: int(k)]


def theta_for_sows(theta_params, theta_realization_ids, realizations_per_sow,
                   sow_labels) -> np.ndarray:
    """One DU-factor row per SOW, joined on the SOW label (never positional).

    The forcing npz stores one theta row per REALIZATION; realization ``k``
    belongs to SOW ``k // realizations_per_sow`` (``src.reeval_core.sow_grouping``).
    Asserts theta is constant within each SOW block and every cube SOW label
    has at least one theta row.

    Args:
        theta_params: ``(n_realizations, n_theta)`` per-realization DU factors.
        theta_realization_ids: Realization id of each theta row.
        realizations_per_sow: R realizations pooled into each SOW value.
        sow_labels: The cube's SOW labels (ascending).

    Returns:
        ``(n_sow, n_theta)`` array in ``sow_labels`` order.
    """
    theta_params = np.asarray(theta_params, dtype=float)
    r = int(realizations_per_sow)
    if r < 1:
        raise ValueError(f"realizations_per_sow must be >= 1, got {r}")
    by_sow: dict[int, list[int]] = {}
    for row, rid in enumerate(theta_realization_ids):
        by_sow.setdefault(int(rid) // r, []).append(row)

    out = np.full((len(sow_labels), theta_params.shape[1]), np.nan)
    for g, s in enumerate(sow_labels):
        rows = by_sow.get(int(s))
        if not rows:
            raise ValueError(f"SOW {s} is missing from the forcing npz")
        block = theta_params[rows]
        if not np.allclose(block, block[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"theta rows are not constant within SOW {s}")
        out[g] = block[0]
    return out


def candidate_menu(name, kind, sow_values, default_thr, anchor_val, *,
                   floors=None, flood_anchors=None,
                   quantiles=scfg.RTD_CANDIDATE_QUANTILES) -> dict:
    """Candidate threshold placements for one objective.

    current + historic-trace anchor + per-SOW distribution quantiles, plus the
    NYC stakeholder floor and the external flood anchor where they apply.
    """
    floors = STAKEHOLDER_FLOORS if floors is None else floors
    flood_anchors = scfg.RTD_FLOOD_ANCHORS if flood_anchors is None else flood_anchors

    menu = {"current": float(default_thr)}
    if anchor_val is not None and np.isfinite(anchor_val):
        menu["historic_anchor"] = float(anchor_val)
    v = np.asarray(sow_values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    for q in quantiles:
        menu[f"sow_p{int(round(q * 100)):02d}"] = float(np.quantile(v, q))
    if name in floors:
        menu["stakeholder_floor"] = float(floors[name])
    if name == FLOOD_OBJ:
        for key, val in flood_anchors.items():
            menu[f"anchor_{key}"] = float(val)
    return menu


def build_recommendation_table(obj_names, kinds, current_thresholds,
                               sow_values_by_name, recommended, basis,
                               headline_delta) -> pd.DataFrame:
    """Per-objective recommendation summary, plus the joint Starr row.

    ``headline_impact`` flags a recommendation that changes a headline result:
    |delta fraction| > ``headline_delta``, or a degenerate current fraction
    (within RTD_DEGENERACY_LIMIT of 0/1) moving out of degeneracy. NaN
    recommendation columns on pass 1 (``recommended`` empty). The final row is
    the Starr conjunction over all criteria — the metric the thresholds serve.
    """
    lim = scfg.RTD_DEGENERACY_LIMIT
    n_sow = np.asarray(sow_values_by_name[obj_names[0]]).size
    merged = None
    if recommended:
        merged = {n: float(recommended.get(n, current_thresholds[n]))
                  for n in obj_names}

    def _row(name, kind, cur, frac_cur, rec, frac_rec, basis_str):
        degen_cur = frac_cur < lim or frac_cur > 1 - lim
        if rec is None and not np.isfinite(frac_rec):
            headline = False
        else:
            degen_rec = frac_rec < lim or frac_rec > 1 - lim
            headline = (abs(frac_rec - frac_cur) > headline_delta
                        or (degen_cur and not degen_rec))
        lo_c, hi_c = wilson_ci(frac_cur, n_sow)
        lo_r, hi_r = ((np.nan, np.nan) if not np.isfinite(frac_rec)
                      else wilson_ci(frac_rec, n_sow))
        return {
            "objective": name, "kind": kind,
            "current_threshold": np.nan if cur is None else float(cur),
            "frac_sow_at_current": frac_cur,
            "frac_sow_at_current_ci_lo": float(lo_c),
            "frac_sow_at_current_ci_hi": float(hi_c),
            "recommended_threshold": np.nan if rec is None else float(rec),
            "basis": basis_str,
            "frac_sow_at_recommended": frac_rec,
            "frac_sow_at_recommended_ci_lo": float(lo_r),
            "frac_sow_at_recommended_ci_hi": float(hi_r),
            "headline_impact": headline,
        }

    rows = []
    for name in obj_names:
        kind = kinds[name]
        cur = float(current_thresholds[name])
        v = sow_values_by_name[name]
        frac_cur = float(sweep_fractions(v, [cur], kind)[0])
        rec = recommended.get(name)
        frac_rec = (np.nan if merged is None
                    else float(sweep_fractions(v, [merged[name]], kind)[0]))
        row = _row(name, kind, cur, frac_cur, rec, frac_rec, basis.get(name, ""))
        row["stringency_of_current"] = stringency_coordinate(v, cur, kind)
        rows.append(row)

    # The conjunction the thresholds exist to serve (Starr 1962), SOW unit.
    sat_cur = satisfaction_matrix(sow_values_by_name, obj_names,
                                  current_thresholds, kinds)
    joint_cur = float(sat_cur.all(axis=1).mean())
    joint_rec = np.nan
    if merged is not None:
        sat_rec = satisfaction_matrix(sow_values_by_name, obj_names,
                                      merged, kinds)
        joint_rec = float(sat_rec.all(axis=1).mean())
    joint = _row("ALL__joint_starr", "joint", None, joint_cur, None, joint_rec,
                 "Starr conjunction over all criteria")
    joint["stringency_of_current"] = np.nan
    rows.append(joint)
    return pd.DataFrame(rows)


###############################################################################
# Loaders
###############################################################################

def load_cube():
    """Baseline cube: ``(raw, sow_values (n_sow, M), sow_labels)``.

    ``sow_values`` is the per-SOW annual-unit objective matrix itself — the
    substrate carries one value per (SOW, objective), no collapse step.
    """
    raw = rob.load_raw(scfg.RTD_REEVAL_BASELINE_DIR)
    if raw.cube.shape[0] != 1:
        sys.exit(f"[rtd] expected the 1-solution baseline cube, got "
                 f"S={raw.cube.shape[0]}")
    if not raw.is_ensemble or raw.n_sow <= 1:
        sys.exit("[rtd] the baseline cube carries no SOW structure; threshold "
                 "placement needs the E_test re-eval, not a single trace.")
    return raw, raw.cube[0], list(raw.sow_labels)


def load_theta(raw) -> tuple[np.ndarray, list]:
    """DU-factor matrix aligned to the cube's SOW labels; ``(theta, names)``."""
    if not raw.realizations_per_sow:
        sys.exit("[rtd] the cube meta records no realizations_per_sow; cannot "
                 "join theta rows to SOW labels.")
    with np.load(scfg.RTD_FORCING_NPZ) as npz:
        names = [str(n) for n in npz["theta_param_names"]]
        theta = theta_for_sows(npz["theta_params"], npz["realization_ids"],
                               raw.realizations_per_sow, raw.sow_labels)
    if tuple(names) != tuple(scfg.RTD_THETA_NAMES):
        sys.exit(f"[rtd] theta names {names} != expected {scfg.RTD_THETA_NAMES}")
    return theta, names


def load_anchor(obj_names) -> dict:
    """Historic-trace annual-unit anchor from the JSON cache (hard requirement)."""
    if not scfg.RTD_ANCHOR_CACHE.exists():
        sys.exit(
            f"[rtd] missing anchor cache {scfg.RTD_ANCHOR_CACHE} — run\n"
            "    sbatch workflow/supplemental/robustness_threshold_diagnostics.sh\n"
            "(or scripts/supplemental/robustness_threshold_anchor.py on a "
            "compute node) first.")
    with open(scfg.RTD_ANCHOR_CACHE) as f:
        payload = json.load(f)
    anchor = payload["anchor"]
    missing = [n for n in obj_names if n not in anchor]
    if missing:
        sys.exit(f"[rtd] anchor cache missing objectives {missing} — rerun the "
                 "anchor script with NYCOPT_RTD_REFRESH=1")
    return anchor


###############################################################################
# Tables
###############################################################################

def write_sow_summary(obj_names, sow_values_by_name) -> pd.DataFrame:
    qs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    rows = []
    for name in obj_names:
        v = np.asarray(sow_values_by_name[name], dtype=float)
        v = v[np.isfinite(v)]
        row = {"objective": name, "n_sow": v.size,
               "min": v.min(), "mean": v.mean(), "max": v.max()}
        row.update({f"p{int(q * 100):02d}": np.quantile(v, q) for q in qs})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_sow_summary"), index=False)
    return df


def write_default_stringency(raw, obj_names, sow_values_by_name,
                             sat_sow) -> pd.DataFrame:
    """Placement of each current threshold: SOW fraction + Wilson CI,
    stringency, guardrail margin (distance to the worst per-SOW value;
    positive = headroom), and the criterion's role inside the conjunction
    (sole vs co-failure)."""
    sole, co = sole_cofailure(sat_sow)
    rows = []
    for k, name in enumerate(obj_names):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = np.asarray(sow_values_by_name[name], dtype=float)
        frac_sow = float(sweep_fractions(sow_v, [thr], kind)[0])
        lo, hi = wilson_ci(frac_sow, sow_v.size)
        fin = sow_v[np.isfinite(sow_v)]
        margin = (fin.min() - thr) if kind == "ge" else (thr - fin.max())
        iqr = float(np.quantile(fin, 0.75) - np.quantile(fin, 0.25))
        rows.append({
            "objective": name, "kind": kind, "current_threshold": thr,
            "frac_sow": frac_sow,
            "frac_sow_ci_lo": float(lo), "frac_sow_ci_hi": float(hi),
            "stringency_sow": stringency_coordinate(sow_v, thr, kind),
            "margin_worst_natural": float(margin),
            "margin_worst_iqr": float(margin / iqr) if iqr > 0 else np.nan,
            "sole_failure_frac_sow": float(sole[k]),
            "cofailure_frac_sow": float(co[k]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_default_stringency"), index=False)
    return df


def write_threshold_sweep(raw, obj_names, sow_values_by_name, menus) -> pd.DataFrame:
    frames = []
    for name in obj_names:
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = sow_values_by_name[name]
        menu = menus[name]
        grid = sweep_grid(sow_v, list(menu.values()) + [thr],
                          scfg.RTD_SWEEP_POINTS)
        frac_sow = sweep_fractions(sow_v, grid, kind)
        labels = ["" for _ in grid]
        for lab, val in menu.items():
            j = int(np.argmin(np.abs(grid - val)))
            labels[j] = f"{labels[j]}+{lab}" if labels[j] else lab
        frames.append(pd.DataFrame({
            "objective": name, "kind": kind, "threshold": grid,
            "frac_sow": frac_sow,
            "stringency_sow": [stringency_coordinate(sow_v, t, kind)
                               for t in grid],
            "is_default": np.isclose(grid, thr),
            "candidate_label": labels,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(scfg.rtd_table_path("rtd_threshold_sweep"), index=False)
    return df


def write_candidate_placements(raw, obj_names, sow_values_by_name,
                               menus) -> pd.DataFrame:
    rows = []
    for name in obj_names:
        kind = raw.kinds[name]
        sow_v = np.asarray(sow_values_by_name[name], dtype=float)
        for lab, val in menus[name].items():
            frac_sow = float(sweep_fractions(sow_v, [val], kind)[0])
            lo, hi = wilson_ci(frac_sow, sow_v.size)
            rows.append({
                "objective": name, "kind": kind, "candidate": lab,
                "threshold": val,
                "frac_sow": frac_sow,
                "frac_sow_ci_lo": float(lo), "frac_sow_ci_hi": float(hi),
                "stringency_sow": stringency_coordinate(sow_v, val, kind),
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_candidate_placements"), index=False)
    return df


def write_theta_spearman(obj_names, sow_values, theta, theta_names) -> pd.DataFrame:
    mat = np.hstack([sow_values, theta])         # (n_sow, M + n_theta)
    corr = spearmanr(mat).correlation            # (M+3, M+3)
    labels = list(obj_names) + list(theta_names)
    df = pd.DataFrame(corr, index=labels, columns=labels)
    df.to_csv(scfg.rtd_table_path("rtd_theta_spearman"))
    return df


def write_anchor_comparison(raw, obj_names, sow_values_by_name, anchor,
                            theta) -> pd.DataFrame:
    """Historic-trace anchor vs the cube's per-SOW distribution.

    Anchor and cube live in ONE metric space (annual-unit objective values), so
    each row carries the anchor's quantile position, explicit support flags,
    the near-historic consistency check (anchor's percentile within the K SOWs
    closest to theta = 0), and — where the persisted baseline CSV has the
    column — the same-space ``baseline_csv_value`` cross-check.
    """
    near = nearest_to_zero_theta(theta, scfg.RTD_NEAR_HISTORIC_K)
    csv_vals = {}
    if scfg.RTD_BASELINE_ANNUAL_CSV.exists():
        annual = pd.read_csv(scfg.RTD_BASELINE_ANNUAL_CSV)
        csv_vals = {n: float(annual[n].iloc[0]) for n in obj_names
                    if n in annual.columns}
    rows = []
    for name in obj_names:
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        val = float(anchor[name])
        v = np.asarray(sow_values_by_name[name], dtype=float)
        fin = v[np.isfinite(v)]
        passes = val >= thr if kind == "ge" else val <= thr
        in_support = bool(fin.min() <= val <= fin.max())
        dist = 0.0 if in_support else float(min(abs(val - fin.min()),
                                                abs(val - fin.max())))
        v_near = v[near]
        v_near = v_near[np.isfinite(v_near)]
        rows.append({
            "objective": name, "anchor_value": val,
            "baseline_csv_value": csv_vals.get(name, np.nan),
            "current_threshold": thr, "kind": kind,
            "passes_current": bool(passes),
            "sow_quantile_of_anchor": float(np.mean(fin <= val)),
            "in_sow_support": in_support,
            "dist_outside_support": dist,
            "near_historic_quantile": float(np.mean(v_near <= val)),
            "n_near_historic": int(v_near.size),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_historic_anchor_comparison"), index=False)
    return df


def write_joint_satisficing(obj_names, sat_sow_by_vector) -> pd.DataFrame:
    """Starr conjunction summary per threshold vector (SOW unit, Wilson CI)."""
    rows = []
    for vector, sat in sat_sow_by_vector.items():
        stats = joint_satisficing_stats(sat, obj_names)
        row = {"vector": vector, "n_sow": int(sat.shape[0])}
        row.update(stats)
        lo, hi = wilson_ci(stats["joint_frac"], sat.shape[0])
        row["joint_frac_ci_lo"] = float(lo)
        row["joint_frac_ci_hi"] = float(hi)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_joint_satisficing"), index=False)
    return df


def write_failure_combinations(obj_names, sat_sow_by_vector) -> pd.DataFrame:
    """Top failing-criteria combinations + the failing-count distribution
    (SOW unit — the adopted primary)."""
    frames = []
    dist_rows = []
    for vector, sat in sat_sow_by_vector.items():
        comb = failure_combinations(sat, obj_names,
                                    scfg.RTD_TOP_FAILURE_COMBOS)
        comb = comb.drop(columns=["criteria"])
        comb.insert(0, "vector", vector)
        frames.append(comb)
        dist = failing_count_distribution(sat)
        for k, frac in enumerate(dist):
            dist_rows.append({"vector": vector, "n_failing": k,
                              "frac_sow": float(frac)})
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(scfg.rtd_table_path("rtd_failure_combinations"), index=False)
    pd.DataFrame(dist_rows).to_csv(
        scfg.rtd_table_path("rtd_failing_count_distribution"), index=False)
    return df


def write_critical_m(obj_names, sow_values_by_name, theta_m, thresholds_by_vector,
                     kinds) -> pd.DataFrame:
    """Hydrologic reading of each placement: the m at which the local pass rate
    crosses 0.5, per threshold vector. NaN = degenerate (no boundary in the
    E_test box)."""
    rows = []
    for vector, thresholds in thresholds_by_vector.items():
        for name in obj_names:
            v = np.asarray(sow_values_by_name[name], dtype=float)
            ok = sweep_fractions_mask(v, float(thresholds[name]), kinds[name])
            rows.append({
                "vector": vector, "objective": name,
                "pass_frac_sow": float(ok.mean()),
                "m_star": critical_m(theta_m, ok, scfg.RTD_CRITICAL_M_WINDOW),
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_critical_m"), index=False)
    return df


###############################################################################
# Figures (clean-minimal: no in-panel text annotations; values live in tables)
###############################################################################

def _ecdf(v):
    v = np.sort(np.asarray(v, dtype=float))
    v = v[np.isfinite(v)]
    return v, np.arange(1, v.size + 1) / v.size


def _ref_line(ax, val, lo, hi, **style):
    """Reference line clipped to the data axis: a vertical line when ``val``
    lies inside [lo, hi], otherwise a small edge chevron in the same style
    (the exact value lives in the tables)."""
    if lo <= val <= hi:
        ax.axvline(val, **style)
    else:
        edge = hi if val > hi else lo
        marker = ">" if val > hi else "<"
        ax.plot([edge], [0.5], marker=marker, color=style.get("color", "black"),
                ms=6, clip_on=False, zorder=6)


def fig_sow_cdfs(raw, obj_names, sow_values_by_name, anchor) -> None:
    n_sow = raw.n_sow
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    for k, (name, ax) in enumerate(zip(obj_names, axes.ravel())):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = np.asarray(sow_values_by_name[name], dtype=float)
        fin = sow_v[np.isfinite(sow_v)]
        pad = 0.04 * (fin.max() - fin.min())
        xlo, xhi = fin.min() - pad, fin.max() + pad

        # Pass-side shading at the current threshold (clipped to the axis).
        lo_span = max(xlo, thr) if kind == "ge" else xlo
        hi_span = xhi if kind == "ge" else min(xhi, thr)
        if lo_span < hi_span:
            ax.axvspan(lo_span, hi_span, color=SOW_COLOR, alpha=0.06, zorder=0)

        xs, ys = _ecdf(sow_v)
        ax.step(xs, ys, where="post", color=SOW_COLOR, lw=1.6)

        _ref_line(ax, thr, xlo, xhi, color="black", lw=1.2)
        _ref_line(ax, float(anchor[name]), xlo, xhi, color=ANCHOR_COLOR,
                  lw=1.2, ls="--")
        if name in STAKEHOLDER_FLOORS:
            _ref_line(ax, STAKEHOLDER_FLOORS[name], xlo, xhi,
                      color="0.35", lw=1.2, ls=":")
        if name == FLOOD_OBJ:
            for val in scfg.RTD_FLOOD_ANCHORS.values():
                _ref_line(ax, val, xlo, xhi, color="0.35", lw=1.2, ls="-.")

        frac = float(sweep_fractions(sow_v, [thr], kind)[0])
        ax.set_title(f"{label_for(name)}\npass {frac:.2f}", fontsize=9)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-0.02, 1.02)
        if k % 4 == 0:
            ax.set_ylabel("fraction of SOWs ≤ x")
    handles = [
        Line2D([], [], color=SOW_COLOR, lw=1.6,
               label=f"per-SOW value (n={n_sow:,})"),
        Line2D([], [], color="black", lw=1.2, label="current threshold"),
        Line2D([], [], color=ANCHOR_COLOR, lw=1.2, ls="--",
               label="historic-trace anchor"),
        Line2D([], [], color="0.35", lw=1.2, ls=":", label="stakeholder floor"),
        Line2D([], [], color="0.35", lw=1.2, ls="-.",
               label="observed flood anchor"),
        Patch(facecolor=SOW_COLOR, alpha=0.12, label="pass region"),
        Line2D([], [], color="black", marker=">", ls="",
               label="beyond axis (see tables)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Baseline FFMP on $E_{test}$: per-SOW annual-unit objective "
                 "distributions vs satisficing thresholds", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_baseline_sow_cdfs"))
    plt.close(fig)


def _candidate_style(label) -> dict:
    """Marker style per candidate class (legend carries the mapping)."""
    if label == "current":
        return dict(marker="o", color="black", ms=5.5, zorder=6)
    if label == "historic_anchor":
        return dict(marker="D", color=ANCHOR_COLOR, ms=4.5, zorder=5)
    if label.startswith("sow_p"):
        return dict(marker="d", color=QUANTILE_COLOR, ms=4, zorder=4,
                    markerfacecolor="none")
    return dict(marker="^", color="0.35", ms=4.5, zorder=5)  # floor / external


def fig_threshold_sensitivity(raw, obj_names, sow_values_by_name, menus) -> None:
    lim = scfg.RTD_DEGENERACY_LIMIT
    n_sow = raw.n_sow
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    for k, (name, ax) in enumerate(zip(obj_names, axes.ravel())):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = np.asarray(sow_values_by_name[name], dtype=float)
        menu = menus[name]
        grid = sweep_grid(sow_v, list(menu.values()) + [thr],
                          scfg.RTD_SWEEP_POINTS)
        for span in ((-0.04, lim), (1 - lim, 1.04)):
            ax.axhspan(*span, color="0.94", zorder=0)
        frac_sow = sweep_fractions(sow_v, grid, kind)
        lo_b, hi_b = wilson_ci(frac_sow, sow_v.size)
        ax.fill_between(grid, lo_b, hi_b, color=SOW_COLOR, alpha=0.15, lw=0)
        ax.plot(grid, frac_sow, color=SOW_COLOR, lw=1.6)
        for lab, val in menu.items():
            frac = float(sweep_fractions(sow_v, [val], kind)[0])
            ax.plot(val, frac, ls="", **_candidate_style(lab))
        if kind == "le":
            ax.invert_xaxis()  # stringency increases rightward on every panel
        ax.set_title(label_for(name), fontsize=9)
        ax.set_ylim(-0.04, 1.04)
        if k % 4 == 0:
            ax.set_ylabel("satisficing fraction")
        if k >= 4:
            ax.set_xlabel("threshold (natural units)")
    handles = [
        Line2D([], [], color=SOW_COLOR, lw=1.6, label="fraction of SOWs"),
        Patch(facecolor=SOW_COLOR, alpha=0.15,
              label=f"95% Wilson band (n={n_sow:,})"),
        Line2D([], [], color="black", marker="o", ls="", label="current threshold"),
        Line2D([], [], color=ANCHOR_COLOR, marker="D", ls="",
               label="historic-trace anchor"),
        Line2D([], [], color="0.35", marker="^", ls="",
               label="stakeholder floor / external anchor"),
        Line2D([], [], color=QUANTILE_COLOR, marker="d", ls="",
               markerfacecolor="none",
               label="SOW quantile (report-only)"),
        Patch(facecolor="0.94", label="degenerate zone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Satisficing fraction vs threshold placement (stringency "
                 "increases rightward; values in rtd_candidate_placements.csv)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_threshold_sensitivity"))
    plt.close(fig)


def fig_theta_heatmap(corr_df, obj_names, theta_names) -> None:
    """Objectives x DU-factor block only; the full matrix stays in the CSV."""
    block = corr_df.loc[list(obj_names), list(theta_names)].values
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    im = ax.imshow(block, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(theta_names)))
    ax.set_xticklabels([THETA_LABELS.get(n, n) for n in theta_names],
                       rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(obj_names)))
    ax.set_yticklabels([label_for(n) for n in obj_names], fontsize=8)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            v = block[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Baseline per-SOW objectives vs DU forcing factors",
                 fontsize=10)
    fig.tight_layout()
    save_figure(fig, scfg.rtd_figure_path("S_rtd_theta_spearman"))
    plt.close(fig)


def fig_factor_maps(raw, obj_names, sow_values_by_name, theta,
                    theta_names) -> None:
    """Pass/fail at the current thresholds for every criterion, in the single
    informative theta plane (RTD_FACTOR_MAP_PLANE); dashed line = the m at
    which the local pass rate crosses 0.5 (rtd_critical_m.csv)."""
    objectives = [n for n in scfg.RTD_FACTOR_MAP_OBJECTIVES if n in obj_names]
    i = theta_names.index(scfg.RTD_FACTOR_MAP_PLANE[0])
    j = theta_names.index(scfg.RTD_FACTOR_MAP_PLANE[1])
    rec = scfg.RTD_RECOMMENDED_THRESHOLDS
    nrows = (len(objectives) + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(12.6, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(objectives):]:
        ax.set_visible(False)
    for name, ax in zip(objectives, axes.ravel()):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        v = np.asarray(sow_values_by_name[name], dtype=float)
        ok = sweep_fractions_mask(v, thr, kind)
        rescued = np.zeros_like(ok)
        if name in rec:
            ok_rec = sweep_fractions_mask(v, float(rec[name]), kind)
            rescued = ~ok & ok_rec
        ax.scatter(theta[~ok & ~rescued, i], theta[~ok & ~rescued, j],
                   s=6, c=FAIL_COLOR, alpha=0.6, lw=0)
        if rescued.any():
            ax.scatter(theta[rescued, i], theta[rescued, j], s=6,
                       c=RESCUED_COLOR, alpha=0.75, lw=0)
        ax.scatter(theta[ok, i], theta[ok, j], s=6, c=SOW_COLOR,
                   alpha=0.6, lw=0)
        if scfg.RTD_FACTOR_MAP_PLANE[0] == "m":
            m_star = critical_m(theta[:, i], ok, scfg.RTD_CRITICAL_M_WINDOW)
            if np.isfinite(m_star):
                ax.axvline(m_star, color="0.3", lw=1.0, ls="--")
        ax.set_title(f"{label_for(name)}\npass {ok.mean():.2f}", fontsize=8.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(THETA_LABELS[theta_names[i]], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(THETA_LABELS[theta_names[j]], fontsize=8)
    handles = [
        Line2D([], [], color=SOW_COLOR, marker="o", ls="", label="pass"),
        Line2D([], [], color=FAIL_COLOR, marker="o", ls="", label="fail"),
        Line2D([], [], color="0.3", lw=1.0, ls="--",
               label="local pass rate = 0.5"),
    ]
    if rec:
        handles.insert(2, Line2D([], [], color=RESCUED_COLOR, marker="o",
                                 ls="", label="pass at recommended only"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pass/fail at the current thresholds across the DU box",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_factor_maps"))
    plt.close(fig)


def fig_conjunction(obj_names, sat_sow_by_vector) -> None:
    """The Starr conjunction, decomposed: which failing-criteria combinations
    occur (left) and each criterion's sole- vs co-failure role (right)."""
    vectors = list(sat_sow_by_vector)
    fig, axes = plt.subplots(len(vectors), 2,
                             figsize=(11.5, 3.9 * len(vectors)),
                             squeeze=False)
    for r, vector in enumerate(vectors):
        sat = sat_sow_by_vector[vector]
        comb = failure_combinations(sat, obj_names,
                                    scfg.RTD_TOP_FAILURE_COMBOS)
        plot = comb[comb["failing_criteria"] != "(none)"]
        labels = [
            " + ".join(SHORT_LABELS.get(n, n) for n in crit) if crit
            else row_lab
            for crit, row_lab in zip(plot["criteria"],
                                     plot["failing_criteria"])
        ]
        ax = axes[r, 0]
        y = np.arange(len(plot))[::-1]
        ax.barh(y, plot["frac"].values, color=FAIL_COLOR, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("fraction of SOWs", fontsize=8)
        ax.set_title(f"failing-criteria combinations ({vector})", fontsize=9)

        sole, co = sole_cofailure(sat)
        ax = axes[r, 1]
        y = np.arange(len(obj_names))[::-1]
        ax.barh(y, sole, color=FAIL_COLOR, alpha=0.9, label="sole failure")
        ax.barh(y, co, left=sole, color=FAIL_COLOR, alpha=0.4,
                label="co-failure")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_LABELS.get(n, n) for n in obj_names],
                           fontsize=7.5)
        ax.set_xlabel("fraction of SOWs failing", fontsize=8)
        ax.set_title(f"sole vs co-failure per criterion ({vector})", fontsize=9)
        if r == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Starr conjunction decomposition (SOW unit; "
                 "rtd_joint_satisficing.csv)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_conjunction"))
    plt.close(fig)


###############################################################################
# Verification against the shipped re-eval summary
###############################################################################

def verify_against_summary(raw, obj_names) -> None:
    """The SOW-mean of the cube must reproduce objectives_summary.csv.

    ``objectives_summary.csv`` carries ``sowmean__{objective}`` columns — the
    mean over SOWs of the per-SOW annual-unit objective values, derived from
    the SAME matrix at persist time (``src.reeval_core.persist_reeval_raw``).
    Recomputing that mean from the loaded cube and requiring equality
    (atol 1e-9) is the end-to-end check that this script scores the same
    substrate the pipeline persisted.
    """
    path = scfg.RTD_REEVAL_BASELINE_DIR / "objectives_summary.csv"
    if not path.exists():
        print(f"[rtd] WARNING: {path} missing; skipping cross-check", flush=True)
        return
    summary = pd.read_csv(path)
    bad = []
    for k, name in enumerate(obj_names):
        col = f"sowmean__{name}"
        if col not in summary.columns:
            continue
        expected = float(summary[col].iloc[0])
        got = float(np.nanmean(raw.cube[0, :, k]))
        status = "OK" if abs(got - expected) <= 1e-9 else "MISMATCH"
        print(f"[rtd] verify {name}: summary={expected:.5f} "
              f"recomputed={got:.5f} {status}", flush=True)
        if status == "MISMATCH":
            bad.append(name)
    if bad:
        sys.exit(f"[rtd] FAIL: SOW-mean recomputation mismatches shipped "
                 f"summary for {bad}")
    print("[rtd] PASS: cube SOW-means reproduce objectives_summary.csv",
          flush=True)


###############################################################################
# Main
###############################################################################

def main() -> None:
    apply_style()
    for d in (scfg.RTD_TABLES_DIR, scfg.RTD_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw, sow_values, sow_labels = load_cube()
    obj_names = list(raw.obj_names)
    sow_values_by_name = {n: sow_values[:, k] for k, n in enumerate(obj_names)}
    anchor = load_anchor(obj_names)
    theta, theta_names = load_theta(raw)

    verify_against_summary(raw, obj_names)

    menus = {
        n: candidate_menu(n, raw.kinds[n], sow_values_by_name[n],
                          raw.thresholds[n], anchor.get(n))
        for n in obj_names
    }

    # Threshold vectors under study: always the cube's snapshotted currents;
    # plus the recommended vector once pass 2 fills it in (merged over currents
    # so a partial recommendation stays a full vector).
    current_thr = {n: float(raw.thresholds[n]) for n in obj_names}
    thresholds_by_vector = {"current": current_thr}
    rec = {n: float(v) for n, v in scfg.RTD_RECOMMENDED_THRESHOLDS.items()
           if n in obj_names}
    if rec:
        thresholds_by_vector["recommended"] = {**current_thr, **rec}

    sat_sow_by_vector = {
        vec: satisfaction_matrix(sow_values_by_name, obj_names, thr, raw.kinds)
        for vec, thr in thresholds_by_vector.items()
    }

    write_sow_summary(obj_names, sow_values_by_name)
    stringency = write_default_stringency(raw, obj_names, sow_values_by_name,
                                          sat_sow_by_vector["current"])
    write_threshold_sweep(raw, obj_names, sow_values_by_name, menus)
    write_candidate_placements(raw, obj_names, sow_values_by_name, menus)
    corr_df = write_theta_spearman(obj_names, sow_values, theta, theta_names)
    write_anchor_comparison(raw, obj_names, sow_values_by_name, anchor, theta)
    write_joint_satisficing(obj_names, sat_sow_by_vector)
    write_failure_combinations(obj_names, sat_sow_by_vector)
    m_col = theta[:, theta_names.index("m")]
    write_critical_m(obj_names, sow_values_by_name, m_col,
                     thresholds_by_vector, raw.kinds)
    recommendation = build_recommendation_table(
        obj_names, raw.kinds, raw.thresholds, sow_values_by_name,
        scfg.RTD_RECOMMENDED_THRESHOLDS, scfg.RTD_RECOMMENDATION_BASIS,
        scfg.RTD_HEADLINE_IMPACT_DELTA)
    recommendation.to_csv(scfg.rtd_table_path("rtd_threshold_recommendation"),
                          index=False)

    fig_sow_cdfs(raw, obj_names, sow_values_by_name, anchor)
    fig_threshold_sensitivity(raw, obj_names, sow_values_by_name, menus)
    fig_theta_heatmap(corr_df, obj_names, theta_names)
    fig_factor_maps(raw, obj_names, sow_values_by_name, theta, theta_names)
    fig_conjunction(obj_names, sat_sow_by_vector)

    print("\n[rtd] === summary (SOW unit: per-SOW annual-unit objective "
          "values) ===", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 24):
        print(stringency.to_string(index=False), flush=True)
        stage = ("pass 2" if scfg.RTD_RECOMMENDED_THRESHOLDS
                 else "pass 1 — recommendations pending")
        print(f"\n[rtd] recommendation table ({stage}):", flush=True)
        print(recommendation.to_string(index=False), flush=True)
    print(f"\n[rtd] tables  -> {scfg.RTD_TABLES_DIR}", flush=True)
    print(f"[rtd] figures -> {scfg.RTD_FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
