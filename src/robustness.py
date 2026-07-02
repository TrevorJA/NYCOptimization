"""
robustness.py - Offline robustness scoring from the persisted re-eval matrix.

The re-eval drivers (``src.reevaluate`` / ``src.reevaluate_mpi``) persist the raw
per-realization base-objective matrix (``reeval_raw.parquet`` + a self-describing
``reeval_raw_meta.json``) rather than a single collapsed satisficing fraction.
This module scores robustness metrics from that matrix *offline*, so different
metrics — which rank solutions differently (McPhail et al. 2018; Giuliani &
Castelletti 2016; Herman et al. 2015; Bonham et al. 2024) — are computed without
re-simulating. The matrix is the McPhail T1×T2×T3 substrate: persisting natural
units preserves enough to recompute any T1 (identity/regret/threshold), T2
(scenario subset), and T3 (moment) composition later.

Implemented now (per project decision):
  - **Univariate satisficing** — fraction of realizations meeting a per-objective
    threshold (reproduces the search-time ``SatisficingAgg`` exactly).
  - **Multivariate (Starr 1962) domain criterion [PRIMARY]** — fraction of
    realizations meeting *all* thresholds jointly (Herman et al. 2015's
    recommended measure; univariate is its per-objective decomposition).
  - **Regret-from-best** — per-realization deviation from the best policy in the
    pooled set, range-normalized so it is dimensionless and cross-objective
    comparable (Herman et al. 2015; Kasprzyk et al. 2013 percent-deviation).
  - **Regret-from-baseline** — deviation from a baseline policy ("vs current
    operations") evaluated on the *same* re-eval ensemble.
  - **Threshold spectrum** — satisficing as a function of the magnitude threshold,
    since robustness depends on the threshold (Hadjimichael et al. 2020).
  - **Ranking-stability** — Kendall τ_b across metrics (McPhail 2020; Bonham 2024).

Composable by design: ``aggregate_over_realizations(values, agg)`` makes maximin
(``agg=min``), Hurwicz, Laplace / expected-value (``agg=mean``) and percentile
metrics thin future wrappers over the same matrix — not implemented here.

Self-describing: thresholds/kinds/directions and the base-objective column order
are read from ``reeval_raw_meta.json`` (snapshotted at simulation time), so
scoring never depends on the live objective registry or a changed
``NYCOPT_SAT_THRESHOLDS`` (the moving-measuring-stick guard, McPhail et al. 2020).
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


###############################################################################
# Loading
###############################################################################

@dataclass
class RawCube:
    """Dense ``(S, R, M)`` re-eval matrix plus its self-describing metadata.

    Attributes:
        cube: float array ``(n_solutions, n_realizations, n_base_objs)`` in
            natural units; NaN where a (solution, realization) is missing/failed.
        solution_ids: length-S solution ids (sorted).
        realization_ids: length-R realization indices (sorted union).
        base_names: length-M base objective names (column order = meta order).
        thresholds: base_name -> satisficing magnitude threshold (or None).
        kinds: base_name -> "ge" | "le" (satisficing direction).
        directions: base_name -> "maximize" | "minimize".
        is_ensemble: False for single-trace re-eval (R == 1, robustness N/A).
        meta: the full parsed ``reeval_raw_meta.json``.
    """

    cube: np.ndarray
    solution_ids: list
    realization_ids: list
    base_names: list
    thresholds: dict
    kinds: dict
    directions: dict
    is_ensemble: bool
    meta: dict

    @property
    def n_realizations(self) -> int:
        return self.cube.shape[1]

    def direction_signs(self) -> np.ndarray:
        """+1 for maximize, -1 for minimize, aligned to ``base_names``."""
        return np.array(
            [1 if self.directions.get(n) == "maximize" else -1
             for n in self.base_names],
            dtype=int,
        )


def _read_long(reeval_dir: Path) -> pd.DataFrame:
    parquet = reeval_dir / "reeval_raw.parquet"
    csvgz = reeval_dir / "reeval_raw.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csvgz.exists():
        return pd.read_csv(csvgz)
    raise FileNotFoundError(
        f"No reeval_raw.parquet or reeval_raw.csv.gz in {reeval_dir}"
    )


def load_raw(reeval_dir) -> RawCube:
    """Load the long-format raw matrix + meta and densify to an ``(S,R,M)`` cube.

    Missing ``(solution, realization, objective)`` cells (failed solutions,
    ragged realization coverage) are NaN-filled so metrics align on the union of
    realization ids — never on positional index.
    """
    reeval_dir = Path(reeval_dir)
    meta_path = reeval_dir / "reeval_raw_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    df = _read_long(reeval_dir)
    base_names = list(meta["base_names"])
    solution_ids = sorted(int(x) for x in df["solution_id"].unique())
    realization_ids = sorted(int(x) for x in df["realization_id"].unique())

    s_ix = {s: i for i, s in enumerate(solution_ids)}
    r_ix = {r: i for i, r in enumerate(realization_ids)}
    o_ix = {o: k for k, o in enumerate(base_names)}

    cube = np.full((len(solution_ids), len(realization_ids), len(base_names)),
                   np.nan, dtype=float)
    for sid, rid, obj, val in zip(
        df["solution_id"], df["realization_id"], df["objective"], df["value"]
    ):
        k = o_ix.get(obj)
        if k is None:
            continue
        cube[s_ix[int(sid)], r_ix[int(rid)], k] = val

    return RawCube(
        cube=cube,
        solution_ids=solution_ids,
        realization_ids=realization_ids,
        base_names=base_names,
        thresholds={k: meta.get("thresholds", {}).get(k) for k in base_names},
        kinds={k: meta.get("kinds", {}).get(k) for k in base_names},
        directions={k: meta.get("directions", {}).get(k) for k in base_names},
        is_ensemble=bool(meta.get("is_ensemble", True)),
        meta=meta,
    )


###############################################################################
# Composable aggregation
###############################################################################

def aggregate_over_realizations(values: np.ndarray,
                                agg: Callable = np.nanmean) -> np.ndarray:
    """Reduce an ``(S, R)`` slice over realizations (axis 1), NaN-safe.

    The single seam that makes risk attitudes composable: ``agg=np.nanmean`` is
    expected value / Laplace, ``agg=np.nanmin`` (on a maximize objective) is
    maximin/Wald, ``lambda v: np.nanpercentile(v, q, axis=1)`` is percentile, and
    a best/worst blend is Hurwicz. Only mean is used by the metrics below; the
    rest are future wrappers.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return agg(values, axis=1)


###############################################################################
# Satisficing (univariate + multivariate Starr domain criterion)
###############################################################################

def _satisfaction_cube(raw: RawCube, thresholds: dict = None,
                       kinds: dict = None) -> np.ndarray:
    """Boolean ``(S, R, M)`` cube: does cell meet its objective's threshold?

    Replicates ``src.objectives_ensemble.SatisficingAgg``: non-finite values are
    **unsatisfied** (a degenerate realization can't masquerade as satisficing).
    """
    thresholds = thresholds if thresholds is not None else raw.thresholds
    kinds = kinds if kinds is not None else raw.kinds
    S, R, M = raw.cube.shape
    sat = np.zeros((S, R, M), dtype=bool)
    for k, name in enumerate(raw.base_names):
        thr = thresholds.get(name)
        kind = kinds.get(name)
        if thr is None or kind is None:
            continue  # leaves column all-False (handled by R==1 N/A gate)
        slab = raw.cube[:, :, k]
        finite = np.isfinite(slab)
        if kind == "ge":
            sat[:, :, k] = finite & (slab >= thr)
        else:
            sat[:, :, k] = finite & (slab <= thr)
    return sat


def satisficing_univariate(raw: RawCube, thresholds: dict = None,
                           kinds: dict = None) -> pd.DataFrame:
    """Per-objective satisficing fraction. Reproduces the search-time metric.

    Returns a ``(S × M)`` DataFrame indexed by solution_id; NaN denominator
    includes missing realizations, so a failed realization counts as unsatisfied.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    frac = sat.mean(axis=1)  # (S, M)
    return pd.DataFrame(
        frac, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"sat_uni__{n}" for n in raw.base_names],
    )


def satisficing_multivariate(raw: RawCube, thresholds: dict = None,
                             kinds: dict = None) -> pd.Series:
    """Starr (1962) domain criterion: fraction of realizations meeting ALL
    thresholds jointly. The PRIMARY robustness measure (Herman et al. 2015).

    A non-finite value in any objective fails that realization's joint criterion,
    mirroring the univariate non-finite-as-unsatisfied rule.
    """
    sat = _satisfaction_cube(raw, thresholds, kinds)
    joint = sat.all(axis=2).mean(axis=1)  # (S,)
    return pd.Series(
        joint, index=pd.Index(raw.solution_ids, name="solution_id"),
        name="sat_multivariate",
    )


###############################################################################
# Regret
###############################################################################

def _safe_nan(reduce: Callable, slab: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reduce(slab, axis=axis)


def regret_from_best(raw: RawCube, normalize: str = "range",
                     agg: Callable = np.nanmean) -> pd.DataFrame:
    """Per-objective regret vs the best policy in the POOLED set, per realization.

    For each realization the "best" is the max (maximize) / min (minimize) over
    solutions, EXCLUDING failed solutions (NaN). Regret is ``best - value``
    (maximize) or ``value - best`` (minimize), so regret ≥ 0. ``normalize``:
      - ``"range"`` (default): divide by the across-solution range in that
        realization → dimensionless [0, 1], cross-objective comparable. Zero range
        (all solutions equal) → 0 regret.
      - ``"best"``: divide by ``|best|`` (Herman/Kasprzyk percent-deviation).
      - ``"none"``: raw natural-unit regret.
    Aggregated over realizations with ``agg`` (default mean; lower is better).
    """
    signs = raw.direction_signs()
    S, R, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    for k in range(M):
        slab = raw.cube[:, :, k]  # (S, R)
        if signs[k] > 0:  # maximize
            best = _safe_nan(np.nanmax, slab, 0)   # (R,)
            worst = _safe_nan(np.nanmin, slab, 0)
            raw_reg = best[None, :] - slab
        else:             # minimize
            best = _safe_nan(np.nanmin, slab, 0)
            worst = _safe_nan(np.nanmax, slab, 0)
            raw_reg = slab - best[None, :]

        if normalize == "range":
            rng = np.abs(worst - best)
            denom = np.where(rng > 0, rng, np.nan)
            norm = raw_reg / denom[None, :]
            # Zero-range realizations: regret is genuinely 0 where value finite.
            norm = np.where(~np.isfinite(norm) & np.isfinite(raw_reg),
                            0.0, norm)
        elif normalize == "best":
            denom = np.where(np.abs(best) > 0, np.abs(best), np.nan)
            norm = raw_reg / denom[None, :]
        else:
            norm = raw_reg
        out[:, k] = aggregate_over_realizations(norm, agg)
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"regret_best__{n}" for n in raw.base_names],
    )


def regret_from_baseline(raw: RawCube, baseline: RawCube,
                         normalize: str = "best",
                         agg: Callable = np.nanmean) -> pd.DataFrame:
    """Per-objective regret vs a baseline policy on the SAME realizations.

    "How much worse than current operations." The baseline cube must carry the
    same objectives; realizations are joined on ``realization_id`` (not position).
    Regret is the directional shortfall vs baseline, clipped at 0 (a policy better
    than baseline has 0 regret), normalized by ``|baseline|`` (``"best"``) or left
    raw (``"none"``). Where the baseline lacks a realization, that cell is NaN.
    """
    signs = raw.direction_signs()
    # Baseline is one policy: collapse its solution axis to a per-realization
    # vector keyed by realization_id.
    base_by_rid = {
        rid: baseline.cube[:, j, :]
        for j, rid in enumerate(baseline.realization_ids)
    }
    S, R, M = raw.cube.shape
    out = np.full((S, M), np.nan)
    # Build aligned baseline matrix (R, M): mean over baseline solutions (usually 1).
    base_aligned = np.full((R, M), np.nan)
    for j, rid in enumerate(raw.realization_ids):
        if rid in base_by_rid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base_aligned[j, :] = np.nanmean(base_by_rid[rid], axis=0)

    for k in range(M):
        slab = raw.cube[:, :, k]            # (S, R)
        bvec = base_aligned[:, k]           # (R,)
        if signs[k] > 0:                    # maximize: worse = lower than baseline
            raw_reg = bvec[None, :] - slab
        else:                               # minimize: worse = higher than baseline
            raw_reg = slab - bvec[None, :]
        raw_reg = np.clip(raw_reg, 0.0, None)
        if normalize == "best":
            denom = np.where(np.abs(bvec) > 0, np.abs(bvec), np.nan)
            norm = raw_reg / denom[None, :]
        else:
            norm = raw_reg
        out[:, k] = aggregate_over_realizations(norm, agg)
    return pd.DataFrame(
        out, index=pd.Index(raw.solution_ids, name="solution_id"),
        columns=[f"regret_baseline__{n}" for n in raw.base_names],
    )


###############################################################################
# Threshold spectrum (Hadjimichael et al. 2020)
###############################################################################

def threshold_spectrum(raw: RawCube, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)
                       ) -> pd.DataFrame:
    """Satisficing fraction as a function of the magnitude threshold.

    Robustness depends on the threshold (Hadjimichael et al. 2020): a solution
    robust at one magnitude can be fragile at a neighbor. For each objective the
    threshold grid is the pooled-distribution quantiles (plus the labeled default
    from the meta), and satisficing is reported at each. Returns a tidy DataFrame
    (solution_id, objective, threshold, is_default, satisficing).
    """
    rows = []
    for k, name in enumerate(raw.base_names):
        kind = raw.kinds.get(name)
        if kind is None:
            continue
        slab = raw.cube[:, :, k]  # (S, R)
        pooled = slab[np.isfinite(slab)]
        if pooled.size == 0:
            continue
        grid = list(np.quantile(pooled, quantiles))
        default = raw.thresholds.get(name)
        labeled = {round(float(g), 6): False for g in grid}
        if default is not None:
            labeled[round(float(default), 6)] = True
        for thr, is_default in sorted(labeled.items()):
            finite = np.isfinite(slab)
            if kind == "ge":
                sat = (finite & (slab >= thr)).mean(axis=1)
            else:
                sat = (finite & (slab <= thr)).mean(axis=1)
            for sid, frac in zip(raw.solution_ids, sat):
                rows.append((sid, name, thr, is_default, float(frac)))
    return pd.DataFrame(
        rows,
        columns=["solution_id", "objective", "threshold",
                 "is_default", "satisficing"],
    )


###############################################################################
# Distributional reporting (Hadjimichael et al. 2023)
###############################################################################

def realization_quantiles(raw: RawCube,
                          quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)
                          ) -> pd.DataFrame:
    """Per-solution per-objective distribution of base values over realizations.

    A scalar scorecard discards the distribution the matrix exists to preserve;
    this keeps it (Hadjimichael et al. 2023). Tidy: (solution_id, objective, qXX...).
    """
    rows = []
    qcols = [f"q{int(q * 100):02d}" for q in quantiles]
    for k, name in enumerate(raw.base_names):
        slab = raw.cube[:, :, k]  # (S, R)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qs = np.nanquantile(slab, quantiles, axis=1)  # (Q, S)
        for si, sid in enumerate(raw.solution_ids):
            rows.append([sid, name] + [float(qs[qi, si])
                                       for qi in range(len(quantiles))])
    return pd.DataFrame(rows, columns=["solution_id", "objective"] + qcols)


###############################################################################
# Ranking stability (McPhail 2020; Bonham 2024)
###############################################################################

def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a, b = a[mask], b[mask]
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b).correlation)
    except Exception:  # noqa: BLE001 - scipy missing -> O(n^2) fallback
        n = len(a)
        num = 0
        for i in range(n):
            for j in range(i + 1, n):
                num += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        denom = n * (n - 1) / 2
        return float(num / denom) if denom else np.nan


def ranking_stability(scorecard: pd.DataFrame,
                      higher_better: dict) -> pd.DataFrame:
    """Kendall τ_b between every pair of metric columns over the solution set.

    Metric rankings disagree (McPhail 2020); this quantifies how much. Columns
    where lower is better (regret) are negated so all are "higher = more robust"
    before correlating. Bonham (2024) treats τ_b ≥ 0.975 as effectively stable.
    """
    cols = list(scorecard.columns)
    mat = scorecard.to_numpy(dtype=float).copy()
    for ci, c in enumerate(cols):
        if not higher_better.get(c, True):
            mat[:, ci] = -mat[:, ci]
    n = len(cols)
    tau = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            tau[i, j] = 1.0 if i == j else _kendall_tau(mat[:, i], mat[:, j])
    return pd.DataFrame(tau, index=cols, columns=cols)


###############################################################################
# Overfitting gap (Brodeur et al. 2020)
###############################################################################

def overfitting_gap(insample: pd.DataFrame, reeval_summary: pd.DataFrame
                    ) -> pd.DataFrame:
    """In-sample minus re-eval satisficing per objective (search overfitting).

    Both frames are indexed by solution_id with the same objective columns
    (search satisficing vs held-out re-eval satisficing). A large positive gap =
    the design overfit its search ensemble (Brodeur et al. 2020). Only computed
    for objective columns present in both.
    """
    shared = [c for c in insample.columns if c in reeval_summary.columns]
    gap = insample[shared].subtract(reeval_summary[shared])
    gap.columns = [f"overfit_gap__{c}" for c in shared]
    return gap


###############################################################################
# Orchestration
###############################################################################

_DEFAULT_METRICS = (
    "satisficing_univariate",
    "satisficing_multivariate",
    "regret_from_best",
    "regret_from_baseline",
)


def score_robustness(raw: RawCube, baseline: Optional[RawCube] = None,
                     metrics=_DEFAULT_METRICS, thresholds: dict = None
                     ) -> tuple[pd.DataFrame, dict]:
    """Assemble the per-solution scorecard for the requested metrics.

    Returns ``(scorecard, higher_better)`` where ``higher_better`` maps each
    column to whether larger = more robust (for ranking-stability orientation).
    Realization-defined satisficing metrics are N/A (NaN) for single-trace
    re-eval (``is_ensemble`` False / R == 1) — a historical-record design is a
    reference, not a controlled robustness comparison.
    """
    pieces = []
    higher_better: dict = {}
    r1 = (not raw.is_ensemble) or raw.n_realizations <= 1

    if "satisficing_univariate" in metrics:
        df = satisficing_univariate(raw, thresholds)
        if r1:
            df[:] = np.nan
        pieces.append(df)
        higher_better.update({c: True for c in df.columns})

    if "satisficing_multivariate" in metrics:
        s = satisficing_multivariate(raw, thresholds)
        if r1:
            s[:] = np.nan
        pieces.append(s.to_frame())
        higher_better[s.name] = True

    if "regret_from_best" in metrics:
        df = regret_from_best(raw)
        pieces.append(df)
        higher_better.update({c: False for c in df.columns})

    if "regret_from_baseline" in metrics:
        if baseline is None:
            warnings.warn("regret_from_baseline requested but no baseline "
                          "provided; skipping.")
        else:
            df = regret_from_baseline(raw, baseline)
            pieces.append(df)
            higher_better.update({c: False for c in df.columns})

    scorecard = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(
        index=pd.Index(raw.solution_ids, name="solution_id"))
    return scorecard, higher_better


def run(reeval_dir, baseline_dir=None, metrics=_DEFAULT_METRICS,
        insample_csv=None) -> Path:
    """Score a re-eval output dir and write the robustness artifacts.

    Writes ``robustness_scorecard.csv`` (+ optional overfitting columns),
    ``robustness_ranking_stability.csv``, ``robustness_threshold_spectrum.csv``,
    and ``robustness_quantiles.csv``. Returns the scorecard path.
    """
    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    baseline = load_raw(baseline_dir) if baseline_dir else None

    scorecard, higher_better = score_robustness(raw, baseline, metrics)

    # Overfitting gap (optional): in-sample search objectives vs re-eval summary.
    if insample_csv:
        insample = pd.read_csv(insample_csv, index_col="solution_id")
        summary_csv = reeval_dir / "objectives_summary.csv"
        if summary_csv.exists():
            reeval_summary = pd.read_csv(summary_csv, index_col="solution_id")
            gap = overfitting_gap(insample, reeval_summary)
            scorecard = scorecard.join(gap)

    out = reeval_dir / "robustness_scorecard.csv"
    scorecard.to_csv(out)

    rank = ranking_stability(
        scorecard[[c for c in scorecard.columns
                   if not c.startswith("overfit_gap__")]],
        higher_better,
    )
    rank.to_csv(reeval_dir / "robustness_ranking_stability.csv")
    threshold_spectrum(raw).to_csv(
        reeval_dir / "robustness_threshold_spectrum.csv", index=False)
    realization_quantiles(raw).to_csv(
        reeval_dir / "robustness_quantiles.csv", index=False)

    print(f"[robustness] scorecard       -> {out}")
    print(f"[robustness] ranking-stability, threshold-spectrum, quantiles "
          f"-> {reeval_dir}")
    return out


def _resolve_default_reeval_dir(formulation: str, seed=None) -> Path:
    from config import (REEVAL_ENSEMBLE_SPEC, active_scenario_name,
                        derive_slug)
    from src.reeval_core import reeval_output_dir
    return reeval_output_dir(active_scenario_name(), derive_slug(formulation),
                             REEVAL_ENSEMBLE_SPEC, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reeval-dir", default=None,
                        help="Re-eval output dir. Default: resolved from config.")
    parser.add_argument("--formulation", default=None,
                        help="Used to resolve --reeval-dir when omitted.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline-dir", default=None,
                        help="Baseline re-eval dir (enables regret-from-baseline).")
    parser.add_argument("--insample-csv", default=None,
                        help="Search-objective CSV for the overfitting gap.")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated metric ids. Default: config "
                             "REEVALUATION_SETTINGS['robustness_metrics'].")
    args = parser.parse_args()

    if args.metrics:
        metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    else:
        from config import REEVALUATION_SETTINGS
        metrics = tuple(REEVALUATION_SETTINGS.get("robustness_metrics",
                                                  _DEFAULT_METRICS))

    if args.reeval_dir:
        reeval_dir = Path(args.reeval_dir)
    elif args.formulation:
        reeval_dir = _resolve_default_reeval_dir(args.formulation, args.seed)
    else:
        parser.error("provide --reeval-dir or --formulation")

    run(reeval_dir, args.baseline_dir, metrics, args.insample_csv)


if __name__ == "__main__":
    main()
